import os
import json
import re
import traceback
import logging
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from .orchestrator_state import ReasoningResult

logger = logging.getLogger("reasoning_agent")
logger.setLevel(logging.INFO)

load_dotenv()


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON safely from LLM output."""
    try:
        m = re.search(r"BEGIN_JSON(.*?)END_JSON", text, re.S | re.I)
        if m:
            return json.loads(m.group(1).strip())

        m2 = re.search(r"\{.*\}", text, re.S)
        if m2:
            return json.loads(m2.group(0))
    except Exception:
        return None
    return None


def _build_prompt(payload: Dict[str, Any], user_query: Optional[str] = None) -> str:
    """Build a deep reasoning prompt for the LLM."""
    summary = payload or {}

    return f"""
You are an Operations Intelligence Analyst.

Your mission is to produce deep, evidence-based reasoning,
combining analytics, historical trends, and operational guidance.

---

USER QUERY:
{json.dumps(user_query, ensure_ascii=False)}

AVAILABLE INPUTS:
{json.dumps(summary, indent=2, ensure_ascii=False)}

---

OBJECTIVE:
Provide a cohesive, technically sound explanation of *why* the event occurred.
If data is incomplete, synthesize a realistic and professional explanation.
"""


def _normalize(obj: Any) -> Any:
    """Recursively normalize objects to JSON-safe forms."""
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if hasattr(obj, "content"):
        return obj.content
    return obj


class ReasoningAgent:
    """
    Reasoning Agent:
    - Normalizes input data
    - Builds advanced reasoning prompts
    - Calls an LLM for deep analysis
    - Produces a consistent output schema
    """

    def __init__(self, llm_tool: Optional[Any] = None):
        self.llm_tool = llm_tool
        self.version = "1.0"

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    def _call_tool(self, tool: Any, method_names: list, *args, **kwargs) -> Dict[str, Any]:
        """Safely call a tool across multiple possible method names."""
        if not tool:
            return {"error": "tool_not_provided"}

        # Add 'invoke' for LangChain compatibility
        extended_methods = method_names + ["invoke"]

        for name in extended_methods:
            fn = getattr(tool, name, None)
            if callable(fn):
                try:
                    # For LangChain LLMs, invoke takes a string or messages
                    if name == "invoke" and "prompt" in kwargs:
                        result = fn(kwargs["prompt"])
                        # Handle LangChain response (AIMessage or string)
                        if hasattr(result, "content"):
                            return {"response_text": result.content, "text": result.content}
                        return {"response_text": str(result), "text": str(result)}
                    return fn(*args, **kwargs)
                except Exception as e:
                    return {
                        "error": f"tool_call_failed:{name}",
                        "exception": str(e),
                        "trace": traceback.format_exc(),
                    }

        if callable(tool):
            try:
                return tool(*args, **kwargs)
            except Exception as e:
                return {
                    "error": "tool_callable_failed",
                    "exception": str(e),
                    "trace": traceback.format_exc(),
                }

        return {"error": "no_callable_method_found"}

    def _validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final output schema."""
        required_keys = {"version", "input_summary", "llm_analysis"}
        missing = required_keys - output.keys()
        if missing:
            output["validation_error"] = f"Missing keys: {sorted(missing)}"
        return output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, payload: Dict[str, Any], user_query: Optional[str] = None) -> Dict[str, Any]:
        """Perform reasoning analysis on the input payload."""
        if not isinstance(payload, dict):
            return {"error": "payload_must_be_dict"}

        normalized_payload = _normalize(payload)

        prompt = _build_prompt(normalized_payload, user_query=user_query)

        llm_response = self._call_tool(
            self.llm_tool,
            method_names=["generate_response", "call_llm"],
            prompt=prompt,
        )

        response_text = llm_response.get("response_text") or llm_response.get("text", "")
        llm_analysis = _safe_json_extract(response_text)

        if llm_analysis is None:
            llm_analysis = {
                "error": "llm_response_not_json",
                "raw_response": response_text[:500],
            }

        output = {
            "version": self.version,
            "input_summary": normalized_payload,
            "llm_analysis": llm_analysis,
        }

        return self._validate_output(output)

    def analyze_batch(
        self, inputs: List[Dict[str, Any]], user_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple payloads."""
        return [self.analyze(payload, user_query) for payload in inputs]

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def set_llm_tool(self, tool: Any) -> None:
        self.llm_tool = tool

    def reset(self) -> None:
        self.llm_tool = None

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": self.version,
            "llm_tool_present": self.llm_tool is not None,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_name": self.__class__.__name__,
            "version": self.version,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)