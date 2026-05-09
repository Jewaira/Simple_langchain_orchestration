import json
import traceback
from typing import Dict, Any, Optional, List

from app.config import call_gpt_json


def _normalize(obj: Any) -> Any:
    """Recursively normalize objects to JSON-safe forms."""
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if hasattr(obj, "content"):
        return obj.content
    return obj


def _build_prompt(
    payload: Dict[str, Any],
    document_result: Dict[str, Any],
    user_query: Optional[str] = None,
) -> str:
    """Build a summarization prompt for the LLM."""
    return f"""
You are a professional summarization agent.

USER QUERY:
{json.dumps(user_query, ensure_ascii=False)}

DOCUMENT ANALYSIS RESULT:
{json.dumps(document_result, indent=2, ensure_ascii=False)}

ORIGINAL PAYLOAD:
{json.dumps(payload, indent=2, ensure_ascii=False)}

OBJECTIVE:
Produce a concise, accurate, and structured summary capturing the key points.
"""


class SummaryAgent:
    """
    Summary Agent.

    - Normalizes input data
    - Calls a document analysis tool
    - Uses an LLM to produce a concise summary
    - Produces a consistent output schema
    """

    def __init__(
        self,
        document_tool: Optional[Any] = None,
        llm_tool: Optional[Any] = None,
    ):
        self.document_tool = document_tool
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

    def summarize(self, payload: Dict[str, Any], user_query: Optional[str] = None) -> Dict[str, Any]:
        """Perform summary analysis on the input payload."""
        if not isinstance(payload, dict):
            return {"error": "payload_must_be_dict"}

        normalized_payload = _normalize(payload)

        # Step 1: Document-level summarization / analysis
        document_result = self._call_tool(
            self.document_tool,
            method_names=["summarize_document", "analyze_document"],
            document=normalized_payload,
        )

        # Step 2: Build LLM prompt
        prompt = _build_prompt(
            normalized_payload,
            document_result,
            user_query=user_query,
        )

        # Step 3: Call LLM
        llm_response = self._call_tool(
            self.llm_tool,
            method_names=["generate_response", "call_llm"],
            prompt=prompt,
        )

        # Step 4: Output schema
        output = {
            "version": self.version,
            "input_summary": document_result,
            "llm_analysis": llm_response,
        }

        return self._validate_output(output)

    # ------------------------------------------------------------------
    # Optional helpers (consistent with other agents)
    # ------------------------------------------------------------------

    def set_document_tool(self, tool: Any) -> None:
        self.document_tool = tool

    def set_llm_tool(self, tool: Any) -> None:
        self.llm_tool = tool

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": self.version,
            "document_tool_present": self.document_tool is not None,
            "llm_tool_present": self.llm_tool is not None,
        }

    def analyze_batch(
        self, inputs: List[Dict[str, Any]], user_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [self.summarize(payload, user_query) for payload in inputs]
