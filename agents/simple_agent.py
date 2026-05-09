import json
import traceback
from typing import Dict, Any, Optional, List

from app.config import call_gpt_json
from .orchestrator_state import SimpleAgentResult


def _normalize(obj: Any) -> Any:
    """Recursively normalize objects to JSON-safe forms."""
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if hasattr(obj, "content"):  # LangChain HumanMessage / AIMessage
        return obj.content
    return obj


class SimpleAgent:
    """
    Simple Agent.

    - Normalizes input data
    - Calls a document analysis tool
    - Uses RAG for content retrieval
    - Augments results using an LLM
    - Produces a consistent output schema
    """

    def __init__(self, document_tool: Optional[Any] = None, rag_tool: Optional[Any] = None):
        self.document_tool = document_tool
        self.rag_tool = rag_tool
        self.version = "1.0"

    # ---------------------------------------------------------------------
    # Core utilities
    # ---------------------------------------------------------------------

    def _call_tool(self, tool: Any, method_names: list, *args, **kwargs) -> Dict[str, Any]:
        """Safely call a tool across multiple possible method names."""
        if not tool:
            return {"error": "tool_not_provided"}

        for name in method_names:
            fn = getattr(tool, name, None)
            if callable(fn):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    return {
                        "error": f"tool_call_failed:{name}",
                        "exception": str(e),
                        "trace": traceback.format_exc(),
                    }

        if callable(tool):  # fallback if tool itself is callable
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
        """Validate the final output schema."""
        required_keys = {
            "version",
            "input_summary",
            "tool_result",
            "llm_analysis",
        }
        missing = required_keys - output.keys()
        if missing:
            output["validation_error"] = f"Missing keys: {sorted(missing)}"
        return output

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data using the document tool and augment results with LLM insights."""
        if not isinstance(input_data, dict):
            return {"error": "input_data_must_be_dict"}

        # Step 1: Normalize input
        normalized_data = _normalize(input_data)
        query = normalized_data.get("query", "")

        # Step 2: Call RAG tool for retrieval if available
        rag_result = {}
        if self.rag_tool and query:
            rag_result = self._call_tool(
                self.rag_tool,
                method_names=["retrieve"],
                query=query,
            )

        # Step 3: Call document tool for analysis
        tool_result = self._call_tool(
            self.document_tool,
            method_names=["analyze_document", "process_document"],
            document=normalized_data,
        )

        # Step 4: Prepare output schema
        output = {
            "version": self.version,
            "input_summary": str(normalized_data)[:200],
            "tool_result": tool_result,
            "rag_result": rag_result,
        }

        # Step 5: Augment with LLM analysis
        context = ""
        if rag_result.get("context"):
            context = f"\n\nRELEVANT DOCUMENT CONTEXT:\n{rag_result['context']}"
        
        llm_prompt = f"""
You are a helpful assistant answering questions about Jewaira's professional background and experience.

USER QUERY:
{query}{context}

Given the relevant context from the document, provide a comprehensive and accurate answer to the user's query.
Make sure to cite the relevant information from the document.
"""

        try:
            llm_response = call_gpt_json(
                llm_prompt,
                schema={
                    "answer": "string",
                    "key_points": "string",
                },
            )
            output["llm_analysis"] = llm_response
        except Exception as e:
            output["llm_analysis"] = {
                "error": "llm_call_failed",
                "exception": str(e),
                "trace": traceback.format_exc(),
            }

        return self._validate_output(output)

    def analyze_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze a batch of inputs."""
        return [self.analyze(item) for item in inputs]

    # ---------------------------------------------------------------------
    # Lifecycle & orchestration helpers
    # ---------------------------------------------------------------------

    def set_document_tool(self, tool: Any) -> None:
        """Safely update the document tool."""
        self.document_tool = tool

    def reset(self) -> None:
        """Reset agent state."""
        self.document_tool = None

    def health_check(self) -> Dict[str, Any]:
        """Health check for orchestrators and monitoring."""
        return {
            "status": "ok",
            "version": self.version,
            "document_tool_present": self.document_tool is not None,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {
            "agent_name": self.__class__.__name__,
            "version": self.version,
        }

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------

    def __getstate__(self):
        """Support pickling / serialization."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Restore state from serialization."""
        self.__dict__.update(state)

    # ---------------------------------------------------------------------
    # Orchestrator integration
    # ---------------------------------------------------------------------

    def to_simpleagent(self) -> SimpleAgentResult:
        """Convert to OrchestratorState SimpleAgentResult representation."""
        from .orchestrator_state import AgentMeta
        return SimpleAgentResult(
            version=self.version,
            meta=AgentMeta(agent="simple_agent", version=self.version),
        )

    def from_simpleagent(self, state: SimpleAgentResult):
        """Load from OrchestratorState SimpleAgentResult representation."""
        self.version = state.version
        # Document tool restoration intentionally not handled here
        return self
