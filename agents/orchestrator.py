import os
import logging
from typing import Any, Dict
from langgraph.graph import StateGraph
from .simple_agent import SimpleAgent
from .reasoning_agent import ReasoningAgent
from .summary_agent import SummaryAgent
from .orchestrator_state import OrchestratorState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools.pdf_tool import DocumentTool
from tools.retrieval_tool import RAGTool

load_dotenv()

logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)

# Initialize tools
document_tool = DocumentTool()
rag_tool = RAGTool()


def has_recent_output(state: OrchestratorState) -> bool:
    """Return True if the last agent in history produced user-visible output."""
    if not state.history:
        return False

    last = state.history[-1]
    agent = last.get("agent")
    if agent == "simple_agent":
        return state.simple is not None
    if agent == "reasoning_agent":
        return state.reasoning is not None
    if agent == "summary_agent":
        return state.summary is not None
    return False


def supervisor_node(state: OrchestratorState) -> OrchestratorState:
    """Supervisor node that decides which agent to invoke next."""
    user_query = state.user_query or state.last_user_query
    if not user_query:
        logger.warning("⚠️ Supervisor invoked with EMPTY user_query")
        state.supervisor_explanation = "no query provided"
        state.next_agent = "end"
        return state

    # Default routing logic - can be enhanced with LLM-based routing
    state.supervisor_explanation = "Routing to simple agent for initial analysis"
    state.next_agent = "simple_agent"
    return state


def build_orchestrator():
    """Build and return the orchestrator graph."""
    graph = StateGraph(OrchestratorState)

    # Initialize agents with tools
    simple_agent = SimpleAgent(document_tool=document_tool, rag_tool=rag_tool)
    reasoning_agent = ReasoningAgent(llm_tool=llm)
    summary_agent = SummaryAgent(document_tool=document_tool, llm_tool=llm)

    # Define nodes
    def simple_node(state: OrchestratorState) -> OrchestratorState:
        result = simple_agent.analyze({"query": state.user_query})
        state.history.append({"agent": "simple_agent", "result": result})
        state.simple = result  # Store result
        return state

    def reasoning_node(state: OrchestratorState) -> OrchestratorState:
        result = reasoning_agent.analyze({"query": state.user_query}, state.user_query)
        state.history.append({"agent": "reasoning_agent", "result": result})
        state.reasoning = result  # Store result
        return state

    def summary_node(state: OrchestratorState) -> OrchestratorState:
        result = summary_agent.summarize({"query": state.user_query}, state.user_query)
        state.history.append({"agent": "summary_agent", "result": result})
        state.summary = result  # Store result
        return state

    # Add nodes to graph
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("simple_agent", simple_node)
    graph.add_node("reasoning_agent", reasoning_node)
    graph.add_node("summary_agent", summary_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Add edges
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.next_agent,
        {
            "simple_agent": "simple_agent",
            "reasoning_agent": "reasoning_agent",
            "summary_agent": "summary_agent",
            "end": "__end__",
        },
    )

    # Default flow: simple -> reasoning -> summary -> end
    graph.add_edge("simple_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", "summary_agent")
    graph.add_edge("summary_agent", "__end__")

    return graph.compile()


def orchestrator(input_text: str) -> str:
    """Main orchestrator function that processes user input."""
    try:
        graph = build_orchestrator()
        initial_state = OrchestratorState(
            user_query=input_text,
            last_user_query=input_text,
        )
        result = graph.invoke(initial_state)

        # Extract final answer from state
        # Try to get summary result
        if result.get("summary"):
            summary = result["summary"]
            if isinstance(summary, dict) and summary.get("llm_analysis"):
                return str(summary["llm_analysis"])
            elif hasattr(summary, "llm_analysis") and summary.llm_analysis:
                return str(summary.llm_analysis)
        
        # Try to get reasoning result
        if result.get("reasoning"):
            reasoning = result["reasoning"]
            if isinstance(reasoning, dict) and reasoning.get("llm_analysis"):
                return str(reasoning["llm_analysis"])
            elif hasattr(reasoning, "llm_analysis") and reasoning.llm_analysis:
                return str(reasoning.llm_analysis)
        
        # Try to get simple result
        if result.get("simple"):
            simple = result["simple"]
            if isinstance(simple, dict) and simple.get("llm_analysis"):
                return str(simple["llm_analysis"])
            elif hasattr(simple, "llm_analysis") and simple.llm_analysis:
                return str(simple.llm_analysis)

        return "I processed your request but couldn't generate a specific answer."
    except Exception as e:
        logger.exception("Orchestrator error")
        return f"Error processing request: {str(e)}"
