# orchestrator_state.py

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Base agent result (shared pattern)
# ---------------------------------------------------------------------

class AgentMeta(BaseModel):
    agent: str
    version: str


class BaseAgentResult(BaseModel):
    version: str
    input_summary: Optional[Any] = None
    error: Optional[str] = None
    trace: Optional[str] = None
    meta: AgentMeta


# ---------------------------------------------------------------------
# Simple Agent (document + LLM analysis)
# ---------------------------------------------------------------------

class SimpleAgentResult(BaseAgentResult):
    tool_result: Optional[Dict[str, Any]] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    meta: AgentMeta = Field(
        default_factory=lambda: AgentMeta(agent="simple_agent", version="1.0")
    )


# ---------------------------------------------------------------------
# Summary Agent
# ---------------------------------------------------------------------

class SummaryResult(BaseAgentResult):
    summary: Optional[str] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    meta: AgentMeta = Field(
        default_factory=lambda: AgentMeta(agent="summary_agent", version="1.0")
    )


# ---------------------------------------------------------------------
# RAG Agent (manual / document retrieval)
# ---------------------------------------------------------------------

class RAGResult(BaseModel):
    summary: Optional[str] = None
    references: List[Dict[str, Any]] = Field(default_factory=list)
    coverage: Optional[str] = Field(
        None, description="direct | partial | none"
    )
    error: Optional[str] = None
    meta: AgentMeta = Field(
        default_factory=lambda: AgentMeta(agent="rag_agent", version="1.0")
    )


# ---------------------------------------------------------------------
# Reasoning Agent
# ---------------------------------------------------------------------

class ReasoningResult(BaseAgentResult):
    summary: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[str] = None
    root_causes: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    llm_analysis: Optional[Dict[str, Any]] = None
    meta: AgentMeta = Field(
        default_factory=lambda: AgentMeta(agent="reasoning_agent", version="1.0")
    )


# ---------------------------------------------------------------------
# Orchestrator State
# ---------------------------------------------------------------------

class OrchestratorState(BaseModel):
    # User context
    user_query: Optional[str] = None
    last_user_query: Optional[str] = None

    # Orchestration decisions
    supervisor_explanation: Optional[str] = None
    next_agent: Optional[str] = None

    # Agent outputs
    simple: Optional[SimpleAgentResult] = None
    summary: Optional[SummaryResult] = None
    rag: Optional[RAGResult] = None
    reasoning: Optional[ReasoningResult] = None

    # Execution history & raw data
    history: List[Dict[str, Any]] = Field(default_factory=list)
    raw_flight_data: Dict[str, Any] = Field(default_factory=dict)
