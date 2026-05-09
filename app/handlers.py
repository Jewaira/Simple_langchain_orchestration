# handlers.py (with extra RAG debug logging)
import chainlit as cl
import traceback
import logging
import time
import json
from agents.orchestrator_state import OrchestratorState
from agents.orchestrator import build_orchestrator

logger = logging.getLogger("handlers")
logger.setLevel(logging.INFO)

orchestrator = build_orchestrator()


@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content.strip()
    logger.info(" User query: %s", user_query)

    init_state = OrchestratorState(user_query=user_query, last_user_query=user_query)

    simple_text = None
    reasoning_text = None
    summary_text = None
    manual_refs_text = None
    last_state = init_state

    seen_steps = {}
    try:
        async for event in orchestrator.astream(init_state, stream_mode="updates"):
            logger.info(" EVENT RECEIVED: keys=%s", list(event.keys()))
            logger.debug("RAW EVENT: %s", event)

            state_dict = next((val for val in event.values() if isinstance(val, dict)), None)
            if state_dict:
                try:
                    state = OrchestratorState(**state_dict)
                    last_state = state

                    # Supervisor
                    if state.supervisor_explanation and "supervisor" not in seen_steps:
                        await cl.Message(content=" Supervisor: Deciding best agent path… ").send()
                        seen_steps["supervisor"] = time.time()

                    # Simple Agent
                    if state.simple and state.simple.llm_analysis and "simple" not in seen_steps:
                        simple_text = str(state.simple.llm_analysis)
                        await cl.Message(content="Simple Agent: Running query… ⏳").send()
                        seen_steps["sql"] = time.time()

                    # Reasoning
                    if state.reasoning and state.reasoning.summary and "reasoning" not in seen_steps:
                        reasoning_text = state.reasoning.summary.strip()

                        try:
                            reasoning_dict = (
                                state.reasoning.dict() if hasattr(state.reasoning, "dict") else state.reasoning
                            )
                            # 🔹 DEBUG: log full evidence block
                            logger.info("🔎 FULL REASONING EVIDENCE: %s", json.dumps(reasoning_dict.get("evidence", {}), indent=2))

                            ev_manual = reasoning_dict.get("evidence", {}).get("manual")
                            logger.info("🔎 RAW MANUAL EVIDENCE: %s", ev_manual)

                            if ev_manual:
                                if isinstance(ev_manual, str):
                                    manual_refs_text = ev_manual
                                elif isinstance(ev_manual, list):
                                    refs = []
                                    for ref in ev_manual:
                                        logger.info("📖 Found manual reference: %s", ref)
                                        title = ref.get("title", "Reference")
                                        snippet = ref.get("snippet", "")
                                        url = ref.get("url")
                                        if url:
                                            refs.append(f"- [{title}]({url}) — {snippet}")
                                        else:
                                            refs.append(f"- {title}: {snippet}")
                                    manual_refs_text = "\n".join(refs)
                        except Exception as e:
                            logger.warning("⚠️ Could not extract manual evidence: %s", e)

                        await cl.Message(content="🧠 Reasoning Agent: Interpreting results… ⏳").send()
                        seen_steps["reasoning"] = time.time()

                    # Summary Agent
                    if state.summary and state.summary.summary and "summary" not in seen_steps:
                        summary_text = state.summary.summary.strip()
                        await cl.Message(content="📋 Summary Agent: Generating summary… ⏳").send()
                        seen_steps["flight"] = time.time()

                except Exception as ve:
                    logger.error("❌ State validation failed: %s", ve)
                    await cl.Message(content=f"❌ State validation failed: {ve}").send()

            # Show elapsed time when node completes
            if event.get("node"):
                node = event["node"]
                if node in seen_steps:
                    elapsed = time.time() - seen_steps[node]
                    await cl.Message(content=f"✅ **{node} completed** in {elapsed:.2f}s").send()
                else:
                    await cl.Message(content=f"✅ **{node} completed.**").send()

        final_state = last_state or init_state

    except Exception as e:
        logger.exception("❌ Orchestration crashed")
        await cl.Message(
            content=f"❌ Orchestration error: {str(e)}\n\n{traceback.format_exc()}"
        ).send()
        return

    # ✅ Build final structured output
    parts = ["🎯 **Final Answer (All Agents Completed):**\n"]
    if simple_text:
        parts.append(f"📊 **Simple Agent Findings:**\n{simple_text}\n")
    if reasoning_text:
        parts.append(f"🧠 **Reasoning:**\n{reasoning_text}\n")

    if manual_refs_text:
        parts.append(f"📖 **Manual References (RAG):**\n{manual_refs_text}\n")

    if summary_text:
        parts.append(f"🛫 **Flight Analysis:**\n{summary_text}\n")

    final_output = "\n".join(parts).strip()
    if not final_output:
        final_output = "⚠️ No insights could be extracted, but agents ran without error."

    logger.info("🚀 FINAL OUTPUT (cleaned): %s", final_output[:300])
    await cl.Message(content=final_output).send()