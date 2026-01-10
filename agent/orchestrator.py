from app import chain, model
from agent.simple_agent import simple_prompt
from agent.reasoning_agent import reasoning_prompt
from agent.summary_agent import summary_prompt
from langchain_core.output_parsers import StrOutputParser

# Define sub-chains
# 1. Extraction Chain: Extracts key facts
extraction_chain = simple_prompt | model | StrOutputParser()

# 2. Reasoning Chain: Explains logic
reasoning_chain = reasoning_prompt | model | StrOutputParser()

# 3. Summary Chain: Summarizes analysis
summary_chain = summary_prompt | model | StrOutputParser()

def orchestrator(input_text):
    """
    Orchestrates the flow:
    1. Extract key facts (Simple Agent)
    2. Retrieve Context & Answer (RAG Agent / Chain)
    3. Reason about the answer (Reasoning Agent)
    4. Summarize (Summary Agent)
    
    Note: 'chain' from app.py is the RAG chain (Context retrieval + generation)
    """
    
 
    key_facts = extraction_chain.invoke({"input": input_text})
    
    rag_input = f"{input_text}\nKey Context/Facts: {key_facts}"
    rag_result = chain.invoke({"input": rag_input})
    
    # Step 3: Reasoning - Analyze the RAG result
    analysis = reasoning_chain.invoke({"input": rag_result})
    
    # Step 4: Summary - Summarize the analysis
    final_summary = summary_chain.invoke({"analysis": analysis})
    
    return final_summary
