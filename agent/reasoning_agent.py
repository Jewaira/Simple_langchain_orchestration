from langchain_core.prompts import PromptTemplate

reasoning_prompt=PromptTemplate.from_template(
    """You are a Reasoning Agent. Based on these input: {input}, 
    explain the 'why' and the logical implications."""
)