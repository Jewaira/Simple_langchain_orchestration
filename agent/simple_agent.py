
from langchain_core.prompts import PromptTemplate
simple_prompt=PromptTemplate.from_template(""""You are a Query Agent. Find the key facts in this request: {input}

"""
)