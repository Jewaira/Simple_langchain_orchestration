from langchain_core.prompts import PromptTemplate

summary_prompt=PromptTemplate.from_template("""You are a Summary Agent. Take this analysis: {analysis}
and write a concise 2-sentence summary for a busy executive."""
)