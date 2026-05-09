from .base_tool import BaseTool
from app.config import call_gpt_json


class LLMTool(BaseTool):
    name = "llm_tool"

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    def generate_response(self, prompt: str, schema: dict) -> dict:
        if self.provider == "openai":
            return call_gpt_json(prompt, schema=schema)
        else:
            return {"error": "unsupported_llm_provider"}
