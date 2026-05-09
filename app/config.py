import os
import json
import traceback
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

OPEN_AI_API_KEY=os.getenv("OPEN_AI_API_KEY")




client = OpenAI()

def call_gpt_json(
    prompt: str,
    schema: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Call OpenAI GPT model and enforce strict JSON output.
    Schema guides expected keys with lightweight validation.
    """

    system_prompt = f"""
You are a strict JSON generator.

Rules:
- Respond ONLY with valid JSON
- Do NOT include explanations, markdown, or extra text
- Output must be a single JSON object
- Keys MUST match exactly

Expected JSON keys:
{list(schema.keys())}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()

    except Exception as e:
        return {
            "error": "llm_invoke_failed",
            "detail": str(e),
            "trace": traceback.format_exc(),
        }

    # -----------------------------
    # Safe JSON parse
    # -----------------------------
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {
            "error": "invalid_json",
            "raw_output": raw_text,
            "parsed": {k: None for k in schema.keys()},
        }

    # -----------------------------
    # Lightweight schema validation
    # -----------------------------
    validated: Dict[str, Any] = {}

    for key, typ in schema.items():
        val = parsed.get(key)

        if val is None:
            validated[key] = None

        elif typ == "string":
            validated[key] = str(val)

        elif typ == "int":
            try:
                validated[key] = int(val)
            except Exception:
                validated[key] = None

        elif typ == "list":
            validated[key] = val if isinstance(val, list) else [val]

        elif typ == "dict":
            validated[key] = val if isinstance(val, dict) else {}

        else:
            validated[key] = val

    return validated