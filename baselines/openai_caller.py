"""GPT-4o-mini baseline via the OpenAI SDK.

Uses the same SYSTEM_PROMPT as every other baseline (fair comparison).
Temperature 0 for determinism, max_tokens 512.

We do NOT use JSON-object mode — the schema is a JSON ARRAY and JSON
mode forces an object. Relying on the system prompt's "JSON only"
instruction; schema_validity_rate is the metric that surfaces any failures.

Pricing constants are USD per 1M tokens. Update if rates shift; current
values are for end-of-2025 GPT-4o-mini list price. The eval report
records the constants used so future re-runs are auditable.
"""

from __future__ import annotations

import os

from baselines.base import Predictor, Pricing
from inference.prompts import SYSTEM_PROMPT

# Public list pricing — update when rates change. Recorded in eval output.
GPT_4O_MINI_PRICING = Pricing(input_per_1m=0.15, output_per_1m=0.60)


class OpenAIPredictor(Predictor):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Either pass api_key= or export the env var."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        super().__init__(name=f"openai:{model}", pricing=GPT_4O_MINI_PRICING)

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        return content, in_tok, out_tok
