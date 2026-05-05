"""Gemini Flash baseline via google-generativeai.

Same SYSTEM_PROMPT as other baselines; passed via system_instruction so
the user message is just the input text. Temperature 0; max_output_tokens 512.

Pricing constants in USD per 1M tokens — Gemini 2.5 Flash list price.
Update if rates change; recorded in eval output.

Safety filters: certain inputs (medical contexts, sensitive PII) may
trigger Gemini's safety system and return an empty response. We catch
this in the base class as a regular error → Prediction.error is set,
schema_validity_rate counts it as failed.
"""

from __future__ import annotations

import os
from dataclasses import replace

from baselines.base import Prediction, Predictor, Pricing, realign_to_input
from inference.prompts import SYSTEM_PROMPT

# Update when rates change.
GEMINI_FLASH_PRICING = Pricing(input_per_1m=0.30, output_per_1m=2.50)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiPredictor(Predictor):
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Either pass api_key= or export the env var."
            )
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model_obj = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        self.max_tokens = max_tokens
        super().__init__(name=f"gemini:{model}", pricing=GEMINI_FLASH_PRICING)

    def predict(self, text: str) -> Prediction:
        # LLM-based predictors emit correct PII text but unreliable offsets;
        # realign to actual input positions. Same treatment as AegisPredictor
        # for fairness — see baselines/base.py::realign_to_input.
        pred = super().predict(text)
        if pred.spans:
            pred = replace(pred, spans=realign_to_input(text, pred.spans))
        return pred

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        resp = self.model_obj.generate_content(
            text,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": self.max_tokens,
            },
        )
        # resp.text raises if blocked by safety filters; caller handles via base.
        content = resp.text if hasattr(resp, "text") else ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) if usage else None
        out_tok = getattr(usage, "candidates_token_count", None) if usage else None
        return content or "", in_tok, out_tok
