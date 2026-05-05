"""Base Qwen 2.5 7B-Instruct baseline (no fine-tuning) — local inference.

Self-hosted; no per-token cost. On Windows CPU expect ~15-60s per call —
intended for adversarial + heldout (~100 examples) rather than the full
test set. For the 7,946-row main test, run on Colab/cloud and feed the
results into eval/harness.py.

If GPU is available, pass device="cuda" or "auto" — torch_dtype switches
to bfloat16 to fit in VRAM.
"""

from __future__ import annotations

from dataclasses import replace

from baselines.base import Prediction, Predictor, Pricing, realign_to_input
from inference.prompts import SYSTEM_PROMPT

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class BaseQwenPredictor(Predictor):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(name=f"qwen-base:{model_id}", pricing=Pricing())
        # Lazy import so the module loads even if transformers is unavailable
        # (e.g. when only Presidio + APIs are exercised).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def predict(self, text: str) -> Prediction:
        # LLM-based predictors emit correct PII text but unreliable offsets;
        # realign to actual input positions. Same treatment as AegisPredictor
        # for fairness — see baselines/base.py::realign_to_input.
        pred = super().predict(text)
        if pred.spans:
            pred = replace(pred, spans=realign_to_input(text, pred.spans))
        return pred

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        in_tok = int(inputs.input_ids.shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,  # ignored when do_sample=False, but suppresses warnings
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0, in_tok:]
        out_tok = int(generated.shape[0])
        content = self.tokenizer.decode(generated, skip_special_tokens=True)
        return content, in_tok, out_tok
