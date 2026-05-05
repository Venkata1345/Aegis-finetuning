"""Aegis predictor — base Qwen 2.5 7B-Instruct + the trained LoRA adapter.

Loads via plain transformers + PEFT (not Unsloth — Unsloth is training-focused).
Self-hosted, no per-token cost. On a T4/A100 GPU expect ~1-3s per call;
on Windows CPU at MAX_SEQ=2048 it's ~30-60s per call (intended only for
adversarial + heldout sample sizes there).

Set AEGIS_ADAPTER_REPO in env to point at your HF Hub adapter repo.
"""

from __future__ import annotations

import os

from baselines.base import Predictor, Pricing
from inference.prompts import SYSTEM_PROMPT

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class AegisPredictor(Predictor):
    def __init__(
        self,
        adapter_repo: str | None = None,
        base_model: str = DEFAULT_BASE_MODEL,
        device: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        adapter_repo = adapter_repo or os.environ.get("AEGIS_ADAPTER_REPO")
        if not adapter_repo:
            raise RuntimeError(
                "AEGIS_ADAPTER_REPO not set. Either pass adapter_repo= or "
                "export the env var (e.g. 'username/aegis-qwen-7b-lora')."
            )

        super().__init__(name=f"aegis:{adapter_repo}", pricing=Pricing())

        # Lazy imports so this module loads on machines without transformers/peft
        # (the eval harness imports it through the registry).
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model = PeftModel.from_pretrained(base, adapter_repo)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _predict_impl(self, text: str) -> tuple[str, int | None, int | None]:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        in_tok = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0, in_tok:]
        out_tok = int(generated.shape[0])
        content = self.tokenizer.decode(generated, skip_special_tokens=True)
        return content, in_tok, out_tok
