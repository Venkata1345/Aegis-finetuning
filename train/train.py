"""Aegis QLoRA training on Qwen 2.5 7B-Instruct.

Designed to run on Colab Pro (A100 40GB). On Colab, follow
scripts/colab_setup.md. Locally, run `make setup-train` first (Linux/CUDA only).

CLI:
    python -m train.train                # full run
    python -m train.train --dry-run      # 50 steps on 100 examples
    python -m train.train --resume       # resume from latest Drive checkpoint
    python -m train.train --help

Env vars (set in a Colab cell before running):
    HF_TOKEN          — required to push the LoRA adapter to Hugging Face Hub
    WANDB_API_KEY     — optional; enables W&B run tracking (free tier)
    AEGIS_HUB_REPO    — e.g. "yourname/aegis-qwen-7b-lora", destination repo
    AEGIS_RUN_NAME    — optional; auto-generated from timestamp if missing

Hyperparameters (per project spec; constants below — edit and re-run to sweep):
  lr=2e-4, batch=4, grad_accum=4, 2 epochs, cosine, 5% warmup
  LoRA r=16 alpha=32 on all linear layers, 4-bit base quantization

Data: reads data/processed/{train,val}.chat.jsonl. Run the data pipeline
on Colab once before training:
    python -m data.download && python -m data.convert && python -m data.format

Outputs:
  - Local checkpoints under <ckpt_root>/<run_name>/checkpoint-N
  - Best-by-eval_loss restored at end of training
  - Final LoRA adapter pushed to AEGIS_HUB_REPO (if set)
  - Periodic pushes every HUB_PUSH_HOURS during training
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# --- constants ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_DEFAULT = PROJECT_ROOT / "data" / "processed"
DRIVE_CKPT_ROOT = Path("/content/drive/MyDrive/aegis-checkpoints")
LOCAL_CKPT_ROOT = PROJECT_ROOT / "checkpoints"

# Unsloth-optimized variant of Qwen 2.5 7B-Instruct. Falls through to
# Qwen/Qwen2.5-7B-Instruct on the HF side; both work via FastLanguageModel.
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.05
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 10
HUB_PUSH_HOURS = 4
SEED = 42

DRY_RUN_STEPS = 50
DRY_RUN_TRAIN_EXAMPLES = 100
DRY_RUN_VAL_EXAMPLES = 50


# --- runtime helpers ------------------------------------------------------


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401  pylint: disable=unused-import

        return True
    except ImportError:
        return False


def maybe_mount_drive() -> None:
    if not is_colab():
        return
    try:
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
            print("Mounting Google Drive...")
            drive.mount("/content/drive")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: drive.mount failed: {e}", file=sys.stderr)


def resolve_ckpt_root() -> Path:
    if is_colab() and Path("/content/drive/MyDrive").exists():
        DRIVE_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
        return DRIVE_CKPT_ROOT
    LOCAL_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    return LOCAL_CKPT_ROOT


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    candidates = [
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: int(d.name.split("-")[-1]))


def load_chat_dataset(path: Path, limit: int | None = None):
    """Load a chat-format JSONL into a HF Dataset with 'messages' column."""
    from datasets import Dataset

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


# --- model / formatting ---------------------------------------------------


def build_model_and_tokenizer():
    from unsloth import FastLanguageModel

    print(f"Loading {MODEL_ID} (4-bit, max_seq={MAX_SEQ_LENGTH})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Unsloth selects bfloat16 on A100
        load_in_4bit=True,
    )
    print(f"Wrapping in LoRA r={LORA_R} alpha={LORA_ALPHA} on all linear layers")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=list(TARGET_MODULES),
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model, tokenizer


def format_with_chat_template(dataset, tokenizer):
    """Apply Qwen chat template; output a single 'text' column for SFTTrainer."""

    def _apply(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in batch["messages"]
            ]
        }

    return dataset.map(_apply, batched=True, remove_columns=["messages"])


# --- callback: periodic Hub push -----------------------------------------


def make_periodic_hub_push_callback(repo_id: str, every_hours: float):
    """Push the LoRA adapter to HF Hub every `every_hours` hours during training."""
    from transformers import TrainerCallback

    class _PeriodicHubPushCallback(TrainerCallback):
        def __init__(self) -> None:
            self.repo_id = repo_id
            self.interval_seconds = every_hours * 3600
            self.last_push = time.time()

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            now = time.time()
            if now - self.last_push < self.interval_seconds:
                return
            model = kwargs.get("model")
            if model is None:
                return
            print(f"\n[step {state.global_step}] periodic push -> {self.repo_id}")
            try:
                model.push_to_hub(
                    self.repo_id,
                    commit_message=f"checkpoint at step {state.global_step}",
                )
                print(f"[step {state.global_step}] push complete")
            except Exception as e:  # noqa: BLE001
                print(f"[step {state.global_step}] push FAILED: {e}", file=sys.stderr)
            self.last_push = now

    return _PeriodicHubPushCallback()


# --- main -----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            f"Train {DRY_RUN_STEPS} steps on {DRY_RUN_TRAIN_EXAMPLES} examples "
            "to verify the pipeline. No Hub push. ~3-5 min on A100."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in <ckpt_root>/<run_name>/.",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help=f"Override default data directory ({DATA_DIR_DEFAULT}).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run name (used as subdirectory under ckpt_root and as W&B run id). "
            "Default: $AEGIS_RUN_NAME or aegis-<timestamp>."
        ),
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    hub_repo = os.environ.get("AEGIS_HUB_REPO") or None
    hf_token = os.environ.get("HF_TOKEN") or None
    wandb_key = os.environ.get("WANDB_API_KEY") or None

    if hub_repo and not hf_token:
        print(
            "WARNING: AEGIS_HUB_REPO set but HF_TOKEN missing; hub push disabled.",
            file=sys.stderr,
        )
        hub_repo = None
    if not args.dry_run and not hub_repo:
        print(
            "WARNING: AEGIS_HUB_REPO not set; final adapter will NOT be pushed to Hub.",
            file=sys.stderr,
        )

    maybe_mount_drive()

    if wandb_key:
        os.environ["WANDB_PROJECT"] = "aegis"

    data_dir = Path(args.data_dir).resolve() if args.data_dir else DATA_DIR_DEFAULT
    train_path = data_dir / "train.chat.jsonl"
    val_path = data_dir / "val.chat.jsonl"
    if not train_path.exists() or not val_path.exists():
        print(
            f"Missing chat-format JSONL under {data_dir}.\n"
            "Run: python -m data.download && python -m data.convert && python -m data.format",
            file=sys.stderr,
        )
        return 1

    ckpt_root = resolve_ckpt_root()
    run_name = (
        args.run_name
        or os.environ.get("AEGIS_RUN_NAME")
        or datetime.now().strftime("aegis-%Y%m%d-%H%M%S")
    )
    if args.dry_run:
        run_name = f"{run_name}-dryrun"
    run_dir = ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {run_name}")
    print(f"Checkpoint root: {run_dir}")

    train_limit = DRY_RUN_TRAIN_EXAMPLES if args.dry_run else None
    val_limit = DRY_RUN_VAL_EXAMPLES if args.dry_run else None
    print(f"Loading train data from {train_path}")
    train_ds = load_chat_dataset(train_path, limit=train_limit)
    print(f"Loading val data from {val_path}")
    val_ds = load_chat_dataset(val_path, limit=val_limit)
    print(f"  train: {len(train_ds)} examples")
    print(f"  val:   {len(val_ds)} examples")

    model, tokenizer = build_model_and_tokenizer()
    train_ds = format_with_chat_template(train_ds, tokenizer)
    val_ds = format_with_chat_template(val_ds, tokenizer)

    from trl import SFTConfig, SFTTrainer

    eval_steps = EVAL_STEPS if not args.dry_run else max(1, DRY_RUN_STEPS // 2)
    save_steps = SAVE_STEPS if not args.dry_run else max(1, DRY_RUN_STEPS // 2)

    sft_config = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=NUM_EPOCHS if not args.dry_run else 1,
        max_steps=DRY_RUN_STEPS if args.dry_run else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=LOGGING_STEPS,
        report_to=["wandb"] if wandb_key else [],
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        push_to_hub=False,  # we push manually
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        seed=SEED,
        data_seed=SEED,
    )

    callbacks = []
    if hub_repo and not args.dry_run:
        callbacks.append(make_periodic_hub_push_callback(hub_repo, HUB_PUSH_HOURS))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
    )

    resume_path: str | None = None
    if args.resume:
        latest = find_latest_checkpoint(run_dir)
        if latest:
            resume_path = str(latest)
            print(f"Resuming from {latest}")
        else:
            print("No checkpoint found in run_dir; starting from scratch")

    print(f"\nStarting training at {datetime.now().isoformat(timespec='seconds')}")
    trainer.train(resume_from_checkpoint=resume_path)
    print(f"Training complete at {datetime.now().isoformat(timespec='seconds')}")

    if hub_repo and not args.dry_run:
        print(f"\nPushing final adapter to {hub_repo}...")
        try:
            model.push_to_hub(hub_repo, commit_message="final adapter")
            tokenizer.push_to_hub(hub_repo)
            print(f"Pushed to https://huggingface.co/{hub_repo}")
        except Exception as e:  # noqa: BLE001
            print(f"Final push FAILED: {e}", file=sys.stderr)
            print(f"Adapter saved locally at {run_dir}", file=sys.stderr)
            return 2

    print(f"\nFinal artifact: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
