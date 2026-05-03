# Aegis training — Colab Pro setup

Training runs on Colab Pro (A100 40GB). Claude Code does **not** run training;
this guide is what you paste into a Colab notebook to launch it.

## Prerequisites

- Colab Pro subscription (for A100 access)
- Hugging Face account with a **write** token (HF_TOKEN)
- W&B account with API key (free tier; WANDB_API_KEY)
- A target HF repo name for the adapter, e.g. `yourname/aegis-qwen-7b-lora`
  (does not need to exist; the script will create it on first push)

## 1. Open in browser VS Code

1. Start a Colab Pro notebook
2. Runtime → Change runtime type → **A100 GPU**
3. (Optional) Tools → Connect to a hosted runtime → use the VS Code browser interface

## 2. Setup cell — paste into the first code cell

```python
# Clone repo (substitute your fork's URL)
!git clone https://github.com/<your-username>/fine-tuning.git
%cd fine-tuning

# Install training stack (uses requirements-train.txt: Unsloth, bitsandbytes, trl)
!make setup-train

# Set env vars — DO NOT commit these. Paste your real values here.
import os
os.environ["HF_TOKEN"]          = "hf_..."
os.environ["WANDB_API_KEY"]     = "..."
os.environ["AEGIS_HUB_REPO"]    = "<your-username>/aegis-qwen-7b-lora"
os.environ["AEGIS_RUN_NAME"]    = "aegis-qwen-7b-v1"   # optional but useful

# Mount Drive — checkpoints land at /content/drive/MyDrive/aegis-checkpoints/<run_name>/
from google.colab import drive
drive.mount("/content/drive")

# Build the processed dataset (only needed once per run; ~5 min total)
!python -m data.download
!python -m data.convert
!python -m data.format
```

## 3. Dry-run cell — verify the pipeline before committing to a long run

```python
!python -m train.train --dry-run
```

This trains **50 steps on 100 examples** (~3–5 min on A100). It verifies:

- Model loads correctly via Unsloth (4-bit Qwen 2.5 7B-Instruct)
- LoRA adapter is wired up with the project's hyperparams
- Chat-format JSONL is accepted by SFTTrainer
- Checkpoints write to `/content/drive/MyDrive/aegis-checkpoints/<run-name>-dryrun/`
- W&B reports the run

If the dry-run succeeds, the next cell is the real run.

## 4. Full training cell

```python
!python -m train.train
```

Expected timing on A100:
- ~28k train examples × 2 epochs / (batch 4 × grad_accum 4) ≈ **3,500 optimizer steps**
- Wall-clock: **3–5 hours** depending on actual sequence lengths
- Checkpoints every 500 steps to Drive
- Eval every 500 steps; best checkpoint restored at end of training
- Periodic adapter pushes to HF Hub every 4 hours (so a disconnect doesn't lose progress)
- Final adapter pushed to `huggingface.co/<AEGIS_HUB_REPO>`

## 5. Resume after a Colab disconnection

```python
# Re-mount Drive + re-set env vars + re-cd into repo if the runtime restarted
%cd fine-tuning
import os
os.environ["HF_TOKEN"]       = "hf_..."
os.environ["WANDB_API_KEY"]  = "..."
os.environ["AEGIS_HUB_REPO"] = "<your-username>/aegis-qwen-7b-lora"
os.environ["AEGIS_RUN_NAME"] = "aegis-qwen-7b-v1"   # MUST match the original run name
from google.colab import drive
drive.mount("/content/drive")

!python -m train.train --resume
```

`--resume` picks up from the latest `checkpoint-N` directory in
`/content/drive/MyDrive/aegis-checkpoints/<AEGIS_RUN_NAME>/`. If no
checkpoint is found, it starts from scratch.

## 6. After training — pull the adapter for local eval

On your local machine (where Claude Code runs):

```bash
huggingface-cli login    # if needed
huggingface-cli download <your-username>/aegis-qwen-7b-lora \
    --local-dir adapters/aegis
```

Then eval the fine-tuned adapter via `eval/run_baseline.py` (Aegis predictor —
to be added in step 7) and re-inject the comparison table:

```bash
python -m eval.run_baseline aegis
python -m eval.harness --inject README.md
```

The delta between the Presidio/baseline numbers (already in the README) and
the Aegis numbers is the project's headline result.

## Troubleshooting

- **"OOM" / CUDA out of memory.** Drop `MAX_SEQ_LENGTH` in `train/train.py`
  from 2048 to 1024. Or reduce `BATCH_SIZE` from 4 to 2 (and bump
  `GRAD_ACCUM` to 8 to keep the effective batch).
- **"unsloth not found"** after `make setup-train`. Restart the Colab runtime
  (Runtime → Restart) so the freshly installed Python packages are picked up.
- **Hub push fails with 401.** The HF_TOKEN doesn't have write scope. Issue
  a new token from huggingface.co/settings/tokens with **write** access.
- **Hub push fails with 403 on first run.** The target repo doesn't exist or
  isn't owned by the token's user. Either create the repo at
  huggingface.co/new or change `AEGIS_HUB_REPO`.
- **W&B run not appearing.** Confirm `WANDB_API_KEY` is set in the env
  before training starts; W&B is configured at trainer init time.
