# Aegis training on Colab

Training lives in [train/train.ipynb](../train/train.ipynb). Open it in
VS Code with the Jupyter extension and connect to a **Colab Pro A100**
runtime as the kernel.

## Quick steps

1. Open `train/train.ipynb` in VS Code locally.
2. **Kernel → Select Kernel → Existing Jupyter Server → paste your Colab URL.**
   (Or: Colab → Connect → "Connect to a local runtime" / VS Code Jupyter integration.)
3. Edit cell 4: replace `<your-username>` in the `git clone` URL with your GitHub username.
4. Edit cell 6: paste your real `HF_TOKEN`, `WANDB_API_KEY`, and `AEGIS_HUB_REPO`.
5. Run cells **top-to-bottom**. The configuration cell (cell 10) has a `DRY_RUN` toggle —
   leave it `True` for the first pass, then re-run the train cell with `DRY_RUN = False` for the real run.

## What the notebook does

| Cell | Purpose |
|---|---|
| 0 | Title + hyperparameter summary |
| 1 | GPU + Python version check |
| 3 | Clone repo into `/content/fine-tuning` |
| 4 | Install eval + train deps |
| 5 | Set env vars (paste your keys here) |
| 6 | Mount Google Drive |
| 7 | Build processed dataset (`download` → `convert` → `format`) |
| 9 | Configuration (constants, run dir, `DRY_RUN`) |
| 10 | Load Qwen 2.5 7B-Instruct + LoRA wrap |
| 13 | Load chat-format data + apply chat template |
| 15 | Build SFTTrainer + `PeriodicHubPush` callback |
| 16 | Run training (`RESUME` toggle here) |
| 18 | Final adapter push to HF Hub |

## After training

On your local machine:

```bash
huggingface-cli login
huggingface-cli download <your-username>/aegis-qwen-7b-lora --local-dir adapters/aegis
python -m eval.run_baseline aegis            # Aegis predictor — wired up in step 7
python -m eval.harness --inject README.md    # refresh the comparison table
```

The delta between the Presidio / GPT-4o-mini / Gemini Flash baselines and the
Aegis numbers is the project's headline result.

## Troubleshooting

- **OOM on A100.** Drop `MAX_SEQ_LENGTH` from 2048 → 1024, or `BATCH_SIZE` 4 → 2 (and bump `GRAD_ACCUM` to 8 to keep the effective batch size).
- **`unsloth` ImportError after pip install.** Restart the Colab runtime so the freshly installed packages are picked up.
- **Hub push 401.** Token lacks write scope — issue a new one at huggingface.co/settings/tokens.
- **Hub push 403 on first run.** The target repo doesn't exist or isn't owned by the token's user. Create it at huggingface.co/new or change `AEGIS_HUB_REPO`.
