"""Real QLoRA training run for OTSentinel AI's generator model (step 12).

Fine-tunes unsloth/Llama-3.2-3B-Instruct-bnb-4bit on the full train_v1.jsonl
dataset (44 train / 10 validation examples), logs the run to MLflow, and saves
the resulting LoRA adapter.

Dataset is tiny (44 train examples) -- overfitting is the real risk here, not
VRAM. eval_dataset uses the existing validation split so eval_loss is tracked
alongside train_loss during training; load_best_model_at_end keeps whichever
checkpoint had the lowest eval_loss rather than just the final one. Watch the
printed metrics for eval_loss starting to climb while train_loss keeps falling --
that's the overfitting signal, and if you see it, reduce num_train_epochs or
lora_rank on the next run rather than trusting the final checkpoint.

Usage:
    python scripts/train_qlora.py
"""

from __future__ import annotations

# unsloth must be imported before trl/transformers/peft -- see
# https://github.com/unslothai/unsloth/issues/2797
from unsloth import FastLanguageModel

import hashlib
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "finetuning" / "train_v1.jsonl"
ADAPTER_OUTPUT_DIR = PROJECT_ROOT / "models" / "qlora_v1_adapter"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "qlora_v1_checkpoints"

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_LENGTH = 768  # covers all 54 examples (max observed: 711 tokens) with margin
LORA_RANK = 16
LORA_ALPHA = 16
NUM_TRAIN_EPOCHS = 4  # 44 train examples x 4 epochs = 44 optimizer steps at batch 4
LEARNING_RATE = 2e-4
SEED = 3407

# Points at the same MLflow server used for retrieval evals (docker-compose mlflow
# service); this run gets its own experiment so it doesn't mix with retrieval runs.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "otsentinel-finetuning")


def compute_dataset_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_split(path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    return rows


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_hash = compute_dataset_hash(DATASET_PATH)
    print(f"Dataset SHA-256: {dataset_hash}")

    train_rows = load_split(DATASET_PATH, "train")
    validation_rows = load_split(DATASET_PATH, "validation")
    print(f"Train examples: {len(train_rows)}  Validation examples: {len(validation_rows)}")

    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    def format_row(row: dict) -> dict:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_dataset = Dataset.from_list([format_row(r) for r in train_rows])
    eval_dataset = Dataset.from_list([format_row(r) for r in validation_rows])

    training_args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        optim="adamw_8bit",
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=NUM_TRAIN_EPOCHS,  # keep all epoch checkpoints, dataset is tiny
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to=["mlflow"],
        run_name=f"qlora-v1-r{LORA_RANK}-lr{LEARNING_RATE}-ep{NUM_TRAIN_EPOCHS}",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print(f"\nStarting training: {NUM_TRAIN_EPOCHS} epochs, "
          f"{len(train_rows)} train / {len(validation_rows)} validation examples...\n")
    result = trainer.train()

    print(f"\nTraining complete. Final train loss: {result.training_loss:.4f}")
    eval_metrics = trainer.evaluate()
    print(f"Final eval_loss (best checkpoint, per load_best_model_at_end): "
          f"{eval_metrics['eval_loss']:.4f}")
    print(
        "If eval_loss is noticeably higher than train_loss, that's expected on a "
        "44-example dataset -- what matters is whether eval_loss stayed flat/improved "
        "across epochs rather than climbing. Check the MLflow run for the per-epoch curve."
    )

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {ADAPTER_OUTPUT_DIR}")
    print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nView the run in MLflow: {os.environ['MLFLOW_TRACKING_URI']} "
          f"(experiment: {os.environ['MLFLOW_EXPERIMENT_NAME']})")


if __name__ == "__main__":
    main()
