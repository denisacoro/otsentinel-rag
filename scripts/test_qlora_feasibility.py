"""Quick local feasibility test for QLoRA fine-tuning on a 4GB GPU.

Loads a small sample of data/finetuning/train_v1.jsonl, attaches a LoRA adapter to
the 4-bit quantized Llama 3.2 3B Instruct model, and runs a handful of training
steps. This is NOT the real training run -- it only checks whether QLoRA fits in
the VRAM available on this machine. If it completes without a CUDA out-of-memory
error, local training is viable; if it OOMs, we move the real run to Colab instead.

Usage:
    python scripts/test_qlora_feasibility.py
"""

from __future__ import annotations

# unsloth must be imported before trl/transformers/peft so its patches apply
# correctly (importing it later causes subtle bugs, e.g. an eos_token mismatch
# error from SFTTrainer) -- see https://github.com/unslothai/unsloth/issues/2797
from unsloth import FastLanguageModel

import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "finetuning" / "train_v1.jsonl"
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024  # conservative for this test; the real run may need more
SAMPLE_SIZE = 8  # just enough examples to exercise a few real optimizer steps


def load_sample_examples(path: Path, sample_size: int) -> list[dict]:
    examples: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if len(examples) >= sample_size:
                break
    return examples


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\nLoading {MODEL_NAME} in 4-bit (first run will download ~2GB)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded. VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"\nLoading {SAMPLE_SIZE} sample examples from {DATASET_PATH.name}...")
    raw_examples = load_sample_examples(DATASET_PATH, SAMPLE_SIZE)

    def format_example(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    formatted = [format_example(ex) for ex in raw_examples]
    dataset = Dataset.from_list(formatted)
    lengths = [len(tokenizer(ex["text"]).input_ids) for ex in formatted]
    print(f"Sample token lengths: {lengths} (max_seq_length={MAX_SEQ_LENGTH})")
    if max(lengths) > MAX_SEQ_LENGTH:
        print(
            "NOTE: at least one sampled example exceeds MAX_SEQ_LENGTH and will be "
            "truncated for this test only. That's fine for a feasibility check -- the "
            "real training run will need a longer max_seq_length to avoid truncating "
            "your two-chunk / comparison examples."
        )

    training_args = SFTConfig(
        output_dir=str(PROJECT_ROOT / "outputs" / "qlora_feasibility_test"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=5,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\nStarting 5-step feasibility test...\n")
    trainer.train()

    print("\nSUCCESS -- 5 training steps completed without a CUDA OOM error.")
    print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Total VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


if __name__ == "__main__":
    main()