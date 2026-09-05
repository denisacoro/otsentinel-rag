"""Generator client that runs a model directly via Unsloth/PEFT (base or the
QLoRA adapter), as an alternative to OllamaClient -- built for step 13's "model
selector" goal without needing vLLM (see conversation notes / docs for why:
vLLM has no proper native Windows support and its default memory management
targets server-class GPUs, not a 4GB laptop card).

Not used by the production pipeline by default. The fine-tuned adapter showed
no measured improvement over baseline and is not deployed (see
docs/FINETUNING_RESULTS.md). This class exists so
app.generation.rag_pipeline.answer_question() can optionally be pointed at
either the base model or the adapter for comparison runs (step 14's full
system evaluation), without adding unsloth/torch/bitsandbytes as a hard
dependency of the production code path -- rag_pipeline.py never imports this
module itself; the caller imports it explicitly when it wants this path.

IMPORTANT import-order constraint: unsloth must be imported before trl,
transformers or peft are touched anywhere in the same process. sentence-
transformers (used by app.retrieval.embedder.DenseEmbedder) imports
transformers internally, so if a script needs both retrieval and this client
in the same process, import this module (or `unsloth` directly) BEFORE
importing anything from app.retrieval or app.generation.rag_pipeline. See
scripts/compare_qlora_finetuning.py for the proven working pattern -- unsloth
imported as the first line of the file.
"""

from __future__ import annotations

from unsloth import FastLanguageModel

import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "models" / "qlora_v1_adapter"
DEFAULT_BASE_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
DEFAULT_MAX_SEQ_LENGTH = 2560  # covers real top_k=5 retrieval context (~2400 tokens
# observed in scripts/compare_qlora_finetuning.py); the 768 used during fine-tuning
# only ever needed to cover 1-2 chunk training examples.
DEFAULT_MAX_NEW_TOKENS = 400


class AdapterClient:
    """Generates answers using a locally loaded Unsloth model, matching
    OllamaClient's chat() interface so the two are interchangeable from
    rag_pipeline.answer_question()'s point of view.

    Pass model_path=DEFAULT_BASE_MODEL_NAME to run the unmodified base model,
    or leave the default to run the fine-tuned adapter -- either way this is
    the "model selector" step 13 asks for, just without vLLM underneath it.

    Loads the model once at construction and keeps it resident in VRAM for
    reuse across multiple chat() calls -- construct one instance per process,
    not one per question. Call unload() before constructing a second instance
    in the same process (4GB VRAM does not fit two 3B models at once).
    """

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_ADAPTER_PATH,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.model_name = Path(model_path).name if Path(str(model_path)).exists() else str(model_path)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, float]:
        """Generate a reply for a chat-format message list.

        Returns (answer_text, generation_latency_ms), matching OllamaClient.chat().
        """
        start = time.perf_counter()

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[1] :]
        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        latency_ms = (time.perf_counter() - start) * 1000
        return answer_text, latency_ms

    def unload(self) -> None:
        """Free VRAM -- call explicitly before loading another model in the
        same process (e.g. switching from base to adapter for comparison)."""
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
