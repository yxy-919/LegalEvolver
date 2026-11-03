# Released checkpoint

Primary weights: [Hugging Face — LegalEvolver-Qwen3-8B](https://huggingface.co/yxy919/LegalEvolver-Qwen3-8B).

```bash
hf download yxy919/LegalEvolver-Qwen3-8B --local-dir checkpoints/LegalEvolver-Qwen3-8B
```

Backup: [GitHub Release checkpoint-best-v1](https://github.com/yxy-919/LegalEvolver/releases/tag/checkpoint-best-v1).

The Release distributes the original merged step-149 Qwen3-8B model, including all four safetensors shards, tokenizer files, model configuration and generation configuration. Optimizer states and raw FSDP training shards are not needed for inference and are not included.

Run `bash checkpoints/download.sh` from the repository root. The download uses 16 numbered archive parts, verifies `SHA256SUMS`, extracts them, then verifies the original files against `model.SHA256SUMS`. The weight bytes are unchanged from the evaluated merged model.

`cail_metrics.json` and `cmdl_metrics.json` are the recorded formal results corresponding to Table 1. Machine-specific dataset and output paths were replaced by portable paths; numeric results are unchanged.

Historical provenance: original merged SFT initialization; fresh GRPO with resume disabled; one epoch / 149 steps; legal-task reward weight 0.01 together with ordinal penalty reward. This is the specific checkpoint associated with the recorded metrics, rather than a claim about any later training run.

Inference: `python -m Eval.inference.run_agentic --help`.

Retrieval corpora, prebuilt FAISS indexes, the SAILER encoder and the optional BGE normalization encoder must be supplied separately. Use `Eval/data/` charge vocabularies, aliases and embeddings with the original BGE model when reproducing charge normalization.
