# Inference and reproduction notes

## Released artifact

Hugging Face `yxy919/LegalEvolver-Qwen3-8B` and GitHub Release `checkpoint-best-v1` contain the same merged inference checkpoint: four safetensors shards, their index, tokenizer, model configuration and generation configuration. Raw FSDP training shards and optimizer states are excluded.

[`checkpoints/model.SHA256SUMS`](../checkpoints/model.SHA256SUMS) records hashes of the 13 original files. Model bytes are unchanged from the evaluated merged checkpoint.

## Recorded results

Numbers in `checkpoints/cail_metrics.json` and `checkpoints/cmdl_metrics.json` are recorded formal results. Only machine-specific dataset/output paths were replaced with portable paths. Rounded values match manuscript Table 1.

Release preparation did not rerun the full benchmark. Exact reproduction depends on the original test subsets, retrieval resources, model/serving versions, decoding, retry passes and scoring resources.

## Inference settings

- Original prompts and the `<tool_call>`, `<tool_call_result>` and `<answer>` protocol are retained.
- The historical runner capped retrieval at **three hits per call**. `--retrieval_top_k 3` remains its default; standalone `ToolManager` defaults to four. Use `--retrieval_top_k 4` for the paper-facing setting.
- Default sampling: temperature 0.6, top-p 0.95, top-k 20 and min-p 0.
- `--max_tokens` defaults to 8,192 **per model request**, not per full trace. Historical retry passes used budgets including 5,000 and 5,800 tokens.
- Serving defaults to a 40,960-token context. The manuscript's 20,480-token training-rollout context is a different setting.
- The runner resumes outputs by case ID and sample count. Use a fresh output directory for comparisons.

## External retrieval and scoring resources

This release supplies original retrieval modules with portable imports/configuration. It excludes corpora, indexes, corpus preprocessing/index building, SAILER weights and BGE weights.

The manuscript describes source-specific lexical reranking, query limits, candidate expansion and snippet truncation. Packaged modules implement dense retrieval and source-specific formatting/version handling; the complete paper environment preprocessing/reranking pipeline is not provided. The packaged encoder maximum length defaults to 512 tokens in `Retrieval/config.yaml`; the manuscript environment reports 256. External indexes alone do not reconstruct every paper-side setting.

Scoring preserves the original first-label single-defendant article/charge convention and multi-defendant name alignment/weighting. `Eval/data/` includes the charge vocabularies, aliases and embeddings. Supply the original encoder via `BGE_MODEL_PATH` for semantic charge normalization; otherwise that fallback is disabled.
