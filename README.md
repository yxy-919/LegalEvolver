# LegalEvolver

Legal retrieval tools, the released Qwen3-8B checkpoint, and matching inference and evaluation code for **LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution**.

This release focuses on running the checkpoint. Training, preference-pair construction and rubric-generation code are outside the current release scope. Earlier repository versions remain in Git history.

## Contents

```text
Retrieval/                 Four original retrieval implementations and SAILER/FAISS utilities
  config.yaml              Portable paths to externally supplied retrieval resources
Eval/
  inference/
    run_agentic.py         Multi-turn inference for single/multiple defendants
    prompts.py            Original tool-use and judgment prompts
    serve.sh              Serve the merged checkpoint with vLLM
  new_eval_single_defendant.py
  eval_multi_defendant.py
  common.py               Shared parsing, label normalization and metrics
  data/                   Charge vocabularies, alias mapping and charge embeddings
checkpoints/
  download.sh             Download, verify and restore the merged model
  model.SHA256SUMS         Checksums of the original model files
  cail_metrics.json        Recorded formal evaluation metrics
  cmdl_metrics.json
```

`Eval/inference/` contains inference and `Eval/` contains scoring. A single directory spelling avoids collisions between `eval/` and `Eval/` on case-insensitive filesystems.

## Checkpoint

Download the merged, inference-ready weights from [the checkpoint Release](https://github.com/yxy-919/LegalEvolver/releases/tag/checkpoint-best-v1). The approximately 16 GB model is distributed as 1 GiB archive parts; do not load a part directly with Transformers.

```bash
# Requires GitHub CLI. For a private repository, run gh auth login first.
bash checkpoints/download.sh
```

The script verifies archive checksums, extracts the model, then verifies every original model file. Allow approximately 34 GB of free disk space for both the archive and extracted model. The resulting directory is `checkpoints/downloads/LegalEvolver-Qwen3-8B/`.

The checkpoint is the step-149 **legal-task reward (lambda=0.01) + ordinal penalty reward** run. Historical training records identify it as a fresh GRPO run from the original merged SFT initialization, with resume disabled. The ordinal penalty score was `0.5 ** abs(predicted_class - gold_class)` for valid classes 0–10. This checkpoint's provenance should not be interpreted as evidence of additional sequential training rounds.

Recorded results (percentage; Acc. / Macro-F1):

| Dataset | Law article | Charge | Prison term |
|---|---:|---:|---:|
| CAIL2018 | 82.1 / 70.7 | 79.4 / 71.9 | 22.1 / 21.0 |
| CMDL | 85.7 / 87.6 | 84.4 / 85.1 | 25.6 / 22.8 |

These match the LegalEvolver row in Table 1 (Overall 61.5). Full precision metrics are provided in `checkpoints/`; only machine-specific source paths were replaced by portable paths.

## Installation

Python 3.10 or newer:

```bash
pip install -e .
# In a GPU serving environment, install a compatible vLLM build separately:
pip install vllm
```

Inference uses an OpenAI-compatible vLLM endpoint. Retrieval can run on CPU separately from model serving.

## Retrieval resources

The tools are `statute_search`, `case_search`, `guideline_search`, and `judicial_interpretation_search`. Corpus processing, index building, raw cases and retrieval databases are not distributed. Supply compatible prebuilt resources:

```text
$LEGAL_INDEX_ROOT/
  statutes.faiss
  statutes_metadata.jsonl
  cases.faiss
  cases_metadata.jsonl
  judicial_interpretations.faiss
  judicial_interpretations_metadata.jsonl
  guidelines_national.faiss
  guidelines_national_metadata.jsonl
  guidelines_local/             Optional province-specific indexes
```

Each index needs metadata in the identical row order used to build it. Metadata fields are consumed directly by the corresponding retriever, including statute article/version identifiers, case facts and judgment metadata, interpretation clause identifiers, and guideline section identifiers/version tags.

```bash
export SAILER_MODEL_PATH=/path/to/SAILER_zh
export LEGAL_INDEX_ROOT=/path/to/prebuilt/indexes
export SENTENCING_CONTEXT_PATH=/path/to/sentencing_general_context.jsonl
# Optional custom YAML using the same schema:
export RUBRIC_REACT_CONFIG=/path/to/retrieval-config.yaml
```

Relative paths in the configuration are resolved against the repository root. `Retrieval/config.yaml` defaults to CPU retrieval. Guideline retrieval retains the original 2017-first version filtering and optional provincial-index behavior.

## Serve and infer

```bash
TENSOR_PARALLEL_SIZE=2 bash Eval/inference/serve.sh \
  checkpoints/downloads/LegalEvolver-Qwen3-8B
```

In another shell, configure the retrieval paths above, then run:

```bash
python -m Eval.inference.run_agentic \
  --test_dataset_path data/cail_test.jsonl --dataset_type single \
  --output_dir outputs/cail --model_name LegalEvolver \
  --base_urls http://127.0.0.1:8000/v1

python -m Eval.inference.run_agentic \
  --test_dataset_path data/cmdl_test.jsonl --dataset_type multi \
  --output_dir outputs/cmdl --model_name LegalEvolver \
  --base_urls http://127.0.0.1:8000/v1
```

Input is JSONL with `caseID` and `fact`. Single-defendant inputs may provide `meta.criminals`; multiple-defendant inputs provide `defendants`. Explicit `--dataset_type multi` supports inference without gold `outcomes`. Gold judgments are not included in the prompt.

The output contains `caseID`, `predict` (the generated trace and answer), and `tool_call` records. The original `<tool_call>`, `<tool_call_result>` and `<answer>` protocol and prompts are retained. Existing output is resumed by case ID and sample count.

The historical inference implementation capped retrieval at **3 hits per call**; this remains the default of `--retrieval_top_k 3`. Use `--retrieval_top_k 4` for the paper-facing four-result setting. The standalone `ToolManager` defaults to 4. The inference generation budget is configurable with `--max_tokens`; use 5000 or 5800 when replaying the historical retry passes. Model context defaults to 40,960 tokens. Decoding, retry passes, retrieval resources and charge-normalization assets can affect exact reproduction; the packaged code has not been rerun on the full benchmark as part of this release preparation.

## Score predictions

```bash
# Optional, for the original semantic charge-normalization fallback:
export BGE_MODEL_PATH=/path/to/BAAI_bge-m3

python -m Eval.new_eval_single_defendant \
  --gold_path data/cail_test.jsonl \
  --pred_path outputs/cail/predictions.jsonl \
  --output_path outputs/cail/metrics.json --include_summary

python -m Eval.eval_multi_defendant \
  --gold_path data/cmdl_test.jsonl \
  --pred_path outputs/cmdl/predictions.jsonl \
  --output_path outputs/cmdl/metrics.json --include_summary
```

These are the original experiment scoring implementations, with imports and environment-specific paths adapted. Single-defendant scoring retains the original first-label article/charge convention. Multiple-defendant scoring aligns defendants by name and retains its original set metrics and case/defendant weighting. Charge vocabularies, alias mapping and precomputed embeddings are included under `Eval/data/`; the BGE encoder itself must be supplied separately. Without that encoder, semantic fallback is disabled and some normalized charge predictions may differ from the recorded benchmark.
