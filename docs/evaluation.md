# Evaluation input formats

Run the evaluators from the repository root. They match gold cases and prediction rows by `caseID`; use unique, consistent IDs.

## Single-defendant gold

The scorer reads `meta.relevant_articles`, `meta.accusation` and `meta.term_of_imprisonment`. The following is a synthetic schema illustration, not a benchmark example:

```json
{
  "caseID": "schema-single-001",
  "fact": "<case fact description>",
  "meta": {
    "criminals": ["甲"],
    "relevant_articles": [264],
    "accusation": ["盗窃"],
    "term_of_imprisonment": {
      "imprisonment": 12,
      "life_imprisonment": false,
      "death_penalty": false
    }
  }
}
```

Gold imprisonment is expressed in **months** and converted to the original 11 interval classes. Single-defendant article/charge scoring uses the first normalized label, preserving the experiment's scoring convention.

## Multiple-defendant gold

The scorer reads a list under `outcomes`, containing `name` and a **`judgment`** list. Each judgment supplies `article`, `standard_accusation` (or `accusation`) and `penalty`:

```json
{
  "caseID": "schema-multi-001",
  "fact": "<case fact description>",
  "defendants": ["甲", "乙"],
  "outcomes": [
    {
      "name": "甲",
      "judgment": [{
        "article": [264],
        "standard_accusation": "盗窃",
        "penalty": {"imprisonment": 12, "life_imprisonment": false, "death_penalty": false}
      }]
    },
    {
      "name": "乙",
      "judgment": [{
        "article": [264],
        "standard_accusation": "盗窃",
        "penalty": {"imprisonment": 6, "life_imprisonment": false, "death_penalty": false}
      }]
    }
  ]
}
```

Keep defendant names consistent between `defendants`, `outcomes` and predictions: alignment is by name, not list position. The original set metrics and case/defendant weighting are retained.

## Predictions

Pass the runner's `predictions.jsonl` directly using `--pred_path`. Each row includes `caseID` and `predict`; the latter contains an `<answer>...</answer>` block with JSON. Predicted fields differ from gold fields:

| Item | Gold | Prediction |
|:---|:---|:---|
| Single article | `meta.relevant_articles` | `relevant_articles` |
| Single charge | `meta.accusation` | `accusation` |
| Single penalty | `meta.term_of_imprisonment` in months | `sentence_interval`, ID 0–10 |
| Multi records | `outcomes[].judgment[]` | `defendants[].judgments[]` |
| Multi article | `article` | `relevant_articles` |
| Multi charge | `standard_accusation` or `accusation` | `accusation` |
| Multi penalty | `penalty` in months | `sentence_interval`, ID 0–10 |

Ground-truth outcomes are used for scoring, not supplied to the model prompt. The inference runner can process unlabeled inputs when `--dataset_type single` or `--dataset_type multi` is explicit.

## Commands

```bash
export BGE_MODEL_PATH=/path/to/BAAI_bge-m3

python -m Eval.new_eval_single_defendant \
  --gold_path data/cail_test.jsonl --pred_path outputs/cail/predictions.jsonl \
  --output_path outputs/cail/metrics.json --include_summary

python -m Eval.eval_multi_defendant \
  --gold_path data/cmdl_test.jsonl --pred_path outputs/cmdl/predictions.jsonl \
  --output_path outputs/cmdl/metrics.json --include_summary
```

The included charge vocabularies, aliases and embeddings live in `Eval/data/`. The optional semantic normalization fallback requires the original BGE encoder. Without that encoder it is disabled; exact normalization may differ from the recorded benchmark.
