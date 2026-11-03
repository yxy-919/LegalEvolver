# Retrieval resource guide

The four tools share a SAILER encoder and cached FAISS index bundles. They can be used by the agent or imported independently.

## External files

Download and verify the complete index archive, then set these variables before inference/tool use:

```bash
hf download CSHaitao/SAILER_zh --local-dir checkpoints/SAILER_zh
bash Retrieval/download_full_data.sh
```

```bash
export SAILER_MODEL_PATH="$PWD/checkpoints/SAILER_zh"
export LEGAL_INDEX_ROOT="$PWD/data/retrieval-full/indexes"
export SENTENCING_CONTEXT_PATH="$PWD/data/retrieval-full/sentencing_general_context.jsonl"
```

Expected layout:

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
  guidelines_local/                  Optional province-specific indexes
```

Indexes must match the encoder used to build them. Metadata rows must remain in exactly the same order as index vectors. The complete index files and metadata are distributed through the retrieval data release. Small subsets remain available under `Retrieval/demo_data/`; SAILER weights are downloaded separately.

Source metadata is consumed directly by the retrievers: statute article/version identifiers and text; case facts/judgment metadata; interpretation clause identifiers/related articles; guideline section identifiers, source scope and version tags. Inspect the corresponding module for complete field accesses. Arbitrary indexes with a different schema are not interchangeable.

## Configuration

[`Retrieval/config.yaml`](../Retrieval/config.yaml) defaults to CPU. Environment variables and `~` are expanded; relative paths resolve against the repository root. To use another YAML with the same schema:

```bash
export RUBRIC_REACT_CONFIG=/path/to/retrieval-config.yaml
# Or pass --retrieval_config /path/to/retrieval-config.yaml to the runner.
```

Guideline retrieval retains 2017-first version filtering and optional provincial-index behavior. Resource/version choices affect returned evidence.

## Call a tool

```python
from Retrieval.tool_manager import ToolManager

manager = ToolManager(topk_default=4, topk_max=4)
result = manager.call_tool(
    "judicial_interpretation_search",
    {"query": "盗窃罪 数额较大 认定标准", "topk": 4},
)
if result.get("error"):
    print(result["error"])
else:
    for card in result.get("results", []):
        print(card["evidence_id"], card["title"], card["score"])
        print(card["text"])
```

Available names: `statute_search`, `case_search`, `judicial_interpretation_search`, `guideline_search`. `ToolManager` sanitizes query/top-k inputs and records dropped extra fields as warnings.

Direct imports work too:

```python
from Retrieval.case_retriever import case_search
result = case_search("盗窃 手机 退赃 自首", topk=4)
```

## Evidence cards

Tool results contain `tool_name`, `query` and `results`. Each result card has:

| Field | Meaning |
|:---|:---|
| `evidence_id` | Source-specific identifier |
| `source_type` | Legal evidence type |
| `title` | Source title/section heading |
| `text` | Evidence text |
| `score` | Retrieval score |
| `metadata` | Original source metadata |

The runner formats cards as observations and preserves call records in prediction JSONL. A model-server-side native function-calling parser is not required.

## Paper environment and this release

The paper describes SAILER-zh, FAISS and source-specific reranking over external corpora. This release supplies dense-search modules and source formatting/version handling, rather than the entire corpus preprocessing/reranking pipeline. See [Reproduction Notes](reproduction.md) for configuration differences, including the historical three-hit runner cap.
