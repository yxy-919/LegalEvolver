# LegalEvolver

This repository provides a compact reference implementation of LegalEvolver,
a framework that improves legal judgment prediction by optimizing the retrieval
queries generated during agent reasoning.

The released code contains the method's essential components:

- four legal retrieval tools backed by a shared SAILER encoder and normalized
  FAISS inner-product indexes;
- step-level construction of supervised examples from complete agent
  trajectories;
- rubric-based query evaluation conditioned on retrieval tool and reasoning
  focus; and
- a combined outcome, format, and rubric reward for GRPO training.

The datasets, retrieval corpora, indexes, model checkpoints, generated
trajectories, and learned rubric bank are not distributed in this repository.
Users should provide these resources through local paths or environment
variables.

## Repository layout

```text
configs/                 Example SFT, GRPO, and retrieval configurations
scripts/                 Minimal data-construction and training entry points
src/legalevolver/
  retrieval/             Shared encoder, FAISS index, and four legal tools
  rewards/               Judge client, rubric reward, and combined reward
  training/              Step-level SFT data construction
```

## Judge model

Rubric scoring uses an OpenAI-compatible judge endpoint. Any sufficiently
capable instruction-following model can be configured:

```bash
export JUDGE_API_KEY="..."
export JUDGE_BASE_URL="https://your-endpoint.example/v1"
export JUDGE_MODEL="judge-model"
```

For every retrieval query, the judge receives the local interaction context and
all rubric groups associated with the selected tool. It first identifies the
query's reasoning focus, selects the matching rubric group, and then scores the
query against those criteria.

## Training

Prepare local data and index paths in `configs/`, then run:

```bash
bash scripts/build_sft_data.sh
bash scripts/train_sft.sh
bash scripts/train_grpo.sh
```

The shell scripts intentionally expose only portable entry points. Cluster
scheduling, model-serving infrastructure, and experiment-specific paths are
left to the user.

