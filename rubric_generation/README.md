# Preference Pairs and Rubric Construction

This directory implements the paper's **Initial Rubric Construction** and
**Rubric Refinement** stages. Both stages use the same workflow:

1. `preference_pairs.py` constructs within-case preferred/rejected trajectory pairs.
2. `induce_rubrics.py` performs pairwise criterion discovery.
3. `aggregate_rubrics.py` merges and abstracts overlapping criteria by retrieval
   tool and reasoning focus.
4. `validate_rubrics.py` performs held-out revalidation within each retrieval
   tool and reasoning focus. Natural pairs compare same-tool query sets from the
   two trajectories, while query-rewriting pairs directly compare the queries at
   the rewritten intervention point. Criteria are retained only when they score
   preferred queries higher across sufficient held-out examples.

For initial construction, run the workflow on the seed trajectories used for
policy initialization. For iterative refinement, rerun it on trajectories from
the updated policy, aggregate the new candidates with the current rubric bank,
and revalidate the combined bank. No trajectory data are included in this
release.
