# Database subsets

Small extracts of the project's existing processed legal retrieval databases, with corresponding original SAILER vectors reconstructed in identical row order into FAISS IndexFlatIP indexes. The source environment draws on criminal statutes, China Judgments Online precedents, judicial interpretations and sentencing guidelines. Original source identifiers/URLs and legal version fields are retained where available; machine-local source filenames and processing-debug fields are omitted.

Counts and source row indices are recorded in manifest.json. Statutes combine selected criminal-law articles with evenly spaced articles. Other databases take the first record per category/document, then fill in original order. These are deterministic convenience subsets, not randomly sampled evaluation sets. Precedent text retains the source database's anonymized name notation. No new legal outcomes or statutory text were generated.

Use ../demo_config.yaml and the original SAILER encoder. Small-subset results need not match full-corpus retrieval or paper scores. National guideline version filtering is preserved; local provincial indexes are not included.
