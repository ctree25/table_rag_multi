# CMoney TableRAG: precomputed table retrieval + hybrid schema retrieval + SQL API

This patch keeps the original DataBench/TableRAGMulti path untouched and adds a
CMoney-specific path.

## Files

Copy these files into the repository root while preserving paths:

```text
table_rag_multi/
├── run_cmoney_api.py
├── table_query.py
├── agent/
│   ├── cmoney_schema_retriever.py
│   └── rag_agent_multi_cmoney_api.py
└── prompts/
    └── cmoney_api.py
```

Expected inputs:

```text
data/cmoney/retrieval_eval.jsonl
data/cmoney/table_schema_20250122.csv
```

## Retrieval / reasoning flow

```text
retrieval_eval.jsonl
  -> retrieved_tables[:candidate_k]
  -> candidate table names
  -> table_schema_20250122.csv
  -> per-candidate schema retrieval only
       dense: jinaai/jina-embeddings-v3
         document task = retrieval.passage
         query task    = retrieval.query
       + BM25
       -> 0.5 / 0.5 EnsembleRetriever hybrid
  -> schema evidence
  -> SQL ReAct prompt
  -> Action {table, sql}
  -> QueryTableLight(sql, [table])
  -> Observation rows
  -> repeat SQL actions as needed
  -> Final Answer {table_source, answer}
```

There is intentionally no local table loading, no cell retrieval, and no
`build_db.py` in this CMoney API path. Cell values become available only after
SQL is executed through the API.

## Persistent schema cache

Schema retrieval is lazy and persistent.

For each candidate table, the first use builds:

- a FAISS dense index using Jina v3 document embeddings;
- a BM25 retriever cache.

Later questions/runs reuse the cache instead of recomputing the same table
schema. The cache identity includes:

- table schema rows/content;
- `jinaai/jina-embeddings-v3` model name;
- embedding dimension;
- retrieval tasks;
- retrieval mode/cache version.

Therefore changing the schema or embedding configuration creates a new cache,
while identical settings produce cache hits. This also prevents an old OpenAI
FAISS index from being accidentally reused after switching to Jina.

Default cache root:

```text
db/cmoney_api/cmoney_schema_only/
```

With `--verbose=True`, the inherited retriever prints messages such as `Load
schema database ...`, `BM25 disk cache hit ...`, or `Build BM25 disk cache ...`.

## Install dependencies

If missing from the current environment:

```bash
pip install -r requirements_cmoney_api.txt
```

`jinaai/jina-embeddings-v3` requires SentenceTransformers support for the Jina
remote model code. The patch loads it with `trust_remote_code=True`.

## First test: one question, candidate top-5

From `table_rag_multi/`:

```bash
python run_cmoney_api.py \
  --retrieval_path=data/cmoney/retrieval_eval.jsonl \
  --schema_path=data/cmoney/table_schema_20250122.csv \
  --candidate_k=5 \
  --schema_top_k=5 \
  --retrieve_mode=hybrid \
  --embed_model_name=jinaai/jina-embeddings-v3 \
  --schema_embed_device=cpu \
  --model_name=gpt-4.1-mini \
  --stop_at=1 \
  --load_exist=False \
  --verbose=True \
  --log_dir=output/cmoney_api_top5_test
```

PDM equivalent:

```bash
pdm run python run_cmoney_api.py \
  --retrieval_path=data/cmoney/retrieval_eval.jsonl \
  --schema_path=data/cmoney/table_schema_20250122.csv \
  --candidate_k=5 \
  --schema_top_k=5 \
  --retrieve_mode=hybrid \
  --embed_model_name=jinaai/jina-embeddings-v3 \
  --schema_embed_device=cpu \
  --model_name=gpt-4.1-mini \
  --stop_at=1 \
  --load_exist=False \
  --verbose=True \
  --log_dir=output/cmoney_api_top5_test
```

If CUDA is available and you want faster initial schema embedding, use:

```text
--schema_embed_device=cuda
```

After the first schema index build, later runs reuse the disk cache.

## Important K values

- `candidate_k`: number of table-level retrieval results used as candidate
  tables. Example: `5` means `retrieved_tables[:5]`.
- `schema_top_k`: number of schema/column hits returned by each retrieval query
  within each candidate table.

## API endpoint

Default:

```text
http://125.227.50.167:4444/CMoneyAdox/AdoxcService.svc
```

Override if needed:

```bash
export CM_ADOXC_BASE="http://125.227.50.167:4444/CMoneyAdox/AdoxcService.svc"
```
