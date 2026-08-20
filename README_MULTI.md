# README — TableRAG Multi-Table

## Repository

```bash
cd ~/table_rag_multi
conda activate tablerag
```

The repository currently supports two reasoning settings.

### Multi-table

```text
question
→ outer table retrieval / candidate tables
→ query expansion once per question
→ schema/cell BM25 independently inside each candidate table
→ multi-table ReAct over df1 ... dfK
→ {"table_source": "dfN", "answer": ...}
```

### Oracle

```text
question
→ annotated GT table directly
→ normal TableRAG schema/cell retrieval inside that table
→ single-table ReAct over df
→ answer
```

Oracle only removes outer table-selection uncertainty. Schema retrieval, cell retrieval, and downstream reasoning are still performed normally.

---

## 1. Environment

If the environment already exists:

```bash
conda activate tablerag
cd ~/table_rag_multi
```

If rebuilding:

```bash
conda create -n tablerag python=3.10 -y
conda activate tablerag

pip install \
  langchain==0.2.16 \
  langchain-core==0.2.38 \
  langchain-community==0.2.16 \
  langchain-openai==0.1.23 \
  langchain-google-vertexai==1.0.10 \
  rank-bm25==0.2.2 \
  openai \
  google-auth \
  pyarrow
```

---

## 2. LLM backends

`agent/model.py` currently supports both Azure/OpenAI-compatible GPT and Vertex AI Gemini.

### 2.1 Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT="https://development-table-qa-resource.openai.azure.com/openai/v1/"
export AZURE_OPENAI_API_KEY="<YOUR_AZURE_OPENAI_API_KEY>"
export AZURE_OPENAI_CONTEXT_LIMIT="128000"
```

Current deployment used in experiments:

```text
gpt-5.6-sol
```

### 2.2 Vertex AI Gemini

Current tested model:

```text
gemini-3-flash-preview
```

The Vertex OpenAI-compatible endpoint uses the global location. Keep credentials in a local config such as `agent/gemini_key.py`; do not commit the service-account private key.

Conceptually:

```python
GCP_PROJECT_ID = "<YOUR_PROJECT>"
GCP_LOCATION = "global"
GEMINI_MODEL_NAME = "gemini-3-flash-preview"
GCP_SERVICE_ACCOUNT_INFO = {...}
```

Direct connectivity test:

```bash
python - <<'PY'
from agent.model import Model

m = Model("gemini-3-flash-preview")
text = m.query(
    "Reply with exactly: GEMINI_OK",
    temperature=0,
    max_tokens=32,
)

print("response:", text)
print("usage:", m.last_usage)
PY
```

Expected response:

```text
GEMINI_OK
```

For a GPT/Gemini experiment pair, keep all TableRAG settings identical and change only `--model_name` unless the experiment explicitly studies model-specific reasoning settings.

---

## 3. Important parameters

```text
candidate_k = number of OUTER candidate tables given to TableRAGMulti

top_k      = number of INNER schema/cell BM25 hits per expansion query

sc         = self-consistency runs
```

Do not confuse `candidate_k` with `top_k`.

`TableRAGOracle` does not use `candidate_k` because the GT table is supplied directly.

---

# DataBench

## 4. DataBench paths

Tables:

```text
data/databench_test80/data/<table_id>/all.parquet
```

OpenTab retrieval:

```text
data/databench_test80/retrieval/hybrid_title_columns/opentab.json
```

Prepared OpenTab top-10 dataset:

```text
data/wtq_databench_multi_opentab_top10.jsonl
```

Prepared all-80 dataset currently used by Oracle and all-80 multi-table experiments:

```text
data/databench_test80/wtq_databench_multi_all80.jsonl
```

Expected corpus/evaluation size:

```text
522 questions
80 tables
```

The all-80 JSONL contains, per question:

```text
gold_table_id
candidate_tables[0..79]
```

For DataBench Oracle, `TableRAGOracle` exact-matches `gold_table_id` against `candidate_tables` and loads the corresponding GT parquet. There is no LLM table selection and no outer retrieval in this branch.

---

## 5. Prepare DataBench OpenTab top-10 candidates

```bash
cd ~/table_rag_multi
conda activate tablerag

python prepare_databench_multi.py
```

Expected output:

```text
data/wtq_databench_multi_opentab_top10.jsonl
```

Check:

```bash
wc -l data/wtq_databench_multi_opentab_top10.jsonl
```

Expected:

```text
522
```

The same prepared top-10 JSONL can be used with `--candidate_k=1` or `--candidate_k=10`.

---

## 6. DataBench main experiment — candidate K = 10

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/databench_multi_opentab_top10_gpt56sol_bm25"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path=data/wtq_databench_multi_opentab_top10.jsonl \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGMulti" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --candidate_k=10 \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=522 \
  --resume_from=0 \
  --n_worker=4 \
  --verbose=False \
  2>&1 | tee -a "${LOG_DIR}_run.log"
```

Resume with the same command and same `LOG_DIR`, keeping:

```text
--load_exist=True
```

Do not reuse one output directory across different model, candidate K, retrieval source, or prompt conditions.

---

## 7. DataBench candidate K = 1

No new dataset preparation is needed.

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/databench_multi_opentab_top1_gpt56sol_bm25"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path=data/wtq_databench_multi_opentab_top10.jsonl \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGMulti" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --candidate_k=1 \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=522 \
  --resume_from=0 \
  --n_worker=4 \
  --verbose=False \
  2>&1 | tee -a "${LOG_DIR}_run.log"
```

---

## 8. Prepare DataBench all-80 candidates

```bash
cd ~/table_rag_multi
conda activate tablerag

python prepare_databench_all80.py
```

Expected output in the current layout:

```text
data/databench_test80/wtq_databench_multi_all80.jsonl
```

Checks:

```bash
wc -l data/databench_test80/wtq_databench_multi_all80.jsonl
find data/databench_test80/data -name all.parquet | wc -l
```

Expected:

```text
522
80
```

For all-80:

```text
gold_in_candidates = True for every question
```

`gold_rank` / `selected_rank` in this file is the deterministic all-80 position, not an OpenTab retrieval rank.

---

## 9. DataBench all-80 one-question test

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/databench_multi_all80_one_gpt56sol_bm25"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path=data/databench_test80/wtq_databench_multi_all80.jsonl \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGMulti" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --candidate_k=80 \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=1 \
  --resume_from=0 \
  --n_worker=1 \
  --verbose=True \
  2>&1 | tee "${LOG_DIR}_run.log"
```

---

## 10. DataBench full all-80 run

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/databench_multi_all80_gpt56sol_bm25"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path=data/databench_test80/wtq_databench_multi_all80.jsonl \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGMulti" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --candidate_k=80 \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=522 \
  --resume_from=0 \
  --n_worker=4 \
  --verbose=False \
  2>&1 | tee -a "${LOG_DIR}_run.log"
```

---

## 11. DataBench Oracle

Agent:

```text
TableRAGOracle
```

Prompt:

```text
tablerag_databench_oracle_solve_table_prompt
```

Oracle definition:

```text
question
→ data["gold_table_id"]
→ exact match inside data["candidate_tables"]
→ GT parquet
→ normal TableRAG schema/cell retrieval
→ single-table df ReAct
→ answer
```

`run.py` raw-reads JSONL for `TableRAGOracle` instead of passing it through the normal WTQ loader. This preserves fields such as `gold_table_id`, `candidate_tables`, and CMoney `table_path`.

### One-question smoke test

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/databench_oracle_gpt56sol_test1"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/databench_test80/wtq_databench_multi_all80.jsonl" \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGOracle" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=1 \
  --resume_from=0 \
  --n_worker=1 \
  --verbose=True \
  2>&1 | tee "${LOG_DIR}_run.log"
```

### Full Oracle run

Change:

```text
--stop_at=522
--n_worker=4
```

Oracle does not use `--candidate_k`.

---

## 12. DataBench responses.txt and official evaluation

When `dataset_path` contains `databench` and `sc=1`, `run.py` writes:

```text
<LOG_DIR>/responses.txt
```

Top-10 example:

```bash
cp \
  ~/table_rag_multi/output/databench_multi_opentab_top10_gpt56sol_bm25/responses.txt \
  ~/databench_eval/ours_output.txt

cd ~/databench_eval
conda activate tablerag
python eval_output.py ours_output.txt
```

For final reported DataBench answer accuracy, use the official DataBench/SemEval evaluator rather than simple exact-string comparison.

---

# CMoney

## 13. CMoney source files

```text
data/cmoney/basic_problems_20260116_answer.csv
data/cmoney/basic_problems_20260614_table_path.csv
data/cmoney/basic_problems_20260616_retrieval.jsonl
data/cmoney/basic_problems_20260616_extended_gt_tables.csv
data/cmoney/tables/
```

Prepared multi-table dataset:

```text
data/cmoney/wtq_cmoney_multi_top10.jsonl
```

Prepared candidate parquets:

```text
data/cmoney/prepared_tables/
```

Current eligible experiment set:

```text
245 questions
```

The `wtq_` prefix is intentional because `run.py` infers the task from the dataset path and routes these records through the WTQ-compatible task path.

---

## 14. Prepare CMoney multi-table top-10

The preparation logic uses the existing eligible set based on a non-empty/resolvable base GT CSV and preserves the original retrieval order.

```bash
cd ~/table_rag_multi
conda activate tablerag

python data/cmoney/prepare_cmoney_multi.py
```

Expected prepared dataset:

```text
data/cmoney/wtq_cmoney_multi_top10.jsonl
```

Check:

```bash
wc -l data/cmoney/wtq_cmoney_multi_top10.jsonl
```

Expected:

```text
245
```

CMoney candidates preserve the raw retrieval order. Do not rerank or backfill a missing raw top-K table with rank K+1 when evaluating a fixed raw top-K condition.

---

## 15. CMoney multi-table run

### One-question smoke test

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/cmoney_top10_gpt56sol_test1"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/cmoney/wtq_cmoney_multi_top10.jsonl" \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGMulti" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --candidate_k=10 \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=1 \
  --resume_from=0 \
  --n_worker=1 \
  --verbose=True \
  2>&1 | tee "${LOG_DIR}_run.log"
```

### Full 245-question run

Use the same command with:

```text
--stop_at=245
--n_worker=4
```

CMoney multi-table final output uses a source-table-aware form internally, e.g.:

```json
{"table_source": "dfN", "answer": "..."}
```

CMoney answer accuracy should be evaluated with the CMoney-specific semantic evaluator, not the generic WTQ exact evaluator.

---

## 16. Prepare CMoney Oracle

Script:

```text
data/cmoney/prepare_cmoney_oracle.py
```

The Oracle preparation uses the existing `wtq_cmoney_multi_top10.jsonl` as the only question universe, so Oracle and multi-table experiments use exactly the same 245 questions.

```bash
cd ~/table_rag_multi
conda activate tablerag

python data/cmoney/prepare_cmoney_oracle.py
```

Outputs:

```text
data/cmoney/wtq_cmoney_oracle.jsonl
data/cmoney/wtq_cmoney_oracle_prepare_summary.json
data/cmoney/prepared_tables_oracle/*.parquet
```

The Oracle JSONL contains only one table-related field per question:

```text
table_path
```

It intentionally does not contain candidate tables, retrieval ranks, retrieval scores, or extended-GT alternatives.

Strict preparation guarantees:

```text
exactly 245 source records
unique question IDs
all 245 have a non-empty GT path
all GT CSVs exist
all GT parquets are readable
all final table_path values exist
final qid set exactly matches the 245-question source set
```

If any check fails, the script aborts and does not leave an incomplete final Oracle JSONL.

Quick validation:

```bash
wc -l data/cmoney/wtq_cmoney_oracle.jsonl

python - <<'PY'
import json
from pathlib import Path

p = Path("data/cmoney/wtq_cmoney_oracle.jsonl")
rows = [
    json.loads(line)
    for line in p.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print("records:", len(rows))
print("with table_path:", sum(bool(x.get("table_path")) for x in rows))
print("unique qids:", len({x["question_id"] for x in rows}))
print(
    "missing files:",
    [x["question_id"] for x in rows if not Path(x["table_path"]).exists()],
)
PY
```

Expected:

```text
records: 245
with table_path: 245
unique qids: 245
missing files: []
```

---

## 17. CMoney Oracle run

Agent:

```text
TableRAGOracle
```

Prompt:

```text
tablerag_cmoney_oracle_solve_table_prompt
```

The CMoney Oracle branch reads `data["table_path"]` directly and reasons over one dataframe named `df`.

The Multi and Oracle prompts should keep the same general reasoning instructions. Domain-specific financial hints were removed from the Multi prompt, so do not give Oracle extra financial hints either. The intended experimental difference is single GT table vs multiple retrieved candidates, not prompt quality.

### One-question GPT smoke test

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/cmoney_oracle_gpt56sol_test1"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/cmoney/wtq_cmoney_oracle.jsonl" \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGOracle" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=1 \
  --resume_from=0 \
  --n_worker=1 \
  --verbose=True \
  2>&1 | tee "${LOG_DIR}_run.log"
```

### Full Oracle run

Use the same command with:

```text
--stop_at=245
--n_worker=4
```

Oracle does not use `--candidate_k`.

---

## 18. Run the same experiment with Gemini

For either DataBench or CMoney, keep the same command and change:

```text
--model_name="gemini-3-flash-preview"
```

Example: CMoney Oracle one-question test

```bash
cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/cmoney_oracle_gemini3flash_test1"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/cmoney/wtq_cmoney_oracle.jsonl" \
  --model_name="gemini-3-flash-preview" \
  --agent_type="TableRAGOracle" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --top_k=5 \
  --sc=1 \
  --max_encode_cell=1000 \
  --max_depth=10 \
  --max_tokens=512 \
  --temperature=0 \
  --stop_at=1 \
  --resume_from=0 \
  --n_worker=1 \
  --verbose=True \
  2>&1 | tee "${LOG_DIR}_run.log"
```

When changing the effective Gemini reasoning configuration, use a fresh query-expansion cache or clear the old expansion cache if the cache identity would otherwise no longer reflect the effective model setting.

---

# Caches and efficiency

## 19. Persistent BM25 cache

BM25 now uses two cache levels:

```text
process memory
→ disk cache
→ build and save on miss
```

Persistent cache files are stored under the BM25 cache directory, e.g.:

```text
db/bm25_cache_v1/
  schema_<hash>.pkl
  cell_<hash>.pkl
  .lock
```

With `--verbose=True`, expect messages such as:

```text
Build BM25 disk cache: ...
BM25 disk cache hit: ...
BM25 memory cache hit: ...
```

The cache key includes the stable table identity and table representation metadata, including `max_encode_cell`. `top_k` is not part of the underlying BM25 index identity, so changing only retrieval top-K does not require rebuilding the index.

Typical reuse:

```text
same process                  → memory hit
restart run.py                → disk hit
change LLM model              → BM25 cache reusable
change top_k                  → BM25 cache reusable
reuse same table in new run   → BM25 cache reusable
```

If the underlying table representation or a cache-affecting encoding setting changes, clear it:

```bash
rm -rf db/bm25_cache_v1
```

---

## 20. Persistent query-expansion cache

`TableRAGMulti` also has a persistent query-expansion cache:

```text
db/query_expansion_cache_v1/
  column_<hash>.json
  cell_<hash>.json
  .lock
```

The cache identity includes important generation settings such as:

```text
expansion prompt
column vs cell expansion
remove_numeric
model_name
temperature
top_p
max_tokens
reasoning_effort
cache version
```

Therefore GPT and Gemini expansions are intentionally kept separate.

Cached expansion usage is re-applied to token accounting so logs remain comparable with the original generation call.

Clear when intentionally changing the expansion behavior or effective model reasoning configuration:

```bash
rm -rf db/query_expansion_cache_v1
```

Current implementation note: the persistent expansion cache was added to the multi-table agent. `TableRAGOracle` still follows the base single-table TableRAG expansion path unless the cache logic is explicitly refactored into a shared component.

---

## 21. ReAct evidence-token optimization

`agent/rag_agent_multi.py` sends the full retrieved table evidence only in ReAct round 1.

```text
Round 1:
question + candidate map + all retrieved evidence

Round 2+:
question + candidate map
+ previous Thought/Action/Observation history
+ no repeated full table evidence
```

Important token fields include:

```text
init_prompt_token_count
continuation_prompt_token_count
input_token_count
output_token_count
total_token_count
```

`run.py` also aggregates completed per-question logs into:

```text
<LOG_DIR>/token_usage.json
```

---

# Logs and evaluation

## 22. Important multi-table per-question log fields

Example:

```text
output/<experiment>/log/databench-0-0.json
```

Useful fields:

```text
id
query
answer
label
raw_final_answer
table_source
selected_table_id
selected_rank
gold_table_id
selected_is_gold
gold_in_candidates
candidate_k
candidate_table_map
candidate_table_ids
column_queries
cell_queries
retrieved_columns
retrieved_cells
solution
n_iter
init_prompt_token_count
continuation_prompt_token_count
input_token_count
output_token_count
total_token_count
```

Oracle logs instead describe a single GT table and do not need multi-table source-selection fields.

---

## 23. Analyze DataBench table selection

`experiment_dir` is a positional argument.

```bash
python analyze_table_selection.py \
  output/databench_multi_opentab_top10_gpt56sol_bm25
```

Do not use:

```text
--experiment_dir
```

The script can report values such as:

```text
selected-gold rate
answer accuracy
answer accuracy given selected gold
answer accuracy given selected wrong table
```

For final DataBench answer accuracy, still use the official evaluator.

---

## 24. Completion count

DataBench top-10 example:

```bash
LOG_DIR="output/databench_multi_opentab_top10_gpt56sol_bm25"

find "$LOG_DIR/log" \
  -maxdepth 1 \
  -name 'databench-*-0.json' \
  | wc -l
```

Expected full completion:

```text
522
```

CMoney:

```bash
LOG_DIR="output/cmoney_oracle_gpt56sol"

find "$LOG_DIR/log" \
  -maxdepth 1 \
  -name 'cmoney-*-0.json' \
  | wc -l
```

Expected full completion:

```text
245
```

---

# Maintenance

## 25. Syntax checks

```bash
cd ~/table_rag_multi
conda activate tablerag

python -m py_compile run.py
python -m py_compile agent/model.py
python -m py_compile agent/retriever.py
python -m py_compile agent/rag_agent_multi.py
python -m py_compile agent/rag_agent_oracle.py
python -m py_compile data/cmoney/prepare_cmoney_oracle.py
```

---

## 26. CPU / RAM / disk checks

Current process:

```bash
PID=$(pgrep -n -f "python run.py")
ps -p "$PID" -o pid,etime,%cpu,%mem,rss,cmd
```

Continuous:

```bash
watch -n 2 'PID=$(pgrep -n -f "python run.py"); ps -p "$PID" -o pid,etime,%cpu,%mem,rss,cmd'
```

Disk:

```bash
df -h
```

Useful directory sizes:

```bash
du -sh db/* 2>/dev/null | sort -h
du -sh output/* 2>/dev/null | sort -h
```

---

## 27. Suggested output-folder naming

```text
DataBench top-1 GPT:
output/databench_multi_opentab_top1_gpt56sol_bm25

DataBench top-10 GPT:
output/databench_multi_opentab_top10_gpt56sol_bm25

DataBench all-80 GPT:
output/databench_multi_all80_gpt56sol_bm25

DataBench Oracle GPT:
output/databench_oracle_gpt56sol

CMoney top-10 GPT:
output/cmoney_top10_gpt56sol_bm25

CMoney Oracle GPT:
output/cmoney_oracle_gpt56sol

Gemini runs:
use a distinct name containing gemini3flash
```

Always use a different output folder for different experiment conditions.

---

## 28. Experiment-comparison rules

For fair comparison:

```text
Multi vs Oracle:
- same LLM
- same top_k
- same max_encode_cell
- same max_depth / max_tokens / temperature
- same general reasoning instructions
- only table availability differs

GPT vs Gemini:
- same dataset
- same candidate_k when applicable
- same TableRAG retrieval settings
- same prompt condition
- report any model-specific reasoning setting explicitly
```

CMoney Oracle uses the annotated/base GT table. Extended GT tables are useful for outer table-retrieval/table-selection evaluation, but should not turn the single-table Oracle baseline back into a multi-table selection problem.
