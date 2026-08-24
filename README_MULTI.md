**# README — TableRAG Multi-Table**

**## Repository**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

\`\`\`

The repository currently supports two reasoning settings.

**### Multi-table**

\`\`\`text

question

→ outer table retrieval / candidate tables

→ query expansion once per question

→ schema/cell BM25 independently inside each candidate table

→ multi-table ReAct over df1 ... dfK

→ {"table\_source": "dfN", "answer": ...}

\`\`\`

**### Oracle**

\`\`\`text

question

→ annotated GT table directly

→ normal TableRAG schema/cell retrieval inside that table

→ single-table ReAct over df

→ answer

\`\`\`

Oracle only removes outer table-selection uncertainty. Schema retrieval, cell retrieval, and downstream reasoning are still performed normally.

\---

**## 1. Environment**

If the environment already exists:

\`\`\`bash

conda activate tablerag

cd \~/table\_rag\_multi

\`\`\`

If rebuilding:

\`\`\`bash

conda create -n tablerag python=3.10 -y

conda activate tablerag

pip install \\

  langchain==0.2.16 \\

  langchain-core==0.2.38 \\

  langchain-community==0.2.16 \\

  langchain-openai==0.1.23 \\

  langchain-google-vertexai==1.0.10 \\

  rank-bm25==0.2.2 \\

  openai \\

  google-auth \\

  pyarrow

\`\`\`

\---

**## 2. LLM backends**

\`agent/model.py\` currently supports both Azure/OpenAI-compatible GPT and Vertex AI Gemini.

**### 2.1 Azure OpenAI**

\`\`\`bash

export AZURE\_OPENAI\_ENDPOINT="https\://development-table-qa-resource.openai.azure.com/openai/v1/"

export AZURE\_OPENAI\_API\_KEY="\<YOUR\_AZURE\_OPENAI\_API\_KEY>"

export AZURE\_OPENAI\_CONTEXT\_LIMIT="128000"

\`\`\`

Current deployment used in experiments:

\`\`\`text

gpt-5.6-sol

\`\`\`

**### 2.2 Vertex AI Gemini**

Current tested model:

\`\`\`text

gemini-3-flash-preview

\`\`\`

The Vertex OpenAI-compatible endpoint uses the global location. Keep credentials in a local config such as \`agent/gemini\_key.py\`; do not commit the service-account private key.

Conceptually:

\`\`\`python

GCP\_PROJECT\_ID = "\<YOUR\_PROJECT>"

GCP\_LOCATION = "global"

GEMINI\_MODEL\_NAME = "gemini-3-flash-preview"

GCP\_SERVICE\_ACCOUNT\_INFO = {...}

\`\`\`

Direct connectivity test:

\`\`\`bash

python - <<'PY'

from agent.model import Model

m = Model("gemini-3-flash-preview")

text = m.query(

    "Reply with exactly: GEMINI\_OK",

    temperature=0,

    max\_tokens=32,

)

print("response:", text)

print("usage:", m.last\_usage)

PY

\`\`\`

Expected response:

\`\`\`text

GEMINI\_OK

\`\`\`

For a GPT/Gemini experiment pair, keep all TableRAG settings identical and change only \`--model\_name\` unless the experiment explicitly studies model-specific reasoning settings.

\---

**## 3. Important parameters**

\`\`\`text

candidate\_k = number of OUTER candidate tables given to TableRAGMulti

top\_k      = number of INNER schema/cell BM25 hits per expansion query

sc         = self-consistency runs

\`\`\`

Do not confuse \`candidate\_k\` with \`top\_k\`.

\`TableRAGOracle\` does not use \`candidate\_k\` because the GT table is supplied directly.

\---

**# DataBench**

**## 4. DataBench paths**

Tables:

\`\`\`text

data/databench\_test80/data/\<table\_id>/all.parquet

\`\`\`

OpenTab retrieval:

\`\`\`text

data/databench\_test80/retrieval/hybrid\_title\_columns/opentab.json

\`\`\`

Prepared OpenTab top-10 dataset:

\`\`\`text

data/wtq\_databench\_multi\_opentab\_top10.jsonl

\`\`\`

Prepared all-80 dataset currently used by Oracle and all-80 multi-table experiments:

\`\`\`text

data/databench\_test80/wtq\_databench\_multi\_all80.jsonl

\`\`\`

Expected corpus/evaluation size:

\`\`\`text

522 questions

80 tables

\`\`\`

The all-80 JSONL contains, per question:

\`\`\`text

gold\_table\_id

candidate\_tables[0..79]

\`\`\`

For DataBench Oracle, \`TableRAGOracle\` exact-matches \`gold\_table\_id\` against \`candidate\_tables\` and loads the corresponding GT parquet. There is no LLM table selection and no outer retrieval in this branch.

\---

**## 5. Prepare DataBench OpenTab top-10 candidates**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

python prepare\_databench\_multi.py

\`\`\`

Expected output:

\`\`\`text

data/wtq\_databench\_multi\_opentab\_top10.jsonl

\`\`\`

Check:

\`\`\`bash

wc -l data/wtq\_databench\_multi\_opentab\_top10.jsonl

\`\`\`

Expected:

\`\`\`text

522

\`\`\`

The same prepared top-10 JSONL can be used with \`--candidate\_k=1\` or \`--candidate\_k=10\`.

\---

**## 6. DataBench main experiment — candidate K = 10**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/databench\_multi\_opentab\_top10\_gpt56sol\_bm25"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path=data/wtq\_databench\_multi\_opentab\_top10.jsonl \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGMulti" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --candidate\_k=10 \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=522 \\

  --resume\_from=0 \\

  --n\_worker=4 \\

  --verbose=False \\

  2>&1 | tee -a "${LOG\_DIR}\_run.log"

\`\`\`

Resume with the same command and same \`LOG\_DIR\`, keeping:

\`\`\`text

\--load\_exist=True

\`\`\`

Do not reuse one output directory across different model, candidate K, retrieval source, or prompt conditions.

\---

**## 7. DataBench candidate K = 1**

No new dataset preparation is needed.

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/databench\_multi\_opentab\_top1\_gpt56sol\_bm25"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path=data/wtq\_databench\_multi\_opentab\_top10.jsonl \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGMulti" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --candidate\_k=1 \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=522 \\

  --resume\_from=0 \\

  --n\_worker=4 \\

  --verbose=False \\

  2>&1 | tee -a "${LOG\_DIR}\_run.log"

\`\`\`

\---

**## 8. DataBench candidate K = 20

Use the prepared OpenTab top-20 dataset:

```text

data/databench_test80/wtq_databench_multi_opentab_top20.jsonl

```

Check:

```bash

wc -l data/databench_test80/wtq_databench_multi_opentab_top20.jsonl

```

Expected:

```text

522

```

Run GPT:

```bash

cd ~/table_rag_multi
conda activate tablerag

LOG_DIR="output/databench_multi_opentab_top20_gpt56sol_bm25"

mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/databench_test80/wtq_databench_multi_opentab_top20.jsonl" \
  --model_name="gpt-5.6-sol" \
  --agent_type="TableRAGMulti" \
  --retrieve_mode="bm25" \
  --embed_model_name="unused" \
  --log_dir="$LOG_DIR" \
  --db_dir="db" \
  --candidate_k=20 \
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

Do not obtain K=20 by taking the first 20 entries of the all-80 dataset. The all-80 ordering is deterministic table order, not OpenTab retrieval rank.

\---

## 9. Prepare DataBench all-80 candidates**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

python prepare\_databench\_all80.py

\`\`\`

Expected output in the current layout:

\`\`\`text

data/databench\_test80/wtq\_databench\_multi\_all80.jsonl

\`\`\`

Checks:

\`\`\`bash

wc -l data/databench\_test80/wtq\_databench\_multi\_all80.jsonl

find data/databench\_test80/data -name all.parquet | wc -l

\`\`\`

Expected:

\`\`\`text

522

80

\`\`\`

For all-80:

\`\`\`text

gold\_in\_candidates = True for every question

\`\`\`

\`gold\_rank\` / \`selected\_rank\` in this file is the deterministic all-80 position, not an OpenTab retrieval rank.

\---

**## 10. DataBench all-80 one-question test**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/databench\_multi\_all80\_one\_gpt56sol\_bm25"

rm -rf "$LOG\_DIR"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path=data/databench\_test80/wtq\_databench\_multi\_all80.jsonl \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGMulti" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --candidate\_k=80 \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=1 \\

  --resume\_from=0 \\

  --n\_worker=1 \\

  --verbose=True \\

  2>&1 | tee "${LOG\_DIR}\_run.log"

\`\`\`

\---

**## 11. DataBench full all-80 run**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/databench\_multi\_all80\_gpt56sol\_bm25"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path=data/databench\_test80/wtq\_databench\_multi\_all80.jsonl \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGMulti" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --candidate\_k=80 \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=522 \\

  --resume\_from=0 \\

  --n\_worker=4 \\

  --verbose=False \\

  2>&1 | tee -a "${LOG\_DIR}\_run.log"

\`\`\`

\---

**## 12. DataBench Oracle**

Agent:

\`\`\`text

TableRAGOracle

\`\`\`

Prompt:

\`\`\`text

tablerag\_databench\_oracle\_solve\_table\_prompt

\`\`\`

Oracle definition:

\`\`\`text

question

→ data["gold\_table\_id"]

→ exact match inside data["candidate\_tables"]

→ GT parquet

→ normal TableRAG schema/cell retrieval

→ single-table df ReAct

→ answer

\`\`\`

\`run.py\` raw-reads JSONL for \`TableRAGOracle\` instead of passing it through the normal WTQ loader. This preserves fields such as \`gold\_table\_id\`, \`candidate\_tables\`, and CMoney \`table\_path\`.

**### One-question smoke test**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/databench\_oracle\_gpt56sol\_test1"

rm -rf "$LOG\_DIR"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path="data/databench\_test80/wtq\_databench\_multi\_all80.jsonl" \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGOracle" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=1 \\

  --resume\_from=0 \\

  --n\_worker=1 \\

  --verbose=True \\

  2>&1 | tee "${LOG\_DIR}\_run.log"

\`\`\`

**### Full Oracle run**

Change:

\`\`\`text

\--stop\_at=522

\--n\_worker=4

\`\`\`

Oracle does not use \`--candidate\_k\`.

\---

**## 13. DataBench responses.txt and official evaluation**

When \`dataset\_path\` contains \`databench\` and \`sc=1\`, \`run.py\` writes:

\`\`\`text

\<LOG\_DIR>/responses.txt

\`\`\`

Top-10 example:

\`\`\`bash

cp \\

  \~/table\_rag\_multi/output/databench\_multi\_opentab\_top10\_gpt56sol\_bm25/responses.txt \\

  \~/databench\_eval/ours\_output.txt

cd \~/databench\_eval

conda activate tablerag

python eval\_output.py ours\_output.txt

\`\`\`

For final reported DataBench answer accuracy, use the official DataBench/SemEval evaluator rather than simple exact-string comparison.

\---

**# CMoney**

**## 14. CMoney source files**

\`\`\`text

data/cmoney/basic\_problems\_20260116\_answer.csv

data/cmoney/basic\_problems\_20260614\_table\_path.csv

data/cmoney/basic\_problems\_20260616\_retrieval.jsonl

data/cmoney/basic\_problems\_20260616\_extended\_gt\_tables.csv

data/cmoney/tables/

\`\`\`

Prepared multi-table dataset:

\`\`\`text

data/cmoney/wtq\_cmoney\_multi\_top10.jsonl

\`\`\`

Prepared candidate parquets:

\`\`\`text

data/cmoney/prepared\_tables/

\`\`\`

Current eligible experiment set:

\`\`\`text

245 questions

\`\`\`

The \`wtq\_\` prefix is intentional because \`run.py\` infers the task from the dataset path and routes these records through the WTQ-compatible task path.

\---

**## 15. Prepare CMoney multi-table top-10**

The preparation logic uses the existing eligible set based on a non-empty/resolvable base GT CSV and preserves the original retrieval order.

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

python data/cmoney/prepare\_cmoney\_multi.py

\`\`\`

Expected prepared dataset:

\`\`\`text

data/cmoney/wtq\_cmoney\_multi\_top10.jsonl

\`\`\`

Check:

\`\`\`bash

wc -l data/cmoney/wtq\_cmoney\_multi\_top10.jsonl

\`\`\`

Expected:

\`\`\`text

245

\`\`\`

CMoney candidates preserve the raw retrieval order. Do not rerank or backfill a missing raw top-K table with rank K+1 when evaluating a fixed raw top-K condition.

\---

**## 16. CMoney multi-table run**

**### One-question smoke test**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/cmoney\_top10\_gpt56sol\_test1"

rm -rf "$LOG\_DIR"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path="data/cmoney/wtq\_cmoney\_multi\_top10.jsonl" \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGMulti" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --candidate\_k=10 \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=1 \\

  --resume\_from=0 \\

  --n\_worker=1 \\

  --verbose=True \\

  2>&1 | tee "${LOG\_DIR}\_run.log"

\`\`\`

**### Full 245-question run**

Use the same command with:

\`\`\`text

\--stop\_at=245

\--n\_worker=4

\`\`\`

CMoney multi-table final output uses a source-table-aware form internally, e.g.:

\`\`\`json

{"table\_source": "dfN", "answer": "..."}

\`\`\`

CMoney answer accuracy should be evaluated with the CMoney-specific semantic evaluator, not the generic WTQ exact evaluator.

\---

**## 17. Prepare CMoney Oracle**

Script:

\`\`\`text

data/cmoney/prepare\_cmoney\_oracle.py

\`\`\`

The Oracle preparation uses the existing \`wtq\_cmoney\_multi\_top10.jsonl\` as the only question universe, so Oracle and multi-table experiments use exactly the same 245 questions.

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

python data/cmoney/prepare\_cmoney\_oracle.py

\`\`\`

Outputs:

\`\`\`text

data/cmoney/wtq\_cmoney\_oracle.jsonl

data/cmoney/wtq\_cmoney\_oracle\_prepare\_summary.json

data/cmoney/prepared\_tables\_oracle/\*.parquet

\`\`\`

The Oracle JSONL contains only one table-related field per question:

\`\`\`text

table\_path

\`\`\`

It intentionally does not contain candidate tables, retrieval ranks, retrieval scores, or extended-GT alternatives.

Strict preparation guarantees:

\`\`\`text

exactly 245 source records

unique question IDs

all 245 have a non-empty GT path

all GT CSVs exist

all GT parquets are readable

all final table\_path values exist

final qid set exactly matches the 245-question source set

\`\`\`

If any check fails, the script aborts and does not leave an incomplete final Oracle JSONL.

Quick validation:

\`\`\`bash

wc -l data/cmoney/wtq\_cmoney\_oracle.jsonl

python - <<'PY'

import json

from pathlib import Path

p = Path("data/cmoney/wtq\_cmoney\_oracle.jsonl")

rows = [

    json.loads(line)

    for line in p.read\_text(encoding="utf-8").splitlines()

    if line.strip()

]

print("records:", len(rows))

print("with table\_path:", sum(bool(x.get("table\_path")) for x in rows))

print("unique qids:", len({x["question\_id"] for x in rows}))

print(

    "missing files:",

    [x["question\_id"] for x in rows if not Path(x["table\_path"]).exists()],

)

PY

\`\`\`

Expected:

\`\`\`text

records: 245

with table\_path: 245

unique qids: 245

missing files: []

\`\`\`

\---

**## 18. CMoney Oracle run**

Agent:

\`\`\`text

TableRAGOracle

\`\`\`

Prompt:

\`\`\`text

tablerag\_cmoney\_oracle\_solve\_table\_prompt

\`\`\`

The CMoney Oracle branch reads \`data["table\_path"]\` directly and reasons over one dataframe named \`df\`.

The Multi and Oracle prompts should keep the same general reasoning instructions. Domain-specific financial hints were removed from the Multi prompt, so do not give Oracle extra financial hints either. The intended experimental difference is single GT table vs multiple retrieved candidates, not prompt quality.

**### One-question GPT smoke test**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

LOG\_DIR="output/cmoney\_oracle\_gpt56sol\_test1"

rm -rf "$LOG\_DIR"

mkdir -p "$LOG\_DIR"

PYTHONUNBUFFERED=1 python run.py \\

  --dataset\_path="data/cmoney/wtq\_cmoney\_oracle.jsonl" \\

  --model\_name="gpt-5.6-sol" \\

  --agent\_type="TableRAGOracle" \\

  --retrieve\_mode="bm25" \\

  --embed\_model\_name="unused" \\

  --log\_dir="$LOG\_DIR" \\

  --db\_dir="db" \\

  --top\_k=5 \\

  --sc=1 \\

  --max\_encode\_cell=1000 \\

  --max\_depth=10 \\

  --max\_tokens=512 \\

  --temperature=0 \\

  --stop\_at=1 \\

  --resume\_from=0 \\

  --n\_worker=1 \\

  --verbose=True \\

  2>&1 | tee "${LOG\_DIR}\_run.log"

\`\`\`

**### Full Oracle run**

Use the same command with:

\`\`\`text

\--stop\_at=245

\--n\_worker=4

\`\`\`

Oracle does not use \`--candidate\_k\`.

\---

**## 19. Gemini 3.7 Flash experiments

Current model:

```text

gemini-3.7-flash

```

Current reasoning setting:

```bash

export GEMINI_REASONING_EFFORT=low

```

For the current GPT/Gemini comparison, use the lowest supported reasoning level selected for each backend and report the backend-specific setting explicitly. Do not describe the levels as numerically equivalent across model families.

### Gemini 3.7 request compatibility

The current `agent/model.py` contains Gemini-3.7-specific compatibility handling:

```text

- strips temperature / top_p / top_k
- validates that response choices contain non-empty text inside the Tenacity retry
- reports finish_reason for empty generations
- adds a plain-text ReAct compatibility system instruction

```

TableRAG uses an external text protocol:

```text

Thought:
Action: <single-line Python command>
Observation:

```

`python_repl_ast` is executed by the local TableRAG program. It is not registered as a Gemini native API tool. The Gemini 3.7 compatibility instruction therefore explicitly tells the model to emit `Action:` as plain text rather than attempting a native function call.

A failure such as:

```text

finish_reason=malformed_function_call

```

should be debugged with `--n_worker=1` on the specific question before restarting a full experiment. Empty/malformed generations are checked inside the Gemini retry loop instead of causing an immediate `NoneType.message.content` crash.

### DataBench Oracle — Gemini 3.7 Flash

```bash

cd ~/table_rag_multi
conda activate tablerag

export GEMINI_REASONING_EFFORT=low

LOG_DIR="output/databench_oracle_gemini37flash_bm25"

mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/databench_test80/wtq_databench_multi_all80.jsonl" \
  --model_name="gemini-3.7-flash" \
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
  --stop_at=522 \
  --resume_from=0 \
  --n_worker=4 \
  --verbose=False \
  2>&1 | tee -a "${LOG_DIR}_run.log"

```

Oracle does not use `--candidate_k`.

### DataBench ours K=10 — Gemini 3.7 Flash

```bash

cd ~/table_rag_multi
conda activate tablerag

export GEMINI_REASONING_EFFORT=low

LOG_DIR="output/databench_multi_opentab_top10_gemini37flash_bm25"

mkdir -p "$LOG_DIR"

PYTHONUNBUFFERED=1 python run.py \
  --dataset_path="data/databench_test80/wtq_databench_multi_opentab_top10.jsonl" \
  --model_name="gemini-3.7-flash" \
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
  --stop_at=522 \
  --resume_from=0 \
  --n_worker=4 \
  --verbose=False \
  2>&1 | tee -a "${LOG_DIR}_run.log"

```

### Resume a partially completed Gemini run

Do not delete the existing output directory. Reuse the exact same `LOG_DIR` and add:

```text

--load_exist=True

```

Completed per-question JSON logs are reused rather than regenerated.

When `n_worker > 1`, the visible tqdm counter may lag behind the actual number of completed JSON logs if the main process is waiting on a slower future. Check real completion with:

```bash

find "$LOG_DIR/log" \
  -maxdepth 1 \
  -name 'databench-*-0.json' \
  | wc -l

```

For DataBench, full completion is:

```text

522

```

When changing the effective Gemini reasoning configuration or model version, use a fresh query-expansion cache or clear the old expansion cache if the cache identity would otherwise no longer reflect the effective generation setting.

\---

# Caches and efficiency**

**## 20. Persistent BM25 cache**

BM25 now uses two cache levels:

\`\`\`text

process memory

→ disk cache

→ build and save on miss

\`\`\`

Persistent cache files are stored under the BM25 cache directory, e.g.:

\`\`\`text

db/bm25\_cache\_v1/

  schema\_\<hash>.pkl

  cell\_\<hash>.pkl

  .lock

\`\`\`

With \`--verbose=True\`, expect messages such as:

\`\`\`text

Build BM25 disk cache: ...

BM25 disk cache hit: ...

BM25 memory cache hit: ...

\`\`\`

The cache key includes the stable table identity and table representation metadata, including \`max\_encode\_cell\`. \`top\_k\` is not part of the underlying BM25 index identity, so changing only retrieval top-K does not require rebuilding the index.

Typical reuse:

\`\`\`text

same process                  → memory hit

restart run.py                → disk hit

change LLM model              → BM25 cache reusable

change top\_k                  → BM25 cache reusable

reuse same table in new run   → BM25 cache reusable

\`\`\`

If the underlying table representation or a cache-affecting encoding setting changes, clear it:

\`\`\`bash

rm -rf db/bm25\_cache\_v1

\`\`\`

\---

**## 21. Persistent query-expansion cache**

\`TableRAGMulti\` also has a persistent query-expansion cache:

\`\`\`text

db/query\_expansion\_cache\_v1/

  column\_\<hash>.json

  cell\_\<hash>.json

  .lock

\`\`\`

The cache identity includes important generation settings such as:

\`\`\`text

expansion prompt

column vs cell expansion

remove\_numeric

model\_name

temperature

top\_p

max\_tokens

reasoning\_effort

cache version

\`\`\`

Therefore GPT and Gemini expansions are intentionally kept separate.

Cached expansion usage is re-applied to token accounting so logs remain comparable with the original generation call.

Clear when intentionally changing the expansion behavior or effective model reasoning configuration:

\`\`\`bash

rm -rf db/query\_expansion\_cache\_v1

\`\`\`

Current implementation note: the persistent expansion cache was added to the multi-table agent. \`TableRAGOracle\` still follows the base single-table TableRAG expansion path unless the cache logic is explicitly refactored into a shared component.

\---

**## 22. ReAct evidence-token optimization**

\`agent/rag\_agent\_multi.py\` sends the full retrieved table evidence only in ReAct round 1.

\`\`\`text

Round 1:

question + candidate map + all retrieved evidence

Round 2+:

question + candidate map

\+ previous Thought/Action/Observation history

\+ no repeated full table evidence

\`\`\`

Important token fields include:

\`\`\`text

init\_prompt\_token\_count

continuation\_prompt\_token\_count

input\_token\_count

output\_token\_count

total\_token\_count

\`\`\`

\`run.py\` also aggregates completed per-question logs into:

\`\`\`text

\<LOG\_DIR>/token\_usage.json

\`\`\`

\---

**# Logs and evaluation**

**## 23. Important multi-table per-question log fields**

Example:

\`\`\`text

output/\<experiment>/log/databench-0-0.json

\`\`\`

Useful fields:

\`\`\`text

id

query

answer

label

raw\_final\_answer

table\_source

selected\_table\_id

selected\_rank

gold\_table\_id

selected\_is\_gold

gold\_in\_candidates

candidate\_k

candidate\_table\_map

candidate\_table\_ids

column\_queries

cell\_queries

retrieved\_columns

retrieved\_cells

solution

n\_iter

init\_prompt\_token\_count

continuation\_prompt\_token\_count

input\_token\_count

output\_token\_count

total\_token\_count

\`\`\`

Oracle logs instead describe a single GT table and do not need multi-table source-selection fields.

\---

**## 24. Analyze DataBench table selection**

\`experiment\_dir\` is a positional argument.

\`\`\`bash

python analyze\_table\_selection.py \\

  output/databench\_multi\_opentab\_top10\_gpt56sol\_bm25

\`\`\`

Do not use:

\`\`\`text

\--experiment\_dir

\`\`\`

The script can report values such as:

\`\`\`text

selected-gold rate

answer accuracy

answer accuracy given selected gold

answer accuracy given selected wrong table

\`\`\`

For final DataBench answer accuracy, still use the official evaluator.

\---

**## 25. Completion count**

DataBench top-10 example:

\`\`\`bash

LOG\_DIR="output/databench\_multi\_opentab\_top10\_gpt56sol\_bm25"

find "$LOG\_DIR/log" \\

  -maxdepth 1 \\

  -name 'databench-\*-0.json' \\

  | wc -l

\`\`\`

Expected full completion:

\`\`\`text

522

\`\`\`

CMoney:

\`\`\`bash

LOG\_DIR="output/cmoney\_oracle\_gpt56sol"

find "$LOG\_DIR/log" \\

  -maxdepth 1 \\

  -name 'cmoney-\*-0.json' \\

  | wc -l

\`\`\`

Expected full completion:

\`\`\`text

245

\`\`\`

\---

**# Maintenance**

**## 26. Syntax checks**

\`\`\`bash

cd \~/table\_rag\_multi

conda activate tablerag

python -m py\_compile run.py

python -m py\_compile agent/model.py

python -m py\_compile agent/retriever.py

python -m py\_compile agent/rag\_agent\_multi.py

python -m py\_compile agent/rag\_agent\_oracle.py

python -m py\_compile data/cmoney/prepare\_cmoney\_oracle.py

\`\`\`

\---

**## 27. CPU / RAM / disk checks**

Current process:

\`\`\`bash

PID=$(pgrep -n -f "python run.py")

ps -p "$PID" -o pid,etime,%cpu,%mem,rss,cmd

\`\`\`

Continuous:

\`\`\`bash

watch -n 2 'PID=$(pgrep -n -f "python run.py"); ps -p "$PID" -o pid,etime,%cpu,%mem,rss,cmd'

\`\`\`

Disk:

\`\`\`bash

df -h

\`\`\`

Useful directory sizes:

\`\`\`bash

du -sh db/\* 2>/dev/null | sort -h

du -sh output/\* 2>/dev/null | sort -h

\`\`\`

\---

**## 28. Suggested output-folder naming**

\`\`\`text

DataBench top-1 GPT:

output/databench\_multi\_opentab\_top1\_gpt56sol\_bm25

DataBench top-10 GPT:

output/databench\_multi\_opentab\_top10\_gpt56sol\_bm25

DataBench all-80 GPT:

output/databench\_multi\_all80\_gpt56sol\_bm25

DataBench Oracle GPT:

output/databench\_oracle\_gpt56sol

CMoney top-10 GPT:

output/cmoney\_top10\_gpt56sol\_bm25

CMoney Oracle GPT:

output/cmoney\_oracle\_gpt56sol

Gemini 3.7 runs:

use a distinct name containing gemini37flash

Examples:

output/databench_oracle_gemini37flash_bm25

output/databench_multi_opentab_top10_gemini37flash_bm25

\`\`\`

Always use a different output folder for different experiment conditions.

\---

**## 29. Experiment-comparison rules**

For fair comparison:

\`\`\`text

Multi vs Oracle:

\- same LLM

\- same top\_k

\- same max\_encode\_cell

\- same max\_depth / max\_tokens / temperature

\- same general reasoning instructions

\- only table availability differs

GPT vs Gemini:

\- same dataset

\- same candidate\_k when applicable

\- same TableRAG retrieval settings

\- same prompt condition

\- report any model-specific reasoning setting explicitly

- omit backend-unsupported sampling parameters rather than forcing identical API arguments

\`\`\`

CMoney Oracle uses the annotated/base GT table. Extended GT tables are useful for outer table-retrieval/table-selection evaluation, but should not turn the single-table Oracle baseline back into a multi-table selection problem.