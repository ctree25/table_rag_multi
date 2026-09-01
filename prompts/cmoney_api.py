"""Prompts for CMoney schema-only retrieval + SQL API reasoning."""

CMONEY_SCHEMA_SQL_SOLVE_PROMPT = r'''
You are working with several candidate CMoney database tables.

Your task is to:
1. Determine which candidate table contains the information needed to answer the question.
2. Use read-only SQL to inspect and reason over the appropriate table.
3. Return both the selected table and the final answer.

Question: {query}

Tool description:
- You can interact with the database only by issuing read-only SQL.
- Input must be a valid single-line SELECT statement.
- You may inspect any of the candidate tables listed below.
- Do not assume that the highest-ranked candidate is always the correct table.
- Do not combine unrelated values from different candidate tables.
- Each question in this dataset is answerable from one gold table.

Candidate tables (retrieval rank order):
{candidate_table_map}

The following schema evidence was independently retrieved from each candidate table.

{table_evidence}

Action format:
Action: SELECT ...;

Example table syntax:
Action: SELECT TOP 20 * FROM [月營收速選];

SQL rules:
- Read-only SELECT statements only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, EXEC, or other write/admin statements.

Strictly follow this interaction format:
Thought: explain what should be checked
Action: a single-line SELECT statement
Observation: the result of the action
... (repeat as needed)
Thought: verify the selected table and final answer
Final Answer: {{"table_source": "EXACT_CANDIDATE_TABLE_NAME", "answer": <answer>}}

Final output requirements:
- The last line must contain only the Final Answer JSON object after "Final Answer:".
- `table_source` must be exactly one of the candidate table names.
- Do not include explanations, labels, or extra text in `answer`
  unless they are explicitly part of the requested answer.
- Apply the answer formatting rules below to the final answer.
- Do not infer or convert units unless required by the question.
- For multiple requested values, preserve the question order.
- Do not use markdown.

【answer格式規則 (務必遵守)】
1. 不四捨五入、不截斷；僅可去除小數末尾多餘 0 (58.524650→58.52465, 0.20→0.2)。
2. 單位僅在題目中「明確出現」或題目含「幾X」量詞時添加：億元、百萬、千、元、%、張、家、天、筆、檔、人等；添加時數值緊貼單位，無空格與括號。
3. 未明示單位且無「幾X」量詞 → 不加單位；EPS 若題目未含 “元”/“(元)” → 不加元。
4. 題目含「比率 / 比例 / 漲幅 / 成長率」或含 %：若答案未帶 % 則補上；已有則保留。
5. 不推測或換算單位，不改變量級。
6. 多重答案題：各部分依題目詢問順序，單獨套用規則後用半形逗號連接，無空格 (例：499張,2039張)。
7. 日期答案正規化：yyyy/m→yyyy/MM；yyyy/m/d→yyyy/MM/DD；不要臆造不存在的日；非日期型答案不得附帶題目日期。
8. 保留負號；不使用千分位逗號。
9. 若答案本身已含正確單位保持原樣；若為括號形式如 6690(百萬) → 改 6690百萬。

Begin!
'''

CMONEY_SCHEMA_SQL_ORACLE_SOLVE_PROMPT = r'''
You are working with one CMoney database table.
The correct table needed to answer the question has already been provided.

Your task is to:
1. Use read-only SQL to inspect and reason over the provided table.
2. Return the provided table and the final answer.

Question: {query}

Table:
{table_name}

The following schema evidence was retrieved from the table.

{table_evidence}

Tool description:
- You can interact with the database only by issuing read-only SQL.
- Input must be a valid single-line SELECT statement.
- Use only the provided table.
- Use the exact table and column names shown above and in the schema evidence.

Action format:
Action: SELECT ...;

Example table syntax:
Action: SELECT TOP 20 * FROM [月營收速選];

SQL rules:
- Read-only SELECT statements only.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, EXEC, or other write/admin statements.
- Query only the provided table.

Strictly follow this interaction format:
Thought: explain what should be checked
Action: a single-line SELECT statement
Observation: the result of the action
... (repeat as needed)
Thought: verify the observations and final answer
Final Answer: {{"table_source": "{table_name}", "answer": <answer>}}

Final output requirements:
- The last line must contain only the Final Answer JSON object after "Final Answer:".
- `table_source` must be exactly "{table_name}".
- Do not include explanations, labels, or extra text in `answer`
  unless they are explicitly part of the requested answer.
- Apply the answer formatting rules below to the final answer.
- Do not infer or convert units unless required by the question.
- For multiple requested values, preserve the question order.
- Do not use markdown.

【answer格式規則 (務必遵守)】
1. 不四捨五入、不截斷；僅可去除小數末尾多餘 0 (58.524650→58.52465, 0.20→0.2)。
2. 單位僅在題目中「明確出現」或題目含「幾X」量詞時添加：億元、百萬、千、元、%、張、家、天、筆、檔、人等；添加時數值緊貼單位，無空格與括號。
3. 未明示單位且無「幾X」量詞 → 不加單位；EPS 若題目未含 “元”/“(元)” → 不加元。
4. 題目含「比率 / 比例 / 漲幅 / 成長率」或含 %：若答案未帶 % 則補上；已有則保留。
5. 不推測或換算單位，不改變量級。
6. 多重答案題：各部分依題目詢問順序，單獨套用規則後用半形逗號連接，無空格 (例：499張,2039張)。
7. 日期答案正規化：yyyy/m→yyyy/MM；yyyy/m/d→yyyy/MM/DD；不要臆造不存在的日；非日期型答案不得附帶題目日期。
8. 保留負號；不使用千分位逗號。
9. 若答案本身已含正確單位保持原樣；若為括號形式如 6690(百萬) → 改 6690百萬。

Begin!
'''