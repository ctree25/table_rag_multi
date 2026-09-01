# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

pyreact_solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

Tool description:
- `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid one-line python command.

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

You are working with the following table:
{table}

Please answer the question: {query}.

Begin!
'''

tablerag_extract_column_prompt = '''

Given a CMoney table with Traditional Chinese column names, I want to answer a question: {query}

Since I cannot view the table directly, please suggest some column names or short column-name phrases that might contain the necessary data to answer this question.

Important:
- Prefer Traditional Chinese terms that are likely to appear directly in the table schema.
- Do not translate Chinese concepts into English.
- Keep abbreviations, ticker symbols, currency codes, or technical terms from the question when they may appear directly in the schema, such as EPS, JPY, USD.
- Use concise column-like terms rather than full sentences.

Please answer with a list in JSON format without any additional explanation.

Example:

["日期", "代號", "名稱", "收盤價", "成交量"]

'''

tablerag_extract_cell_prompt = '''

Given a CMoney table, I want to answer a question: {query}

Please extract some keywords from the question that might appear directly as values in the table cells and help answer the question.

Important:
- Keep the original wording and language used in the question whenever possible.
- Do not translate Chinese keywords into English.
- Keep abbreviations, ticker symbols, company names, currency codes, and other identifiers exactly as they appear in the question.
- The keywords should be categorical or textual values rather than numerical values.
- Do not include dates or other purely numerical values.

Please answer with a list of keywords in JSON format without any additional explanation.

Example:

["日圓", "JPY"]

'''
tablerag_solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

Tool description:
- `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.

Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

{schema_retrieval_result}

{cell_retrieval_result}

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- The answer after "Final Answer:" must follow one of these formats:
  1. Boolean: True or False
  2. Single category or number: output the value directly
  3. Multiple categories or numbers: output them as [item1, item2, ...]
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

Now, please use ``python_repl_ast` with the column names and cell values above to answer the question: {query}

Begin!
'''

tablerag_multi_solve_table_prompt = '''
You are working with {num_tables} candidate pandas dataframes in Python.
The dataframe names are df1, df2, ..., df{num_tables}.

Your task is to:
1. Determine which candidate table contains the information needed to answer the question.
2. Use `python_repl_ast` to inspect and reason over the appropriate dataframe.
3. Return both the selected dataframe and the final answer.

Question: {query}

Tool description:
- `python_repl_ast`: A Python interactive shell.
- Input must be a valid single-line Python command.
- You may inspect any of df1, df2, ..., df{num_tables}.
- Do not assume that df1 is always the correct table.
- Do not combine unrelated values from different candidate tables.
- Each question in this dataset is answerable from one gold table.

Candidate table map:
{candidate_table_map}

The following evidence was independently retrieved from each candidate table.

{table_evidence}

Strictly follow this interaction format:
Thought: explain what should be checked
Action: a single-line Python command
Observation: the result of the action
... (repeat as needed)
Thought: verify the selected table and final answer
Final Answer: {{"table_source": "dfN", "answer": <answer>}}

Final output requirements:
- The last line must contain only the Final Answer JSON object.
- `table_source` must be exactly one of df1, df2, ..., df{num_tables}.
- Do not output the table name in `table_source`; output its dataframe name.
- `answer` must follow one of these formats:
  1. Boolean: true or false
  2. Single category: a JSON string
  3. Single number: a JSON number
  4. Multiple categories or numbers: a JSON list
- Do not include units, explanations, labels, or extra text in `answer`
  unless they are explicitly part of the requested answer.
- Do not output the answer through Python.
- Do not use markdown.

Begin!
'''

tablerag_multi_solve_table_prompt_paper = '''
You are working with {num_tables} candidate pandas dataframes in Python.
The dataframe names are df1, df2, ..., df{num_tables}.

Your task is to:
1. Determine which candidate table contains the information needed to answer the question.
2. Use `python_repl_ast` to inspect and reason over the appropriate dataframe.
3. Return the final answer.

Tool description:
- `python_repl_ast`: A Python interactive shell.
- Input must be a valid single-line Python command.
- You may inspect any of df1, df2, ..., df{num_tables}.

Candidate table map:
{candidate_table_map}

The following evidence was independently retrieved from each candidate table.
{table_evidence}

Strictly follow this interaction format:
Thought: explain what should be checked
Action: a single-line Python command
Observation: the result of the action
... (repeat as needed)
Thought: before giving the final answer, you should think about the observations
Final Answer:  the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.
- `answer` must follow one of these formats:
  1. Boolean: true or false
  2. Single category: a JSON string
  3. Single number: a JSON number
  4. Multiple categories or numbers: a JSON list

Please answer the question: {query}.

Begin!
'''
# CMoney Multi pandas
tablerag_multi_cmoney_solve_table_prompt = '''
You are working with {num_tables} candidate pandas dataframes in Python.
The dataframe names are df1, df2, ..., df{num_tables}.

Your task is to:
1. Determine which candidate table contains the information needed to answer the question.
2. Use `python_repl_ast` to inspect and reason over the appropriate dataframe.
3. Return both the selected dataframe and the final answer.

Question: {query}

Tool description:
- `python_repl_ast`: A Python interactive shell.
- Input must be a valid single-line Python command.
- You may inspect any of df1, df2, ..., df{num_tables}.
- Do not assume that df1 is always the correct table.
- Do not combine unrelated values from different candidate tables.
- Each question in this dataset is answerable from one gold table.

Candidate table map:
{candidate_table_map}

The following evidence was independently retrieved from each candidate table.

{table_evidence}

Strictly follow this interaction format:
Thought: explain what should be checked
Action: a single-line Python command
Observation: the result of the action
... (repeat as needed)
Thought: verify the selected table and final answer
Final Answer: {{"table_source": "dfN", "answer": <answer>}}

Final output requirements:
- The last line must contain only the Final Answer JSON object.
- `table_source` must be exactly one of df1, df2, ..., df{num_tables}.
- Do not output the table name in `table_source`; output its dataframe name.
- Do not include explanations, labels, or extra text in `answer`
  unless they are explicitly part of the requested answer.
- Do not output the answer through Python.
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

# DataBench Oracle
tablerag_databench_oracle_solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

Tool description:
- `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.

Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

{schema_retrieval_result}

{cell_retrieval_result}

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- The answer after "Final Answer:" must follow one of these formats:
  1. Boolean: True or False
  2. Single category or number: output the value directly
  3. Multiple categories or numbers: output them as [item1, item2, ...]
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

Now, please use ``python_repl_ast` with the column names and cell values above to answer the question: {query}

Begin!
'''

# CMoney Oracle pandas
tablerag_cmoney_oracle_solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

Tool description:
- `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.

Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

{schema_retrieval_result}

{cell_retrieval_result}

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

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

Now, please use ``python_repl_ast` with the column names and cell values above to answer the question: {query}

Begin!
'''
