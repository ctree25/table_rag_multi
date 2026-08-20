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

import os
import json
import pickle
import hashlib
import fcntl
from typing import Optional, List, Any
from collections import Counter

import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Process-local BM25 cache.
# The cache survives across questions handled by the same worker process.
# With n_worker > 1, each worker has its own independent memory cache.
#
# Cross-process / cross-run reuse is handled by the persistent disk cache
# in self.db_dir / "bm25_cache_v1".
_BM25_CACHE_VERSION = 1
_BM25_RETRIEVER_CACHE = {}


class Retriever:
    def __init__(self, agent_type, mode, embed_model_name, top_k = 5, max_encode_cell = 10000, db_dir = 'db/', verbose = False):
        self.agent_type = agent_type
        self.mode = mode
        self.embed_model_name = embed_model_name
        self.schema_retriever = None
        self.cell_retriever = None
        self.row_retriever = None
        self.column_retriever = None
        self.top_k = top_k
        self.max_encode_cell = max_encode_cell
        self.db_dir = db_dir
        self.verbose = verbose
        os.makedirs(db_dir, exist_ok=True)

        if self.mode == 'bm25':
            self.embedder = None
        elif 'text-embedding' in self.embed_model_name:
            self.embedder = OpenAIEmbeddings(model=self.embed_model_name)
        elif 'gecko' in self.embed_model_name: # VertexAI
            from langchain_google_vertexai import VertexAIEmbeddings
            self.embedder = VertexAIEmbeddings(model_name=self.embed_model_name)
        else:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)

    def init_retriever(self, table_id, df):
        self.df = df
        if 'TableRAG' in self.agent_type:
            self.schema_retriever = self.get_retriever('schema', table_id, self.df)
            self.cell_retriever = self.get_retriever('cell', table_id, self.df)
        elif self.agent_type == 'TableSampling':
            max_row = max(1, self.max_encode_cell // 2 // len(self.df.columns))
            self.df = self.df.iloc[:max_row]
            self.row_retriever = self.get_retriever('row', table_id, self.df)
            self.column_retriever = self.get_retriever('column', table_id, self.df)

    def _bm25_cache_identity(self, data_type, table_id, df):
        """
        Build a stable cache identity without scanning all cell values.

        The identity includes:
        - cache implementation version
        - data_type
        - stable table_id
        - max_encode_cell
        - dataframe shape / columns / dtypes

        top_k is intentionally excluded because it only changes retrieval-time k.
        """
        payload = {
            "version": _BM25_CACHE_VERSION,
            "data_type": str(data_type),
            "table_id": str(table_id),
            "max_encode_cell": int(self.max_encode_cell),
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns.tolist()],
            "dtypes": [str(t) for t in df.dtypes.tolist()],
        }

        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        digest = hashlib.sha256(raw).hexdigest()

        # Keep the original tuple semantics for the in-process cache.
        memory_key = (
            str(data_type),
            str(table_id),
            int(self.max_encode_cell),
            digest,
        )

        return memory_key, digest, payload

    def _load_or_build_bm25(self, data_type, table_id, df, docs=None):
        """
        BM25 cache lookup order:

            1. process-local memory
            2. persistent disk cache
            3. build from documents and save to disk

        A Linux file lock prevents multiple workers from building the same
        missing cache entry at the same time.
        """
        cache_key, cache_digest, cache_identity = (
            self._bm25_cache_identity(data_type, table_id, df)
        )

        # --------------------------------------------------------------
        # 1. Same-process memory cache
        # --------------------------------------------------------------
        if cache_key in _BM25_RETRIEVER_CACHE:
            bm25_retriever = _BM25_RETRIEVER_CACHE[cache_key]
            bm25_retriever.k = self.top_k

            if self.verbose:
                print(
                    f'BM25 memory cache hit: '
                    f'{table_id} / {data_type}'
                )

            return bm25_retriever

        # --------------------------------------------------------------
        # 2. Persistent disk cache
        # --------------------------------------------------------------
        cache_dir = os.path.join(
            self.db_dir,
            f"bm25_cache_v{_BM25_CACHE_VERSION}",
        )
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(
            cache_dir,
            f"{data_type}_{cache_digest}.pkl",
        )
        lock_path = cache_path + ".lock"

        # `run.py` is Linux-based in this project, so fcntl is appropriate.
        # The lock is per cache entry, so different tables can still build
        # concurrently.
        with open(lock_path, "a+") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)

            try:
                # Another worker may have created the cache while we waited,
                # so always re-check after acquiring the lock.
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "rb") as fp:
                            saved = pickle.load(fp)

                        if (
                            isinstance(saved, dict)
                            and saved.get("identity") == cache_identity
                            and "retriever" in saved
                        ):
                            bm25_retriever = saved["retriever"]
                            bm25_retriever.k = self.top_k

                            _BM25_RETRIEVER_CACHE[cache_key] = (
                                bm25_retriever
                            )

                            if self.verbose:
                                print(
                                    f'BM25 disk cache hit: '
                                    f'{table_id} / {data_type}'
                                )

                            return bm25_retriever

                        if self.verbose:
                            print(
                                f'BM25 disk cache stale: '
                                f'{table_id} / {data_type}; rebuild'
                            )

                    except Exception as exc:
                        if self.verbose:
                            print(
                                f'BM25 disk cache load failed: '
                                f'{table_id} / {data_type} | '
                                f'{type(exc).__name__}: {exc}; rebuild'
                            )

                    # Invalid / incompatible cache file.
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

                # ------------------------------------------------------
                # 3. Cache miss -> build exactly as before
                # ------------------------------------------------------
                if self.verbose:
                    print(
                        f'Build BM25 disk cache: '
                        f'{table_id} / {data_type}'
                    )

                if docs is None:
                    docs = self.get_docs(data_type, df)

                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = self.top_k

                _BM25_RETRIEVER_CACHE[cache_key] = bm25_retriever

                # Atomic write: write temp file then rename.
                tmp_path = (
                    cache_path
                    + f".tmp.{os.getpid()}"
                )

                payload = {
                    "identity": cache_identity,
                    "retriever": bm25_retriever,
                }

                try:
                    with open(tmp_path, "wb") as fp:
                        pickle.dump(
                            payload,
                            fp,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                        fp.flush()
                        os.fsync(fp.fileno())

                    os.replace(tmp_path, cache_path)

                except Exception as exc:
                    # Do not fail the experiment if persistence itself fails.
                    # The in-memory BM25 retriever is still valid.
                    if self.verbose:
                        print(
                            f'Warning: BM25 disk cache save failed: '
                            f'{table_id} / {data_type} | '
                            f'{type(exc).__name__}: {exc}'
                        )

                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                return bm25_retriever

            finally:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def get_retriever(self, data_type, table_id, df):
        docs = None

        if self.mode == 'embed' or self.mode == 'hybrid':
            db_dir = os.path.join(
                self.db_dir,
                f'{data_type}_db_{self.max_encode_cell}_' + table_id
            )

            if os.path.exists(db_dir):
                if self.verbose:
                    print(f'Load {data_type} database from {db_dir}')

                db = FAISS.load_local(
                    db_dir,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
            else:
                docs = self.get_docs(data_type, df)
                db = FAISS.from_documents(docs, self.embedder)
                db.save_local(db_dir)

            embed_retriever = db.as_retriever(
                search_kwargs={'k': self.top_k}
            )

        if self.mode == 'bm25' or self.mode == 'hybrid':
            bm25_retriever = self._load_or_build_bm25(
                data_type=data_type,
                table_id=table_id,
                df=df,
                docs=docs,
            )

        if self.mode == 'hybrid':
            # return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.9, 0.1])
            return EnsembleRetriever(
                retrievers=[embed_retriever, bm25_retriever],
                weights=[0.5, 0.5],
            )
        elif self.mode == 'embed':
            return embed_retriever
        elif self.mode == 'bm25':
            return bm25_retriever

    def get_docs(self, data_type, df):
        if data_type == 'schema':
            return self.build_schema_corpus(df)
        elif data_type == 'cell':
            return self.build_cell_corpus(df)
        elif data_type == 'row':
            return self.build_row_corpus(df)
        elif data_type == 'column':
            return self.build_column_corpus(df)

    def build_schema_corpus(self, df):
        docs = []

        for col_name, col in df.items():
            non_null = col.dropna()

            is_numeric = (
                pd.api.types.is_numeric_dtype(col.dtype)
                and not pd.api.types.is_bool_dtype(col.dtype)
            )

            if is_numeric:
                if non_null.empty:
                    min_value = None
                    max_value = None
                else:
                    min_value = non_null.min()
                    max_value = non_null.max()

                    if isinstance(min_value, np.generic):
                        min_value = min_value.item()

                    if isinstance(max_value, np.generic):
                        max_value = max_value.item()

                result = {
                    "column_name": str(col_name),
                    "dtype": str(col.dtype),
                    "min": min_value,
                    "max": max_value,
                }

            else:
                example_cells = (
                    non_null.astype(str)
                    .value_counts()
                    .index
                    .tolist()[:3]
                )

                result = {
                    "column_name": str(col_name),
                    "dtype": str(col.dtype),
                    "cell_examples": example_cells,
                }

            result_text = json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

            docs.append(
                Document(
                    page_content=str(col_name),
                    metadata={
                        "result_text": result_text,
                    },
                )
            )

        return docs

    def build_cell_corpus(self, df):
        docs = []
        categorical_cell_counts = Counter()

        for col_name, col in df.items():
            non_null = col.dropna()

            is_numeric = (
                pd.api.types.is_numeric_dtype(col.dtype)
                and not pd.api.types.is_bool_dtype(col.dtype)
            )

            if is_numeric:
                if non_null.empty:
                    min_value = None
                    max_value = None
                else:
                    min_value = non_null.min()
                    max_value = non_null.max()

                    if isinstance(min_value, np.generic):
                        min_value = min_value.item()

                    if isinstance(max_value, np.generic):
                        max_value = max_value.item()

                result = {
                    "column_name": str(col_name),
                    "dtype": str(col.dtype),
                    "min": min_value,
                    "max": max_value,
                }

                docs.append(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            else:
                # Only the most frequent max_encode_cell values from
                # each column can possibly enter the global top list.
                value_counts = (
                    non_null.astype(str)
                    .value_counts()
                    .head(self.max_encode_cell)
                )

                for cell_value, count in value_counts.items():
                    result = {
                        "column_name": str(col_name),
                        "cell_value": str(cell_value),
                    }

                    result_text = json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    )

                    categorical_cell_counts[result_text] += int(count)

        remaining_slots = max(
            0,
            self.max_encode_cell - len(docs),
        )

        docs.extend(
            cell
            for cell, _ in categorical_cell_counts.most_common(
                remaining_slots
            )
        )

        return [
            Document(page_content=doc)
            for doc in docs
        ]

    def build_row_corpus(self, df):
        row_docs = []
        for row_id, (_, row) in enumerate(df.iterrows()):
            row_text = '|'.join(str(cell) for cell in row)
            row_doc = Document(page_content=row_text, metadata={'row_id': row_id})
            row_docs.append(row_doc)
        return row_docs

    def build_column_corpus(self, df):
        col_docs = []
        for col_id, (_, column) in enumerate(df.items()):
            col_text = '|'.join(str(cell) for cell in column)
            col_doc = Document(page_content=col_text, metadata={'col_id': col_id})
            col_docs.append(col_doc)
        return col_docs

    def retrieve_schema(self, query):
        results = self.schema_retriever.invoke(query)
        observations = [doc.metadata['result_text'] for doc in results]
        return observations

    def retrieve_cell(self, query):
        results = self.cell_retriever.invoke(query)
        observations = [doc.page_content for doc in results]
        return observations

    def sample_rows_and_columns(self, query):
        # Apply row sampling
        row_results = self.row_retriever.invoke(query)
        row_ids = sorted([doc.metadata['row_id'] for doc in row_results])
        # Apply column sampling
        col_results = self.column_retriever.invoke(query)
        col_ids = sorted([doc.metadata['col_id'] for doc in col_results])
        # Return sampled rows and columns
        return self.df.iloc[row_ids, col_ids]