"""CMoney schema-only retriever with Jina v3 + BM25 hybrid retrieval.

This retriever indexes only table_schema_20250122.csv. It reuses the original
TableRAG Retriever's FAISS/BM25 cache machinery, but bypasses its generic
HuggingFaceEmbeddings initialization so jinaai/jina-embeddings-v3 can be loaded
with trust_remote_code=True and asymmetric retrieval tasks:
- documents/schema columns: retrieval.passage
- queries: retrieval.query
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from agent.retriever import Retriever


class JinaV4Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 64,
        encode_dim: int = 1024,
    ):
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        self.device = device
        self.batch_size = batch_size
        self.encode_dim = encode_dim

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        vectors = self.model.encode(
            sentences=texts,
            task="retrieval",
            prompt_name="passage",
            batch_size=self.batch_size,
            truncate_dim=self.encode_dim,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        vector = self.model.encode(
            sentences=[text],
            task="retrieval",
            prompt_name="query",
            batch_size=1,
            truncate_dim=self.encode_dim,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()


class CMoneySchemaRetriever(Retriever):
    """Retrieve relevant columns independently inside each candidate table."""

    REQUIRED_COLUMNS = {"table_id", "table", "column", "type", "nullable"}
    CACHE_VERSION = 2

    def __init__(
        self,
        *,
        schema_path: str | Path,
        mode: str,
        embed_model_name: str,
        top_k: int,
        db_dir: str,
        embed_device: str = "cpu",
        embed_batch_size: int = 64,
        encode_dim: int = 1024,
        verbose: bool = False,
    ) -> None:
        requested_mode = str(mode)
        requested_model = str(embed_model_name)

        # Important: initialize the inherited cache machinery in BM25 mode so
        # the original Retriever does NOT instantiate HuggingFaceEmbeddings.
        # We install the correct Jina v3 adapter immediately afterward.
        super().__init__(
            agent_type="TableRAGMultiCMoneyAPI",
            mode="bm25",
            embed_model_name=requested_model,
            top_k=top_k,
            max_encode_cell=1,
            db_dir=db_dir,
            verbose=verbose,
        )

        if requested_mode not in {"bm25", "embed", "hybrid"}:
            raise ValueError(
                f"Unsupported schema retrieval mode: {requested_mode!r}. "
                "Use bm25, embed, or hybrid."
            )

        self.mode = requested_mode
        self.embed_model_name = requested_model
        self.embed_device = str(embed_device)
        self.embed_batch_size = int(embed_batch_size)
        self.encode_dim = int(encode_dim)

        if self.mode in {"embed", "hybrid"}:
            if self.embed_model_name != "jinaai/jina-embeddings-v4":
                raise ValueError(
                    "This CMoney retriever is configured for "
                    "jinaai/jina-embeddings-v4. Got: "
                    f"{self.embed_model_name!r}"
                )
            self.embedder = JinaV4Embeddings(
                model_name=self.embed_model_name,
                device=self.embed_device,
                batch_size=self.embed_batch_size,
                encode_dim=self.encode_dim,
            )
        else:
            self.embedder = None

        self.schema_path = Path(schema_path)
        self.schema_df = pd.read_csv(self.schema_path)

        missing = self.REQUIRED_COLUMNS - set(self.schema_df.columns)
        if missing:
            raise ValueError(
                f"Schema CSV is missing required columns: {sorted(missing)}"
            )

        self.schema_df = self.schema_df[
            ["table_id", "table", "column", "type", "nullable"]
        ].copy()
        for col in ["table_id", "table", "column", "type"]:
            self.schema_df[col] = self.schema_df[col].astype(str).str.strip()

        self._table_names = set(self.schema_df["table"].tolist())
        self._retriever_cache: dict[str, Any] = {}

    @staticmethod
    def _strip_brackets(value: str) -> str:
        value = str(value).strip()
        if value.startswith("[") and value.endswith("]"):
            return value[1:-1]
        return value

    @staticmethod
    def _nullable_to_bool(value: Any) -> bool | None:
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        if text in {"0", "false", "no"}:
            return False
        if text in {"1", "true", "yes"}:
            return True
        return None

    def has_table(self, table_name: str) -> bool:
        return table_name in self._table_names

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        sub = self.schema_df[self.schema_df["table"] == table_name].copy()
        return sub.reset_index(drop=True)

    def _table_cache_id(self, table_name: str, sub_df: pd.DataFrame) -> str:
        """Cache key changes when schema/model/task settings change."""
        if sub_df.empty:
            return "missing_" + hashlib.sha256(
                table_name.encode("utf-8")
            ).hexdigest()[:16]

        identity = {
            "version": self.CACHE_VERSION,
            "table_name": table_name,
            "schema_rows": sub_df.fillna("").astype(str).to_dict("records"),
            "mode": self.mode,
            "embed_model_name": self.embed_model_name,
            "document_task": "retrieval.passage",
            "query_task": "retrieval.query",
            "encode_dim": self.encode_dim,
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        table_id = str(sub_df.iloc[0]["table_id"]).strip()
        return f"cmoney_{table_id}_{digest}"

    def build_schema_corpus(self, df: pd.DataFrame) -> list[Document]:
        """Build schema documents from schema rows, not local table cells."""
        docs: list[Document] = []

        for row in df.itertuples(index=False):
            sql_column = str(row.column).strip()
            column_name = self._strip_brackets(sql_column)
            sql_type = str(row.type).strip()
            table_name = str(row.table).strip()
            nullable = self._nullable_to_bool(row.nullable)

            result = {
                "table": table_name,
                "column_name": column_name,
                "sql_column": sql_column,
                "sql_type": sql_type,
                "nullable": nullable,
            }
            result_text = json.dumps(result, ensure_ascii=False)

            page_content = (
                f"table {table_name}; column {column_name}; type {sql_type}"
            )
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={"result_text": result_text},
                )
            )

        return docs

    def get_docs(self, data_type: str, df: pd.DataFrame):
        if data_type == "schema":
            return self.build_schema_corpus(df)
        return super().get_docs(data_type, df)

    def _get_table_retriever(self, table_name: str):
        if table_name in self._retriever_cache:
            return self._retriever_cache[table_name]

        sub_df = self.get_table_schema(table_name)
        if sub_df.empty:
            return None

        cache_id = self._table_cache_id(table_name, sub_df)
        retriever = self.get_retriever("schema", cache_id, sub_df)
        self._retriever_cache[table_name] = retriever
        return retriever

    def retrieve_for_table(
        self,
        *,
        table_name: str,
        queries: list[str],
    ) -> list[str]:
        """Retrieve unique schema evidence for one candidate table."""
        retriever = self._get_table_retriever(table_name)
        if retriever is None:
            return []

        seen: set[str] = set()
        output: list[str] = []
        effective_queries = queries or [table_name]

        for query in effective_queries:
            for doc in retriever.invoke(query):
                text = doc.metadata.get("result_text", doc.page_content)
                if text not in seen:
                    seen.add(text)
                    output.append(text)

        return output
