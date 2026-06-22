# src/retrieval/sparse_retriever.py

from typing import Dict, List, Optional
import psycopg2
from src.config import CONFIG


class SparseRetriever:
    def __init__(self, db_conn, config: dict = CONFIG):
        if db_conn is None:
            raise ValueError("Database connection cannot be None")

        try:
            db_conn.cursor().close()
        except Exception as e:
            raise RuntimeError(
                "Invalid or closed PostgreSQL connection"
            ) from e

        self.db_conn = db_conn

        sparse_cfg = config.get("sparse_retrieval", {})

        self.default_top_k = sparse_cfg.get(
            "default_top_k",
            5,
        )

        self.score_threshold = sparse_cfg.get(
            "score_threshold",
            0.0,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if not query or not query.strip():
            return []

        if top_k is None:
            top_k = self.default_top_k

        sql = """
        SELECT
            id,
            text,
            metadata,
            ts_rank(
                tsv,
                plainto_tsquery('english', %s)
            ) AS score
        FROM chunks
        WHERE tsv @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """

        try:
            with self.db_conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        query,
                        query,
                        top_k,
                    ),
                )

                rows = cur.fetchall()

        except psycopg2.Error as e:
            raise RuntimeError(
                f"PostgreSQL search failed: {e}"
            ) from e

        results = []

        for row in rows:
            doc = {
                "id": str(row[0]),
                "text": row[1],
                "metadata": row[2] or {},
                "score": float(row[3]),
            }

            if doc["score"] >= self.score_threshold:
                results.append(doc)

        return results