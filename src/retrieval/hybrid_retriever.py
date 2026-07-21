from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from typing import List, Dict, Optional
from src.config.config import CONFIG
from src.utils.helper_func import RRF

import logging

class HybridRetriever():
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever, config: dict =CONFIG):
        self.config = config
        self.s_retriever = sparse_retriever
        self.d_retriever = dense_retriever
        
        hybrid_cfg = config.get("hybrid_retrieval", {})

        self.candidate_k = hybrid_cfg.get("candidate_k", 10)
        self.final_k = hybrid_cfg.get("final_k", 5)
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)
        
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if not query or not query.strip():
            return []

        if top_k is None:
            top_k = self.final_k

        try:
            dense_results = self.d_retriever.search(
                query,
                top_k=self.candidate_k,
            )
        except Exception as e:
            logging.exception("Dense retrieval failed")
            raise RuntimeError(
                f"Dense retrieval failed: {e}"
            ) from e

        try:
            sparse_results = self.s_retriever.search(
                query,
                top_k=self.candidate_k,
            )
        except Exception as e:
            logging.exception("Sparse retrieval failed")
            raise RuntimeError(
                f"Sparse retrieval failed: {e}"
            ) from e

        retrieval_result = [dense_results, sparse_results]
        result = RRF(top_k=top_k, rrf_k = self.rrf_k, retrievers=retrieval_result)

        return result

        