from src.config.config import CONFIG
from src.retrieval.hybrid_retriever import HybridRetriever

from sentence_transformers import CrossEncoder
import torch
import numpy as np


class CrossEncoderReranker():
    def __init__(self, base_retriever: HybridRetriever  , config = CONFIG):
        rerank_config = config.get("reranking",{})
        self.model_name = rerank_config.get("model_name")
        self.candidate_k = rerank_config.get("candidate_k", 50)
        self.final_k = rerank_config.get("final_k", 5)
        self.batch_size = rerank_config.get("batch_size", 32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_retriever = base_retriever
        if base_retriever is None: 
            raise ValueError("base_retriever is required")
        
        self.enabled = rerank_config.get("enabled", False)
        self.model = CrossEncoder(self.model_name, device=self.device) if self.enabled else None
    
    def rerank(self, query, candidates, top_k):
        if not query:
            return []

        scores = self.model.predict(
            [[query, doc["text"]] for doc in candidates],
            batch_size=self.batch_size,
        )

        idx = np.argsort(scores)[::-1][:top_k]

        reranked = []
        for i in idx:
            doc = candidates[i].copy()
            doc["score"] = float(scores[i])
            reranked.append(doc)

        return reranked
        
    def search(self, query, top_k = None):
        if not query:
            return []

        if not self.enabled:
            k = top_k if top_k is not None else self.final_k
            return self.base_retriever.search(query, top_k=k)

        final_k = top_k if top_k is not None else self.final_k

        candidates = self.base_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        return self.rerank(query, candidates, final_k)
         

    