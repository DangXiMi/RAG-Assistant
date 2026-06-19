from src.config.config import CONFIG
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from typing import List, Dict, Optional
import logging

class DenseRetriever():
    def __init__(self, embedder: Embedder, indexer: Indexer, config: dict = CONFIG):
        self.config = config
        self.default_top_k = self.config["retrieval"]["default_top_k"]
        self.embedder = embedder
        self.indexer = indexer
        logging.info("Load Dense Retriever")
        
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, float]]:
        if not query:
            return []

        if not top_k:
            top_k = self.config["retrieval"]["default_top_k"]
        
        self.indexer.ensure_collection()
        
        query_embedding = self.embedder.embed([query])[0]
        search_result = self.indexer.client.query_points(
            collection_name=f"{self.indexer.collection_name}",
            query=query_embedding,
            with_payload= True,
            limit=top_k
        ).points
        
        result = []
        for point in search_result:
            if point.score >= self.config["retrieval"]["score_threshold"]:
                result.append({
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload["text"],
                    "metadata": point.payload
                })
                
        return result

        
        
    
        