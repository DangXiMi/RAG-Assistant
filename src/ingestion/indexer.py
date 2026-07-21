# src/ingestion/indexer.py
import os
from src.config.config import CONFIG
from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.ingestion.chunker import Chunk
import logging

class Indexer():
    def __init__(self, config: Dict = CONFIG):
        self.config = config
        
        # Read from environment variables FIRST, then fallback to config
        host = os.getenv("QDRANT_HOST", config["qdrant"]["host"])
        port = int(os.getenv("QDRANT_PORT", config["qdrant"]["port"]))
        
        # Special case for in-memory
        if host in [":memory:", "memory"]:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)
            
        self.collection_name = config["qdrant"]["collection_name"]
        self.vector_size = config["qdrant"]["vector_size"]
        logging.info(f"Loaded Qdrant client: {host}:{port}")

    def ensure_collection(self):
        collection_name = self.config["qdrant"]["collection_name"]

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config["qdrant"]["vector_size"],
                    distance=Distance.COSINE,
                ),
            )
    
    def index(self, chunks: List[Chunk], vectors: List[List[float]]):
        if len(chunks) != len(vectors):
            raise ValueError(
                "must have the same length"
            )
        
        self.ensure_collection()

        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "text": chunk.text,
                     **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        self.client.upsert(
            collection_name=self.config["qdrant"]["collection_name"],
            points=points,
        )