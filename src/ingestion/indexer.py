from src.config.config import CONFIG
from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.ingestion.chunker import Chunk
import logging

class Indexer():
    def __init__(self, config: Dict = CONFIG):
        self.config = config
        host = config["qdrant"]["host"]
        if host == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                host=host,
                port=config["qdrant"]["port"],
            )
        self.collection_name = config["qdrant"]["collection_name"]
        self.vector_size = config["qdrant"]["vector_size"]
        logging.info("Loaded Qdrant client")

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
        

        