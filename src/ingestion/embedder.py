from sentence_transformers import SentenceTransformer
import logging
from typing import Dict
from src.config.config import CONFIG
import torch

class Embedder:
    def __init__(self, config: Dict = CONFIG):
        self.config = config
        self.model_name = config["embedding"]["model_name"]
        self.device = self.config["embedding"]["device"]
        device = config["embedding"]["device"]

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        
        self.batch_size = self.config["embedding"]["batch_size"]
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_embedding_dimension()
        logging.info(f"Loaded {self.model_name} model")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts is None:
            return []
        
        embeddings = self.model.encode(texts).tolist()
        return embeddings
    
