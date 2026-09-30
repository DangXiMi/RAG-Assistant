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

        # Recorded so callers can verify the embedding width matches the vector
        # store's configured size, rather than failing confusingly at upsert time.
        self.dimension = self.model.get_sentence_embedding_dimension()
        if self.dimension is None:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.model_name}"
            )

        logging.info(
            "Loaded %s model (dimension=%s, device=%s)",
            self.model_name,
            self.dimension,
            self.device,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts is None:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        ).tolist()
        return embeddings
    
