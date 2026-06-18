from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

# connect to Qdrant Cloud
client = QdrantClient(host="localhost", port=6333)

if not client.collection_exists("rag-docs"):
    client.create_collection(
        collection_name="rag-docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )