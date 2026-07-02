def RRF(top_k, rrf_k, retrievers):
    fused_docs = {}

    for retriever in retrievers :
        for rank, doc in enumerate(retriever):
            doc_id = str(doc["id"])

            if doc_id not in fused_docs:
                fused_docs[doc_id] = {
                    "id": doc_id,
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": 0.0,
                }

            fused_docs[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

    results = sorted(
        fused_docs.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]