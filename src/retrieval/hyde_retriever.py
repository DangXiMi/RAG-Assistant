from typing import Optional, Dict

from src.config.config import CONFIG
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.embedder import Embedder

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class HyDERetriever:
    def __init__(self, llm, embedder, base_retriever, config=CONFIG):
        self.base_retriever = base_retriever
        self.embedder = embedder

        hyde_config = config.get("hyde", {})

        self.enabled = hyde_config.get("enabled", False)
        self.temp = hyde_config.get("temperature", 0.0)

        self.prompt_template = hyde_config.get(
            "prompt_template",
            "Write a short document that answers: {query}"
        )

        model_name = hyde_config.get("model", "llama3.1:8b")

        self.llm = llm if llm is not None else ChatOllama(
            model=model_name,
            temperature=self.temp
        )

    def _generate_hypothetical(self, query: str):
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        messages = prompt_template.format_messages(query=query)

        response = self.llm.invoke(messages)
        return response.content

    def search(self, query: str, top_k: Optional[int] = None):
        if not query:
            return []

        if not self.enabled:
            return self.base_retriever.search(query, top_k=top_k)

        hyde_query = self._generate_hypothetical(query)
        return self.base_retriever.search(hyde_query, top_k=top_k)

        
        
