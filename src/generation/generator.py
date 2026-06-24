from src.config.config import CONFIG
from typing import Dict, List
from src.retrieval.hybrid_retriever import HybridRetriever

from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document



class Generator():
    def __init__(self, retriever: HybridRetriever, config: dict = CONFIG):
        self.config = config
        generator_cfg = config.get("generator",{})
        self.model_name = generator_cfg.get("model", "llama3.1:8b")
        self.temperature = generator_cfg.get("temperature", 0.0)
        self.max_tokens = generator_cfg.get("max_tokens", 512)
        
        self.retriever = retriever
        self.model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.max_tokens)

        
        
    def _ensure_prompt(self):
        prompt ="""
        System: You are a helpful assistant. Use the provided context to answer the user's question.
        If the answer cannot be found in the context, say "I don't know" and do not make up information.
        When citing sources, mention the document IDs (e.g., "According to [doc_id]").

        Context:
        {context}

        Question: {question}

        Answer:
        
        """
        
        g_prompt = ChatPromptTemplate.from_template(prompt)
        return g_prompt
    
    @traceable(name="retrieve_documents")
    def retrieve(self, query, top_k):
        raw = self.retriever.search(query=query, top_k=top_k)
        return [Document(page_content=d["text"], metadata={"doc_id": d["id"], **d["metadata"]}) for d in raw]
    
    @traceable(name="context_builder")
    def build_context(self, docs):
        return "\n\n".join(
            f"[{d.metadata['doc_id']}]\n{d.page_content}"
            for d in docs
        )
    
    @traceable(name="llm_generation")
    def generate(self, context: str, query: str):

        prompt = self._ensure_prompt()

        messages = prompt.format_messages(
            context=context,
            question=query
        )

        return self.model.invoke(messages)
        
    @traceable(name="rag_pipeline")
    def run(self, query: str, top_k: int = 5):

        docs = self.retrieve(query, top_k)

        context = self.build_context(docs)

        response = self.generate(context, query)

        return {
            "answer": response.content,
            "sources": [d.metadata["doc_id"] for d in docs]
        }


        
        