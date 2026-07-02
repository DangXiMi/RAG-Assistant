from typing import Dict, Optional
from src.config.config import CONFIG
from src.utils.RRF import RRF

from src.retrieval.hybrid_retriever import HybridRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class MultiQueryRetriever():
    def __init__(self, llm: ChatOllama, base_retriever: HybridRetriever, config: dict = CONFIG):
        self.llm = llm if llm is not None else ChatOllama(model_name="llama3.1:8", temperature = 0)
        self.base_retriever = base_retriever
        
        MQR_config = config.get("multi_query", {})
        self.enabled =  MQR_config.get("enabled", False)
        self.num_queries = MQR_config.get("num_queries", 3)
        self.candidate_k =  MQR_config.get("candidate_k", 10)
        self.rrf_k = MQR_config.get("rrf_k", 60)
        self.prompt_template = MQR_config.get("prompt_template", " Generate {num_queries} different variations of the user's question to help retrieve relevant documents. Each variation should be a single sentence, rephrased or expanded, but keeping the same meaning. Return only the variations, one per line, without numbering. Original question: {query}" )
        
        
    def _generate_queries(self, query):
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        messages = prompt.format_messages(num_queries=self.num_queries, query=query)
        response = self.llm.invoke(messages)
        
        return [line.strip() for line in response.content.split("\n") if line.strip()]
    
    def search(self, query: str, top_k = 5):
        if not query:
            return []
        if self.enabled == False:
            return self.base_retriever.search(query, top_k = top_k)
        
        queries = self._generate_queries(query)
        retriever = []
        
        for query in queries:
            res = self.base_retriever.search(query, top_k = self.candidate_k)
            retriever.append(res)

        result = RRF(top_k=top_k, rrf_k = self.rrf_k, retrievers=retriever)
        
        return result[:top_k]

        
        
        