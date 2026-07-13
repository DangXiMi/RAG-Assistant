from src.config.config import CONFIG
from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class Generator:

    def __init__(self, retriever=None, config: dict = CONFIG):

        self.config = config

        generator_cfg = config.get("generator", {})

        self.model_name = generator_cfg.get(
            "model",
            "llama3.1:8b"
        )

        self.temperature = generator_cfg.get(
            "temperature",
            0.0
        )

        self.max_tokens = generator_cfg.get(
            "max_tokens",
            512
        )


        if retriever is None:
            raise ValueError(
                "Retriever must be provided"
            )

        self.retriever = retriever


        self.model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.max_tokens
        )


        self.prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.

        Answer the question using ONLY the provided context.

        If the answer cannot be found in the context,
        say "I don't know".

        Context:
        {context}


        Question:
        {question}


        Answer:
        """
        )


    @traceable(name="retrieve_documents")
    def retrieve(
        self,
        query,
        retriever,
        top_k=5
    ):
        used_retriever = retriever if retriever else self.retriever
        docs = used_retriever.search(
            query=query,
            top_k=top_k
        )

        return [
            Document(
                page_content=d["text"],
                metadata={
                    "doc_id": d["id"]
                }
            )
            for d in docs
        ]


    @traceable(name="context_builder")
    def build_context(
        self,
        docs
    ):

        if not docs:
            return ""

        return "\n\n".join(
            f"[{doc.metadata['doc_id']}]\n{doc.page_content}"
            for doc in docs
        )


    @traceable(name="llm_generation")
    def generate(
        self,
        context,
        query
    ):

        messages = self.prompt.format_messages(
            context=context,
            question=query
        )

        return self.model.invoke(messages)


    @traceable(name="rag_pipeline")
    def run(
        self,
        query,
        top_k=5,
        retriever = None
    ):

        docs = self.retrieve(
            query,
            retriever,
            top_k
            
        )

        context = self.build_context(
            docs
        )

        response = self.generate(
            context,
            query
        )


        return {
            "answer": response.content,

            "contexts": [
                d.page_content
                for d in docs
            ],

            "sources": [
                d.metadata["doc_id"]
                for d in docs
            ]
        }