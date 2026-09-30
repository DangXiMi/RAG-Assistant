from src.config.config import CONFIG
from src.generation.citations import build_source_refs, label, source_labels
from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import logging

logger = logging.getLogger(__name__)

# Returned verbatim when retrieval finds nothing usable, so the assistant
# states its limitation instead of answering from parametric memory.
NOT_FOUND_MESSAGE = "I could not find this in the provided documents."


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

        # Minimum retrieval score for a hit to count as usable evidence.
        # 0.0 keeps every hit, which preserves the previous behaviour.
        self.score_threshold = (
            config.get("retrieval", {}).get("score_threshold", 0.0)
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

        Each context block begins with a numbered source reference in square
        brackets, for example [1] refund_policy.pdf, p.3. When you state a
        fact, cite where it came from using that same [n] form.

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
        """Retrieve chunks while preserving their full metadata.

        The metadata has to survive: `source` and `page`/`sheet` are what make a
        citation possible. Previously only the chunk id was carried through,
        which is why answers cited bare UUIDs instead of file names.
        """
        used_retriever = retriever if retriever else self.retriever
        docs = used_retriever.search(
            query=query,
            top_k=top_k
        )

        documents = []
        for doc in docs:
            metadata = dict(doc.get("metadata") or {})
            # Expose the id under both keys: `doc_id` is the established name,
            # `id` is what the stores return.
            metadata.setdefault("doc_id", doc.get("id"))
            metadata.setdefault("id", doc.get("id"))
            if metadata.get("score") is None:
                metadata["score"] = doc.get("score")

            documents.append(
                Document(
                    page_content=doc.get("text", ""),
                    metadata=metadata,
                )
            )

        return documents


    def filter_by_score(self, docs):
        """Drop hits below the configured score threshold.

        Scores come from RRF fusion, where higher is better. A threshold of
        0.0 keeps everything.
        """
        if not self.score_threshold:
            return list(docs)

        kept = []
        for doc in docs:
            score = doc.metadata.get("score")
            if score is None or float(score) >= self.score_threshold:
                kept.append(doc)

        if len(kept) != len(docs):
            logger.info(
                "Dropped %d of %d retrieved chunk(s) below score threshold %s",
                len(docs) - len(kept),
                len(docs),
                self.score_threshold,
            )
        return kept


    @traceable(name="context_builder")
    def build_context(
        self,
        docs
    ):
        """Render the context with a numbered, human-readable reference per block."""

        if not docs:
            return ""

        blocks = []
        for index, doc in enumerate(docs, start=1):
            reference = label(doc.metadata, prefix=str(index))
            blocks.append(f"{reference}\n{doc.page_content}")

        return "\n\n".join(blocks)


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

        docs = self.filter_by_score(docs)

        # No usable evidence: answer deterministically instead of asking the
        # model to reason over an empty context.
        if not docs:
            logger.info("No usable context retrieved; returning not-found response")
            return {
                "answer": NOT_FOUND_MESSAGE,
                "contexts": [],
                "sources": [],
                "source_refs": [],
                "retrieved": 0,
                "answered": False,
            }

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

            # Human-readable citations, e.g. "refund_policy.pdf, p.3". Kept as
            # plain strings so the existing API contract is unchanged.
            "sources": source_labels(docs),

            # Structured form for richer rendering. Additive, so clients that
            # only read `sources` keep working.
            "source_refs": build_source_refs(docs),

            "retrieved": len(docs),
            "answered": True,
        }
