from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from datasets import Dataset
import json
import logging
import os

import pandas as pd
from pathlib import Path
from src.generation.generator import Generator
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

GOLDEN_FILE = Path("data/evaluation/golden.jsonl")
# Per-question scores, so a low average can be traced to specific questions.
PER_SAMPLE_FILE = Path("data/evaluation/per_sample.csv")

class RAGASEvaluator():
    def __init__(self, generator: Generator, dataset_path: str = GOLDEN_FILE  ):
        self.generator = generator

        # Validate the dataset up front. Loading the judge models below is slow,
        # so failing fast on a missing/misconfigured golden set saves minutes and
        # gives a clear error instead of a confusing one much later.
        self.data_path = Path(dataset_path)
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {self.data_path}. "
                f"Point RAGASEvaluator at a .jsonl file with one record per line."
            )

        self.evaluator_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.run_config = RunConfig(
            timeout=300,
            max_workers=1
        )
        
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOllama(
                model="qwen2.5:7b",
                temperature=0,
                request_timeout=300
            )
        )
        self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        

    def load_dataset(self):
        """Read the golden .jsonl into RAGAS-ready records.

        Accepts either `ground_truth_chunks` (list of reference passages, joined
        into the `ground_truth` string RAGAS expects) or a plain string
        `ground_truth`. Records carrying nothing but ids are skipped, because
        `context_recall` needs actual reference text to compare against and a
        silent empty string would corrupt the metric.
        """
        data = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)

                if "question" not in record:
                    raise ValueError(
                        f"{self.data_path}:{line_number} has no 'question' field"
                    )

                chunks = record.get("ground_truth_chunks")
                if chunks is None:
                    ground_truth = record.get("ground_truth", "")
                else:
                    ground_truth = "\n".join(str(c) for c in chunks if c)

                if not str(ground_truth).strip():
                    logger.warning(
                        "%s:%d has no reference text; skipping this record "
                        "(context_recall cannot be computed without it)",
                        self.data_path,
                        line_number,
                    )
                    continue

                data.append(
                    {
                        "question": record["question"],
                        "ground_truth": ground_truth,
                    }
                )

        if not data:
            raise ValueError(
                f"{self.data_path} produced no evaluable records. Each line needs "
                f"a 'question' and either 'ground_truth_chunks' or 'ground_truth'."
            )
        return data

    def evaluate(self, retriever, top_k: int = 5, save_per_sample: bool = True):
        """Run the RAG pipeline over the golden set and score it with RAGAS.

        Args:
            retriever: Retriever to answer the golden questions with.
            top_k: Number of contexts to retrieve per question.
            save_per_sample: Persist per-sample scores alongside the aggregate,
                so a low average can be traced to the questions that caused it.
                Tests pass False so a mocked run cannot leave synthetic scores
                sitting in data/evaluation/ as if they were real results.

        Returns:
            `{"aggregated", "per_sample", "raw"}`.
        """
        data = self.load_dataset()
        for record in data:
            generation = self.generator.run(
                record["question"],
                top_k=top_k,
                retriever=retriever  
            )
            record["answer"] = generation["answer"]
            record["contexts"] = generation["contexts"]
        
        dataset = Dataset.from_list(data)
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            run_config=self.run_config
        )
        
        # Convert to DataFrame for easier manipulation
        df = result.to_pandas()
        
        # Compute aggregated scores (mean)
        aggregated = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()

        per_sample = df.to_dict(orient="records")

        if save_per_sample:
            self.save_per_sample(per_sample)

        return {
            "aggregated": aggregated,
            "per_sample": per_sample,
            "raw": result
        }

    def save_per_sample(self, per_sample, path: str = PER_SAMPLE_FILE) -> Path:
        """Write per-question scores to CSV so failures can be inspected.

        The aggregate alone cannot show whether a 0.8 came from eight good
        answers or four perfect ones and four disasters.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(per_sample)
        frame.to_csv(destination, index=False)

        logger.info("Per-sample scores saved to %s", destination)
        return destination