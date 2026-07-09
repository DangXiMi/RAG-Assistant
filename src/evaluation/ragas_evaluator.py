from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from datasets import Dataset
import json
import os
from pathlib import Path
from src.generation.generator import Generator
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama


GOLDEN_FILE = Path("data/evaluation/golden.jsonl")

class RAGASEvaluator():
    def __init__(self, generator: Generator, dataset_path: str = GOLDEN_FILE  ):
        self.data_path = dataset_path
        self.generator = generator
        
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
        data = []
        with self.data_path.open("r",encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)


                data.append({
                    "question": record["question"],
                    "ground_truth": "\n".join(record["ground_truth_chunks"]) if record["ground_truth_chunks"] else ""
                 })
        
        return data
    
    def evaluate(self, retriever, top_k: int = 5):
        data = self.load_dataset()
        for record in data:
            generation = self.generator.run(
                record["question"],
                top_k=top_k,
                retriever=retriever  
            )
            record["answer"] = generation["answer"]
            record["contexts"] = generation["contexts"]
            print(record)
        
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
        
        return {
            "aggregated": aggregated,
            "per_sample": df.to_dict(orient="records"),
            "raw": result
        }