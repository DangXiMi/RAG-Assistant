from scripts.run_ragas_evaluation import load_pipeline

import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    pipeline = load_pipeline()
    print(load_pipeline)
    query = "What is the launch date of JWST?"

    d_retrievers = pipeline["retrievers"]["Dense"]
    s_retrievers = pipeline["retrievers"]["Sparse"]
    
    dense_results = d_retrievers.search(
        query,
        top_k=5,
    )

    print("=" * 60)
    print("QUESTION:", query)
    print("Dense:", len(dense_results))
    for d in dense_results[:3]:
        print(d["text"])

    sparse_results = s_retrievers.search(
        query,
        top_k=5,
    )

    print("Sparse:", len(sparse_results))
    for d in sparse_results[:3]:
        print(d["text"])
        
if __name__ == "__main__":
    main()