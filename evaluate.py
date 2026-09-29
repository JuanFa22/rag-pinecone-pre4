"""Evalúa el recuperador con un Golden Set: Precision@k y Recall@k.

Definiciones (una consulta):
  - Se toman los top-k chunks recuperados.
  - Precision@k = (# chunks del top-k que pertenecen a un documento esperado) / k
  - Recall@k    = (# documentos esperados presentes en el top-k) / (# documentos esperados)
Se reporta el promedio sobre todas las preguntas, para vectorial, BM25 e híbrido.

Uso:
    python evaluate.py --k 5
"""
import argparse
import json
from statistics import mean
from typing import Callable, Dict, List

import config


def precision_at_k(retrieved_doc_ids: List[str], expected: List[str], k: int) -> float:
    top = retrieved_doc_ids[:k]
    return sum(1 for d in top if d in expected) / k if k else 0.0


def recall_at_k(retrieved_doc_ids: List[str], expected: List[str], k: int) -> float:
    if not expected:
        return 0.0
    found = set(retrieved_doc_ids[:k]) & set(expected)
    return len(found) / len(set(expected))


def load_golden_set(path=config.GOLDEN_SET_FILE) -> List[Dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for it in items:
        exp = it["documento_id_esperado"]
        it["esperados"] = exp if isinstance(exp, list) else [exp]
    return items


def evaluate(name: str, search: Callable[[str], list], golden: List[Dict], k: int) -> Dict:
    precisions, recalls, misses = [], [], []
    for item in golden:
        docs = search(item["pregunta"])[:k]
        ids = [d.metadata.get("doc_id") for d in docs]
        precisions.append(precision_at_k(ids, item["esperados"], k))
        r = recall_at_k(ids, item["esperados"], k)
        recalls.append(r)
        if r < 1.0:
            misses.append(item["pregunta"])
    return {"modo": name, f"precision@{k}": mean(precisions), f"recall@{k}": mean(recalls), "fallos": misses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.6)
    args = parser.parse_args()
    config.require_keys()

    from hybrid_retriever import build_bm25, build_hybrid_retriever, build_vector_retriever

    golden = load_golden_set()
    k = args.k
    retrievers = {
        "vectorial (Pinecone)": build_vector_retriever(k).invoke,
        "léxico (BM25)": build_bm25(k).invoke,
        "híbrido (Ensemble)": build_hybrid_retriever(k, args.vector_weight).invoke,
    }

    print(f"\nGolden set: {len(golden)} preguntas | k = {k}\n")
    print(f"{'Modo':<24}{'Precision@'+str(k):>14}{'Recall@'+str(k):>12}")
    print("-" * 50)
    results = []
    for name, fn in retrievers.items():
        res = evaluate(name, fn, golden, k)
        results.append(res)
        print(f"{name:<24}{res[f'precision@{k}']:>14.3f}{res[f'recall@{k}']:>12.3f}")
    print()
    for res in results:
        if res["fallos"]:
            print(f"Preguntas sin recall completo en '{res['modo']}':")
            for q in res["fallos"]:
                print(f"  - {q}")


if __name__ == "__main__":
    main()
