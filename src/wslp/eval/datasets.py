"""Benchmark and practice sets as lists of {id, problem, answer}, matching exp8_math/code/evaluate.py prepare()."""
import json

from ..paths import ROOT


def load(name: str) -> list[dict]:
    if name in ("gsm8k_train", "math_train"):
        return [json.loads(l) for l in open(ROOT / "data/practice" / f"{name}.jsonl")]
    from datasets import load_dataset
    if name == "math500":
        return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    if name == "gsm8k":
        return [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
    if name == "aime24":
        return [{"id": f"aime24_{r['id']}", "problem": r["problem"], "answer": str(r["answer"])}
                for r in load_dataset("HuggingFaceH4/aime_2024", split="train")]
    if name == "aime25":
        return [{"id": f"aime25_{r['problem_idx']}", "problem": r["problem"], "answer": str(r["answer"])}
                for r in load_dataset("MathArena/aime_2025", split="train")]
    if name == "amc23":
        return [{"id": f"amc23_{r['id']}", "problem": r["question"], "answer": str(r["answer"])}
                for r in load_dataset("math-ai/amc23", split="test")]
    if name == "minerva":
        return [{"id": f"minerva_{i}", "problem": r["question"], "answer": str(r["answer"])}
                for i, r in enumerate(load_dataset("math-ai/minervamath", split="test"))]
    if name == "olympiad":
        rows = []
        for r in load_dataset("math-ai/olympiadbench", split="test"):
            fa = r["final_answer"]
            fa = fa if isinstance(fa, list) else json.loads(fa.replace("'", '"')) if fa.startswith("[") else [fa]
            rows.append({"id": f"olympiad_{r['id']}", "problem": r["question"], "answer": ", ".join(str(x) for x in fa)})
        return rows
    raise KeyError(name)
