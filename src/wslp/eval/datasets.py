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
    raise KeyError(name)
