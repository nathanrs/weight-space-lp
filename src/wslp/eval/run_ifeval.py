"""IFEval (google/IFEval, 541 prompts) with the think35 sampling protocol, scored by lm-eval's port of the official
checkers. The scored response is the text after the last </think>; a response cut inside the thinking scores empty.

  python -m wslp.eval.run_ifeval <ckpt> <out> [--max-tokens 32768]
Writes <out>/scores.json ({"ifeval": {...}}), <out>/ifeval_per_problem.json (prompt-level strict as "correct")
and <out>/ifeval_samples.jsonl.gz.
"""
import argparse
import gzip
import json
import os
import time

from .run_vllm import THINK35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=THINK35["max_tokens"])
    a = ap.parse_args()

    import lm_eval.tasks.ifeval.utils as U
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(a.out, exist_ok=True)
    sampling = dict(THINK35, max_tokens=a.max_tokens)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.88)),
              max_model_len=sampling["max_tokens"] + 2048, seed=0)
    t0 = time.time()
    docs = list(load_dataset("google/IFEval", split="train"))
    prompts = [tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for d in docs]
    outs = llm.generate(prompts, SamplingParams(n=1, **sampling))
    agg = {"prompt_level_strict_acc": [], "prompt_level_loose_acc": [], "inst_level_strict_acc": [],
           "inst_level_loose_acc": []}
    cut = toks = 0
    per = []
    with gzip.open(os.path.join(a.out, "ifeval_samples.jsonl.gz"), "wt") as f:
        for d, o in zip(docs, outs):
            s = o.outputs[0]
            resp = s.text.rsplit("</think>", 1)[1].strip() if "</think>" in s.text else ""
            r = U.process_results(d, [resp])
            for k in agg:
                agg[k] += r[k] if isinstance(r[k], list) else [r[k]]
            per.append({"id": d["key"], "correct": [int(r["prompt_level_strict_acc"])],
                        "loose": [int(r["prompt_level_loose_acc"])], "cut": s.finish_reason == "length"})
            cut += s.finish_reason == "length"
            toks += len(s.token_ids)
            f.write(json.dumps({"key": d["key"], "cut": s.finish_reason == "length", "tokens": len(s.token_ids),
                                **{k: r[k] for k in agg}, "response": resp, "text": s.text}) + "\n")
    json.dump(per, open(os.path.join(a.out, "ifeval_per_problem.json"), "w"))
    n = len(docs)
    res = {"_protocol": {"sampling": sampling, "ckpt": a.ckpt},
           "ifeval": {**{k: sum(v) / len(v) for k, v in agg.items()}, "acc": sum(agg["prompt_level_strict_acc"]) / n,
                      "cut_share": cut / n, "mean_tokens": toks / n, "n_problems": n, "s": round(time.time() - t0)}}
    print(json.dumps(res["ifeval"]), flush=True)
    json.dump(res, open(os.path.join(a.out, "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
