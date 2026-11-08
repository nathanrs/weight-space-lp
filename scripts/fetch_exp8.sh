#!/usr/bin/env bash
# Fetch the authors' Experiment 8 logs (CC BY 4.0 / code under its own license) into third_party/, and copy the
# pieces our code reads into data/: the practice problems and the text-only Qwen3.5 config used for vLLM checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d third_party/lm_understand-logs ] || git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git third_party/lm_understand-logs
X=third_party/lm_understand-logs/exp8_math
mkdir -p data/practice data/model_text
cp $X/practice/{gsm8k_train.jsonl,math_train.jsonl,README.txt} data/practice/
cp $X/base_model/Qwen3.5-0.8B-text/{config.json,generation_config.json} data/model_text/
