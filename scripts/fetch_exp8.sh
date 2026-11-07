#!/usr/bin/env bash
# Fetch the authors' Experiment 8 logs (CC BY 4.0 / code under its own license) into third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d third_party/lm_understand-logs ] || git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs.git third_party/lm_understand-logs
