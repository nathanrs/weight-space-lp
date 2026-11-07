# Shared setup for GPU jobs: base model download + checkpoint build helper.
set -euo pipefail
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17", local_dir="models/Qwen3.5-0.8B")
PY
ckpt() {  # ckpt <name> <edit.safetensors|->
  [ -f ckpts/$1/model.safetensors ] || python -m wslp.eval.ckpt ckpts/$1 $2
}
ev() {  # ev <ckpt name> <out name> <args...>
  local c=$1 o=$2; shift 2
  [ -f results/gpu/$o/.done ] && return 0
  python -m wslp.eval.run_vllm ckpts/$c results/gpu/$o "$@" 2>&1 | grep -v -E "it/s\]|Processed prompts" | tail -n 40
  touch results/gpu/$o/.done
}
