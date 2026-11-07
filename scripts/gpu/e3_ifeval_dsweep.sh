# Queued after e3_finalists.sh:
#  1. blind-edit push strength: realshift k=8 at 2x / 8x (dev_a; 4x is dev_a_realshift_k8)
#  2. IFEval (capability preservation) for base / Astra / shift_k8 / realshift_k8, think35, 1 sample
source scripts/gpu/common.sh
uv pip install -q lm-eval langdetect immutabledict nltk
evi() {  # evi <ckpt name> <out name>
  [ -f results/gpu/$2/.done ] && return 0
  python -m wslp.eval.run_ifeval ckpts/$1 results/gpu/$2 2>&1 | grep -v -E "it/s\]|Processed prompts" | tail -n 20
  touch results/gpu/$2/.done
}
for m in 2 8; do
  ckpt realshift_d${m}_k8 edits/realshift_d${m}_k8.safetensors
  ev realshift_d${m}_k8 dev_a_realshift_d${m}_k8 --sets gsm8k_train,math_train --n 1 --ids split:dev_a
done
ckpt base -
for e in shift_k8 astra_p50c realshift_k8; do ckpt $e edits/$e.safetensors; done
for e in shift_k8 astra_p50c realshift_k8 base; do evi $e ifeval_$e; done
