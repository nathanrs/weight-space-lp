# Confirmation of the headline candidate on untouched problems (no selection used them):
#  - paired base / Astra / shift_k8 on final_practice (256 GSM8K + 256 MATH practice), 1 sample, 32k
#  - shift_k8 on MATH-500 x1 (compare: base 7.0 / 93.2 cut, Astra 50.0 / 30.6 from E0)
source scripts/gpu/common.sh
ckpt base -
ckpt astra_p50c edits/astra_p50c.safetensors
ckpt shift_k8 edits/shift_k8.safetensors
ev shift_k8 e0_shift_k8_math500 --sets math500 --n 1
for e in shift_k8 astra_p50c base; do
  ev $e final_$e --sets gsm8k_train,math_train --n 1 --ids split:final_practice
done
