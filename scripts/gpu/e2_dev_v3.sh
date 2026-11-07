# E2 dev eval v3: blind LP edits (real-loop u from our own traces, no Astra input) on dev_a.
source scripts/gpu/common.sh
for e in realsel_k512 realsel_k64 realshift_k8 realshift_k64; do
  ckpt $e edits/$e.safetensors
  ev $e dev_a_$e --sets gsm8k_train,math_train --n 1 --ids split:dev_a
done
