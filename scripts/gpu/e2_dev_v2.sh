# E2 dev eval v2: sparsity-sweep edits on dev_a (paired with dev_a_base from v1).
source scripts/gpu/common.sh
for e in shift_k8 shift_k32 shift_k128 sel_k64 sel_k512; do
  ckpt $e edits/$e.safetensors
  ev $e dev_a_$e --sets gsm8k_train,math_train --n 1 --ids split:dev_a
done
