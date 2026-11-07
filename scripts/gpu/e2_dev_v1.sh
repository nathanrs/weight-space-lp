# E2 dev eval v1: paired on dev_a (64 GSM8K + 64 MATH practice), 1 sample, 32k tokens.
source scripts/gpu/common.sh
ckpt base -
for e in astra_p50c lp_shift lp_sel recipe_nosearch random_ctrl; do ckpt $e edits/$e.safetensors; done
for e in base astra_p50c lp_shift lp_sel recipe_nosearch random_ctrl; do
  ev $e dev_a_$e --sets gsm8k_train,math_train --n 1 --ids split:dev_a
done
