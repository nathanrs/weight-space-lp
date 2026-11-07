# Finalist benchmarks, priority order (each eval is skipped if already .done, so the job is resumable):
#  1. blind realshift_k8 on untouched final_practice (paired with final_{base,astra_p50c,shift_k8})
#  2. GSM8K test x2 for shift_k8 / Astra
#  3. MATH-500: 3 more samples (seed 1) for shift_k8 / Astra -> x4 together with e0_*_math500 (seed 0)
#  4. realshift_k8 on MATH-500 x4 and GSM8K test x2
#  5. base on GSM8K test x1 (base MATH-500 x4 is in the authors' logs: 7.9)
source scripts/gpu/common.sh
ckpt base -
ckpt astra_p50c edits/astra_p50c.safetensors
ckpt shift_k8 edits/shift_k8.safetensors
ckpt realshift_k8 edits/realshift_k8.safetensors
ev realshift_k8 final_realshift_k8 --sets gsm8k_train,math_train --n 1 --ids split:final_practice
for e in shift_k8 astra_p50c; do ev $e gsm8k_$e --sets gsm8k --n 2; done
for e in shift_k8 astra_p50c; do ev $e math500s1_$e --sets math500 --n 3 --sampling '{"seed": 1}'; done
ev realshift_k8 math500_realshift_k8 --sets math500 --n 4
ev realshift_k8 gsm8k_realshift_k8 --sets gsm8k --n 2
ev base gsm8k_base --sets gsm8k --n 1
