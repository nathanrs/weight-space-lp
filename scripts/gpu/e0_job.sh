# E0.4 harness gate + E1 fit traces.
#  - base and Astra p50c on MATH-500 x1 (logs: base acc 7.9 / cut 93.0; p50c 48.1 / 33.1, x4 samples)
#  - base on the 512 fit_gen practice problems x1 (LP training traces; = Astra's pass-1 style generation)
source scripts/gpu/common.sh
ckpt base -
ckpt astra_p50c edits/astra_p50c.safetensors
ev astra_p50c e0_astra_math500 --sets math500 --n 1
ev base e0_base_math500 --sets math500 --n 1
ev base e1_base_fitgen --sets gsm8k_train,math_train --n 1 --ids split:fit_gen
