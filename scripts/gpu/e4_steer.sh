# E4 steering baseline: add each edit's mean residual write (mean(v.h) * u, from scripts/e3_overlap.py) as a constant
# vector at layers 15/19 of the *unedited* model (vLLM plugin wslp.eval.steer). dev_a sweep over scale, then the best
# on MATH-500 x1. Compile cache off: its key does not see the patch.
source scripts/gpu/common.sh
uv pip install -q -e . --no-deps   # registers the vllm.general_plugins entry point
ckpt base -
export VLLM_DISABLE_COMPILE_CACHE=1
st() {  # st <steer npz name> <scale> <out> <args...>
  local n=$1 s=$2 o=$3; shift 3
  WSLP_STEER=steer/$n.npz WSLP_STEER_SCALE=$s ev base $o "$@"
}
for s in 1 2 0.5; do st shift_k8 $s dev_a_steer_shift_k8_x$s --sets gsm8k_train,math_train --n 1 --ids split:dev_a; done
st astra_p50c 1 dev_a_steer_astra_x1 --sets gsm8k_train,math_train --n 1 --ids split:dev_a
best=$(python - <<'PY'
import json, glob
r = {d.split("dev_a_steer_shift_k8_x")[1]: sum(v["ok"] for k, v in json.load(open(d + "/scores.json")).items() if not k.startswith("_"))
     for d in glob.glob("results/gpu/dev_a_steer_shift_k8_x*")}
print(max(r, key=r.get))
PY
)
echo "best scale $best"
st shift_k8 $best math500_steer_shift_k8_x$best --sets math500 --n 1
