from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP8 = ROOT / "third_party" / "lm_understand-logs" / "exp8_math"
BASE_MODEL = ROOT / "models" / "Qwen3.5-0.8B"
BASE_ST = BASE_MODEL / "model.safetensors-00001-of-00001.safetensors"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
EDIT_LAYERS = (15, 19)


def down_proj(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"
