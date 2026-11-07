"""Answer extraction and grading, ported from exp8_math/code/mathqa.py (MIT, the authors of "LLMs can understand and
write neural network weights"): last \\boxed{} after the last </think>, math-verify equivalence."""
import functools
import re
import threading

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def build_prompt(problem):
    return problem + "\n" + INSTRUCTION


def answer_text(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))


def _timeouts():
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s, to):
    import math_verify
    return math_verify.parse("\\boxed{" + s + "}", parsing_timeout=to)


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    import math_verify
    to = _timeouts()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        return bool(math_verify.verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001
        return False
