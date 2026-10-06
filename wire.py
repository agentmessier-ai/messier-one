"""The wire contract shared by the front end: how option criteria become model text, and the answer shape.

Confidence follows the TypeSafe definitions:
    choice   (top probability - 1/n) / (1 - 1/n), n options: 0 at chance, 1 when certain
    score    1 - (mean distance of the levels from the most likely level) / (the same under a uniform distribution), floor 0
    noul     none; the answer is P(true)
It is not the top probability. Anything that needs that (the gate, calibration error) calls top_probability().
"""


def options(criteria):
    """Choice criteria -> {option name: text the model reads}. A missing description (null or blank, which the contract allows)
    leaves the option name itself as the text."""
    if isinstance(criteria, list):
        return {str(v): str(v) for v in criteria}
    return {k: k if v is None or (isinstance(v, str) and not v.strip()) else v for k, v in (criteria or {}).items()}


def confidence(typ, probs):
    n = len(probs)
    if n < 2:
        return 1.0
    if typ == "choice":
        return (max(probs.values()) - 1 / n) / (1 - 1 / n)
    p = [probs[k] for k in sorted(probs, key=int)]; mode = max(range(n), key=p.__getitem__)
    spread = sum(x * abs(i - mode) for i, x in enumerate(p)); uniform = sum(abs(i - mode) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform)


def top_probability(a):
    return max(a["noul"], 1 - a["noul"]) if a["type"] == "noul" else max(a["probabilities"].values())


def answer(typ, probs):
    """probs: {option name / level index / "true","false": probability}. typ: choice, score, or noul (called boolean in the prompt)."""
    if typ in ("noul", "boolean"):
        return {"type": "noul", "noul": probs["true"]}
    if typ == "choice":
        return {"type": "choice", "choice": max(probs, key=probs.get), "confidence": confidence(typ, probs), "probabilities": probs}
    return {"type": "score", "score": float(sum(int(k) * v for k, v in probs.items())), "confidence": confidence(typ, probs), "probabilities": probs}
