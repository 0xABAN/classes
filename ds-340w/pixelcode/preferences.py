"""Export explicitly labeled training comparisons; never synthesize human votes."""

from .renderer import render


def training_pair(row: dict, a: str, b: str, choice: str, reviewer: str) -> dict:
    """Return the standard prompt/chosen/rejected fields for a token-level DPO run.

    Both programs must render, and only train scenes with an explicit non-tie
    human choice are eligible. Keep ties in the original annotation log.
    """
    if row["split"] != "train" or choice not in {"a", "b"} or not reviewer.strip():
        raise ValueError("Training needs a train scene, a/b choice, and reviewer ID")
    if render(a).tobytes() == render(b).tobytes():
        raise ValueError("Identical rendered candidates do not form a preference pair")
    chosen, rejected = (a, b) if choice == "a" else (b, a)
    return {"scene_id": row["scene_id"], "prompt": row["prompt"],
            "chosen": chosen, "rejected": rejected, "reviewer": reviewer}
