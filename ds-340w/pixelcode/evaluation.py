"""Report technical checks separately from human judgments of the drawing."""

from .renderer import BACKGROUND, render


def rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))


def check_candidate(row: dict, code: str) -> dict:
    """Check syntax, palette and location, not whether the depicted object is right."""
    try:
        image = render(code)
    except ValueError as error:
        return {"valid": False, "palette_ok": False, "placement_ok": False, "error": str(error)}

    foreground = [(x, y) for y in range(32) for x in range(32)
                  if image.getpixel((x, y)) != BACKGROUND]
    colors = {image.getpixel((x, y)) for y in range(32) for x in range(32)}
    x0, y0, x1, y1 = row["bounds"]
    return {
        "valid": True,
        "palette_ok": colors <= {rgb(c) for c in row["allowed_colors"]}
                      and rgb(row["color_hex"]) in colors,
        "placement_ok": bool(foreground) and all(x0 <= x < x1 and y0 <= y < y1 for x, y in foreground),
        "error": None,
    }


def evaluate(rows: list[dict], predictions: dict[str, str]) -> dict:
    """Require exact coverage, retaining invalid programs in every denominator."""
    expected = {row["id"] for row in rows}
    if not rows or len(expected) != len(rows) or expected != set(predictions):
        raise ValueError("Prediction IDs must exactly match the nonempty evaluation split")
    details = [{"id": row["id"], **check_candidate(row, predictions[row["id"]])} for row in rows]
    return {
        "n": len(rows),
        "valid_rate": sum(item["valid"] for item in details) / len(rows),
        "palette_rate": sum(item["palette_ok"] for item in details) / len(rows),
        "placement_rate": sum(item["placement_ok"] for item in details) / len(rows),
        "details": details,
    }


def check_recolor(before: str, after: str, old_color: str, new_color: str) -> bool:
    """Require every old-color pixel to change and every other pixel to stay put."""
    old, new = rgb(old_color), rgb(new_color)
    first, second = render(before), render(after)
    original = [first.getpixel((x, y)) for y in range(32) for x in range(32)]
    edited = [second.getpixel((x, y)) for y in range(32) for x in range(32)]
    return old != new and old in original and all(
        actual == (new if previous == old else previous)
        for previous, actual in zip(original, edited)
    )
