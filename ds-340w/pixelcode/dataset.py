"""Build the original 120-scene pilot benchmark with fixed, disjoint splits."""

import json
from pathlib import Path

COLORS = {
    "red": "#d94b4b", "orange": "#c65326", "green": "#528c45",
    "blue": "#437ab5", "purple": "#7963af", "teal": "#368f89",
}
DETAIL = "#30342c"
POSITIONS = {"top left": (0, 0), "top right": (16, 0),
             "bottom left": (0, 16), "bottom right": (16, 16)}
PATTERNS = {
    "mushroom": ["..aaaa..", ".aaaaaa.", "aaaaaaaa", "aabaabaa",
                 "aaaaaaaa", "...bb...", "...bb...", "..bbbb.."],
    "tree": ["...aa...", "..aaaa..", ".aaaaaa.", "aaaaaaaa",
             "..aaaa..", ".aaaaaa.", "...bb...", "...bb..."],
    "house": ["...aa...", "..aaaa..", ".aaaaaa.", "aaaaaaaa",
              ".aaaaaa.", ".abaaba.", ".aaabaa.", ".aaabaa."],
    "boat": ["...b....", "...ba...", "...baa..", "...baaa.",
             "...b....", "aaaaaaaa", ".aaaaaa.", "..aaaa.."],
    "heart": [".aa..aa.", "aaaaaaaa", "aaaaaaaa", "aaaaaaaa",
              ".aaaaaa.", "..aaaa..", "...aa...", "........"],
}


def reference_program(object_name: str, color: str, position: str) -> str:
    """Draw a hand-authored sprite template, not a language-model prediction."""
    origin_x, origin_y = POSITIONS[position]
    palette = {"a": color, "b": DETAIL}
    lines = []
    for y, row in enumerate(PATTERNS[object_name]):
        for x, token in enumerate(row):
            if token != ".":
                lines.append(f'rect({origin_x + 2*x}, {origin_y + 2*y}, 2, 2, "{palette[token]}");')
    return "\n".join(lines)


def build_records() -> list[dict]:
    """Use one held-out color per object-position pair; retain all other colors.

    Split by complete scene before making candidate pairs. The same scene must
    never enter both splits through paraphrases or multiple model samples.
    """
    rows = []
    for object_index, object_name in enumerate(PATTERNS):
        for color_index, (color_name, color_hex) in enumerate(COLORS.items()):
            for position_index, (position, (x, y)) in enumerate(POSITIONS.items()):
                scene_id = f"{object_name}-{color_name}-{position.replace(' ', '-')}"
                split = "test" if color_index == (object_index + position_index) % 6 else "train"
                prompt = (
                    f"Draw one {color_name} {object_name} in the {position} quadrant of a "
                    f"32 by 32 pixel canvas. Use {color_hex} as the main color, "
                    f"{DETAIL} for optional details, and #f5f5f4 as the background. "
                    "Keep the other three quadrants empty. Return only rect(x, y, width, "
                    'height, "#rrggbb"); calls with integer coordinates.'
                )
                rows.append({
                    "id": scene_id, "scene_id": scene_id, "split": split,
                    "object": object_name, "color_name": color_name, "color_hex": color_hex,
                    "position": position, "bounds": [x, y, x + 16, y + 16],
                    "allowed_colors": [color_hex, DETAIL, "#f5f5f4"], "prompt": prompt,
                    "reference_code": reference_program(object_name, color_hex, position),
                    "reference_origin": "hand_authored_sprite_template",
                })
    return rows


def read_records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty and have unique IDs")
    return rows
