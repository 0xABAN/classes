"""Interpret a bounded drawing-only subset of JavaScript without executing JS.

Programs consist only of rect(x, y, width, height, "#rrggbb"); calls. Integer
coordinates map directly to a 32 by 32 raster. Reject the entire program if
any token or rectangle is invalid; never repair a candidate during evaluation.
"""

import re
from PIL import Image, ImageDraw

SIZE = 32
BACKGROUND = (245, 245, 244)
MAX_COMMANDS = 512
COMMAND = re.compile(
    r'\s*rect\(\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*'
    r'(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*"(#[0-9a-fA-F]{6})"\s*\)\s*;'
)


def parse(code: str) -> list[tuple[int, int, int, int, str]]:
    """Validate the whole input before rendering, including resource limits."""
    if not isinstance(code, str) or not code.strip() or len(code) > 65536:
        raise ValueError("Expected a nonempty drawing program under 65,537 characters")

    commands = []
    offset = 0
    while code[offset:].strip():
        match = COMMAND.match(code, offset)
        if match is None:
            raise ValueError(f"Unsupported syntax at character {offset}")
        x, y, width, height = map(int, match.groups()[:4])
        if width < 1 or height < 1 or x + width > SIZE or y + height > SIZE:
            raise ValueError("Rectangle must have positive area and stay inside the canvas")
        commands.append((x, y, width, height, match.group(5).lower()))
        if len(commands) > MAX_COMMANDS:
            raise ValueError("Program exceeds 512 rectangles")
        offset = match.end()
    return commands


def render(code: str) -> Image.Image:
    """Render deterministic RGB pixels; overlaps use last-command-wins order."""
    commands = parse(code)
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for x, y, width, height, color in commands:
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill=color)
    return image
