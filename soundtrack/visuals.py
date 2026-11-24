"""Typography and stills: the opening title, cue names, thumbnail, and Shorts overlay.

Text is kept small, letter-spaced, and off the picture where possible: cue names sit in the
lower cinema bar, and the title shows once at the start.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import FONT_ROOT, HEIGHT, WIDTH, YEAR

INK = (238, 234, 226)
INTER_DIRS = [Path(os.environ.get("SOUNDTRACK_INTER_DIR", "/usr/share/fonts/opentype/inter"))]
FADE_STEPS = 6
FADE_STEP_SECONDS = 0.1
TITLE_SECONDS = (1.0, 8.0)   # When the opening title shows.
CUE_SECONDS = 7.0            # How long a cue name stays up.


def typeface(weight, size):
    """Inter at the given weight, falling back to DejaVu Sans when Inter is not installed."""
    for folder in INTER_DIRS:
        path = folder / f"Inter-{weight}.otf"
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    bold = weight in {"SemiBold", "Bold", "ExtraBold", "Black"}
    return ImageFont.truetype(str(FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def tracked_width(text, face, tracking):
    if not text:
        return 0.0
    return sum(face.getlength(char) for char in text) + tracking * face.size * (len(text) - 1)


def draw_tracked(draw, xy, text, face, fill, tracking):
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=face, fill=fill)
        x += face.getlength(char) + tracking * face.size


def fit_tracked(text, weight, width, largest, smallest, tracking):
    for size in range(largest, smallest - 1, -1):
        face = typeface(weight, size)
        if tracked_width(text, face, tracking) <= width:
            return face
    return typeface(weight, smallest)


def split_title(title):
    """'Name: Subtitle (2003)' -> ('NAME', 'SUBTITLE')."""
    text = YEAR.sub("", title).replace("()", "").strip(" -–—()")
    for separator in (": ", " – ", " — ", " - "):
        if separator in text:
            main, sub = text.split(separator, 1)
            return main.strip().upper(), sub.strip().upper()
    return text.upper(), ""


def _with_shadow(layer, radius):
    """Soft dark halo under the text so it reads on bright footage without a box."""
    alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda value: int(value * 0.55)))
    return Image.alpha_composite(shadow, layer)


def title_card(title, composer, size=(WIDTH, HEIGHT)):
    width, height = size
    scale = width / WIDTH
    layer = Image.new("RGBA", size)
    draw = ImageDraw.Draw(layer)
    main, sub = split_title(title)
    main_face = fit_tracked(main, "Light", width * 0.84, round(46 * scale), round(22 * scale), 0.16)
    sub_face = fit_tracked(sub, "Regular", width * 0.80, round(20 * scale), round(11 * scale), 0.32)
    credit = f"ORIGINAL SCORE BY {composer.upper()}" if composer else "ORIGINAL SCORE"
    credit_face = fit_tracked(credit, "Regular", width * 0.80, round(12 * scale), round(8 * scale), 0.36)
    rows = [(main, main_face, 0.16, 255)]
    if sub:
        rows.append((sub, sub_face, 0.32, 235))
    rows.append((credit, credit_face, 0.36, 170))
    gaps = [round(18 * scale)] * (len(rows) - 1)
    gaps[-1] = round(34 * scale)
    total = sum(face.size for _, face, _, _ in rows) + sum(gaps)
    y = (height - total) / 2 - height * 0.02
    for index, (text, face, tracking, alpha) in enumerate(rows):
        x = (width - tracked_width(text, face, tracking)) / 2
        if index == len(rows) - 1:
            # A short rule above the credit line.
            rule = round(36 * scale)
            draw.line(((width - rule) / 2, y - gaps[-1] / 2, (width + rule) / 2, y - gaps[-1] / 2),
                      fill=(*INK, 120), width=1)
        draw_tracked(draw, (x, y), text, face, (*INK, alpha), tracking)
        y += face.size + (gaps[index] if index < len(gaps) else 0)
    # A soft, wide darkening behind the words keeps them legible on bright scenes.
    glow = Image.new("L", size)
    top = (height - total) / 2 - height * 0.02
    ImageDraw.Draw(glow).ellipse((width * 0.12, top - height * 0.12, width * 0.88, top + total + height * 0.12),
                                 fill=110)
    glow = glow.filter(ImageFilter.GaussianBlur(round(70 * scale)))
    backing = Image.new("RGBA", size, (0, 0, 0, 0))
    backing.putalpha(glow)
    return Image.alpha_composite(backing, _with_shadow(layer, max(2, round(10 * scale))))


def cue_card(label, size=(WIDTH, HEIGHT), bar=0.128):
    """Cue name in the lower cinema bar, left-aligned with the picture's safe margin."""
    width, height = size
    scale = width / WIDTH
    layer = Image.new("RGBA", size)
    if not label:
        return layer
    draw = ImageDraw.Draw(layer)
    text = label.upper()
    face = fit_tracked(text, "Regular", width * 0.6, round(15 * scale), round(9 * scale), 0.22)
    bar_px = round(height * bar / 2) * 2
    y = height - bar_px / 2 - face.size * 0.62
    x = round(64 * scale)
    draw.line((x, y + face.size * 0.6, x + round(18 * scale), y + face.size * 0.6), fill=(*INK, 150), width=1)
    draw_tracked(draw, (x + round(30 * scale), y), text, face, (*INK, 225), 0.22)
    return layer


def _faded(image, level):
    if level >= 1:
        return image
    faded = image.copy()
    faded.putalpha(image.getchannel("A").point(lambda value: int(value * level)))
    return faded


def write_timeline(entries, seconds, listing):
    """ffconcat list showing each image from its start time. entries: [(start, path)], sorted."""
    entries = [(start, path) for start, path in entries if start < seconds]
    if not entries:
        return None
    if entries[0][0] > 0:
        entries[0] = (0.0, entries[0][1])
    lines = ["ffconcat version 1.0"]
    for index, (start, path) in enumerate(entries):
        end = entries[index + 1][0] if index + 1 < len(entries) else seconds
        lines += [f"file '{Path(path).as_posix()}'", f"duration {max(end - start, 0.05):.3f}"]
    lines.append(f"file '{Path(entries[-1][1]).as_posix()}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return listing


def text_timeline(title, composer, cues, seconds, scratch, size=(WIDTH, HEIGHT)):
    """Concat list of text overlays: the opening title, then each cue name as it begins.

    cues: [(start, label)]. Each card fades in and out in a few alpha steps; images hold
    between entries, so the list stays short however long the video is.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    blank = scratch / "blank.png"
    Image.new("RGBA", size).save(blank)
    shows = []
    title_start, title_end = TITLE_SECONDS
    if seconds >= 20:
        shows.append((title_start, min(title_end, seconds * 0.4), title_card(title, composer, size)))
    free_from = shows[-1][1] + 1.0 if shows else 0.0
    cards = {}
    for start, label in cues:
        start = max(start + 1.5, free_from)
        end = min(start + CUE_SECONDS, seconds - 4)
        if not label or end - start < 3:
            continue
        if label not in cards:
            cards[label] = cue_card(label, size)
        shows.append((start, end, cards[label]))
        free_from = end + 1.0
    entries, files = [(0.0, blank)], {}
    ramp = FADE_STEPS * FADE_STEP_SECONDS
    for start, end, image in shows:
        key = id(image)
        if key not in files:
            files[key] = []
            for step in range(1, FADE_STEPS + 1):
                path = scratch / f"text-{len(files)}-{step}.png"
                _faded(image, step / FADE_STEPS).save(path)
                files[key].append(path)
        steps = files[key]
        for step, path in enumerate(steps):
            entries.append((start + step * FADE_STEP_SECONDS, path))
        for step, path in enumerate(reversed(steps)):
            entries.append((end - ramp + step * FADE_STEP_SECONDS, path))
        entries.append((end, blank))
    entries.sort(key=lambda entry: entry[0])
    return write_timeline(entries, seconds, scratch / "text.ffconcat")


def _dark_gradient(width, height):
    y = np.linspace(0, 1, height)[:, None, None]
    top, bottom = np.array([18, 22, 28]), np.array([5, 6, 9])
    return np.uint8(np.broadcast_to(top * (1 - y) + bottom * y, (height, width, 3)))


def plain_backdrop(path, size=(WIDTH, HEIGHT)):
    """Used only when no footage is available."""
    Image.fromarray(_dark_gradient(*size)).save(path)
    return path


def short_overlay(title, cta, path, size=(1080, 1920)):
    """Title above the picture and a quiet call to action below it, for the vertical teaser."""
    width, height = size
    scale = width / 1080
    layer = Image.new("RGBA", size)
    draw = ImageDraw.Draw(layer)
    main, sub = split_title(title)
    main_face = fit_tracked(main, "Light", width * 0.86, round(58 * scale), round(28 * scale), 0.14)
    sub_face = fit_tracked(sub, "Regular", width * 0.84, round(26 * scale), round(14 * scale), 0.28)
    y = height * 0.25
    draw_tracked(draw, ((width - tracked_width(main, main_face, 0.14)) / 2, y), main, main_face, (*INK, 255), 0.14)
    if sub:
        y += main_face.size * 1.5
        draw_tracked(draw, ((width - tracked_width(sub, sub_face, 0.28)) / 2, y), sub, sub_face,
                     (*INK, 230), 0.28)
    cta_face = fit_tracked(cta.upper(), "Regular", width * 0.84, round(22 * scale), round(12 * scale), 0.3)
    y = height * 0.73
    rule = round(40 * scale)
    draw.line(((width - rule) / 2, y - 26 * scale, (width + rule) / 2, y - 26 * scale), fill=(*INK, 140), width=2)
    draw_tracked(draw, ((width - tracked_width(cta.upper(), cta_face, 0.3)) / 2, y), cta.upper(), cta_face,
                 (*INK, 210), 0.3)
    _with_shadow(layer, round(12 * scale)).save(path)
    return path
