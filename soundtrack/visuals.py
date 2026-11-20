"""Artwork, portraits, scene selection, and on-screen overlay cards."""
from colorsys import hsv_to_rgb
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .common import HEIGHT, WIDTH, font

CARD_SIZE = (640, 96)
CARD_POS = (64, 462)
BAR_POS, BAR_SIZE = (64, 566), (560, 4)


def create_eclipse(accent):
    eclipse = Image.new("RGBA", (410, 410))
    draw = ImageDraw.Draw(eclipse)
    for radius, alpha in [(201, 35), (177, 64), (156, 110), (145, 230)]:
        bounds = (205 - radius, 205 - radius, 205 + radius, 205 + radius)
        if radius == 145:
            draw.ellipse(bounds, fill=(8, 12, 37, 194))
        draw.ellipse(bounds, outline=(*accent, alpha), width=2)
    draw.ellipse((61, 168, 68, 175), fill=(*accent, 246))
    draw.ellipse((357, 250, 364, 257), fill=(234, 247, 255, 224))
    return eclipse


def character_layer(portrait, accent):
    if portrait is None:
        return Image.new("RGBA", (420, 462))
    mask = Image.new("L", portrait.size)
    ImageDraw.Draw(mask).rounded_rectangle((7, 7, 412, 454), radius=96, fill=255)
    feather = np.asarray(mask.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32)
    vertical = np.linspace(1, 0, 122)
    feather[-122:] *= vertical[:, None]
    portrait = portrait.convert("RGBA")
    portrait.putalpha(Image.fromarray(feather.astype(np.uint8)))
    edge = Image.new("RGBA", portrait.size)
    ImageDraw.Draw(edge).rounded_rectangle((9, 9, 410, 452), radius=94,
                                           outline=(*accent, 110), width=2)
    edge_alpha = np.array(edge.getchannel("A"))
    edge_alpha[-122:] = (edge_alpha[-122:] * vertical[:, None]).astype(np.uint8)
    edge.putalpha(Image.fromarray(edge_alpha))
    return Image.alpha_composite(portrait, edge)


def wrap(text, typeface, width, max_lines):
    lines = [""]
    for word in text.split():
        trial = (lines[-1] + " " + word).strip()
        if lines[-1] and typeface.getlength(trial) > width:
            lines.append(word)
        else:
            lines[-1] = trial
    return lines if len(lines) <= max_lines else None


def fit_title(text, width, max_lines, largest, smallest):
    for size in range(largest, smallest - 1, -3):
        typeface = font("DejaVuSans-Bold.ttf", size)
        lines = wrap(text, typeface, width, max_lines)
        if lines and max(map(typeface.getlength, lines)) <= width:
            return typeface, lines
    typeface = font("DejaVuSans-Bold.ttf", smallest)
    lines = (wrap(text, typeface, width, 99) or [text])[:max_lines]
    while typeface.getlength(lines[-1]) > width and len(lines[-1]) > 2:
        lines[-1] = lines[-1][:-2] + "…"
    return typeface, lines


def accent_for(seed):
    hue = random.Random(seed).randrange(170, 300)
    return tuple(int(v) for v in np.array(hsv_to_rgb(hue / 360, 0.55, 1)) * 255)


def make_art(title, seed, thumbnail, background, eclipse_path, character_path, portrait,
             badge=None, caption="CINEMATIC SCORE MIX"):
    rng = random.Random(seed)
    rng.randrange(170, 300)  # Keep the star field stable for a given title.
    color = accent_for(seed)
    accent = np.array(color, dtype=np.float64)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    center_x, horizon = int(WIDTH * 0.74), int(HEIGHT * 0.59)
    light = np.exp(-((x - center_x) / 390) ** 2 - ((y - horizon) / 210) ** 2)
    pixels = np.array([6, 10, 24]) + light[..., None] * accent * 0.44
    pixels += np.random.default_rng(seed).normal(0, 0.55, (HEIGHT, WIDTH, 1))
    art = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    draw = ImageDraw.Draw(art, "RGBA")
    for _ in range(310):
        px, py = rng.randrange(WIDTH), rng.randrange(horizon)
        radius = rng.choices([0.6, 1, 1.6], [75, 22, 3])[0]
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     fill=(213, 232, 255, rng.randrange(25, 155)))
    for layer in range(4):
        base = (10, 16, 34)
        shade = tuple(int(v + (3 - layer) * 0.085 * a) for v, a in zip(base, accent))
        heights = []
        phase = rng.uniform(0, 6)
        for px in range(0, WIDTH + 1, 8):
            curve = 35 * math.sin(px / 157 + phase) + 17 * math.sin(px / 59 + phase)
            ridge = (6 + layer * 2) * math.sin(px / 18 + 2 * phase)
            heights.append((px, int(horizon + 26 + layer * 47 + curve + ridge + rng.randrange(0, 8))))
        draw.polygon(heights + [(WIDTH, HEIGHT), (0, HEIGHT)], fill=shade)

    draw.line((64, 632, 1216, 632), fill=(*color, 67))
    draw.line((64, 96, 154, 96), fill=(*color, 255), width=3)
    mono = font("DejaVuSansMono.ttf", 21)
    draw.text((64, 116), caption, font=mono, fill=color)
    typeface, lines = fit_title(title, 715, 3, 83, 22)
    for index, line in enumerate(lines):
        draw.text((60, 160 + index * (typeface.size + 4)), line, font=typeface,
                  fill=(244, 245, 255), stroke_width=2, stroke_fill=(10, 13, 33))
    art.save(background)
    eclipse = create_eclipse(color)
    eclipse.save(eclipse_path)
    character = character_layer(portrait, color)
    character.save(character_path)
    if portrait is not None:
        eclipse = eclipse.resize((530, 530), Image.Resampling.LANCZOS)
    art.paste(eclipse, (center_x - eclipse.width // 2, horizon - 95 - eclipse.height // 2), eclipse)
    art.paste(character, (750, 106), character)
    draw = ImageDraw.Draw(art, "RGBA")
    if badge:
        badge_font = font("DejaVuSans-Bold.ttf", 52)
        text = badge.upper()
        box_w = int(badge_font.getlength(text)) + 52
        top = max(176 + len(lines) * (typeface.size + 4), 470)
        draw.rounded_rectangle((64, top, 64 + box_w, top + 84), radius=20, fill=(*color, 255))
        draw.text((90, top + 12), text, font=badge_font, fill=(9, 12, 30))
    draw.text((64, 646), "CONTINUOUS SOUNDTRACK MIX", font=mono, fill=(192, 207, 241))
    art.save(thumbnail, quality=92)
    return color


def _card(label, accent):
    card = Image.new("RGBA", CARD_SIZE)
    if not label:
        return card
    draw = ImageDraw.Draw(card)
    for index, height in enumerate((10, 16, 7)):
        draw.rectangle((index * 6, 18 - height, index * 6 + 3, 18), fill=(*accent, 255))
    draw.text((26, 0), "NOW PLAYING", font=font("DejaVuSansMono.ttf", 17), fill=(*accent, 255))
    typeface, lines = fit_title(label, CARD_SIZE[0] - 8, 1, 34, 22)
    draw.text((0, 34), lines[0], font=typeface, fill=(244, 245, 255, 255),
              stroke_width=2, stroke_fill=(10, 13, 33, 255))
    return card


def scene_shade(path):
    """Transparent-to-dark gradient over the lower third, keeping the waveform legible on scenes."""
    shade = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    rows = int(HEIGHT * 0.36)
    shade[-rows:, :, 3] = (np.linspace(0, 1, rows) ** 1.6 * 185).astype(np.uint8)[:, None]
    shade[..., :3] = (5, 8, 20)
    Image.fromarray(shade, "RGBA").save(path)
    return path


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


def now_playing_cards(entries, accent, scratch, seconds):
    """Concat list that shows each cue name from its start time. entries: [(start, label)]."""
    if not entries:
        return None
    scratch.mkdir(parents=True, exist_ok=True)
    if entries[0][0] > 0:
        entries = [(0.0, "")] + list(entries)
    files, timeline = {}, []
    for start, label in entries:
        if label not in files:
            files[label] = scratch / f"card-{len(files)}.png"
            _card(label, accent).save(files[label])
        timeline.append((start, files[label]))
    return write_timeline(timeline, seconds, scratch / "cards.ffconcat")


def portrait_timeline(layers, scenes, scene_people, seconds, listing):
    """Swap the foreground portrait halfway through each cutaway, where the scene hides the change.

    layers: {person: png path}, most important first. After a scene, the portrait shows that
    scene's character when there is one, otherwise the next character in rotation.
    """
    if not layers:
        return None
    order = list(layers)
    current = order[0]
    timeline = [(0.0, layers[current])]
    for (_, _, length, when), person in zip(scenes, scene_people):
        following = person if person in layers else order[(order.index(current) + 1) % len(order)]
        if following != current:
            current = following
            timeline.append((when + length / 2, layers[current]))
    return write_timeline(timeline, seconds, listing)


def make_vertical_art(title, thumbnail, output, accent, cta):
    """1080x1920 poster for the Shorts teaser, built from the landscape thumbnail."""
    width, height = 1080, 1920
    with Image.open(thumbnail) as image:
        source = image.convert("RGB")
    backdrop = ImageOps.fit(source, (width, height), Image.Resampling.LANCZOS)
    backdrop = backdrop.filter(ImageFilter.GaussianBlur(36))
    backdrop = Image.blend(backdrop, Image.new("RGB", backdrop.size, (6, 9, 22)), 0.55)
    front = source.resize((width - 80, round((width - 80) * source.height / source.width)),
                          Image.Resampling.LANCZOS)
    mask = Image.new("L", front.size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, front.width - 1, front.height - 1), radius=36, fill=255)
    backdrop.paste(front, (40, 640), mask)
    draw = ImageDraw.Draw(backdrop)
    draw.line((60, 250, 170, 250), fill=accent, width=4)
    draw.text((60, 272), "CINEMATIC SCORE MIX", font=font("DejaVuSansMono.ttf", 34), fill=accent)
    typeface, lines = fit_title(title, 960, 3, 96, 40)
    for index, line in enumerate(lines):
        draw.text((56, 330 + index * (typeface.size + 6)), line, font=typeface,
                  fill=(244, 245, 255), stroke_width=3, stroke_fill=(10, 13, 33))
    cta_font = font("DejaVuSans-Bold.ttf", 44)
    cta_w = cta_font.getlength(cta)
    draw.rounded_rectangle(((width - cta_w) / 2 - 36, 1560, (width + cta_w) / 2 + 36, 1660),
                           radius=50, fill=accent)
    draw.text(((width - cta_w) / 2, 1584), cta, font=cta_font, fill=(9, 12, 30))
    backdrop.save(output)
