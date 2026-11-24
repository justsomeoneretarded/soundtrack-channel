"""Click-worthy thumbnails: one sharp frame of a main character and a few words of type.

Each thumbnail is a full-bleed film still, framed so the character's face sits on a
third and the words sit in the quiet part of the picture, never over the face. Three
versions are made, from different frames and in different type styles, so they can be
compared with YouTube's thumbnail test.
"""
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .common import FONT_ROOT, YEAR
from .reel import resolution_weight
from .visuals import typeface

SIZE = (1280, 720)
WHITE = (247, 245, 240)
DISPLAY_DIR = Path(os.environ.get("SOUNDTRACK_DISPLAY_FONT_DIR", Path.home() / ".cache" / "soundtrack-fonts"))
FRAME_W, FRAME_H = 640, 360  # Analysis frame size used by footage.analyse.


# --------------------------------------------------------------------------- type

def display_font(family, size, weight=None):
    """Cinzel (classical serif) or Josefin Sans (thin geometric caps), with fallbacks."""
    files = {"serif": "Cinzel[wght].ttf", "geometric": "JosefinSans[wght].ttf"}
    path = DISPLAY_DIR / files[family]
    if path.is_file():
        face = ImageFont.truetype(str(path), size)
        try:
            face.set_variation_by_name(weight or ("Bold" if family == "serif" else "Light"))
        except (OSError, ValueError):
            pass
        return face
    if family == "serif":
        for name in ("LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf"):
            for folder in (FONT_ROOT, Path("/usr/share/fonts/truetype/liberation")):
                if (folder / name).is_file():
                    return ImageFont.truetype(str(folder / name), size)
    return typeface("Light" if family == "geometric" else "SemiBold", size)


def _width(text, face, tracking):
    if not text:
        return 0.0
    return sum(face.getlength(char) for char in text) + tracking * face.size * (len(text) - 1)


def _draw(draw, x, y, text, face, tracking, fill):
    for char in text:
        draw.text((x, y), char, font=face, fill=fill)
        x += face.getlength(char) + tracking * face.size


def _fit(family, text, width, largest, smallest, tracking, weight=None):
    for size in range(largest, smallest - 1, -2):
        face = display_font(family, size, weight)
        if _width(text, face, tracking) <= width:
            return face
    return display_font(family, smallest, weight)


def short_name(title):
    """'Pirates of the Caribbean: The Curse of the Black Pearl (2003)' -> 'Pirates of the Caribbean'."""
    text = YEAR.sub("", title).replace("()", "").replace("[]", "").strip(" -–—()[]")
    for separator in (": ", " – ", " — ", " - "):
        if separator in text:
            return text.split(separator, 1)[0].strip()
    return text


# --------------------------------------------------------------------------- frames

def crop_bars(image, threshold=22):
    """Remove black letterbox and pillarbox bars."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    rows = np.flatnonzero(gray.mean(axis=1) > threshold)
    cols = np.flatnonzero(gray.mean(axis=0) > threshold)
    if rows.size < gray.shape[0] * 0.4 or cols.size < gray.shape[1] * 0.4:
        return image, (0, 0)
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return image.crop((left, top, right, bottom)), (left, top)


def _source_box(box, size):
    """Map a face box from the padded 640x360 analysis frame to source pixels."""
    width, height = size
    scale = min(FRAME_W / width, FRAME_H / height)
    ox, oy = (FRAME_W - width * scale) / 2, (FRAME_H - height * scale) / 2
    x, y, w, h = box
    return ((x - ox) / scale, (y - oy) / scale, w / scale, h / scale)


def _sharpness(gray):
    import cv2

    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(np.ascontiguousarray(gray, dtype=np.uint8), cv2.CV_64F).var())


def named_people(analysis, characters):
    """{character name: person id}, matched through videos whose titles name the character.

    A scene upload titled after a character shows that character more than anyone else, so
    the person seen most (by face size and count) in those uploads takes the name.
    """
    from .common import terms

    names = {}
    taken = set()
    for character in characters:
        words = {word for word in terms(character) if len(word) > 2}
        if not words:
            continue
        videos = {index for index, video in enumerate(analysis.videos)
                  if words & terms(video.credit[0]) and len(words & terms(video.credit[0])) >= min(2, len(words))}
        if not videos:
            continue
        weight = {}
        for face in analysis.faces:
            if face.video in videos and face.person >= 0 and face.person not in taken:
                weight[face.person] = weight.get(face.person, 0.0) + math.sqrt(face.area) * face.quality
        if weight:
            person = max(weight, key=weight.get)
            if weight[person] >= 0.35 * sum(weight.values()) / max(1, len(videos)):
                names[character] = person
                taken.add(person)
    return names


def candidate_frames(analysis, grab, count=3, examine=18, lead=None):
    """Up to `count` (image, face_box, person) from different moments, best first.

    Faces of the main characters are ranked by detector quality, how directly they face
    the camera, size, and source resolution; the best few are grabbed at full size and
    re-scored by real sharpness and exposure.
    """
    main = [person for person, _, _ in analysis.people]
    if lead is not None and lead not in main:
        main = [lead] + main
    weights = {person: 1.0 - 0.12 * rank for rank, person in enumerate(main)}
    crowd = {}
    for face in analysis.faces:
        key = (face.video, round(face.time * 2))
        crowd[key] = crowd.get(key, 0) + 1
    pool = []
    for face in analysis.faces:
        if analysis.recognised and face.person not in weights:
            continue
        video = analysis.videos[face.video]
        area = face.area
        size_fit = 1.0 if 0.02 <= area <= 0.18 else 0.5
        others = crowd.get((face.video, round(face.time * 2)), 1)
        single = 1.0 if others <= 1 else 0.8 if others == 2 else 0.4  # One clear subject reads best.
        value = (face.quality * (0.4 + 0.6 * face.frontal) * size_fit * min(1.0, math.sqrt(area) * 6)
                 * weights.get(face.person, 0.8) * resolution_weight(video.size) * single)
        pool.append((value, face))
    pool.sort(key=lambda item: -item[0])
    chosen, examined = [], []
    for value, face in pool:
        if len(examined) >= examine:
            break
        if any(f.video == face.video and abs(f.time - face.time) < 4 for _, _, f in examined):
            continue
        video = analysis.videos[face.video]
        image = grab(video.path, face.time)
        if image is None:
            continue
        image = image.convert("RGB")
        x, y, w, h = _source_box(face.box, image.size)
        image, (dx, dy) = crop_bars(image)
        x, y = x - dx, y - dy
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        crop = gray[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)]
        crisp = min(1.0, _sharpness(crop) / 250) if crop.size else 0.0
        lit = 1.0 if 50 < float(crop.mean() if crop.size else 0) < 215 else 0.5
        overall = 1.0 if 35 < float(gray.mean()) < 190 else 0.6
        wide = 1.0 if image.width / image.height >= 1.6 else 0.8
        small = np.asarray(image.convert("L").resize((320, 180)), dtype=np.float32)
        busy = float(np.abs(np.diff(small, axis=1)).mean() + np.abs(np.diff(small, axis=0)).mean())
        calm = 1.0 / (1.0 + max(0.0, busy - 10) / 8)  # Crowds and clutter read as noise at feed size.
        examined.append((value * (0.25 + 0.75 * crisp) * lit * overall * wide * calm,
                         (image, (x, y, w, h), face.person), face))
    examined.sort(key=lambda item: -item[0])
    if lead is not None:
        # The named character's best frame goes first, for the dedication thumbnail.
        named = [item for item in examined if item[1][2] == lead]
        if named:
            examined.remove(named[0])
            examined.insert(0, named[0])
    people_used = {}
    for score, frame, face in examined:
        # Prefer a mix of people and moments across the versions.
        if people_used.get(frame[2], 0) >= 2:
            continue
        chosen.append(frame)
        people_used[frame[2]] = people_used.get(frame[2], 0) + 1
        if len(chosen) == count:
            break
    return chosen


def fallback_frames(shots, grab, count=3):
    """Faceless footage: the sharpest well-exposed frames from the strongest shots."""
    frames = []
    for shot in sorted(shots, key=lambda shot: -shot.score)[:count * 3]:
        image = grab(shot.path, shot.start + shot.length / 2)
        if image is None:
            continue
        image, _ = crop_bars(image.convert("RGB"))
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        frames.append((_sharpness(gray) * (1.0 if 40 < gray.mean() < 200 else 0.4), (image, None, -1)))
    frames.sort(key=lambda item: -item[0])
    return [frame for _, frame in frames[:count]]


# --------------------------------------------------------------------------- layout

def frame_to_canvas(image, face, side):
    """Crop and scale to 1280x720, putting the face on the left or right third.

    Returns (canvas, face box in canvas pixels or None).
    """
    width, height = image.size
    target = SIZE[0] / SIZE[1]
    if face is None:
        return ImageOps.fit(image, SIZE, Image.Resampling.LANCZOS, centering=(0.5, 0.45)), None
    x, y, w, h = face
    cx, cy = x + w / 2, y + h / 2
    # Tighten to the face when it is small, but never crop below the source resolution much.
    crop_h = min(height, max(h * 4.2, height * 0.62))
    crop_w = crop_h * target
    if crop_w > width:
        crop_w, crop_h = width, width / target
    want_x = {"left": 1 / 3, "right": 2 / 3, "centre": 0.5}[side]
    left = min(max(cx - crop_w * want_x, 0), width - crop_w)
    top = min(max(cy - crop_h * 0.38, 0), height - crop_h)
    box = (left, top, left + crop_w, top + crop_h)
    canvas = image.crop(tuple(int(round(v)) for v in box)).resize(SIZE, Image.Resampling.LANCZOS)
    scale = SIZE[0] / crop_w
    return canvas, ((x - left) * scale, (y - top) * scale, w * scale, h * scale)


def grade(canvas):
    canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
    canvas = ImageEnhance.Color(canvas).enhance(0.92)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.25)
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    radius = np.sqrt(((x - SIZE[0] / 2) / (SIZE[0] / 2)) ** 2 + ((y - SIZE[1] / 2) / (SIZE[1] / 2)) ** 2)
    shade = 1 - np.clip((radius - 0.7) / 0.8, 0, 1) ** 1.5 * 0.45
    pixels = np.asarray(canvas, dtype=np.float32) * shade[..., None]
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))


def _busyness(canvas, region):
    left, top, right, bottom = (int(v) for v in region)
    patch = np.asarray(canvas.convert("L").crop((left, top, right, bottom)), dtype=np.float32)
    if patch.size == 0:
        return 1e9
    return float(np.abs(np.diff(patch, axis=1)).mean() + np.abs(np.diff(patch, axis=0)).mean())


def text_region(canvas, face, block_w, block_h):
    """Top-left corner for a text block: clear of the face and hair, over the calmest area."""
    width, height = SIZE
    margin = 56
    spots = {
        "top": ((width - block_w) / 2, height * 0.08),
        "bottom": ((width - block_w) / 2, height * 0.88 - block_h),
        "left": (margin, (height - block_h) / 2),
        "right": (width - margin - block_w, (height - block_h) / 2),
        "upper-left": (margin, height * 0.1),
        "upper-right": (width - margin - block_w, height * 0.1),
    }
    best, best_cost = None, None
    for name, (x, y) in spots.items():
        region = (x - 20, y - 20, x + block_w + 20, y + block_h + 20)
        if region[0] < 0 or region[2] > width or region[1] < 0 or region[3] > height:
            continue
        cost = _busyness(canvas, region)
        if face is not None:
            fx, fy, fw, fh = face
            # Hats and hair reach well above and beside the detected face.
            head = (fx - fw * 0.7, fy - fh * 1.2, fx + fw * 1.7, fy + fh * 1.3)
            overlap_w = max(0, min(region[2], head[2]) - max(region[0], head[0]))
            overlap_h = max(0, min(region[3], head[3]) - max(region[1], head[1]))
            if overlap_w * overlap_h > 0:
                cost += 1000 + overlap_w * overlap_h / 100
        cost += 4 if name in {"left", "right"} else 0  # Centred lines read best at small sizes.
        if best_cost is None or cost < best_cost:
            best, best_cost = (x, y), cost
    return best


def _lettering(lines, width, height):
    """Render [(text, face, tracking, gap_before)] centred in a transparent block."""
    block = Image.new("RGBA", (int(width), int(height)))
    draw = ImageDraw.Draw(block)
    y = 0.0
    for text, face, tracking, gap in lines:
        y += gap
        x = (width - _width(text, face, tracking)) / 2
        _draw(draw, x, y, text, face, tracking, (*WHITE, 255))
        y += face.size
    return block


def _block_size(lines):
    width = max(_width(text, face, tracking) for text, face, tracking, _ in lines)
    height = sum(face.size + gap for _, face, _, gap in lines)
    return width + 8, height + 12


def _place(canvas, face, lines, glow=0.62):
    block_w, block_h = _block_size(lines)
    x, y = text_region(canvas, face, block_w, block_h)
    block = _lettering(lines, block_w, block_h)
    layer = Image.new("RGBA", SIZE)
    layer.paste(block, (int(x), int(y)), block)
    # A soft dark glow under the words, like light falloff rather than a box.
    alpha = layer.getchannel("A")
    halo = alpha.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(22))
    shadow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    shadow.putalpha(halo.point(lambda value: int(value * glow)))
    tight = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    tight.putalpha(alpha.filter(ImageFilter.GaussianBlur(3)).point(lambda value: int(value * 0.5)))
    result = Image.alpha_composite(canvas.convert("RGBA"), shadow)
    result = Image.alpha_composite(result, tight)
    return Image.alpha_composite(result, layer).convert("RGB")


# --------------------------------------------------------------------------- styles

def style_headline(canvas, face, words):
    """'STAY FOCUSED': one or two classical serif words, large."""
    text = words.upper()
    face_font = _fit("serif", text, SIZE[0] * 0.66, 104, 52, 0.02)
    return _place(canvas, face, [(text, face_font, 0.02, 0)])


def style_spaced(canvas, face, word):
    """'F O C U S': a single serif word with very wide letter spacing."""
    text = word.upper()
    face_font = _fit("serif", text, SIZE[0] * 0.42, 78, 44, 0.55, weight="Regular")
    return _place(canvas, face, [(text, face_font, 0.55, 0)])


def style_dedication(canvas, face, name, character):
    """Thin geometric caps: 'SUCCESSION MUSIC / to work with KENDALL ROY' for a short name,
    or 'music to work with / JACK SPARROW' when the work's name is long."""
    if not character:
        head = f"{name} MUSIC".upper() if len(name) <= 12 else "WORKING MUSIC"
        return _place(canvas, face, [(head, _fit("geometric", head, SIZE[0] * 0.8, 96, 44, 0.06), 0.06, 0)])
    if len(name) <= 16:
        head = f"{name} MUSIC".upper()
        head_font = _fit("geometric", head, SIZE[0] * 0.78, 92, 44, 0.06)
        small = display_font("geometric", max(22, round(head_font.size * 0.32)), "Regular")
        name_font = _fit("geometric", character.upper(), SIZE[0] * 0.5, round(head_font.size * 0.5), 24, 0.08,
                         weight="Regular")
        return _place(canvas, face, [(head, head_font, 0.06, 0),
                                     ("to work with", small, 0.12, round(head_font.size * 0.3)),
                                     (character.upper(), name_font, 0.08, round(head_font.size * 0.12))])
    name_font = _fit("geometric", character.upper(), SIZE[0] * 0.72, 100, 48, 0.06)
    small = display_font("geometric", max(24, round(name_font.size * 0.32)), "Regular")
    return _place(canvas, face, [("music to work with", small, 0.16, 0),
                                 (character.upper(), name_font, 0.06, round(name_font.size * 0.22))])


HEADLINES = ["Stay Focused", "Deep Work", "Locked In", "Keep Building"]
SINGLES = ["Focus"]


def make_thumbnails(frames, title, character, out, seed=0, lead_style="dedication"):
    """Write thumbnail.jpg, thumbnail-2.jpg and thumbnail-3.jpg; return their paths.

    frames: [(image, face_box or None, person)], best first. Words stay at three or fewer
    (plus a small dedication line), and lead_style picks the first version so it does not
    repeat the first title: "headline" when that title already names the work's music.
    """
    name = short_name(title)
    dedication = lambda canvas, face: style_dedication(canvas, face, name, character)  # noqa: E731
    headline = lambda canvas, face: style_headline(canvas, face, HEADLINES[seed % len(HEADLINES)])  # noqa: E731
    spaced = lambda canvas, face: style_spaced(canvas, face, SINGLES[seed % len(SINGLES)])  # noqa: E731
    styles = [dedication, headline, spaced] if lead_style == "dedication" else [headline, dedication, spaced]
    paths = []
    sides = ["right", "left", "right"]
    for index, style in enumerate(styles[:3]):
        if not frames:
            break
        image, face, _ = frames[index % len(frames)]
        canvas, box = frame_to_canvas(image, face, sides[index])
        result = style(grade(canvas), box)
        path = out / ("thumbnail.jpg" if index == 0 else f"thumbnail-{index + 1}.jpg")
        result.save(path, quality=94)
        paths.append(path)
    return paths
