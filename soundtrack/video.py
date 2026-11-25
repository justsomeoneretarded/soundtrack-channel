"""FFmpeg filter graphs for the main video and the vertical Shorts teaser."""
import re
import subprocess

from .common import FFMPEG, RATE, media_duration
from .reel import BAR, frame_size

# Pin the program audio to stereo through the filter graph.
STEREO = f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo"


def _check_duration(output, seconds):
    duration = media_duration(output)
    if duration is None or abs(duration - seconds) > 0.1:
        raise RuntimeError(f"Expected {seconds}s, rendered {duration}s")
    picture_duration = media_duration(output, stream="v:0")
    if picture_duration is None or abs(picture_duration - seconds) > 0.1:
        raise RuntimeError(f"Expected {seconds}s of video, rendered {picture_duration}s")


def _check_black_picture(output, seconds, reel=False):
    if seconds < 30:
        return
    # Sample keyframes to catch a sustained blackout without decoding the entire video.
    threshold = 0.085 if reel else 0.02
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-skip_frame", "nokey",
        "-i", str(output), "-map", "0:v:0", "-an",
        "-vf", f"fps=1/2,scale=64:36,blackdetect=d=28:pic_th=0.995:pix_th={threshold}",
        "-f", "null", "-",
    ], capture_output=True, text=True, check=True)
    blackout = re.search(r"black_start:([\d.]+) black_end:([\d.]+)", result.stderr)
    if blackout:
        start, end = blackout.groups()
        raise RuntimeError(f"Video is black from {start}s to {end}s")


def _picture_input(reel, backdrop, fps, start=0.0):
    """Input arguments and the first graph link for the reel, or a still backdrop without one."""
    if reel:
        seek = ["-ss", f"{start:.3f}"] if start else []
        return [*seek, "-i", str(reel)], "setpts=PTS-STARTPTS"
    # Decode the image once and repeat it inside the graph; a looping image input fills memory.
    return ["-framerate", str(fps), "-i", str(backdrop)], "loop=-1:1:0,setpts=N/FRAME_RATE/TB"


def render_video(audio, reel, backdrop, text, output, seconds, draft,
                 audio_filter="anull", metadata=None):
    """reel: assembled MP4 of graded footage (or None to use `backdrop`).
    text: ffconcat list of full-frame RGBA text overlays, or None."""
    width, height, fps = frame_size(draft)
    picture, first = _picture_input(reel, backdrop, fps)
    inputs = [*picture, "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio)]
    graph = [f"[0:v]{first},fps={fps},settb=1/{fps},setpts=N,scale={width}:{height},setsar=1,format=yuv420p[picture]",
             f"[1:a]{audio_filter},{STEREO},afade=t=out:st={max(0, seconds - 5)}:d=5[a]"]
    layer, next_input = "picture", 2
    if text:
        # No fps filter on the image timeline: it would emit every held frame at once.
        inputs += ["-f", "concat", "-safe", "0", "-i", str(text)]
        graph += [f"[{next_input}:v]scale={width}:{height},format=rgba[text]",
                  f"[{layer}][text]overlay=0:0:eof_action=repeat:format=auto[titled]"]
        layer, next_input = "titled", next_input + 1
    graph.append(f"[{layer}]fade=t=in:st=0:d=1.5,fade=t=out:st={max(0, seconds - 4)}:d=4,format=yuv420p[v]")
    meta_args = []
    if metadata:
        inputs += ["-f", "ffmetadata", "-i", str(metadata)]
        meta_args = ["-map_metadata", str(next_input), "-map_chapters", str(next_input)]
    subprocess.run(FFMPEG + [
        "-filter_complex_threads", "2", *inputs,
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", *meta_args,
        "-t", str(seconds), "-r", str(fps), "-c:v", "libx264", "-threads", "4", "-preset", "veryfast",
        "-crf", "20", "-tune", "film", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ], check=True)
    _check_duration(output, seconds)
    _check_black_picture(output, seconds, reel=bool(reel))


def render_short(audio, reel, backdrop, overlay, output, start, seconds, audio_filter="anull", draft=False):
    """Vertical teaser from the loudest stretch: the framed picture over a blurred copy of itself."""
    width, height, fps = (540, 960, 12) if draft else (1080, 1920, 30)
    _, _, reel_fps = frame_size(draft)
    picture, first = _picture_input(reel, backdrop, reel_fps, start)
    visible = 1 - 2 * BAR  # The reel carries cinema bars; both views use the picture between them.
    frame_h = 2 * round(width * 9 / 16 * visible / 2)
    graph = [
        f"[0:v]{first},fps={fps},crop=iw:2*trunc(ih*{visible:.4f}/2),setsar=1,split[full][soft]",
        f"[soft]scale={width // 4}:{height // 4}:force_original_aspect_ratio=increase,"
        f"crop={width // 4}:{height // 4},boxblur=8:2,eq=brightness=-0.16:saturation=0.8,"
        f"scale={width}:{height},setsar=1[back]",
        f"[full]scale={width}:{frame_h},setsar=1[front]",
        "[back][front]overlay=x=0:y=(H-h)/2[framed]",
        f"[2:v]scale={width}:{height},format=rgba,loop=-1:1:0,setpts=N/FRAME_RATE/TB[words]",
        "[framed][words]overlay=0:0:shortest=0,"
        f"fade=t=in:st=0:d=0.6,fade=t=out:st={seconds - 1.2}:d=1.2,format=yuv420p[v]",
        f"[1:a]{audio_filter},{STEREO},afade=t=in:st=0:d=1.5,afade=t=out:st={seconds - 2.5}:d=2.5[a]",
    ]
    subprocess.run(FFMPEG + [
        *picture,
        "-ss", f"{start:.3f}", "-t", str(seconds),
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio),
        "-framerate", str(fps), "-i", str(overlay),
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", "-t", str(seconds),
        "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ], check=True)
    _check_duration(output, seconds)
