"""FFmpeg filter graphs for the main video and the vertical Shorts teaser."""
import subprocess

from .common import FFMPEG, RATE, WAVE_COLOR, WIDTH, media_duration
from .visuals import BAR_POS, BAR_SIZE, CARD_POS

WAVE_HEX = "0x%02x%02x%02x" % WAVE_COLOR
# Pin the program audio so the mono waveform branch cannot downmix it during negotiation.
STEREO = f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo"


def _check_duration(output, seconds):
    duration = media_duration(output)
    if duration is None or abs(duration - seconds) > 0.1:
        raise RuntimeError(f"Expected {seconds}s, rendered {duration}s")


def render_video(audio, backdrop, eclipse, character, scenes, output, seconds, draft,
                 audio_filter="anull", cards=None, metadata=None):
    width, height, fps = (640, 360, 12) if draft else (WIDTH, 720, 24)
    wave_width, wave_height = round(width * 0.90), round(height * 0.083)
    ratio = width / WIDTH
    disc_size = round((530 if character else 410) * ratio)
    inputs = [
        "-framerate", str(fps), "-i", str(backdrop),
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio),
        "-framerate", str(fps), "-i", str(eclipse),
    ]
    graph = [
        f"[0:v]scale={width}:{height},format=yuv420p,loop=-1:1:0,setpts=N[sky]",
        f"[2:v]scale={disc_size}:{disc_size},format=yuva420p[disc]",
        "[sky][disc]overlay=x='W*0.74-w/2+W*0.026*sin(t/7)':"
        "y='H*0.45-h/2+H*0.04*sin(t/10)'[eclipse]",
        f"[1:a]{audio_filter},{STEREO},asplit[waveaudio][mix]",
        f"[waveaudio]aformat=channel_layouts=mono,showwaves=s={wave_width}x{wave_height}:r={fps}:"
        f"mode=cline:scale=sqrt:draw=full:colors={WAVE_HEX},format=rgba[wave]",
    ]
    layer = "eclipse"
    next_input = 3
    if character:
        inputs += ["-framerate", str(fps), "-i", str(character)]
        graph += [
            f"[{next_input}:v]scale={round(420 * ratio)}:{round(462 * ratio)},format=yuva420p[portrait]",
            "[eclipse][portrait]overlay=x=W*0.586:y=H*0.147[characters]",
        ]
        layer = "characters"
        next_input += 1
    if cards:
        inputs += ["-f", "concat", "-safe", "0", "-i", str(cards)]
        graph += [
            f"[{next_input}:v]fps={fps},scale=iw*{ratio:.5f}:-2,format=rgba[cards]",
            f"[{layer}][cards]overlay=x={round(CARD_POS[0] * ratio)}:y={round(CARD_POS[1] * ratio)}:"
            "eof_action=repeat[carded]",
        ]
        layer = "carded"
        next_input += 1
    bar_w, bar_h = round(BAR_SIZE[0] * ratio), max(2, round(BAR_SIZE[1] * ratio))
    graph += [
        f"color=c={WAVE_HEX}@0.16:s={bar_w}x{bar_h}:r={fps},format=rgba[track]",
        f"color=c={WAVE_HEX}:s={bar_w}x{bar_h}:r={fps},format=rgba[fill]",
        f"[track][fill]overlay=x='-w+w*t/{seconds}':shortest=1[progress]",
        f"[{layer}][progress]overlay=x={round(BAR_POS[0] * ratio)}:y={round(BAR_POS[1] * ratio)}"
        "[barred]",
        "[barred][wave]overlay=x=(W-w)/2:"
        "y='H-h-H*0.043+H*0.01*sin(t*1.4)':eof_action=pass[visualizer]",
    ]
    layer = "visualizer"
    scene_w = 2 * round(width * 0.91 / 2)
    scene_h = 2 * round(height * 0.87 / 2)
    for index, (path, offset, length, when) in enumerate(scenes):
        inputs += ["-ss", str(offset), "-t", str(length), "-i", str(path)]
        graph += [
            f"[{next_input + index}:v]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={scene_w}:{scene_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={scene_w}:{scene_h}:(ow-iw)/2:(oh-ih)/2:color=0x090e1e,format=yuva420p,"
            f"fade=t=in:st=0:d=0.55:alpha=1,fade=t=out:st={length - 0.70}:d=0.65:alpha=1,"
            f"setpts=PTS+{when}/TB[cut{index}]",
            f"[{layer}][cut{index}]overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:"
            f"enable='between(t,{when},{when + length})'[edit{index}]",
        ]
        layer = f"edit{index}"
    graph += [
        f"[{layer}]fade=t=in:st=0:d=1,fade=t=out:st={max(0, seconds - 3)}:d=3,setsar=1[v]",
        f"[mix]afade=t=out:st={max(0, seconds - 5)}:d=5[a]",
    ]
    meta_args = []
    if metadata:
        meta_index = next_input + len(scenes)
        inputs += ["-f", "ffmetadata", "-i", str(metadata)]
        meta_args = ["-map_metadata", str(meta_index), "-map_chapters", str(meta_index)]
    subprocess.run(FFMPEG + [
        "-filter_complex_threads", "2", *inputs,
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", *meta_args,
        "-t", str(seconds), "-c:v", "libx264", "-threads", "4", "-preset", "veryfast",
        "-tune", "stillimage", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ], check=True)
    _check_duration(output, seconds)


def render_short(audio, poster, output, start, seconds, audio_filter="anull", draft=False):
    """Vertical teaser cut from the loudest stretch of the arranged mix."""
    width, height, fps = (540, 960, 12) if draft else (1080, 1920, 30)
    wave_w, wave_h = round(width * 0.86), round(height * 0.085)
    graph = [
        f"[0:v]scale={width}:{height},format=yuv420p,loop=-1:1:0,setpts=N[poster]",
        f"[1:a]{audio_filter},{STEREO},asplit[waveaudio][mix]",
        f"[waveaudio]aformat=channel_layouts=mono,showwaves=s={wave_w}x{wave_h}:r={fps}:"
        f"mode=cline:scale=sqrt:draw=full:colors={WAVE_HEX},format=rgba[wave]",
        f"[poster][wave]overlay=x=(W-w)/2:y=H*0.72:eof_action=pass,"
        f"fade=t=in:st=0:d=0.6,fade=t=out:st={seconds - 1.2}:d=1.2,setsar=1[v]",
        f"[mix]afade=t=in:st=0:d=1.5,afade=t=out:st={seconds - 2.5}:d=2.5[a]",
    ]
    subprocess.run(FFMPEG + [
        "-framerate", str(fps), "-i", str(poster),
        "-ss", f"{start:.3f}", "-t", str(seconds),
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio),
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", "-t", str(seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ], check=True)
    _check_duration(output, seconds)
