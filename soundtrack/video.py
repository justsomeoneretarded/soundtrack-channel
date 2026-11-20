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


def _scene_inputs(scenes, first_input, width, height, fps, layer):
    """Overlay full-frame scenes with alpha cross-dissolves. Returns (inputs, graph, layer)."""
    inputs, graph = [], []
    for index, (path, offset, length, when) in enumerate(scenes):
        fade = min(0.6, length / 4)
        inputs += ["-ss", f"{offset:.3f}", "-t", f"{length:.3f}", "-i", str(path)]
        graph += [
            f"[{first_input + index}:v]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
            f"setsar=1,format=yuva420p,fade=t=in:st=0:d={fade:.2f}:alpha=1,"
            f"fade=t=out:st={length - fade - 0.05:.3f}:d={fade:.2f}:alpha=1,"
            f"setpts=PTS+{when}/TB[cut{index}]",
            f"[{layer}][cut{index}]overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:"
            f"enable='between(t,{when},{when + length})'[scene{index}]",
        ]
        layer = f"scene{index}"
    return inputs, graph, layer


def render_video(audio, backdrop, eclipse, characters, scenes, output, seconds, draft,
                 audio_filter="anull", cards=None, metadata=None, shade=None):
    """characters: ffconcat list of portrait layers over time, or None."""
    width, height, fps = (640, 360, 12) if draft else (WIDTH, 720, 24)
    wave_width, wave_height = round(width * 0.90), round(height * 0.083)
    ratio = width / WIDTH
    disc_size = round((530 if characters else 410) * ratio)
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
    if characters:
        inputs += ["-f", "concat", "-safe", "0", "-i", str(characters)]
        graph += [
            f"[{next_input}:v]fps={fps},scale={round(420 * ratio)}:{round(462 * ratio)},format=yuva420p[portrait]",
            "[eclipse][portrait]overlay=x=W*0.586:y=H*0.147:eof_action=repeat[characters]",
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
    # Scenes cover the artwork; the progress bar and waveform stay on top so the music reads
    # as continuous through every cutaway.
    scene_inputs, scene_graph, layer = _scene_inputs(scenes, next_input, width, height, fps, layer)
    inputs += scene_inputs
    graph += scene_graph
    next_input += len(scenes)
    if scenes and shade:
        during = "+".join(f"between(t,{when},{when + length})" for _, _, length, when in scenes)
        inputs += ["-framerate", str(fps), "-loop", "1", "-i", str(shade)]
        graph += [
            f"[{next_input}:v]scale={width}:{height},format=rgba[shade]",
            f"[{layer}][shade]overlay=0:0:shortest=0:enable='{during}'[shaded]",
        ]
        layer = "shaded"
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
    graph += [
        f"[{layer}]fade=t=in:st=0:d=1,fade=t=out:st={max(0, seconds - 3)}:d=3,setsar=1[v]",
        f"[mix]afade=t=out:st={max(0, seconds - 5)}:d=5[a]",
    ]
    meta_args = []
    if metadata:
        meta_index = next_input
        inputs += ["-f", "ffmetadata", "-i", str(metadata)]
        meta_args = ["-map_metadata", str(meta_index), "-map_chapters", str(meta_index)]
    subprocess.run(FFMPEG + [
        "-filter_complex_threads", "2", *inputs,
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", *meta_args,
        "-t", str(seconds), "-c:v", "libx264", "-threads", "4", "-preset", "veryfast",
        "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ], check=True)
    _check_duration(output, seconds)


def render_short(audio, poster, output, start, seconds, audio_filter="anull", draft=False, scenes=()):
    """Vertical teaser cut from the loudest stretch of the arranged mix.

    scenes: up to three (path, source_start, length) shown back to back in the picture panel.
    """
    width, height, fps = (540, 960, 12) if draft else (1080, 1920, 30)
    ratio = width / 1080
    panel_w, panel_h = 2 * round(1000 * ratio / 2), 2 * round(562 * ratio / 2)
    extra_inputs, scene_graph, at = [], [], 1.5
    layer = "poster"
    for index, (path, offset, length) in enumerate(scenes[:3]):
        length = min(length, seconds - at - 2)
        if length < 2:
            break
        extra_inputs += ["-ss", f"{offset:.3f}", "-t", f"{length:.3f}", "-i", str(path)]
        scene_graph += [
            f"[{2 + index}:v]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={panel_w}:{panel_h}:force_original_aspect_ratio=increase,crop={panel_w}:{panel_h},"
            f"setsar=1,format=yuva420p,fade=t=in:st=0:d=0.4:alpha=1,"
            f"fade=t=out:st={length - 0.45:.3f}:d=0.4:alpha=1,setpts=PTS+{at}/TB[shot{index}]",
            f"[{layer}][shot{index}]overlay=x={round(40 * ratio)}:y={round(640 * ratio)}:eof_action=pass:"
            f"enable='between(t,{at},{at + length})'[panel{index}]",
        ]
        layer = f"panel{index}"
        at += length
    wave_w, wave_h = round(width * 0.86), round(height * 0.085)
    graph = [
        f"[0:v]scale={width}:{height},format=yuv420p,loop=-1:1:0,setpts=N[poster]",
        f"[1:a]{audio_filter},{STEREO},asplit[waveaudio][mix]",
        f"[waveaudio]aformat=channel_layouts=mono,showwaves=s={wave_w}x{wave_h}:r={fps}:"
        f"mode=cline:scale=sqrt:draw=full:colors={WAVE_HEX},format=rgba[wave]",
        *scene_graph,
        f"[{layer}][wave]overlay=x=(W-w)/2:y=H*0.72:eof_action=pass,"
        f"fade=t=in:st=0:d=0.6,fade=t=out:st={seconds - 1.2}:d=1.2,setsar=1[v]",
        f"[mix]afade=t=in:st=0:d=1.5,afade=t=out:st={seconds - 2.5}:d=2.5[a]",
    ]
    subprocess.run(FFMPEG + [
        "-framerate", str(fps), "-i", str(poster),
        "-ss", f"{start:.3f}", "-t", str(seconds),
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(audio), *extra_inputs,
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", "-t", str(seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ], check=True)
    _check_duration(output, seconds)
