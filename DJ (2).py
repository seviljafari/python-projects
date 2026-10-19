import argparse
import array
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

SR = 44100
WIDTH, HEIGHT, FPS = 720, 1280, 30


def midi_hz(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def render_audio(bpm, bars, wav_path, sub_scope_path=None, ninja_scope_path=None):
    seconds = bars * 4 * 60.0 / bpm
    samples = int(seconds * SR)
    mix = array.array("f", [0.0]) * samples
    pad_mix = array.array("f", [0.0]) * samples
    arp_mix = array.array("f", [0.0]) * samples
    sub_mix = array.array("f", [0.0]) * samples
    ninja_mix = array.array("f", [0.0]) * samples
    rng = random.Random(17)

    def add(start, length, voice, target=mix):
        first = int(start * SR)
        count = min(int(length * SR), samples - first)
        for j in range(max(0, count)):
            target[first + j] += voice(j / SR)

    beat = 60.0 / bpm

    
    for b in range(bars * 4):
        t = b * beat

        def kick(x):
            phase = 2 * math.pi * (46 * x + (135 / 32) * (1 - math.exp(-32 * x)))
            return 0.88 * math.sin(phase) * math.exp(-13 * x)

        add(t, 0.25, kick)

        if b % 4 in (1, 3):
            seed = rng.randrange(1_000_000)
            local = random.Random(seed)

            def snare(x, local=local):
                noise = local.uniform(-1, 1)
                return (0.20 * noise + 0.13 * math.sin(2 * math.pi * 185 * x)) * math.exp(-22 * x)

            add(t, 0.19, snare)

        bass_notes = [36, 36, 43, 34, 36, 36, 46, 34]
        freq = midi_hz(bass_notes[b % len(bass_notes)])

        def bass(x, freq=freq):
            env = min(1.0, x * 90) * math.exp(-4.7 * x)
            return 0.30 * (math.sin(2 * math.pi * freq * x) +
                           0.22 * math.sin(4 * math.pi * freq * x)) * env

        add(t, 0.34, bass)

        if b % 2 == 0:
            melody = [60, 63, 67, 70, 67, 63, 58, 63]
            lead_freq = midi_hz(melody[(b // 2) % len(melody)])

            def lead(x, freq=lead_freq):
                env = min(1.0, x * 100) * math.exp(-5.5 * x)
                return 0.13 * (math.sin(2 * math.pi * freq * x) +
                               0.25 * math.sin(4 * math.pi * freq * x)) * env

            add(t, 0.38, lead)

    for step in range(bars * 8):
        t = step * beat / 2
        seed = rng.randrange(1_000_000)
        local = random.Random(seed)
        volume = 0.12 if step % 2 == 0 else 0.19

        def hat(x, local=local, volume=volume):
            return volume * local.uniform(-1, 1) * math.exp(-72 * x)

        add(t, 0.055, hat)

    
    chords = [
        [48, 51, 55],  
        [44, 48, 51],  
        [46, 50, 53],  
        [43, 46, 50],  
    ]
    bar_length = 4 * beat
    for bar in range(bars):
        start = bar * bar_length
        length = min(bar_length, seconds - start)
        for note in chords[bar % len(chords)]:
            freq = midi_hz(note)

            def pad(x, freq=freq, length=length):
                attack = min(1.0, x / 0.18)
                release = min(1.0, max(0.0, (length - x) / 0.35))
                envelope = attack * release
                fundamental = math.sin(2 * math.pi * freq * x)
                detuned = math.sin(2 * math.pi * freq * 1.004 * x)
                octave = math.sin(4 * math.pi * freq * x)
                return 0.10 * envelope * (0.56 * fundamental + 0.32 * detuned + 0.12 * octave)

            add(start, length, pad, pad_mix)

    
    arp_notes = [72, 75, 79, 82, 79, 75, 70, 75, 72, 75, 79, 84, 82, 79, 75, 70]
    for step in range(bars * 8):
        start = step * beat / 2
        note = arp_notes[step % len(arp_notes)]
        freq = midi_hz(note)
        accent = 1.0 if step % 4 == 0 else 0.72

        def pluck(x, freq=freq, accent=accent):
            envelope = (1 - math.exp(-125 * x)) * math.exp(-10.5 * x)
            tone = (
                0.52 * math.sin(2 * math.pi * freq * x)
                + 0.27 * math.sin(2 * math.pi * freq * 2 * x)
                + 0.13 * math.sin(2 * math.pi * freq * 3 * x)
                + 0.20 * math.sin(2 * math.pi * freq * 1.006 * x)
            )
            return 0.115 * accent * envelope * tone

        add(start, 0.20, pluck, arp_mix)
        add(start + 0.15, 0.16,
            lambda x, f=freq, a=accent: pluck(x, freq=f, accent=a * 0.28),
            arp_mix)
        add(start + 0.30, 0.13,
            lambda x, f=freq, a=accent: pluck(x, freq=f, accent=a * 0.12),
            arp_mix)

    
    sub_steps = [0, 5, 8, 11, 14]
    sub_notes = [36, 36, 43, 34, 36, 31, 34, 29]
    for bar in range(bars):
        for hit, step in enumerate(sub_steps):
            start = bar * bar_length + step * beat / 4
            freq = midi_hz(sub_notes[(bar + hit) % len(sub_notes)])
            accent = 1.0 if step in (0, 8) else 0.72

            def thump(x, freq=freq, accent=accent):
                phase = 2 * math.pi * (
                    freq * x + (0.70 * freq / 31) * (1 - math.exp(-31 * x))
                )
                envelope = min(1.0, x * 90) * math.exp(-6.4 * x)
                fundamental = math.sin(phase)
                harmonics = 0.18 * math.sin(2 * phase)
                return 0.42 * accent * envelope * (fundamental + harmonics)

            add(start, 0.40, thump, sub_mix)

    
    ninja_steps = [2, 7, 13]
    ninja_notes = [82, 79, 75, 70, 79, 75, 67, 72]
    for bar in range(bars):
        for hit, step in enumerate(ninja_steps):
            start = bar * bar_length + step * beat / 4
            freq = midi_hz(ninja_notes[(bar * 2 + hit) % len(ninja_notes)])
            accent = 1.0 if step == 7 else 0.78
            local = random.Random(900 + bar * 11 + hit)

            def shadow_bell(x, freq=freq, accent=accent, local=local):
                bend = 0.075 * freq / 34
                phase = 2 * math.pi * (freq * x + bend * (1 - math.exp(-34 * x)))
                envelope = (1 - math.exp(-95 * x)) * math.exp(-4.3 * x)
                metal = (
                    0.50 * math.sin(phase)
                    + 0.28 * math.sin(phase * 2.71)
                    + 0.17 * math.sin(phase * 4.13)
                )
                air = 0.025 * local.uniform(-1, 1) * math.exp(-85 * x)
                return accent * envelope * (0.15 * metal + air)

            add(start, 0.62, shadow_bell, ninja_mix)
            add(start + 0.24, 0.48,
                lambda x, voice=shadow_bell: voice(x) * 0.30,
                ninja_mix)
            add(start + 0.50, 0.34,
                lambda x, voice=shadow_bell: voice(x) * 0.14,
                ninja_mix)

    peak = max(abs(x) for x in mix) or 1.0
    pad_peak = max(abs(x) for x in pad_mix) or 1.0
    arp_peak = max(abs(x) for x in arp_mix) or 1.0
    sub_peak = max(abs(x) for x in sub_mix) or 1.0
    ninja_peak = max(abs(x) for x in ninja_mix) or 1.0
    pad_gain = 3600 / pad_peak
    arp_gain = 4400 / arp_peak
    sub_gain = 4800 / sub_peak
    ninja_gain = 3400 / ninja_peak
    combined = array.array("i")
    final_peak = 1
    for base, pad_value, arp_value, sub_value, ninja_value in zip(
        mix, pad_mix, arp_mix, sub_mix, ninja_mix
    ):
        original_track = int(math.tanh((base / peak) * 1.35) * 30000)
        layered = (
            original_track
            + int(pad_value * pad_gain)
            + int(arp_value * arp_gain)
            + int(sub_value * sub_gain)
            + int(ninja_value * ninja_gain)
        )
        combined.append(layered)
        final_peak = max(final_peak, abs(layered))

    output_gain = min(1.0, 30000 / final_peak)
    pcm = array.array("h")
    for value in combined:
        pcm.append(max(-32768, min(32767, int(value * output_gain))))
    if sys.byteorder != "little":
        pcm.byteswap()

    write_mono_wav(wav_path, pcm)
    if sub_scope_path:
        scope_gain = 22000 / sub_peak
        scope_pcm = array.array(
            "h",
            (max(-30000, min(30000, int(value * scope_gain))) for value in sub_mix),
        )
        if sys.byteorder != "little":
            scope_pcm.byteswap()
        write_mono_wav(sub_scope_path, scope_pcm)
    if ninja_scope_path:
        ninja_scope_gain = 22000 / ninja_peak
        ninja_scope_pcm = array.array(
            "h",
            (max(-30000, min(30000, int(value * ninja_scope_gain))) for value in ninja_mix),
        )
        if sys.byteorder != "little":
            ninja_scope_pcm.byteswap()
        write_mono_wav(ninja_scope_path, ninja_scope_pcm)
    return seconds


def write_mono_wav(path, pcm):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(pcm.tobytes())


def escape_filter_path(path):
    return str(path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def make_video(wav_path, sub_scope_path, ninja_scope_path, output, seconds, bpm, camera=None):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg not found. Install FFmpeg, then run this script again.")

    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    font = next((p for p in fonts if Path(p).exists()), None)
    font_arg = f"fontfile='{escape_filter_path(font)}'" if font else "font=monospace"

    with tempfile.TemporaryDirectory(prefix="python_reel_") as temp:
        temp = Path(temp)
        title_file = temp / "title.txt"
        placeholder_file = temp / "placeholder.txt"
        code_file = temp / "code.txt"
        title_file.write_text(f"PYTHON LIVE SET   /   {bpm} BPM   /   5 TRACKS", encoding="utf-8")
        placeholder_file.write_text("CAMERA CLIP\nVIDEO GOES HERE", encoding="utf-8")
        code_file.write_text(
            "# five tracks  |  8 bars\n"
            f"bpm = {bpm}\n"
            'key = "C minor"\n'
            'kick  = "X---" * 4\n'
            'snare = "----X---" * 2\n'
            'hats  = "x-x-" * 4\n'
            'bass  = ["C2", "C2", "G2", "Bb1"] * 2\n'
            'pad = ["Cm", "Ab", "Bb", "Gm"]\n'
            "groove = play(kick, snare, hats, bass)\n"
            'arp = ["C5", "Eb5", "G5", "Bb5"]\n'
            'sub = "808 DOOP / sync"\n'
            'ninja = "shadow bell / echo"\n'
            "music = groove + pad + arp + sub + ninja",
            encoding="utf-8",
        )

        def text_filter(file, x, y, size, color, spacing=17):
            path = escape_filter_path(file)
            return (f"drawtext={font_arg}:textfile='{path}':"
                    f"x={x}:y={y}:fontsize={size}:fontcolor={color}:"
                    f"line_spacing={spacing}")

        graph = [
            f"color=c=0x080910:s={WIDTH}x{HEIGHT}:r={FPS}:d={seconds:.3f}[bg]",
            ("[bg]drawbox=x=0:y=0:w=720:h=420:color=0x8D536C:t=fill,"
             "drawbox=x=0:y=420:w=720:h=510:color=0x11121A:t=fill,"
             "drawbox=x=0:y=930:w=720:h=350:color=0x090A10:t=fill,"
             "drawbox=x=0:y=418:w=720:h=2:color=0xF47AB8@0.8:t=fill,"
             "drawbox=x=0:y=928:w=720:h=2:color=0xF47AB8@0.8:t=fill[p0]"),
        ]
        if camera:
            graph.extend([
                "[3:v]scale=720:420:force_original_aspect_ratio=increase,"
                "crop=720:420,setsar=1,fps=30[cam]",
                "[p0][cam]overlay=0:0[panel]",
            ])
        else:
            graph.extend([
                f"[p0]{text_filter(placeholder_file, 210, 158, 30, 'white')}[panel]",
            ])

        graph.extend([
            f"[panel]{text_filter(title_file, 28, 28, 20, 'white')},"
            "drawbox=x=20:y=450:w=4:h=426:color=0xF47AB8:t=fill[headed]",
            f"[headed]{text_filter(code_file, 42, 456, 18, '0xE8E9F2', spacing=8)}[coded]",
            "[0:a]asplit=2[program][scope]",
            f"[scope]showwaves=s=680x66:mode=cline:colors=0xff69c7:"
            f"r={FPS}:scale=cbrt:draw=full,format=rgba[waves]",
            "[coded][waves]overlay=20:979:format=auto[withwave]",
            f"[1:a]showwaves=s=680x66:mode=cline:colors=0x35efd8:r={FPS}:"
            "scale=cbrt:draw=full,format=rgba[subwaves]",
            "[withwave][subwaves]overlay=20:1078:format=auto[withsub]",
            f"[2:a]showwaves=s=680x66:mode=cline:colors=0xb77cff:r={FPS}:"
            "scale=cbrt:draw=full,format=rgba[ninjawaves]",
            "[withsub][ninjawaves]overlay=20:1177:format=auto[withninja]",
            "[withninja]drawtext=" + font_arg +
            ":text='MIX / MASTER':x=24:y=950:fontsize=17:fontcolor=0xff8bd0[mixlabel]",
            "[mixlabel]drawtext=" + font_arg +
            ":text='808 SUB':x=24:y=1050:fontsize=17:fontcolor=0x35efd8[sublabel]",
            "[sublabel]drawtext=" + font_arg +
            ":text='NINJA SHADOW':x=24:y=1149:fontsize=17:fontcolor=0xb77cff[video]",
            "[program]alimiter=limit=0.95,aresample=44100[audio]",
        ])

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path)]
        cmd += ["-i", str(sub_scope_path)]
        cmd += ["-i", str(ninja_scope_path)]
        if camera:
            cmd += ["-stream_loop", "-1", "-i", str(camera)]
        cmd += [
            "-filter_complex", ";".join(graph),
            "-map", "[video]", "-map", "[audio]",
            "-t", f"{seconds:.3f}", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(output),
        ]
        subprocess.run(cmd, check=True)


def run_live_window(wav_path, sub_scope_path, ninja_scope_path, seconds, bpm, camera=None):
    
    try:
        import pygame
    except ImportError:
        raise SystemExit("Install pygame first: python -m pip install pygame")

    cv2 = None
    capture = None
    if camera is not None:
        try:
            import cv2 as opencv
            cv2 = opencv
        except ImportError:
            raise SystemExit("For a live camera panel, install OpenCV: python -m pip install opencv-python")
        source = int(camera) if camera.isdigit() else camera
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise SystemExit(f"Could not open camera or video source: {camera}")

    with wave.open(str(wav_path), "rb") as audio_file:
        pcm = array.array("h")
        pcm.frombytes(audio_file.readframes(audio_file.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
    with wave.open(str(sub_scope_path), "rb") as audio_file:
        sub_pcm = array.array("h")
        sub_pcm.frombytes(audio_file.readframes(audio_file.getnframes()))
        if sys.byteorder != "little":
            sub_pcm.byteswap()
    with wave.open(str(ninja_scope_path), "rb") as audio_file:
        ninja_pcm = array.array("h")
        ninja_pcm.frombytes(audio_file.readframes(audio_file.getnframes()))
        if sys.byteorder != "little":
            ninja_pcm.byteswap()

    pygame.init()
    pygame.mixer.init(frequency=SR, size=-16, channels=1)
    pygame.mixer.music.load(str(wav_path))
    pygame.mixer.music.play(loops=-1)

    info = pygame.display.Info()
    screen_height = info.current_h if info.current_h > 0 else 900
    window_h = min(1000, max(600, screen_height - 80))
    window_w = round(window_h * WIDTH / HEIGHT)
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Python Live Groove — C minor")

    canvas = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    mono = pygame.font.SysFont("dejavusansmono", 22)
    small = pygame.font.SysFont("dejavusansmono", 18)
    heading = pygame.font.SysFont("dejavusansmono", 25, bold=True)
    white = (237, 235, 244)
    pink = (246, 105, 190)
    pale_pink = (255, 174, 221)
    code_lines = [
        "# five tracks  |  8 bars",
        f"bpm = {bpm}",
        'key = "C minor"',
        'kick  = "X---" * 4',
        'snare = "----X---" * 2',
        'hats  = "x-x-" * 4',
        'bass  = ["C2", "C2", "G2", "Bb1"] * 2',
        'pad = ["Cm", "Ab", "Bb", "Gm"]',
        'arp = ["C5", "Eb5", "G5", "Bb5"]',
        'sub = "808 DOOP / sync"',
        'ninja = "shadow bell / echo"',
        "groove = play(kick, snare, hats, bass)",
        "music = groove + pad + arp + sub + ninja",
    ]

    
    top_gradient = pygame.Surface((WIDTH, 420))
    for y in range(420):
        mix = y / 420
        color = (
            int(100 + 54 * mix),
            int(52 + 24 * mix),
            int(82 + 30 * mix),
        )
        pygame.draw.line(top_gradient, color, (0, y), (WIDTH, y))

    started = time.perf_counter()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            elapsed = time.perf_counter() - started
            loop_time = elapsed % seconds
            beat_position = loop_time * bpm / 60.0
            beat_phase = beat_position % 1.0

            
            if capture is not None:
                ok, frame = capture.read()
                if not ok and isinstance(camera, str) and not camera.isdigit():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if ok:
                    h, w = frame.shape[:2]
                    target_ratio = WIDTH / 420
                    if w / h > target_ratio:
                        crop_w = int(h * target_ratio)
                        left = (w - crop_w) // 2
                        frame = frame[:, left:left + crop_w]
                    else:
                        crop_h = int(w / target_ratio)
                        top = (h - crop_h) // 2
                        frame = frame[top:top + crop_h, :]
                    frame = cv2.resize(frame, (WIDTH, 420))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_surface = pygame.image.frombuffer(frame.tobytes(), (WIDTH, 420), "RGB")
                    canvas.blit(camera_surface, (0, 0))
                else:
                    canvas.blit(top_gradient, (0, 0))
                    no_feed = heading.render("CAMERA SIGNAL LOST", True, white)
                    canvas.blit(no_feed, no_feed.get_rect(center=(WIDTH // 2, 210)))
            else:
                canvas.blit(top_gradient, (0, 0))
                pulse = int(23 + 8 * (1 + math.sin(2 * math.pi * beat_phase)) / 2)
                pygame.draw.circle(canvas, (255, 183, 218), (WIDTH // 2, 220), pulse, 2)
                pygame.draw.circle(canvas, (255, 183, 218), (WIDTH // 2, 220), 4)
                for i in range(18):
                    height = 8 + int(48 * abs(math.sin(beat_position * 0.6 + i * 0.7)))
                    x = 95 + i * 31
                    pygame.draw.rect(canvas, (231, 157, 190), (x, 345 - height, 12, height), border_radius=3)
                hint = small.render("CAMERA PANEL  ·  use --camera 0 for webcam", True, white)
                canvas.blit(hint, hint.get_rect(center=(WIDTH // 2, 384)))

            title = heading.render(f"LIVE PYTHON SET   /   {bpm} BPM   /   5 TRACKS", True, white)
            canvas.blit(title, (25, 24))
            pygame.draw.circle(canvas, (255, 98, 143), (680, 39), 8 + int(4 * (1 - beat_phase)))

            
            pygame.draw.rect(canvas, (16, 17, 25), (0, 420, WIDTH, 510))
            pygame.draw.rect(canvas, pink, (0, 418, WIDTH, 2))
            pygame.draw.rect(canvas, pink, (18, 450, 4, 426))
            active_line = 1 + int(beat_position) % (len(code_lines) - 1)
            for index, line in enumerate(code_lines):
                y = 458 + index * 29
                if index == active_line:
                    pygame.draw.rect(canvas, (37, 27, 42), (31, y - 2, 666, 27), border_radius=3)
                color = (194, 151, 210) if line.startswith("#") else (pale_pink if index == active_line else white)
                rendered = mono.render(f"{index + 1:>2}  {line}", True, color)
                canvas.blit(rendered, (36, y))

            
            pygame.draw.line(canvas, (57, 57, 71), (40, 850), (680, 850), 1)
            note_steps = [0, 2, 4, 7, 8, 10, 12, 15]
            for i, step in enumerate(note_steps):
                x = 48 + step * 38
                y = 875 - (i % 4) * 13
                pygame.draw.rect(canvas, (172, 170, 190), (x, y, 27, 9), border_radius=2)
            playhead_x = 48 + (int(beat_position * 4) % 16) * 38
            pygame.draw.line(canvas, pink, (playhead_x, 860), (playhead_x, 920), 2)

            
            pygame.draw.rect(canvas, (8, 9, 15), (0, 930, WIDTH, 350))
            pygame.draw.line(canvas, pink, (0, 928), (WIDTH, 928), 2)
            mix_label = small.render("MIX / MASTER", True, pale_pink)
            sub_label = small.render("808 SUB", True, (53, 239, 216))
            ninja_label = small.render("NINJA SHADOW", True, (183, 124, 255))
            canvas.blit(mix_label, (25, 948))
            canvas.blit(sub_label, (25, 1042))
            canvas.blit(ninja_label, (25, 1136))
            mix_center = 1007
            sub_center = 1107
            ninja_center = 1195
            pygame.draw.line(canvas, (48, 48, 61), (24, mix_center), (696, mix_center), 1)
            pygame.draw.line(canvas, (31, 58, 58), (24, sub_center), (696, sub_center), 1)
            pygame.draw.line(canvas, (57, 43, 73), (24, ninja_center), (696, ninja_center), 1)
            point_count = 600
            window_samples = int(SR * 0.18)
            end_sample = int(loop_time * SR) % len(pcm)
            start_sample = end_sample - window_samples
            points = []
            sub_points = []
            ninja_points = []
            for i in range(point_count):
                sample_index = (start_sample + int(i * window_samples / point_count)) % len(pcm)
                amplitude = pcm[sample_index] / 32768.0
                sub_amplitude = sub_pcm[sample_index] / 32768.0
                ninja_amplitude = ninja_pcm[sample_index] / 32768.0
                x = 24 + int(i * 672 / (point_count - 1))
                y = mix_center - int(amplitude * 32)
                sub_y = sub_center - int(sub_amplitude * 32)
                ninja_y = ninja_center - int(ninja_amplitude * 32)
                points.append((x, y))
                sub_points.append((x, sub_y))
                ninja_points.append((x, ninja_y))
            if len(points) > 1:
                pygame.draw.lines(canvas, pink, False, points, 2)
                pygame.draw.lines(canvas, (53, 239, 216), False, sub_points, 2)
                pygame.draw.lines(canvas, (183, 124, 255), False, ninja_points, 2)
            footer = small.render(f"BAR {int(beat_position // 4) % 8 + 1:02d} / 08     ESC TO QUIT", True, (147, 145, 160))
            canvas.blit(footer, (25, 1220))

            current_size = screen.get_size()
            scaled = pygame.transform.smoothscale(canvas, current_size)
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.mixer.music.stop()
        if capture is not None:
            capture.release()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Make a Python-generated music Reel.")
    parser.add_argument("--output", default="python_live_groove.mp4")
    parser.add_argument("--camera")
    parser.add_argument("--bpm", type=int, default=140)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--render-video", action="store_true")
    args = parser.parse_args()
    if not 60 <= args.bpm <= 200 or args.bars < 1:
        raise SystemExit("Choose BPM from 60–200 and at least one bar.")

    if args.render_video and args.camera and args.camera.isdigit():
        raise SystemExit("For --render-video, --camera must be a video file path, not a webcam index.")

    with tempfile.TemporaryDirectory(prefix="python_reel_audio_") as temp:
        wav_path = Path(temp) / "groove.wav"
        sub_scope_path = Path(temp) / "sub_scope.wav"
        ninja_scope_path = Path(temp) / "ninja_scope.wav"
        seconds = render_audio(args.bpm, args.bars, wav_path, sub_scope_path, ninja_scope_path)
        if args.render_video:
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            make_video(wav_path, sub_scope_path, ninja_scope_path, output, seconds, args.bpm, args.camera)
            print(f"Done: {output}")
            print(f"Length: {seconds:.1f}s | Size: 720x1280 vertical | {args.bpm} BPM")
        else:
            run_live_window(wav_path, sub_scope_path, ninja_scope_path, seconds, args.bpm, args.camera)


if __name__ == "__main__":
    main()
