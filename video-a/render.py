#!/usr/bin/env python3
"""Render Video A ("I Let 3 AI Agents Run Unsupervised For 30 Days") as a
timed animatic: every shot is drawn frame by frame with Pillow, piped into
ffmpeg as its own clip, and the clips are then merged with ffmpeg's concat
demuxer. Shot lengths come from the voiceover itself (words at READ_WPM), so
the cut lines up with a real read and the .srt doubles as a teleprompter.

    python3 render.py            # all shots -> build/video-a.mp4 + .srt
    python3 render.py --only b1  # re-render a single shot, then re-merge
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BUILD = ROOT / "build"

W, H, FPS = 1920, 1080, 30
READ_WPM = 190          # conversational read; the script targets 190-200
SENTENCE_GAP = 0.35     # breath between sentences, seconds
VO_TAIL = 0.45          # air after the last word of a shot

# Palette lifted from the scripts page so graphics match the rundown.
GROUND = (19, 23, 28)
PANEL = (25, 30, 37)
INK = (232, 230, 224)
DIM = (152, 162, 171)
FAINT = (107, 117, 126)
RULE = (40, 48, 58)
SCREEN = (127, 178, 192)
GRAPHIC = (217, 164, 65)
CUTAWAY = (155, 180, 141)
GHOST = (64, 74, 86)     # unstruck tally mark


def font(name, size):
    return ImageFont.truetype(str(ASSETS / "fonts" / f"{name}.ttf"), size)


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


# ---------------------------------------------------------------- easing

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, start, dur):
    """0 before start, eased 0..1 across dur, 1 after."""
    return ease_out((t - start) / dur) if dur > 0 else float(t >= start)


def mix(a, b, k):
    return tuple(round(a[i] + (b[i] - a[i]) * k) for i in range(3))


# ---------------------------------------------------------------- timing

def words(text):
    return len(text.split())


def sentence_dur(text):
    return words(text) / READ_WPM * 60 + SENTENCE_GAP


class Shot:
    """A run of voiceover lines over one visual. Timing comes from the
    ElevenLabs character alignment in build/vo/<key>.json when it exists,
    otherwise from a READ_WPM estimate."""

    def __init__(self, key, lines, draw, hold=0.0, tally=None):
        self.key = key
        self.lines = lines          # voiceover sentences covered by this shot
        self.draw = draw            # draw(img, t, shot) -> None
        self.hold = hold            # silent tail, seconds
        self.tally = tally          # None, or number of struck failures
        self.offset = 0.0           # start of this shot in the whole cut
        self.starts, self.ends = [], []
        vo = BUILD / "vo" / f"{key}.json"
        if vo.exists():
            al = json.loads(vo.read_text())
            begin, stop = al["character_start_times_seconds"], al["character_end_times_seconds"]
            pos = 0
            for line in lines:
                i = al["text"].index(line, pos)
                self.starts.append(begin[i])
                self.ends.append(stop[i + len(line) - 1])
                pos = i + len(line)
            self.duration = self.ends[-1] + VO_TAIL + hold
        else:
            acc = 0.0
            for line in lines:
                self.starts.append(acc)
                self.ends.append(acc + sentence_dur(line) - SENTENCE_GAP)
                acc += sentence_dur(line)
            self.duration = acc + hold

    def line_start(self, i):
        return self.starts[i]

    def word_time(self, i, frac):
        """Time at which fraction `frac` of sentence i has been read."""
        return self.starts[i] + frac * (self.ends[i] - self.starts[i])


# ---------------------------------------------------------------- shared drawing

_presenter = None


def presenter():
    global _presenter
    if _presenter is None:
        _presenter = Image.open(ASSETS / "img" / "presenter.webp").convert("RGB")
    return _presenter


HEDRA = BUILD / "hedra" / "presenter.mp4"


class FaceReader:
    """Streams the Hedra lip-sync frames for one shot, in step with render."""

    def __init__(self, ff, shot):
        self.proc = subprocess.Popen(
            [ff, "-loglevel", "error", "-ss", f"{shot.offset:.3f}", "-i", str(HEDRA),
             "-t", f"{shot.duration:.3f}", "-vf", f"scale={W}:{H},fps={FPS}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)  # closing early at the shot cut breaks its pipe
        self.last = None

    def next(self):
        buf = self.proc.stdout.read(W * H * 3)
        if len(buf) == W * H * 3:
            self.last = Image.frombytes("RGB", (W, H), buf)
        return self.last            # past the end of the take: hold last frame

    def close(self):
        self.proc.stdout.close()
        self.proc.wait()


def face(shot):
    """Current talking-head frame, or the still when there's no Hedra take."""
    return getattr(shot, "face", None) or presenter()


def draw_pip(img, shot):
    """Presenter keeps talking in the corner while a graphic holds the frame."""
    if getattr(shot, "face", None) is None:
        return
    pw, ph = 400, 225
    x, y = W - pw - 56, H - ph - 56
    src = shot.face
    sw, sh = src.size
    # frame the head and shoulders rather than the whole room
    box = (sw * 0.30, sh * 0.02, sw * 0.72, sh * 0.02 + sw * 0.42 * ph / pw)
    tile = src.resize((pw, ph), Image.BICUBIC, box=box)
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), 18, fill=255)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x - 4, y - 4, x + pw + 3, y + ph + 3), 21, fill=RULE)
    img.paste(tile, (x, y), mask)


def draw_presenter(img, t, shot, focus=(1010, 330), z0=1.0, z1=1.07):
    """Slow push-in towards the face."""
    src = face(shot)
    sw, sh = src.size
    focus = (focus[0] * sw / 2000, focus[1] * sh / 1116)
    k = ease_in_out(t / shot.duration)
    zoom = z0 + (z1 - z0) * k
    ch = sh / zoom
    cw = ch * W / H
    cx = sw / 2 + (focus[0] - sw / 2) * (zoom - 1) / (z1 - 1 + 1e-9) * 0.9
    cy = sh / 2 + (focus[1] - sh / 2) * (zoom - 1) / (z1 - 1 + 1e-9) * 0.9
    x0 = clamp(cx - cw / 2, 0, sw - cw)
    y0 = clamp(cy - ch / 2, 0, sh - ch)
    frame = src.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    img.paste(frame)


def text_center(d, xy, text, fnt, fill, anchor="mm"):
    d.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def draw_tally(img, t, struck, strike_at=None):
    """Persistent failure counter, top right. With strike_at, the five marks
    strike on in amber from that time and then settle to the ghost counter."""
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, gap, bw, bh = W - 330, 64, 30, 9, 58
    settle = 1.0
    if strike_at is not None:
        settle = ramp(t, strike_at + 1.9, 0.8)
    plate = int(170 * (1 if strike_at is None else ramp(t, strike_at, 0.3)))
    d.rounded_rectangle((x0 - 28, y0 - 22, W - 52, y0 + bh + 22), 14,
                        fill=GROUND + (plate,))
    for i in range(5):
        x = x0 + i * gap
        if strike_at is not None:
            grow = ramp(t, strike_at + i * 0.22, 0.28)
            col = mix(GRAPHIC, GHOST, settle) if i >= struck else GRAPHIC
        else:
            grow = 1.0
            col = GRAPHIC if i < struck else GHOST
        if grow > 0:
            d.rectangle((x, y0, x + bw, y0 + bh * grow), fill=col)
    label_a = 1.0 if strike_at is None else settle
    if label_a > 0:
        f = font("Archivo-Bold", 30)
        d.text((x0 + 5 * gap + 14, y0 + bh / 2), f"{struck}/5", font=f,
               fill=INK + (int(255 * label_a),), anchor="lm")


# ---------------------------------------------------------------- shots

def draw_counter(img, t, shot):
    """1 -> 30 over a dimmed dashboard still, then hold dry on 30."""
    d = ImageDraw.Draw(img)
    rng = random.Random(7)
    cols, rows, pad = 6, 4, 36
    cw = (W - pad * (cols + 1)) / cols
    chh = (H - pad * (rows + 1)) / rows
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (cw + pad)
            y = pad + r * (chh + pad)
            d.rounded_rectangle((x, y, x + cw, y + chh), 14, fill=PANEL)
            d.rectangle((x + 24, y + 28, x + 24 + cw * 0.45, y + 40), fill=RULE)
            for j in range(3):
                ww = cw * rng.uniform(0.3, 0.8)
                d.rectangle((x + 24, y + 70 + j * 30, x + 24 + ww, y + 80 + j * 30),
                            fill=(33, 39, 47))
            dot = [CUTAWAY, CUTAWAY, CUTAWAY, GRAPHIC][rng.randrange(4)]
            d.ellipse((x + cw - 44, y + 26, x + cw - 28, y + 42), fill=mix(dot, PANEL, 0.55))
    img.paste(img.filter(ImageFilter.GaussianBlur(3)))
    veil = Image.new("RGB", (W, H), GROUND)
    img.paste(Image.blend(img, veil, 0.55))

    d = ImageDraw.Draw(img)
    tick_end = shot.duration * 0.62
    day = 1 + int(29 * ease_in_out(t / tick_end)) if t < tick_end else 30
    text_center(d, (W / 2, H / 2 - 40), str(day), font("Archivo-Bold", 380), INK)
    text_center(d, (W / 2, H / 2 + 170), "DAYS UNSUPERVISED", font("Archivo-Medium", 40),
                GRAPHIC)
    bar = (day - 1) / 29
    d.rectangle((W / 2 - 300, H / 2 + 222, W / 2 + 300, H / 2 + 226), fill=RULE)
    d.rectangle((W / 2 - 300, H / 2 + 222, W / 2 - 300 + 600 * bar, H / 2 + 226), fill=GRAPHIC)


_run_rows = None


def run_rows():
    global _run_rows
    if _run_rows is None:
        rng = random.Random(30)
        rows = []
        agents = [("Scout", "research run", SCREEN), ("Ridge", "judgment run", GRAPHIC),
                  ("Atlas", "design run", CUTAWAY)]
        for day in range(1, 31):
            weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][(day - 1) % 7]
            for hour in range(7, 23, 2):
                rows.append((day, weekday, f"{hour:02d}:{rng.choice(['00', '30'])}",
                             *agents[0], f"{rng.randint(0, 5)} items", "completed"))
            if weekday == "Mon":
                for a in agents[1:]:
                    rows.append((day, weekday, "07:00", *a, "handoff", "completed"))
        _run_rows = rows
    return _run_rows


def draw_scrub(img, t, shot):
    """A month of thread history flying past: sell volume, not detail."""
    d = ImageDraw.Draw(img)
    rows = run_rows()
    rh, top = 66, 170
    visible = (H - top) // rh + 2
    total = len(rows) - visible
    k = ease_in_out(t / shot.duration)
    offset = k * total * rh
    first = int(offset // rh)
    fy = top - (offset % rh)
    f_day = font("Archivo-Bold", 26)
    f_row = font("Archivo-Medium", 26)
    f_meta = font("Archivo-Medium", 22)
    for i in range(visible):
        idx = first + i
        if idx >= len(rows):
            break
        day, wd, hh, name, kind, col, note, status = rows[idx]
        y = fy + i * rh
        d.line((120, y + rh - 1, W - 120, y + rh - 1), fill=RULE, width=1)
        d.text((120, y + rh / 2), f"Day {day:02d}", font=f_day, fill=FAINT, anchor="lm")
        d.text((270, y + rh / 2), f"{wd} {hh}", font=f_meta, fill=FAINT, anchor="lm")
        d.ellipse((430, y + rh / 2 - 7, 444, y + rh / 2 + 7), fill=col)
        d.text((466, y + rh / 2), name, font=f_row, fill=INK, anchor="lm")
        d.text((600, y + rh / 2), kind, font=f_row, fill=DIM, anchor="lm")
        d.text((W - 420, y + rh / 2), note, font=f_meta, fill=FAINT, anchor="lm")
        d.text((W - 120, y + rh / 2), status, font=f_meta, fill=CUTAWAY, anchor="rm")
    # speed smear: blur scales with scroll velocity
    speed = abs(ease_in_out((t + 0.02) / shot.duration) - k) * total * rh / 0.02
    if speed > 400:
        blurred = img.filter(ImageFilter.BoxBlur(min(6, speed / 1500)))
        img.paste(blurred)
        d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, top - 2), fill=GROUND)
    d.text((120, 96), "Thread history", font=font("Archivo-Bold", 44), fill=INK, anchor="lm")
    d.text((W - 120, 96), f"{len(rows)} runs · 30 days", font=f_meta, fill=DIM, anchor="rm")
    d.line((120, top - 2, W - 120, top - 2), fill=INK, width=2)


def draw_presenter_tally(img, t, shot):
    draw_presenter(img, t, shot)
    draw_tally(img, t, 0, strike_at=shot.word_time(0, 0.08))


def draw_presenter_plain(img, t, shot):
    draw_presenter(img, t, shot, z0=1.03, z1=1.09)


SPINE = [
    ("Scout", "RESEARCH", SCREEN, ["What launched in AI.", "Source + date on every item.",
                                   "Max five, then stop."]),
    ("Ridge", "JUDGMENT", GRAPHIC, ["Which items are worth", "a video, and which are",
                                    "already saturated."]),
    ("Atlas", "DESIGN", CUTAWAY, ["A page with a cover", "image per item, ready", "for an editor."]),
]


def draw_spine(img, t, shot):
    """Scout -> Ridge -> Atlas, each node landing on its own sentence."""
    d = ImageDraw.Draw(img)
    d.text((120, 110), "The team", font=font("Archivo-Bold", 44), fill=INK, anchor="lm")
    d.text((120, 160), "One job each. Deliberately no use at the other two.",
           font=font("Newsreader-Regular", 32), fill=DIM, anchor="lm")
    nw, nh, gap = 460, 380, 150
    x0 = (W - (3 * nw + 2 * gap)) / 2
    y0 = 300
    f_name = font("Archivo-Bold", 64)
    f_role = font("Archivo-Medium", 24)
    f_job = font("Newsreader-Regular", 32)
    for i, (name, role, col, job) in enumerate(SPINE):
        appear = shot.line_start(i + 1)
        a = ramp(t, appear, 0.6)
        if a <= 0:
            continue
        x = x0 + i * (nw + gap)
        y = y0 + (1 - a) * 40
        fade = lambda c: mix(GROUND, c, a)
        d.rounded_rectangle((x, y, x + nw, y + nh), 18, fill=fade(PANEL), outline=fade(RULE), width=2)
        d.rectangle((x, y + 18, x + 6, y + nh - 18), fill=fade(col))
        d.text((x + 44, y + 60), role, font=f_role, fill=fade(col), anchor="lm")
        d.text((x + 44, y + 125), name, font=f_name, fill=fade(INK), anchor="lm")
        for j, ln in enumerate(job):
            d.text((x + 44, y + 215 + j * 44), ln, font=f_job, fill=fade(DIM), anchor="lm")
        if i > 0:
            g = ramp(t, appear - 0.1, 0.5)
            ax0 = x - gap + 24
            ax1 = ax0 + (gap - 48) * g
            ay = y0 + nh / 2
            d.line((ax0, ay, ax1, ay), fill=FAINT, width=4)
            if g > 0.95:
                d.polygon([(ax1 + 4, ay), (ax1 - 16, ay - 12), (ax1 - 16, ay + 12)], fill=FAINT)


def draw_schedule(img, t, shot):
    """Schedule editor mock: interval on agent one, Monday cadence on two and three."""
    d = ImageDraw.Draw(img)
    px, py, pw, ph = 360, 140, W - 720, 620
    d.rounded_rectangle((px, py, px + pw, py + ph), 20, fill=PANEL, outline=RULE, width=2)
    d.text((px + 56, py + 70), "Schedules", font=font("Archivo-Bold", 44), fill=INK, anchor="lm")
    d.line((px + 56, py + 120, px + pw - 56, py + 120), fill=RULE, width=2)
    rows = [("Scout", SCREEN, "Every 30 minutes", "interval"),
            ("Ridge", GRAPHIC, "Mondays · 07:00", "weekly"),
            ("Atlas", CUTAWAY, "Mondays · 07:00", "weekly")]
    seg = shot.word_time(0, 0.62) / 3
    f_name = font("Archivo-Bold", 40)
    f_pill = font("Archivo-Medium", 32)
    f_meta = font("Archivo-Medium", 24)
    for i, (name, col, when, kind) in enumerate(rows):
        y = py + 200 + i * 130
        on = ramp(t, i * seg, 0.5)
        focus = 1.0 if i * seg <= t < (i + 1) * seg else 0.0
        if focus:
            d.rounded_rectangle((px + 30, y - 52, px + pw - 30, y + 52), 14, fill=(31, 37, 45))
        d.ellipse((px + 64, y - 9, px + 82, y + 9), fill=col)
        d.text((px + 110, y), name, font=f_name, fill=INK, anchor="lm")
        d.text((px + 330, y), kind, font=f_meta, fill=FAINT, anchor="lm")
        pill_w = 440
        pxr = px + pw - 70
        d.rounded_rectangle((pxr - pill_w, y - 34, pxr, y + 34), 34,
                            fill=mix(PANEL, mix(PANEL, col, 0.22), on),
                            outline=mix(RULE, col, on), width=2)
        d.text((pxr - pill_w / 2, y), when if on > 0.3 else "Off", font=f_pill,
               fill=mix(FAINT, INK, on), anchor="mm")
    deal = ramp(t, shot.word_time(0, 0.72), 0.6)
    if deal > 0:
        d.text((W / 2, py + ph + 110), "The deal: 4 weeks → 4 pages",
               font=font("Archivo-Bold", 52), fill=mix(GROUND, GRAPHIC, deal), anchor="mm")


def draw_scorecard(img, t, shot):
    """'4 pages expected' struck through, replaced by what actually arrived."""
    d = ImageDraw.Draw(img)
    f_big = font("Archivo-Bold", 104)
    lift = ramp(t, 1.0, 0.7)
    y = H / 2 - 40 - lift * 250
    col = mix(INK, FAINT, lift)
    text = "4 pages expected"
    d.text((W / 2, y), text, font=f_big, fill=col, anchor="mm")
    tw = d.textlength(text, font=f_big)
    s = ramp(t, 0.35, 0.5)
    if s > 0:
        d.line((W / 2 - tw / 2 - 20, y + 4, W / 2 - tw / 2 - 20 + (tw + 40) * s, y + 4),
               fill=GRAPHIC, width=10)
    items = [("2", "pages", CUTAWAY, 0.12), ("1", "empty run", FAINT, 0.30),
             ("1", "unusable page", GRAPHIC, 0.55)]
    f_num = font("Archivo-Bold", 150)
    f_lab = font("Archivo-Medium", 40)
    colw = 480
    x0 = W / 2 - colw
    for i, (n, lab, c, frac) in enumerate(items):
        a = ramp(t, shot.word_time(0, frac), 0.45)
        if a <= 0:
            continue
        x = x0 + i * colw
        yy = H / 2 + 60 + (1 - a) * 30
        d.text((x, yy), n, font=f_num, fill=mix(GROUND, c, a), anchor="mm")
        d.text((x, yy + 115), lab, font=f_lab, fill=mix(GROUND, INK, a), anchor="mm")
    b = ramp(t, shot.word_time(0, 0.8), 0.6)
    if b > 0:
        d.text((W / 2, H - 150), "…and a bill that didn't match any of it",
               font=font("Newsreader-Regular", 40), fill=mix(GROUND, DIM, b), anchor="mm")


# ---------------------------------------------------------------- the cut

SHOTS = [
    # 0:00 cold open
    Shot("a1", [
        "A month ago I built a three agent team, put it on a schedule, and then did the one thing that nobody who makes these videos ever actually does, which is nothing at all.",
        "No corrections. No restarts. I didn't open a single thread to see how it was getting on.",
    ], draw_counter, hold=0.8),
    Shot("a2", [
        "Thirty days later I sat down and read every run it had done, and the useful half of that is not the part that worked.",
    ], draw_scrub),
    Shot("a3", [
        "It's the five things that broke, because every one of them broke quietly, and three of them would have carried on breaking for another month if I hadn't gone looking.",
        "So this is what an agent team does to itself when nobody is watching, in the order it happened to me, and what I'd change in yours before you set it running.",
    ], draw_presenter_tally),
    # 0:30 the setup
    Shot("b1", [
        "The team is three agents and each one has exactly one job.",
        "The first one goes out every day and finds what launched in AI, with a source link and a date on every item and a hard cap of five items so it has a definition of done.",
        "The second one takes that list and works out which of it is worth making a video about and which of it is already saturated.",
        "The third one turns whatever survives into a page with a cover image on each item that I can send to an editor without writing a covering note.",
    ], draw_spine, tally=0),
    Shot("b2", [
        "Research, judgment, design. One each, and each one deliberately no use at the other two.",
    ], draw_presenter_plain, tally=0),
    Shot("b3", [
        "The first agent runs on an interval because noticing things is its whole job, the other two run on a Monday morning schedule because their work only starts once there's something new, and the deal I set up was four weeks, four pages, whether I'm at my desk or not.",
    ], draw_schedule, tally=0),
    Shot("b4", [
        "What I got was two pages, one empty run, one page I couldn't use, and a bill that didn't match any of it.",
    ], draw_scorecard, hold=3.0, tally=0),
]


# ---------------------------------------------------------------- output

def render_shot(shot, ff):
    out = BUILD / "shots" / f"{shot.key}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = round(shot.duration * FPS)
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    reader = FaceReader(ff, shot) if HEDRA.exists() else None
    for f in range(n):
        t = f / FPS
        shot.face = reader.next() if reader else None
        img = Image.new("RGB", (W, H), GROUND)
        shot.draw(img, t, shot)
        if shot.draw not in (draw_presenter_tally, draw_presenter_plain):
            draw_pip(img, shot)
        if shot.tally is not None:
            draw_tally(img, t, shot.tally)
        # short fade from/to black at the head and tail of the whole cut
        if shot is SHOTS[0] and t < 0.4:
            img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img, t / 0.4)
        if shot is SHOTS[-1] and shot.duration - t < 0.6:
            img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), img,
                              clamp((shot.duration - t) / 0.6))
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if reader:
        reader.close()
    if proc.wait() != 0:
        sys.exit(f"ffmpeg failed on shot {shot.key}")
    print(f"  {shot.key}: {shot.duration:5.1f}s  {n} frames")


def layout():
    clock = 0.0
    for shot in SHOTS:
        shot.offset = clock
        clock += shot.duration
    return clock


def build_voiceover(ff):
    """One continuous voice track: each shot's ElevenLabs read, padded to the
    shot's length. Hedra lip-syncs to this, so it must match the cut exactly."""
    out = BUILD / "vo" / "full.wav"
    missing = [s.key for s in SHOTS if not (BUILD / "vo" / f"{s.key}.mp3").exists()]
    if missing:
        print(f"  no voiceover for {', '.join(missing)}: run voice.py; merging silent")
        return None
    cmd = [ff, "-y", "-loglevel", "error"]
    for s in SHOTS:
        cmd += ["-i", str(BUILD / "vo" / f"{s.key}.mp3")]
    chains = "".join(f"[{i}:a]aresample=48000,apad,atrim=0:{s.duration:.3f}[a{i}];"
                     for i, s in enumerate(SHOTS))
    chains += "".join(f"[a{i}]" for i in range(len(SHOTS)))
    chains += f"concat=n={len(SHOTS)}:v=0:a=1[out]"
    subprocess.run(cmd + ["-filter_complex", chains, "-map", "[out]", "-ac", "1", str(out)],
                   check=True)
    return out


def merge(ff):
    listing = BUILD / "shots.txt"
    listing.write_text("".join(f"file 'shots/{s.key}.mp4'\n" for s in SHOTS))
    out = BUILD / "video-a.mp4"
    total = sum(s.duration for s in SHOTS)
    vo = build_voiceover(ff)
    audio = ["-i", str(vo)] if vo else ["-f", "lavfi", "-t", f"{total:.3f}",
                                        "-i", "anullsrc=r=48000:cl=stereo"]
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), *audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    return out


def srt_time(s):
    ms = round(s * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt():
    cues, clock = [], 0.0
    for shot in SHOTS:
        for i, line in enumerate(shot.lines):
            a = clock + shot.starts[i]
            b = clock + shot.ends[i]
            cues.append((a, b, line))
        clock += shot.duration
    out = BUILD / "video-a.srt"
    out.write_text("\n".join(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n"
                             for n, (a, b, txt) in enumerate(cues, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated shot keys to re-render")
    ap.add_argument("--voice-only", action="store_true",
                    help="just build build/vo/full.wav (for hedra.py)")
    args = ap.parse_args()
    ff = ffmpeg_exe()
    layout()
    if args.voice_only:
        print(f"voice track -> {build_voiceover(ff)}")
        return
    keys = set(args.only.split(",")) if args.only else {s.key for s in SHOTS}
    print("rendering shots")
    for shot in SHOTS:
        if shot.key in keys:
            render_shot(shot, ff)
    out = merge(ff)
    srt = write_srt()
    total = sum(s.duration for s in SHOTS)
    print(f"merged -> {out.relative_to(ROOT)} ({total:.1f}s), captions -> {srt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
