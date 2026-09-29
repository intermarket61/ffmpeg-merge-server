#!/usr/bin/env python3
"""Render Video A ("I Let 3 AI Agents Run Unsupervised For 30 Days").

Pipeline:
    python3 voice.py                 # ElevenLabs read per shot, with timestamps
    python3 render.py --voice-only   # one continuous voice track, build/vo/full.wav
    python3 hedra.py                 # lip-synced presenter, build/hedra/presenter.mp4
    python3 render.py                # motion graphics + merge -> build/video-a.mp4

The motion graphics live in scenes/index.html. Each shot is loaded into
headless Chromium, stepped frame by frame with the Hedra frame for that
moment handed in, screenshotted, and piped into ffmpeg as its own clip. The
clips are then merged with ffmpeg's concat demuxer under the voice track.

    python3 render.py --only b1,b2   # re-render some shots, then re-merge
"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BUILD = ROOT / "build"
SCENES = ROOT / "scenes" / "index.html"
HEDRA = BUILD / "hedra" / "presenter.mp4"

W, H, FPS = 1920, 1080, 30
READ_WPM = 190          # estimate used only before a shot has been voiced
SENTENCE_GAP = 0.35     # breath between sentences in that estimate
VO_TAIL = 0.45          # air after the last word of a shot
WORKERS = 3


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


def sentence_dur(text):
    return len(text.split()) / READ_WPM * 60 + SENTENCE_GAP


class Shot:
    """A run of voiceover lines over one scene. Timing comes from the
    ElevenLabs character alignment in build/vo/<key>.json when it exists."""

    def __init__(self, key, scene, lines, hold=0.0):
        self.key = key
        self.scene = scene          # scene name in scenes/index.html
        self.lines = lines          # voiceover sentences covered by this shot
        self.hold = hold            # silent tail, seconds
        self.offset = 0.0           # start of this shot in the whole cut
        self.starts, self.ends, self.words = [], [], []
        vo = BUILD / "vo" / f"{key}.json"
        if vo.exists():
            self._from_alignment(json.loads(vo.read_text()))
            self.duration = self.ends[-1] + VO_TAIL + hold
        else:
            acc = 0.0
            for line in lines:
                self.starts.append(acc)
                self.ends.append(acc + sentence_dur(line) - SENTENCE_GAP)
                acc += sentence_dur(line)
            self.duration = acc + hold

    def _from_alignment(self, al):
        text = al["text"]
        begin, stop = al["character_start_times_seconds"], al["character_end_times_seconds"]
        spans, pos = [], 0
        for line in self.lines:
            i = text.index(line, pos)
            spans.append((i, i + len(line)))
            self.starts.append(begin[i])
            self.ends.append(stop[i + len(line) - 1])
            pos = i + len(line)
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            line = next(n for n, (a, b) in enumerate(spans) if a <= i < b)
            self.words.append({"w": text[i:j], "t0": begin[i], "t1": stop[j - 1], "line": line})
            i = j

    def spec(self, first, last):
        return {"key": self.key, "scene": self.scene, "duration": self.duration,
                "lines": [{"t0": a, "t1": b} for a, b in zip(self.starts, self.ends)],
                "words": self.words, "fadeIn": first, "fadeOut": last}


SHOTS = [
    # 0:00 cold open
    Shot("a1", "counter", [
        "A month ago I built a three agent team, put it on a schedule, and then did the one thing that nobody who makes these videos ever actually does, which is nothing at all.",
        "No corrections. No restarts. I didn't open a single thread to see how it was getting on.",
    ], hold=0.8),
    Shot("a2", "scrub", [
        "Thirty days later I sat down and read every run it had done, and the useful half of that is not the part that worked.",
    ]),
    Shot("a3", "presenterFive", [
        "It's the five things that broke, because every one of them broke quietly, and three of them would have carried on breaking for another month if I hadn't gone looking.",
        "So this is what an agent team does to itself when nobody is watching, in the order it happened to me, and what I'd change in yours before you set it running.",
    ]),
    # 0:30 the setup
    Shot("b1", "spine", [
        "The team is three agents and each one has exactly one job.",
        "The first one goes out every day and finds what launched in AI, with a source link and a date on every item and a hard cap of five items so it has a definition of done.",
        "The second one takes that list and works out which of it is worth making a video about and which of it is already saturated.",
        "The third one turns whatever survives into a page with a cover image on each item that I can send to an editor without writing a covering note.",
    ]),
    Shot("b2", "presenterWords", [
        "Research, judgment, design. One each, and each one deliberately no use at the other two.",
    ]),
    Shot("b3", "schedule", [
        "The first agent runs on an interval because noticing things is its whole job, the other two run on a Monday morning schedule because their work only starts once there's something new, and the deal I set up was four weeks, four pages, whether I'm at my desk or not.",
    ]),
    Shot("b4", "scorecard", [
        "What I got was two pages, one empty run, one page I couldn't use, and a bill that didn't match any of it.",
    ], hold=3.0),
]


def layout():
    clock = 0.0
    for shot in SHOTS:
        shot.offset = clock
        clock += shot.duration
    return clock


# ---------------------------------------------------------------- rendering

def face_frames(ff, shot):
    """Hedra frames for this shot as JPEG data URLs, one per output frame."""
    if not HEDRA.exists():
        while True:
            yield None
    proc = subprocess.Popen(
        [ff, "-loglevel", "error", "-ss", f"{shot.offset:.3f}", "-i", str(HEDRA),
         "-t", f"{shot.duration:.3f}", "-vf", f"fps={FPS}", "-q:v", "3",
         "-f", "image2pipe", "-c:v", "mjpeg", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf, last = b"", None
    while True:
        chunk = proc.stdout.read(1 << 16)
        if chunk:
            buf += chunk
        # split on JPEG end-of-image markers
        while True:
            end = buf.find(b"\xff\xd9")
            if end < 0:
                break
            last = "data:image/jpeg;base64," + base64.b64encode(buf[:end + 2]).decode()
            buf = buf[end + 2:]
            yield last
        if not chunk:
            break
    proc.wait()
    while True:                     # past the end of the take: hold last frame
        yield last


def render_shot(index):
    from playwright.sync_api import sync_playwright

    layout()
    shot = SHOTS[index]
    ff = ffmpeg_exe()
    out = BUILD / "shots" / f"{shot.key}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = round(shot.duration * FPS)
    enc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
         "-r", str(FPS), str(out)], stdin=subprocess.PIPE)
    faces = face_frames(ff, shot)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(SCENES.as_uri())
        page.evaluate("s => setup(s)", shot.spec(index == 0, index == len(SHOTS) - 1))
        for f in range(n):
            page.evaluate("f => setFace(f)", next(faces))
            page.evaluate("([t, f]) => seek(t, f)", [f / FPS, f])
            enc.stdin.write(page.screenshot(type="jpeg", quality=93))
        browser.close()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on shot {shot.key}")
    return f"  {shot.key}: {shot.duration:5.1f}s  {n} frames"


# ---------------------------------------------------------------- audio + merge

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
    cues = [(s.offset + a, s.offset + b, line)
            for s in SHOTS for a, b, line in zip(s.starts, s.ends, s.lines)]
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
    total = layout()
    if args.voice_only:
        print(f"voice track -> {build_voiceover(ff)}")
        return
    keys = set(args.only.split(",")) if args.only else {s.key for s in SHOTS}
    todo = [i for i, s in enumerate(SHOTS) if s.key in keys]
    # longest first so the pool finishes together
    todo.sort(key=lambda i: -SHOTS[i].duration)
    print("rendering shots")
    with ProcessPoolExecutor(WORKERS) as pool:
        for line in pool.map(render_shot, todo):
            print(line, flush=True)
    out = merge(ff)
    srt = write_srt()
    print(f"merged -> {out.relative_to(ROOT)} ({total:.1f}s), captions -> {srt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
