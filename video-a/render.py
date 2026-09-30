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
HEDRA_DIR = BUILD / "hedra"      # one lip-synced take per part: <part>.mp4

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


# Each part gets its own Hedra take (a take tops out at ten minutes, and
# splitting lets the takes generate in parallel).
PART_OF = {**dict.fromkeys("ab", "p1"), **dict.fromkeys("cdef", "p2"),
           **dict.fromkeys("ghijk", "p3")}


class Shot:
    """A run of voiceover lines over one scene. Timing comes from the
    ElevenLabs character alignment in build/vo/<key>.json when it exists."""

    def __init__(self, key, scene, lines, hold=0.0, trim=0.0, **params):
        self.key = key
        self.part = PART_OF[key[0]]  # Hedra take this shot's face comes from
        self.params = params        # scene options, passed through to the page
        self.scene = scene          # scene name in scenes/index.html
        self.lines = lines          # voiceover sentences covered by this shot
        self.hold = hold            # silent tail, seconds
        self.trim = trim            # seconds cut from the tail in the edit only;
                                    # the Hedra take keeps them, so sync holds
        self.offset = 0.0           # start of this shot in the whole cut
        self.part_offset = 0.0      # start of this shot in its part's take
        self.starts, self.ends, self.words = [], [], []
        vo = BUILD / "vo" / f"{key}.json"
        if not lines:
            self.duration = hold
        elif vo.exists():
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

    @property
    def length(self):
        """Seconds this shot occupies in the finished cut."""
        return self.duration - self.trim

    def spec(self, first, last):
        return {"key": self.key, "scene": self.scene, "duration": self.duration,
                "lines": [{"t0": a, "t1": b} for a, b in zip(self.starts, self.ends)],
                "words": self.words, "fadeIn": first, "fadeOut": last,
                "params": self.params}


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
    ], hold=3.0, trim=2.6),
    # 1:10 week one is the trap
    Shot("c1", "weekOne", [
        "Week one is genuinely good and that is the problem with week one.",
        "The page turns up on Monday morning, five items on it, sources on all five, images that aren't embarrassing, and I remember thinking the hard part was over.",
        "What's actually happened is that week one is the week the job most closely resembles the job I described when I wrote the prompts.",
        "Everything I'd seen while testing was in the prompts, so week one is the agent being marked on the exact paper it revised for.",
    ]),
    Shot("c2", "presenter", [
        "Every failure after this one comes from the same place, which is a situation I never described, and an agent that has no way of telling me it's in one.",
        "That's the thing to hold on to for the rest of this, because none of these failures look like an error.",
    ], pills=[[0, "never", "A situation I <em>never described</em>"]]),
    Shot("c3", "textCard", [
        "They all look like output.",
    ], hold=3.0, theme="dark", blocks=[
        {"html": "None of these look like errors.", "at": [0, "they"], "size": 104},
        {"html": "They look like <span style='color:var(--orange)'>output.</span>", "at": [0, "output"], "size": 104},
    ]),
    # 1:50 failure one
    Shot("d1", "failureTitle", [
        "Week two, Monday morning, nothing.",
        "No page, no error, no email telling me a run had failed, because nothing had failed.",
    ], n=1, title="The run that finished<br>and never arrived"),
    Shot("d2", "runChain", [
        "The research agent ran, the judgment agent ran, the design agent ran, and then it stopped, because publishing the page counts as writing to somewhere outside the app and I'd set every external write to ask me first.",
        "Which is the correct setting.",
    ], tally=1),
    Shot("d3", "approvalQueue", [
        "The approval request went into a dashboard I had promised myself I would not open for thirty days, and it sat there.",
        "The work was all done.",
        "It was sitting one click away from being finished for six days.",
    ], tally=1),
    Shot("d4", "presenter", [
        "And the reason I'm putting this first is that it's the failure mode nobody warns you about, because it isn't the agent doing something bad, it's the agent doing everything right and then waiting politely for a person who isn't coming.",
    ], tally=1, pills=[[0, "everything", "Doing <em>everything right</em>"],
                       [0, "politely", "Waiting for <em>a person who isn't coming</em>"]]),
    Shot("d5", "textCard", [
        "If your approval requests land somewhere you don't already look every day, you don't have an approval step.",
        "You have a queue where work goes to die.",
    ], hold=1.2, tally=1, theme="orange", blocks=[
        {"html": "No approval step.", "at": [0, "don't", 1], "size": 150},
        {"html": "Just a queue where<br>work goes to die.", "at": [1, "queue"], "size": 150, "color": "#fff"},
    ]),
    # 2:45 failure two
    Shot("e1", "failureTitle", [
        "Week three is where it gets more interesting, because the page turns up and it's wrong in a way that took me a while to see.",
    ], n=2, title="An agent with<br>no way to say no"),
    Shot("e2", "funnel", [
        "Five items, all sourced, all dated, all correct, and every single one of them marked as worth covering.",
        "So I go back through weeks one and two and it's the same.",
        "The judgment agent has never once returned a list shorter than the list it was given.",
    ], tally=2),
    Shot("e3", "presenter", [
        "Its job description told it to work out which items are worth covering and come back with the angles, and I gave it a definition of done that says come back with angles and a reason on each, and nothing in any of that gives it permission to hand back nothing.",
    ], tally=2, pills=[[0, "permission", "No <em>permission</em> to hand back nothing"]]),
    Shot("e4", "textCard", [
        "An agent with no way to fail an item will never fail one.",
        "It will write you a reason instead, because writing a plausible reason is easier than returning an empty list and it has no idea that an empty list is an acceptable answer.",
    ], tally=2, theme="dark", blocks=[
        {"html": "No way to fail an item?", "at": [0, "an"], "size": 110},
        {"html": "It will <span style='color:var(--orange)'>never fail one.</span>", "at": [0, "never"], "size": 110},
        {"html": "It writes you a plausible reason instead.", "at": [1, "reason"], "size": 52, "serif": True},
    ]),
    Shot("e5", "promptDiff", [
        "That whole agent was a formatting step for three weeks and it cost me a model call every Monday to do it.",
        "The fix is one line, which is that returning zero items is a successful run, and here is what a zero week looks like.",
    ], tally=2),
    Shot("e6", "textCard", [
        "And you have to say it explicitly, because be selective is not a threshold, it's a mood.",
    ], hold=1.5, trim=1.2, tally=2, theme="paper", blocks=[
        {"html": "&ldquo;Be selective&rdquo;", "at": [0, "selective"], "size": 150, "serif": True, "italic": True},
        {"html": "is not a threshold.", "at": [0, "threshold"], "size": 110},
        {"html": "It's a mood.", "at": [0, "mood"], "size": 110, "color": "var(--orange)"},
    ]),
    # 3:45 failure three
    Shot("f1", "failureTitle", [
        "The third one only showed up because I was reading four weeks of runs in one sitting, which is the only way you'd catch it.",
        "Somewhere in week two, the source links stop arriving on the final page.",
        "Not all of them, and not consistently, which is why it doesn't look like a bug.",
    ], n=3, title="Everything the<br>handoff left behind"),
    Shot("f2", "presenter", [
        "What's happening is in the handoff.",
        "The child agent only sees the brief it was handed, not the conversation the parent was having, and the brief I wrote said to take the research agent's items and pick the best ones.",
    ], tally=3, pills=[[1, "brief", "It only sees <em>the brief</em>"]]),
    Shot("f3", "handoff", [
        "So the research agent, being reasonable, summarised its items into a paragraph and passed the paragraph, because that reads like items.",
        "The links were in the thread.",
        "They just weren't in the handoff.",
        "And the design agent at the end of the chain can't put back a link it was never given, so it writes the page without one, and nothing anywhere in that chain has done anything wrong.",
    ], tally=3),
    Shot("f4", "textCard", [
        "Every handoff between two agents is a lossy compression step, and whatever you don't name in the brief is what gets compressed out.",
    ], tally=3, theme="orange", blocks=[
        {"html": "Every handoff<br>is lossy.", "at": [0, "handoff"], "size": 170},
        {"html": "What you don't name gets compressed out.", "at": [0, "name"], "size": 56, "serif": True, "italic": True, "color": "#2a130a"},
    ]),
    Shot("f5", "checklist", [
        "The brief has to carry the goal, the inputs, the format of the inputs, and what finished looks like, written so it makes sense to somebody with nothing else in front of them, because that is literally the situation the agent on the other end is in.",
    ], tally=3),
    # 4:45 failure four
    Shot("g1", "failureTitle", [
        "Then there's the cost, and it's the failure that's easiest to ignore because it's still small.",
    ], n=4, title="The bill nobody reads<br>until the end of the month"),
    Shot("g2", "costChart", [
        "My spend per usable page roughly tripled over the month while the number of usable pages went down.",
        "Two things did that.",
    ], tally=4),
    Shot("g3", "nothingChanged", [
        "The first is that the research agent runs on a short interval and it was running the expensive model to find out that nothing had changed, which is most of the runs it does, all day, every day.",
        "There's a cheaper model you can put on the recurring check for exactly that reason, and I hadn't set it.",
    ], tally=4),
    Shot("g4", "presenter", [
        "The second one is subtler.",
        "A scheduled run opens a fresh thread by default, so every Monday the whole team starts from zero and re-reads and re-summarises everything it needs, and as the amount of material grows, every run pays for all of it again from the beginning.",
    ], tally=4, pills=[[1, "zero", "Every Monday starts <em>from zero</em>"],
                       [1, "pays", "Every run <em>pays for all of it again</em>"]]),
    Shot("g5", "textCard", [
        "None of this shows up as an error.",
        "It shows up as a number going up in a place you look at once a month, which is exactly why you put a cap on what a single run is allowed to spend before you set any of it running.",
        "The cap is the only part of this that tells you.",
    ], tally=4, theme="dark", blocks=[
        {"html": "Not an error.", "at": [0, "error"], "size": 120},
        {"html": "A number going up.", "at": [1, "number"], "size": 120, "color": "var(--orange)"},
        {"html": "Cap what a single run can spend.", "at": [2, "cap"], "size": 56, "serif": True, "italic": True},
    ]),
    # 5:40 failure five
    Shot("h1", "failureTitle", [
        "The last one didn't cost me anything, and it's the one I think about.",
    ], n=5, title="The tool it<br>never used"),
    Shot("h2", "toolSwitches", [
        "When I built the research agent I switched on web search and the browser, and I left the page publishing tool on as well, because at the time the research agent was the only agent I had and I was testing whether it could publish.",
        "It kept that tool for thirty days.",
    ], tally=5),
    Shot("h3", "presenter", [
        "Nothing in its job description mentions publishing and it never published anything, so as far as the runs are concerned, nothing happened.",
        "But I had an agent running unattended on a short interval with the ability to put a page on the public internet and no reason in its instructions ever to do it, which means the only thing standing between me and a published half-finished research dump was that the situation never came up.",
    ], tally=5, pills=[[1, "public", "Able to publish to <em>the public internet</em>"],
                       [1, "situation", "Only luck: <em>it never came up</em>"]]),
    Shot("h4", "textCard", [
        "That's not a safeguard.",
        "A tool an agent doesn't need is not neutral, it's a thing that can happen.",
    ], tally=5, theme="orange", blocks=[
        {"html": "Not a safeguard.", "at": [0, "safeguard"], "size": 150},
        {"html": "An unneeded tool is<br>a thing that can happen.", "at": [1, "thing"], "size": 96, "color": "#fff"},
    ]),
    Shot("h5", "presenter", [
        "So go through every agent you've got and switch off anything that isn't part of the one job you gave it, and do it before the schedule, not after something turns up.",
    ], tally=5, pills=[[0, "switch", "<em>Switch off</em> what the job doesn't need"]]),
    # 6:35 what I'd change
    Shot("i1", "fourFixes", [
        "So, four changes, and they're all cheap.",
        "Approvals have to land where you already are, which for me means a phone notification and a Slack channel, not a tab.",
        "Every agent needs an explicit successful-but-empty outcome, described, so that handing back nothing is a result rather than a failure it will work around.",
        "Every handoff brief gets written as if the receiving agent has amnesia, because it does, and that includes the format the inputs arrive in and not just their existence.",
        "And every agent gets the smallest tool list that lets it finish, with a spend cap per run and the cheap model on anything recurring.",
    ], hold=2.0, trim=1.6, tally=5),
    # 7:10 the actual lesson
    Shot("j1", "presenter", [
        "But the thing underneath all four of those is about what supervision actually means, and I had it wrong for a month.",
        "I thought supervising the team meant reading its work, and reading its work is the one thing that doesn't scale, because the whole reason you built it is that you didn't want to read every version.",
    ], tally=5, pills=[[1, "scale", "Reading its work <em>doesn't scale</em>"]]),
    Shot("j2", "presenter", [
        "Three of my five failures produced output that looked completely normal.",
        "The empty week didn't produce anything at all, and that one I'd have caught in a day if anything had been set up to tell me.",
    ], tally=5, pills=[[0, "normal", "3 of 5 looked <em>completely normal</em>"]]),
    Shot("j3", "alarms", [
        "So you don't audit the output, you build alarms on the shape of it.",
        "Tell me when a run produces nothing.",
        "Tell me when the judgment step passes everything through.",
        "Tell me when a page goes out without a source on an item.",
        "Tell me when a run costs more than a number.",
    ], hold=1.0, tally=5),
    Shot("j4", "presenter", [
        "Four conditions, each of which is a thing that cannot be true if the team is working, and the team is then allowed to run unwatched, because it's got a way of shouting.",
    ], tally=5, pills=[[0, "shouting", "It's got <em>a way of shouting</em>"]]),
    # 7:50 close
    Shot("k1", "presenter", [
        "Thirty days, five failures, and not one of them was the agent doing something it wasn't told to do.",
        "Every single one was the agent doing exactly what I told it, in a situation I never described, with no way of letting me know it was in one.",
    ], tally=5, pills=[[1, "exactly", "Doing <em>exactly what I told it</em>"]]),
    Shot("k2", "textCard", [
        "Which means the work isn't watching them.",
        "The work is writing down, in advance, what cannot be true if this is going well, and wiring that to something that reaches you.",
    ], tally=5, theme="paper", blocks=[
        {"html": "The work isn't watching.", "at": [0, "watching"], "size": 110},
        {"html": "It's writing down what <span style='color:var(--orange)'>cannot be true</span>,", "at": [1, "cannot"], "size": 76, "serif": True},
        {"html": "and wiring it to something that reaches you.", "at": [1, "wiring"], "size": 76, "serif": True, "italic": True},
    ]),
    Shot("k3", "presenter", [
        "Do that and you can leave it alone for a month.",
        "I'm running mine for another thirty days with all four of those changes in, and if it breaks in a new way I'll come back and tell you.",
    ], tally=5, pills=[[1, "another", "<em>Another 30 days.</em>"]]),
    Shot("k4", "endCard", [], hold=6.0, tally=5),
]


def layout():
    """offset is where a shot sits in the cut (trimmed lengths); part_offset
    is where it sits in its Hedra take (full durations, as voiced)."""
    clock, part_clock = 0.0, {}
    for shot in SHOTS:
        shot.offset = clock
        shot.part_offset = part_clock.get(shot.part, 0.0)
        part_clock[shot.part] = shot.part_offset + shot.duration
        clock += shot.length
    return clock


def part_shots(part=None):
    return [s for s in SHOTS if part is None or s.part == part]


# ---------------------------------------------------------------- rendering

def face_frames(ff, shot):
    """Hedra frames for this shot as JPEG data URLs, one per output frame."""
    take = HEDRA_DIR / f"{shot.part}.mp4"
    if not take.exists():
        while True:
            yield None
    proc = subprocess.Popen(
        [ff, "-loglevel", "error", "-ss", f"{shot.part_offset:.3f}", "-i", str(take),
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
    n = round(shot.length * FPS)
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
    return f"  {shot.key}: {shot.length:5.1f}s  {n} frames"


# ---------------------------------------------------------------- audio + merge

def build_voiceover(ff, part=None):
    """One continuous voice track: each shot's ElevenLabs read, padded to the
    shot's length. Per part it is what Hedra lip-syncs to, so it must match
    the cut exactly; without a part it is the whole film's track."""
    shots = part_shots(part)
    out = BUILD / "vo" / (f"full-{part}.wav" if part else "full.wav")
    missing = [s.key for s in shots if s.lines and not (BUILD / "vo" / f"{s.key}.mp3").exists()]
    if missing:
        print(f"  no voiceover for {', '.join(missing)}: run voice.py; merging silent")
        return None
    cmd = [ff, "-y", "-loglevel", "error"]
    for s in shots:
        cmd += (["-i", str(BUILD / "vo" / f"{s.key}.mp3")] if s.lines else
                ["-f", "lavfi", "-t", f"{s.duration:.3f}", "-i", "anullsrc=r=48000:cl=mono"])
    # a part's track feeds Hedra, so it keeps full durations; the film's
    # track follows the cut
    dur = (lambda s: s.duration) if part else (lambda s: s.length)
    chains = "".join(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,apad,"
                     f"atrim=0:{dur(s):.3f}[a{i}];" for i, s in enumerate(shots))
    chains += "".join(f"[a{i}]" for i in range(len(shots)))
    chains += f"concat=n={len(shots)}:v=0:a=1[out]"
    subprocess.run(cmd + ["-filter_complex", chains, "-map", "[out]", "-ac", "1", str(out)],
                   check=True)
    return out


def merge(ff):
    listing = BUILD / "shots.txt"
    listing.write_text("".join(f"file 'shots/{s.key}.mp4'\n" for s in SHOTS))
    out = BUILD / "video-a.mp4"
    total = sum(s.length for s in SHOTS)
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
                    help="just build build/vo/full-<part>.wav per part (for hedra.py)")
    args = ap.parse_args()
    ff = ffmpeg_exe()
    total = layout()
    if args.voice_only:
        for part in sorted({s.part for s in SHOTS}):
            print(f"voice track {part} -> {build_voiceover(ff, part)}")
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
