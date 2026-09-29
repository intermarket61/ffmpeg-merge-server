#!/usr/bin/env python3
"""Lip-sync the presenter photo to the full voice track with Hedra.

    python3 render.py --voice-only        # builds build/vo/full-<part>.wav
    python3 hedra.py p2                   # -> build/hedra/p2.mp4
    python3 hedra.py p2 --resume <id>     # pick up a generation already queued

Needs HEDRA_API_KEY. Each take spans one part of the cut, so render.py can
take any shot's frames from it by that shot's offset within the part.
"""
import json
import mimetypes
import os
import sys
import time
import urllib.request
import uuid

import render

API = "https://api.hedra.com/web-app/public"
MODEL = "d1dd37a3-e39a-4854-a298-6510289f9cf2"   # Hedra Character 3
PROMPT = ("A confident presenter speaking directly to camera in a boardroom, "
          "calm and conversational, natural head movement and blinking, hands resting on the table.")


def call(method, path, body=None, files=None):
    headers = {"X-API-Key": os.environ["HEDRA_API_KEY"]}
    data = None
    if files:
        boundary = uuid.uuid4().hex
        name, blob = files
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode() + blob + \
            f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"Hedra {method} {path}: {e.code} {e.read().decode()[:500]}")


def upload(path, kind):
    asset = call("POST", "/assets", {"name": path.name, "type": kind})
    call("POST", f"/assets/{asset['id']}/upload", files=(path.name, path.read_bytes()))
    return asset["id"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    part = sys.argv[1]
    if len(sys.argv) > 3 and sys.argv[2] == "--resume":
        return wait(sys.argv[3], part)
    audio = render.BUILD / "vo" / f"full-{part}.wav"
    if not audio.exists():
        sys.exit(f"no {audio.name}: run `python3 render.py --voice-only` first")
    image_id = upload(render.ASSETS / "img" / "presenter.webp", "image")
    audio_id = upload(audio, "audio")
    gen = call("POST", "/generations", {
        "type": "video",
        "ai_model_id": MODEL,
        "start_keyframe_id": image_id,
        "audio_id": audio_id,
        "generated_video_inputs": {"text_prompt": PROMPT, "resolution": "1080p",
                                   "aspect_ratio": "16:9"},
    })
    print(f"generation {gen['id']} queued (resume with: {part} --resume {gen['id']})", flush=True)
    wait(gen["id"], part)


def wait(gen_id, part):
    while True:
        st = call("GET", f"/generations/{gen_id}/status")
        if st.get("status") == "complete":
            break
        if st.get("status") == "error":
            sys.exit(f"Hedra failed: {st.get('error_message') or st}")
        print(f"  {st.get('status')} {st.get('progress', '')}", flush=True)
        time.sleep(15)
    out = render.BUILD / "hedra" / f"{part}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(st.get("download_url") or st["url"], out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
