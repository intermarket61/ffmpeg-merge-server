#!/usr/bin/env python3
"""Voice every shot with ElevenLabs, keeping character timestamps so
render.py can time shots and graphics to the actual read.

Needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID. Existing takes are kept;
pass shot keys to re-voice just those (python3 voice.py a3 b1).
"""
import base64
import json
import os
import sys
import urllib.request

import render

MODEL = "eleven_multilingual_v2"


def voice(shot):
    text = " ".join(shot.lines)
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{os.environ['ELEVENLABS_VOICE_ID']}"
           "/with-timestamps?output_format=mp3_44100_128")
    req = urllib.request.Request(url, data=json.dumps({"text": text, "model_id": MODEL}).encode(),
                                 headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                                          "Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req))
    out = render.BUILD / "vo"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{shot.key}.mp3").write_bytes(base64.b64decode(data["audio_base64"]))
    (out / f"{shot.key}.json").write_text(json.dumps({"text": text, **data["alignment"]}))
    return data["alignment"]["character_end_times_seconds"][-1]


def main():
    only = set(sys.argv[1:])
    for shot in render.SHOTS:
        done = (render.BUILD / "vo" / f"{shot.key}.mp3").exists()
        if (only and shot.key in only) or (not only and not done):
            print(f"  {shot.key}: {voice(shot):.1f}s")


if __name__ == "__main__":
    main()
