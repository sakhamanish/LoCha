"""
Piper text-to-speech worker for LoCha's natural voice.

voice.py runs this file in its own process: Piper uses onnxruntime, whose
DLLs crash on Windows when loaded into the same process as torch and
PySide6. Nothing from LoCha is imported here.

Protocol (stdin/stdout):
- After loading the voice, the worker writes b"RDY0".
- For each request line {"text": "..."} on stdin, it writes one frame per
  sentence: <uint32 sample_rate><uint32 byte_length><int16 mono PCM>, and
  then an end frame with both values 0.
"""

import json
import struct
import sys


def main():
    from piper import PiperVoice

    voice = PiperVoice.load(sys.argv[1])
    out = sys.stdout.buffer
    out.write(b"RDY0")
    out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line)["text"]
            for chunk in voice.synthesize(text):
                pcm = chunk.audio_int16_bytes
                out.write(struct.pack("<II", chunk.sample_rate, len(pcm)))
                out.write(pcm)
                out.flush()
        except Exception as e:
            print(f"Piper synthesis failed: {e}", file=sys.stderr, flush=True)
        out.write(struct.pack("<II", 0, 0))
        out.flush()


if __name__ == "__main__":
    main()
