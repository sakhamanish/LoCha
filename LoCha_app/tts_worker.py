"""
Piper text-to-speech worker for LoCha's natural voice.

voice.py runs this file in its own process: Piper uses onnxruntime, whose
DLLs crash on Windows when loaded into the same process as torch and
PySide6. Nothing from LoCha is imported here. In the Windows build this
file is its own program, LoChaVoice.exe (see LoCha.spec).

Protocol (stdin/stdout):
- After loading the voice, the worker writes b"RDY0".
- For each request line {"text": "..."} on stdin, it writes one frame per
  sentence: <uint32 sample_rate><uint32 byte_length><int16 mono PCM>, and
  then an end frame with both values 0.
"""

import io
import os
import json
import struct
import sys


def _std_stream(stream, fd, mode, win_handle):
    """Binary stdin/stdout. In LoCha.exe (a windowed app) sys.stdin and
    sys.stdout can be None even though LoCha passed pipes, so fall back to
    the process's standard handles."""
    if stream is not None:
        return stream.buffer
    try:
        return os.fdopen(fd, mode)
    except OSError:
        import ctypes
        import msvcrt
        handle = ctypes.windll.kernel32.GetStdHandle(win_handle)
        return os.fdopen(msvcrt.open_osfhandle(handle, 0), mode)


def main(model_path=None):
    from piper import PiperVoice

    if sys.stderr is None:  # windowed exe: keep error messages for LoCha's log
        try:
            sys.stderr = io.TextIOWrapper(
                _std_stream(None, 2, "wb", -12), encoding="utf-8", line_buffering=True
            )
        except Exception:
            sys.stderr = io.StringIO()
    voice = PiperVoice.load(model_path or sys.argv[1])
    out = _std_stream(sys.stdout, 1, "wb", -11)  # STD_OUTPUT_HANDLE
    inp = io.TextIOWrapper(_std_stream(sys.stdin, 0, "rb", -10), encoding="utf-8")
    out.write(b"RDY0")
    out.flush()

    for line in inp:
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
    if len(sys.argv) < 2:
        # Opened directly (e.g. double-clicking LoChaVoice.exe). LoCha starts
        # this helper itself and passes it the voice file.
        message = "This is LoCha's voice helper; it runs in the background.\nOpen LoCha.exe instead."
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "LoCha", 0x40)  # information icon
        else:
            print(message, file=sys.stderr)
        sys.exit(1)
    main()
