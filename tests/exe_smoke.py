"""
End-to-end test that runs *inside* a PyInstaller build made from LoCha.spec
(see .github/workflows/windows-exe.yml). It exercises the real pipeline in
the frozen app, in the order the app uses it, so packaging problems (missing
modules or data, Windows DLL clashes) show up as failures:

- load a PDF: PDF parsing, the real embedding model (torch), FAISS
- ask, with streaming, and a follow-up: LangChain + Ollama (a stand-in server)
- summarize, then reopen the document from the disk cache
- the natural voice: LoChaVoice.exe speaks a question (Piper/onnxruntime)
- speech-to-text: Whisper transcribes that audio (ctranslate2, after torch)
- open the main window

Usage: LoChaSmoke.exe <sample.pdf>
Optional, for running without internet: LOCHA_SMOKE_EMBEDDING (local
sentence-transformers folder), LOCHA_SMOKE_VOICE (local .onnx voice),
LOCHA_SMOKE_SKIP_WHISPER=1.
"""

import http.server
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""), flush=True)


class FakeOllama(http.server.BaseHTTPRequestHandler):
    """Stands in for the Ollama server (not available on the build machine)."""

    def do_GET(self):
        body = json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for token in ["Dynamo ", "saves ", "about ", "12 ", "hours. ", "It ", "helps."]:
            self.wfile.write((json.dumps({"response": token, "done": False}) + "\n").encode())
            self.wfile.flush()
        self.wfile.write((json.dumps({"response": "", "done": True}) + "\n").encode())

    def log_message(self, *args):
        pass


def piper_speak(voice_exe, model, text):
    """Asks LoChaVoice.exe to synthesize text; returns (pcm_bytes, rate)."""
    proc = subprocess.Popen(
        [voice_exe, model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        if proc.stdout.read(4) != b"RDY0":
            raise RuntimeError("voice worker did not start: " + proc.stderr.read().decode(errors="replace")[-500:])
        proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
        proc.stdin.flush()
        pcm, rate = b"", 0
        while True:
            frame_rate, length = struct.unpack("<II", proc.stdout.read(8))
            if frame_rate == 0 and length == 0:
                return pcm, rate
            rate = frame_rate
            pcm += proc.stdout.read(length)
    finally:
        proc.kill()


def main():
    pdf = os.path.abspath(sys.argv[1])
    work = tempfile.mkdtemp()
    os.environ["LOCALAPPDATA"] = work  # fresh index cache and log folder
    os.environ.setdefault("LOCHA_WHISPER_MODEL", "tiny.en")  # small download

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["OLLAMA_HOST"] = f"127.0.0.1:{server.server_port}"

    import numpy as np
    from LoCha_app import rag_engine, voice

    if os.environ.get("LOCHA_SMOKE_EMBEDDING"):
        rag_engine.EMBEDDING_MODEL = os.environ["LOCHA_SMOKE_EMBEDDING"]

    # ---- documents and answers ----
    engine = rag_engine.LoChaEngine()
    engine.load_document(pdf)
    check("load PDF and build the search index", engine.index_ready,
          f"{engine.vectorstore.index.ntotal} passages")
    tokens = []
    qa = engine.ask("How much time does Dynamo save?", on_token=tokens.append)
    check("ask with streaming", qa["answer"] == "Dynamo saves about 12 hours. It helps." and len(tokens) == 7,
          f"source match {rag_engine.format_match(qa)}")
    check("follow-up question", engine.ask("And per project?")["answer"])
    check("summarize", engine.summarize())
    reopened = rag_engine.LoChaEngine()
    reopened.load_document(pdf)
    check("reopen from the disk cache", reopened.summary == engine.summary)

    # ---- natural voice, then speech-to-text of what it said ----
    check("natural voice program found", voice.PIPER_AVAILABLE, voice.VOICE_EXE)
    model = os.environ.get("LOCHA_SMOKE_VOICE") or voice._piper._voice_path()
    pcm, rate = piper_speak(voice.VOICE_EXE, model, "What are the payment terms?")
    check("natural voice speaks (LoChaVoice)", len(pcm) > rate, f"{len(pcm) / 2 / max(rate, 1):.1f}s of audio")

    if os.environ.get("LOCHA_SMOKE_SKIP_WHISPER"):
        print("SKIP speech-to-text (LOCHA_SMOKE_SKIP_WHISPER)", flush=True)
    else:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        audio = voice._resample(audio, rate, voice.TARGET_SAMPLE_RATE)
        text = voice.Transcriber().transcribe(audio)
        check("speech-to-text (Whisper) after torch is loaded", "payment" in text.lower(), repr(text))

    # ---- window ----
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    from LoCha_app import app as locha_app

    window = locha_app.LoChaApp()
    window.show()
    for _ in range(50):
        QCoreApplication.processEvents()
        time.sleep(0.01)
    check("main window opens", window.isVisible())
    window.close()
    voice.shutdown()

    passed = all(results)
    print(f"RESULT: {'ALL PASSED' if passed else 'FAILURES'} ({sum(results)}/{len(results)})", flush=True)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("RESULT: CRASHED", flush=True)
        os._exit(2)
