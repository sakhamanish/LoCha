"""
Offline voice support for LoCha.

- Speech-to-text: microphone capture via `sounddevice`, transcription via
  `faster-whisper` running locally on the CPU.
- Text-to-speech: a natural-sounding neural voice (Piper), run in a separate
  process. If Piper is unavailable, the voices installed in the OS are used:
  SAPI on Windows (driven through `comtypes`), otherwise `pyttsx3`
  (NSSpeechSynthesizer on macOS, eSpeak on Linux).

Everything runs on the local machine. The Whisper model and the Piper voice
are downloaded once from Hugging Face on first use and cached, exactly like
the embedding model.
The voice dependencies are optional: if they are missing, LoCha still runs
and the voice controls are disabled.
"""

import os
import re
import sys
import json
import queue
import struct
import logging
import threading
import subprocess
import importlib.util
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# small.en is noticeably more accurate than base.en for spoken questions.
# Override with e.g. LOCHA_WHISPER_MODEL=base.en (faster) or medium.en.
WHISPER_MODEL = os.environ.get("LOCHA_WHISPER_MODEL", "small.en")
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio
MAX_RECORD_SECONDS = 60

# Natural voice: a Piper voice name from https://huggingface.co/rhasspy/piper-voices
# (e.g. en_US-amy-medium, en_GB-alba-medium) or a path to a local .onnx voice.
PIPER_VOICE = os.environ.get("LOCHA_PIPER_VOICE", "en_US-lessac-medium")

try:
    import sounddevice as sd
    SD_IMPORT_ERROR = ""
except Exception as e:  # ImportError, or OSError when PortAudio is missing
    sd = None
    SD_IMPORT_ERROR = str(e)

try:
    if sd is None:
        raise ImportError(SD_IMPORT_ERROR)
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    STT_IMPORT_ERROR = ""
except Exception as e:
    STT_AVAILABLE = False
    STT_IMPORT_ERROR = str(e)

try:
    import pyttsx3
    TTS_AVAILABLE = True
    TTS_IMPORT_ERROR = ""
except Exception as e:
    TTS_AVAILABLE = False
    TTS_IMPORT_ERROR = str(e)

# In LoCha.exe the natural voice is the separate LoChaVoice.exe next to it
# (see LoCha.spec); from source it's "python tts_worker.py".
FROZEN = getattr(sys, "frozen", False)
VOICE_EXE = os.path.join(
    os.path.dirname(sys.executable),
    "LoChaVoice.exe" if sys.platform == "win32" else "LoChaVoice",
)
PIPER_AVAILABLE = sd is not None and (
    os.path.exists(VOICE_EXE) if FROZEN else importlib.util.find_spec("piper") is not None
)
READ_ALOUD_AVAILABLE = PIPER_AVAILABLE or TTS_AVAILABLE


# ---------------- Speech-to-text ----------------
class VoiceRecorder:
    """Records mono audio from the default microphone until stop() is called."""

    def __init__(self):
        self._stream = None
        self._frames = []
        self._sample_rate = TARGET_SAMPLE_RATE
        self._max_frames = 0
        self._lock = threading.Lock()

    @property
    def is_recording(self):
        return self._stream is not None

    def start(self):
        if not STT_AVAILABLE:
            raise RuntimeError(f"Voice input is unavailable: {STT_IMPORT_ERROR}")

        # Record at the device's native rate and resample later; many
        # microphones/drivers reject 16 kHz directly.
        device_info = sd.query_devices(kind="input")
        self._sample_rate = int(device_info["default_samplerate"])
        self._max_frames = self._sample_rate * MAX_RECORD_SECONDS
        self._frames = []

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"Recording started at {self._sample_rate} Hz")

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio input status: {status}")
        with self._lock:
            if sum(len(f) for f in self._frames) < self._max_frames:
                self._frames.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        """Stops recording and returns 16 kHz mono float32 audio."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)

        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            audio = (
                np.concatenate(self._frames)
                if self._frames else np.zeros(0, dtype=np.float32)
            )
            self._frames = []

        logger.info(f"Recording stopped ({len(audio) / self._sample_rate:.1f}s)")
        return _resample(audio, self._sample_rate, TARGET_SAMPLE_RATE)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / src_rate
    dst_len = int(duration * dst_rate)
    src_t = np.linspace(0, duration, num=len(audio), endpoint=False)
    dst_t = np.linspace(0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def _trim_silence(audio: np.ndarray, rate=TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Trims leading/trailing silence with a simple energy gate; returns an
    empty array if the clip has no speech-level sound at all.

    Used instead of faster-whisper's built-in VAD (vad_filter), which loads
    onnxruntime; on Windows that crashes when torch/PySide6 DLLs are
    already loaded in the process.
    """
    frame = rate // 50  # 20 ms
    n = len(audio) // frame
    if n == 0:
        return audio[:0]
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    if rms.max() < 0.003:  # nothing louder than background hiss
        return audio[:0]
    active = np.where(rms > max(0.002, rms.max() * 0.05))[0]
    pad = 15  # keep 300 ms around the speech
    start = max(0, active[0] - pad) * frame
    end = min(n, active[-1] + 1 + pad) * frame
    return audio[start:end]


_COMMON_WORDS = set("""
a an and are as at be but by can do does for from has have how if in into is
it its may more most no not of on or our so such than that the their then there
these they this those to use used using was we what when where which who why
will with you your yes also all any each both other one two three new page
""".split())


def extract_vocabulary(text: str, max_terms=40):
    """
    Picks the document's distinctive terms (product names, acronyms, codes),
    i.e. words that appear capitalised but rarely in lower case. They are
    given to Whisper as hints so it spells domain words the document's way.
    """
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]\b", text or "")
    lower = Counter(w for w in words if w.islower())
    capitalised = Counter(w for w in words if not w.islower())
    candidates = [
        (count, word) for word, count in capitalised.items()
        if count >= 2
        and lower.get(word.lower(), 0) * 3 <= count
        and word.lower() not in _COMMON_WORDS
    ]
    terms, seen = [], set()
    for _, word in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if word.lower() not in seen:
            seen.add(word.lower())
            terms.append(word)
        if len(terms) == max_terms:
            break
    return terms


class Transcriber:
    """Lazily loads the Whisper model on first use and reuses it."""

    def __init__(self, model_name=WHISPER_MODEL):
        self.model_name = model_name
        self._model = None
        self._hotwords = None
        self._lock = threading.Lock()

    def set_vocabulary(self, terms):
        self._hotwords = ", ".join(terms) if terms else None
        logger.info(f"Speech vocabulary hints: {self._hotwords}")

    def transcribe(self, audio: np.ndarray) -> str:
        if not STT_AVAILABLE:
            raise RuntimeError(f"Voice input is unavailable: {STT_IMPORT_ERROR}")
        audio = _trim_silence(audio)
        if len(audio) < TARGET_SAMPLE_RATE // 2:  # under half a second
            return ""

        with self._lock:
            if self._model is None:
                logger.info(f"Loading Whisper model '{self.model_name}'")
                try:
                    self._model = WhisperModel(
                        self.model_name, device="cpu", compute_type="int8"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Could not load the speech model '{self.model_name}'. "
                        "The first use of voice input needs an internet "
                        "connection to download it; after "
                        f"that it works offline.\n\nDetails: {e}"
                    ) from e
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
                hotwords=self._hotwords,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        logger.info(f"Transcribed: {text}")
        return text


# ---------------- Text-to-speech ----------------
TTS_CHUNK_CHARS = 200


def _clean_for_speech(text: str) -> str:
    """Removes markdown symbols that voices would otherwise read out."""
    text = re.sub(r"[*_`#>]+", "", text or "")
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"[ \t]+", " ", text).strip()


class _PiperUnavailable(Exception):
    """The natural voice could not be started; use the system voice."""


class _PiperProcess:
    """
    Runs tts_worker.py in a separate process (see that file for why) and
    plays the audio it streams back. Started on first use and kept running,
    so only the first answer pays the start-up cost.
    """

    def __init__(self):
        self._proc = None
        self._failed = None  # reason, once starting has failed this session
        self._lock = threading.Lock()  # one utterance at a time

    def _voice_path(self):
        if PIPER_VOICE.endswith(".onnx"):
            if not os.path.exists(PIPER_VOICE):
                raise _PiperUnavailable(f"voice file not found: {PIPER_VOICE}")
            return PIPER_VOICE
        try:
            lang_code, name, quality = PIPER_VOICE.split("-")
        except ValueError:
            raise _PiperUnavailable(f"invalid voice name: {PIPER_VOICE}")
        from huggingface_hub import hf_hub_download

        base = f"{lang_code.split('_')[0]}/{lang_code}/{name}/{quality}/{PIPER_VOICE}"
        logger.info(f"Loading natural voice '{PIPER_VOICE}' (downloaded once, about 60 MB)")
        try:
            model = hf_hub_download("rhasspy/piper-voices", base + ".onnx")
            hf_hub_download("rhasspy/piper-voices", base + ".onnx.json")
        except Exception as e:
            raise _PiperUnavailable(f"could not download voice: {e}")
        return model

    def _read(self, n):
        data = b""
        while len(data) < n:
            part = self._proc.stdout.read(n - len(data))
            if not part:
                raise EOFError("natural voice process ended unexpectedly")
            data += part
        return data

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._failed:
            raise _PiperUnavailable(self._failed)
        try:
            self._start()
        except _PiperUnavailable as e:
            self._failed = str(e)  # don't retry (and delay) every answer
            raise

    def _start(self):
        if FROZEN:
            command = [VOICE_EXE]
        else:
            command = [sys.executable, os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "tts_worker.py"
            )]
        self._proc = subprocess.Popen(
            command + [self._voice_path()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._log_errors, args=(self._proc,), daemon=True).start()
        try:
            ready = self._read(4)
        except EOFError:
            ready = b""
        if ready != b"RDY0":
            self._proc.kill()
            self._proc = None
            raise _PiperUnavailable("natural voice failed to start (see log)")

    @staticmethod
    def _log_errors(proc):
        for line in proc.stderr:
            logger.warning(f"Natural voice: {line.decode(errors='replace').rstrip()}")

    def speak(self, texts, stop_event):
        """
        Speaks each text from the iterable in turn through one open audio
        stream, so sentences that arrive while an answer is still being
        written play back to back.
        """
        with self._lock:
            if stop_event.is_set():
                return
            self._ensure_started()  # before reading texts: on failure the
                                    # caller can still speak them all
            # Texts are sent from a separate thread as soon as they are
            # available, so Piper synthesizes the next sentence while the
            # current one plays (no pause between sentences).
            counts = {"sent": 0, "done": 0}
            counts_lock = threading.Lock()
            sender_done = threading.Event()

            def send_texts():
                try:
                    for text in texts:
                        if stop_event.is_set():
                            break
                        self._proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
                        self._proc.stdin.flush()
                        with counts_lock:
                            counts["sent"] += 1
                except Exception as e:  # worker died; the reader sees EOF
                    logger.warning(f"Natural voice: could not send text: {e}")
                finally:
                    sender_done.set()

            threading.Thread(target=send_texts, daemon=True).start()

            stream = None
            try:
                while True:
                    with counts_lock:
                        outstanding = counts["sent"] - counts["done"]
                    if outstanding == 0:
                        if sender_done.is_set():
                            with counts_lock:
                                if counts["sent"] == counts["done"]:
                                    break
                            continue
                        sender_done.wait(0.02)
                        continue
                    rate, length = struct.unpack("<II", self._read(8))
                    if length == 0 and rate == 0:
                        with counts_lock:
                            counts["done"] += 1  # end of one text
                        continue
                    pcm = self._read(length)
                    if stop_event.is_set():
                        continue  # drain so the next request starts clean
                    if stream is None or stream.samplerate != rate:
                        if stream is not None:
                            stream.stop()
                            stream.close()
                        stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16")
                        stream.start()
                    step = rate // 10 * 2  # 100 ms of audio, so Stop reacts quickly
                    for i in range(0, len(pcm), step):
                        if stop_event.is_set():
                            stream.abort()
                            break
                        stream.write(pcm[i:i + step])
            except EOFError:
                self._proc = None
                raise
            finally:
                if stream is not None:
                    if not stream.stopped:
                        if stop_event.is_set():
                            stream.abort()
                        else:
                            stream.stop()  # let the last buffer finish playing
                    stream.close()

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


_piper = _PiperProcess()


def shutdown():
    """Stops the natural-voice process; call when the app closes."""
    _piper.close()


def _split_sentences(text: str, max_chars=TTS_CHUNK_CHARS):
    """Splits text into sentences, merging short ones up to max_chars."""
    sentences = [
        p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()
    ]
    chunks = []
    for sentence in sentences:
        if chunks and len(chunks[-1]) + 1 + len(sentence) <= max_chars:
            chunks[-1] += " " + sentence
        else:
            chunks.append(sentence)
    return chunks


class LiveSpeech:
    """
    Receives an answer's text while it is being written (feed) and hands
    complete sentences to the speaker (sentences), so reading aloud starts
    with the first sentence instead of after the whole answer.
    feed() and finish() are called from the UI thread; sentences() is
    consumed by the speaking thread.
    """

    # A sentence ends at . ! ? followed by whitespace (not after a digit, so
    # list numbers like "1. " stay with their item), or at a line break.
    _BOUNDARY = re.compile(r"(?<=[^\d\s][.!?])\s+|\n+")

    def __init__(self):
        self._pending = ""
        self._queue = queue.Queue()

    def feed(self, text):
        self._pending += text
        parts = self._BOUNDARY.split(self._pending)
        for sentence in parts[:-1]:
            if sentence.strip():
                self._queue.put(sentence)
        self._pending = parts[-1]

    def finish(self):
        if self._pending.strip():
            self._queue.put(self._pending)
        self._pending = ""
        self._queue.put(None)

    def sentences(self, stop_event):
        while not stop_event.is_set():
            try:
                sentence = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if sentence is None:
                return
            yield sentence


class Speaker:
    """
    Speaks text with the natural Piper voice, or the OS voices if Piper is
    unavailable. speak() blocks; stop() is thread-safe. Each Speaker is
    meant for a single utterance.

    Piper audio is played in 100 ms slices, so stop() takes effect almost
    immediately. For the OS voices:
    on Windows, SAPI speaks asynchronously and stop() is polled every
    100 ms, so speech stops almost immediately. Elsewhere (pyttsx3), text is
    spoken in sentence-sized chunks so stop() always takes effect at the
    next chunk boundary, or mid-sentence where the driver reports words as
    they are spoken (macOS). The Linux eSpeak driver synthesises a whole
    chunk before playing it, so there the chunk boundary is the limit.
    """

    def __init__(self):
        self._stop_requested = threading.Event()

    def speak(self, text: str):
        """Speaks a complete text."""
        self._speak_texts(iter([text]))

    def speak_live(self, live: "LiveSpeech"):
        """Speaks sentences as they arrive until live.finish() is called."""
        self._speak_texts(live.sentences(self._stop_requested))

    def _speak_texts(self, texts):
        if not READ_ALOUD_AVAILABLE:
            raise RuntimeError(f"Read-aloud is unavailable: {TTS_IMPORT_ERROR}")
        texts = (t for t in map(_clean_for_speech, texts) if t)
        if PIPER_AVAILABLE:
            try:
                _piper.speak(texts, self._stop_requested)
                return
            except _PiperUnavailable as e:
                logger.warning(f"Natural voice unavailable, using system voice: {e}")
            except EOFError as e:
                logger.warning(f"Natural voice stopped: {e}")
                return
            if self._stop_requested.is_set() or not TTS_AVAILABLE:
                return
        for text in texts:
            if self._stop_requested.is_set():
                break
            self._speak_system(_split_sentences(text))

    def _speak_system(self, chunks):
        if not chunks:
            return
        if sys.platform == "win32":
            self._speak_windows(" ".join(chunks), chunks)
        else:
            self._speak_pyttsx3(chunks)

    def _speak_windows(self, text, chunks):
        import comtypes
        import comtypes.client

        SVSF_ASYNC, SVSF_PURGE, SVSF_IS_NOT_XML = 1, 2, 16
        comtypes.CoInitialize()  # COM must be initialised in this thread
        try:
            try:
                sapi = comtypes.client.CreateObject("SAPI.SpVoice", dynamic=True)
            except Exception as e:
                logger.warning(f"SAPI unavailable, using pyttsx3: {e}")
                self._speak_pyttsx3(chunks)
                return
            if self._stop_requested.is_set():
                return
            sapi.Speak(text, SVSF_ASYNC | SVSF_IS_NOT_XML)
            while not sapi.WaitUntilDone(100):
                if self._stop_requested.is_set():
                    sapi.Speak("", SVSF_ASYNC | SVSF_PURGE)  # cut speech off
                    break
            del sapi
        finally:
            comtypes.CoUninitialize()

    def _speak_pyttsx3(self, chunks):
        # SAPI5 is a COM API; COM must be initialised in this worker thread.
        com_initialised = False
        if sys.platform == "win32":
            try:
                import comtypes
                comtypes.CoInitialize()
                com_initialised = True
            except Exception as e:
                logger.warning(f"COM initialisation failed: {e}")

        try:
            # A fresh engine per call: pyttsx3 engines are bound to the
            # thread that created them.
            tts = pyttsx3.init()
            tts.setProperty("rate", 175)

            def on_word(name, location, length):
                if self._stop_requested.is_set():
                    tts.stop()

            tts.connect("started-word", on_word)
            for chunk in chunks:
                if self._stop_requested.is_set():
                    break
                tts.say(chunk)
                tts.runAndWait()
            del tts
        finally:
            if com_initialised:
                import comtypes
                comtypes.CoUninitialize()

    def stop(self):
        self._stop_requested.set()
