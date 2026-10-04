"""
Offline voice support for LoCha.

- Speech-to-text: microphone capture via `sounddevice`, transcription via
  `faster-whisper` running locally on the CPU.
- Text-to-speech: `pyttsx3`, which uses the voices installed in the OS
  (SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux).

Everything runs on the local machine. The Whisper model is downloaded once
from Hugging Face on first use and cached, exactly like the embedding model.
The voice dependencies are optional: if they are missing, LoCha still runs
and the voice controls are disabled.
"""

import sys
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_MODEL = "base.en"
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio
MAX_RECORD_SECONDS = 60

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    STT_IMPORT_ERROR = ""
except Exception as e:  # ImportError, or OSError when PortAudio is missing
    STT_AVAILABLE = False
    STT_IMPORT_ERROR = str(e)

try:
    import pyttsx3
    TTS_AVAILABLE = True
    TTS_IMPORT_ERROR = ""
except Exception as e:
    TTS_AVAILABLE = False
    TTS_IMPORT_ERROR = str(e)


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


class Transcriber:
    """Lazily loads the Whisper model on first use and reuses it."""

    def __init__(self, model_name=WHISPER_MODEL):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        if not STT_AVAILABLE:
            raise RuntimeError(f"Voice input is unavailable: {STT_IMPORT_ERROR}")
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
                        "connection to download it (about 150 MB); after "
                        f"that it works offline.\n\nDetails: {e}"
                    ) from e
            segments, _ = self._model.transcribe(
                audio, language="en", beam_size=5, vad_filter=True
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        logger.info(f"Transcribed: {text}")
        return text


# ---------------- Text-to-speech ----------------
class Speaker:
    """Speaks text with the OS voices. speak() blocks; stop() is thread-safe."""

    def __init__(self):
        self._stop_requested = threading.Event()

    def speak(self, text: str):
        if not TTS_AVAILABLE:
            raise RuntimeError(f"Read-aloud is unavailable: {TTS_IMPORT_ERROR}")
        if not text.strip():
            return

        self._stop_requested.clear()

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
            tts.say(text)
            tts.runAndWait()
            del tts
        finally:
            if com_initialised:
                import comtypes
                comtypes.CoUninitialize()

    def stop(self):
        self._stop_requested.set()
