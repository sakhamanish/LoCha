LoCha is a local chat assistant skilled to answer the questions relevant to the given document. It uses retrieval-augmented generation (RAG) method to provide pertinent answers. The primary objective of LoCha is to maintain privacy of your documents. Once installed in your system, LoCha uses your local computer resources to generate the answers to your uploaded documents.

Prerequisites:
- Ollama installed
- llama3.2:latest model downloaded

Install:
1. Download Ollama from https://ollama.com
2. Run: ollama pull llama3.2:latest
3. Launch LoCha






Run from source:
1. Use Python 3.11 (the pinned packages do not support newer versions):
   py -3.11 -m venv .venv
   .venv\Scripts\activate
2. pip install -r requirements.txt
3. From the project folder: python -m LoCha_app.app

Voice mode (optional):
You can ask questions by voice and have answers read aloud. Everything runs offline on your computer.
1. The voice packages are included in requirements.txt (sounddevice, faster-whisper, pyttsx3).
   On Linux also install PortAudio and eSpeak, e.g. sudo apt install libportaudio2 libespeak1
2. Load a document as usual.
3. Click "🎤 Speak", ask your question out loud, then click "⏹ Stop".
   LoCha shows what it heard in the question box. Check or edit it, then press Enter or Ask.
4. Tick "🔊 Read answers aloud" to hear each answer. Use "Stop speaking" to cut it short.
   While LoCha is answering or speaking, Speak is disabled so the microphone never records LoCha's own voice.
The first time you use voice input, the Whisper speech model (small.en, about 480 MB) is downloaded once and cached. After that it works without internet.
To trade accuracy for speed, set LOCHA_WHISPER_MODEL=base.en (smaller, faster) or medium.en (larger, more accurate) before starting LoCha.
LoCha also gives Whisper the document's distinctive terms (names, acronyms) as hints, so domain words are recognised more reliably.
If the voice packages are not installed, the voice controls are greyed out and the rest of the app works as before.
