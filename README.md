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
1. The voice packages are included in requirements.txt (sounddevice, faster-whisper, piper-tts, pyttsx3).
   On Linux also install PortAudio and eSpeak, e.g. sudo apt install libportaudio2 libespeak1
2. Load a document as usual.
3. Click "🎤 Speak", ask your question out loud, then click "⏹ Stop".
   LoCha transcribes your question and answers it straight away; no need to press Ask.
4. Tick "🔊 Read answers aloud" to hear each answer in a natural-sounding voice. Use "Stop speaking" to cut it short.
   While LoCha is answering or speaking, Speak is disabled so the microphone never records LoCha's own voice.
The first time you use voice input, the Whisper speech model (small.en, about 480 MB) is downloaded once and cached. After that it works without internet.
The natural voice (Piper, en_US-lessac-medium, about 60 MB) is also downloaded once, the first time an answer is read aloud. If it can't be used, LoCha falls back to your computer's built-in voice.
To try a different voice, set LOCHA_PIPER_VOICE to any voice name from https://huggingface.co/rhasspy/piper-voices (for example en_US-amy-medium or en_GB-alba-medium) before starting LoCha.
To trade accuracy for speed, set LOCHA_WHISPER_MODEL=base.en (smaller, faster) or medium.en (larger, more accurate) before starting LoCha.
LoCha also gives Whisper the document's distinctive terms (names, acronyms) as hints, so domain words are recognised more reliably.
If the voice packages are not installed, the voice controls are greyed out and the rest of the app works as before.

Summarize:
After loading a document, click "Summarize" for a brief overview: a short paragraph with the gist of the document, then its key points.
The summary appears as it is written. For long documents LoCha picks about ten passages that represent the document's different topics (plus the opening) and summarizes those in one go, so it stays quick. The summary is kept, even after restarting LoCha, so clicking again shows it instantly, and "Save Conversation" includes it.

Source match:
Each answer shows a "Source match" score from 0.00 to 1.00: how closely the best passage in the document matches your question (cosine similarity).
With this search model even close matches usually score 0.55-0.75, so a word is shown next to it: strong (0.55 and up), moderate (0.35-0.54) or weak (below 0.35).
A weak match means the document probably doesn't cover the question, so treat the answer with caution.

Troubleshooting:
If LoCha says "Ollama isn't running", start the Ollama app (or run "ollama serve"), then ask again.
If it says the model isn't downloaded, run: ollama pull llama3.2:latest

Faster answers and reloading:
Answers appear as they are written, at a steady reading pace (about the speed of the voice, so the two stay in step), and with read-aloud on the voice starts with the first sentence. To change the pace, set LOCHA_TEXT_SPEED to the characters per second you want (default 22) before starting LoCha.
The first time you load a document, LoCha indexes it and saves the index on your computer. Loading the same file again (even after restarting LoCha) skips re-indexing and is almost instant; a summary you already made is kept too. A changed file counts as a new document.
The saved indexes are in %LOCALAPPDATA%\LoCha\index on Windows (~/.cache/LoCha/index on macOS/Linux). They are small, stay on your computer, and can be deleted at any time to free space.

Build LoCha.exe (Windows):
1. Set up the project as in "Run from source" (Python 3.11, .venv, pip install -r requirements.txt) and check LoCha runs.
2. In PowerShell, from the LoCha folder: .\build_exe.ps1
   (or: pip install pyinstaller==6.22.3, then: pyinstaller LoCha.spec --noconfirm --clean)
3. The app is in dist\LoCha. Start it with dist\LoCha\LoCha.exe. Keep the whole dist\LoCha folder together: LoCha.exe needs the _internal folder and LoChaVoice.exe (the natural voice) next to it.
To use LoCha on another PC, copy (or zip) the dist\LoCha folder. That PC also needs Ollama with the model (ollama pull llama3.2:latest), and an internet connection the first time, to download the search, speech and voice models.
LoCha.exe has no console window: its messages go to %LOCALAPPDATA%\LoCha\locha.log, which is the place to look if something goes wrong.
Windows may show a SmartScreen warning the first time, because the exe isn't code-signed: click "More info", then "Run anyway".
The build is also tested automatically on a Windows machine by GitHub Actions (.github/workflows/windows-exe.yml). It runs only when started from the Actions tab or by a commit whose message contains [build-exe].
