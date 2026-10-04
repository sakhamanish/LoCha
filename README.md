LoCha is a local chat assistant skilled to answer the questions relevant to the given document. It uses retrieval-augmented generation (RAG) method to provide pertinent answers. The primary objective of LoCha is to maintain privacy of your documents. Once installed in your system, LoCha uses your local computer resources to generate the answers to your uploaded documents.

Prerequisites:
- Ollama installed
- llama3.2:latest model downloaded

Install:
1. Download Ollama from https://ollama.com
2. Run: ollama pull llama3.2:latest
3. Launch LoCha






Voice mode (optional):
You can ask questions by voice and have answers read aloud. Everything runs offline on your computer.
1. Install the voice packages: pip install sounddevice faster-whisper pyttsx3
   (on Linux also install PortAudio and eSpeak, e.g. sudo apt install libportaudio2 libespeak1)
2. Load a document as usual.
3. Click "🎤 Speak", ask your question out loud, then click "⏹ Stop". LoCha transcribes it and asks it for you.
4. Tick "🔊 Read answers aloud" to hear each answer. Use "Stop speaking" to cut it short.
The first time you use voice input, the Whisper speech model (base.en, about 150 MB) is downloaded once and cached. After that it works without internet.
If the voice packages are not installed, the voice controls are greyed out and the rest of the app works as before.
