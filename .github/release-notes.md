LoCha is a local chat assistant for your documents: it answers questions about a PDF or DOCX using AI that runs entirely on your computer.

## Downloads
- **LoCha-…-windows.zip**: ready to run on Windows, no Python needed. Unzip it and double-click `LoCha.exe`. Keep the whole folder together (`LoCha.exe` needs the `_internal` folder and `LoChaVoice.exe` next to it).
- **Source code**: the Python project, to run with `python -m LoCha_app.app` (see the README).

## Before you start
- Install [Ollama](https://ollama.com) and download the language model once: `ollama pull llama3.2:latest`. Ollama must be running while you use LoCha.
- The first start needs internet once, to download the search, speech-recognition and voice models. After that LoCha works offline.
- Windows may show a SmartScreen warning because the app isn't code-signed: click *More info*, then *Run anyway*.
- If something goes wrong, check `%LOCALAPPDATA%\LoCha\locha.log`.

## What's in this version
- Ask questions about a PDF or DOCX, with references and a source match score (0.00–1.00).
- Answers appear as they're written, at a steady reading pace.
- Voice mode: ask by speaking (🎤 Speak, then ⏹ Stop) and hear answers in a natural-sounding voice.
- Summarize: a brief overview of the whole document.
- Documents you've loaded before reopen almost instantly.
- Save the conversation (including the summary) to a text file.
