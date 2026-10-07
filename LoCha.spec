# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for LoCha.exe. Build on Windows with build_exe.ps1
# (or: pyinstaller LoCha.spec --noconfirm). Output: dist\LoCha\LoCha.exe
#
# Two programs share the dist\LoCha folder:
# - LoCha.exe: the app (PySide6, torch, LangChain, speech recognition).
# - LoChaVoice.exe: the natural-voice worker (Piper + onnxruntime).
# They must be built separately: on Windows onnxruntime crashes when loaded
# alongside torch/PySide6, and PyInstaller imports all of an exe's packages
# together while building, so a single exe with both fails to build.
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)

# ---------------- LoCha.exe ----------------
datas = [("LoCha_app/LoCha_icon.ico", ".")]  # app.py looks for it next to the exe
binaries = []
hiddenimports = [
    # Imported lazily (by name) by langchain_community.
    "langchain_community.llms.ollama",
    "langchain_community.embeddings.huggingface",
    "langchain_community.vectorstores.faiss",
    "langchain_community.document_loaders.pdf",
    "langchain_community.document_loaders.word_document",
    # Windows system voice (fallback when the natural voice is unavailable).
    "pyttsx3.drivers",
    "pyttsx3.drivers.sapi5",
    "comtypes.client",
]

# langchain.chains loads its chains by name (lazy __getattr__).
hiddenimports += collect_submodules("langchain.chains.conversational_retrieval")

# The embedding model's config names these modules; they're loaded by name.
hiddenimports += collect_submodules("sentence_transformers")

# Speech recognition.
datas += collect_data_files("faster_whisper")
binaries += collect_dynamic_libs("ctranslate2")

app = Analysis(
    ["run_locha.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # onnxruntime and Piper belong to LoChaVoice.exe only (see above). LoCha
    # doesn't use faster-whisper's VAD or transformers' ONNX export, which
    # are what would pull onnxruntime in.
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "onnxruntime", "piper"],
    noarchive=False,
)
app_exe = EXE(
    PYZ(app.pure),
    app.scripts,
    [],
    exclude_binaries=True,
    name="LoCha",
    icon="LoCha_app/LoCha_icon.ico",
    console=False,  # windowed app; output goes to %LOCALAPPDATA%\LoCha\locha.log
    upx=False,
)

# ---------------- LoChaVoice.exe ----------------
voice = Analysis(
    ["LoCha_app/tts_worker.py"],
    pathex=[],
    binaries=collect_dynamic_libs("piper"),
    datas=collect_data_files("piper", excludes=["**/train/**"]),
    hiddenimports=collect_submodules("piper", filter=lambda name: not name.startswith("piper.train")),
    # Keep the app's heavy packages out (and away from onnxruntime).
    excludes=[
        "tkinter", "matplotlib", "IPython", "pytest", "piper.train",
        "torch", "torchvision", "transformers", "sentence_transformers",
        "PySide6", "shiboken6", "langchain", "langchain_core", "langchain_community",
        "scipy", "sklearn", "pandas", "faster_whisper", "ctranslate2",
    ],
    noarchive=False,
)
voice_exe = EXE(
    PYZ(voice.pure),
    voice.scripts,
    [],
    exclude_binaries=True,
    name="LoChaVoice",
    console=False,
    upx=False,
)

coll = COLLECT(
    app_exe, app.binaries, app.datas,
    voice_exe, voice.binaries, voice.datas,
    name="LoCha",
    upx=False,
)
