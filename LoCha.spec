# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for LoCha.exe. Build on Windows with build_exe.ps1
# (or: pyinstaller LoCha.spec --noconfirm). Output: dist\LoCha\LoCha.exe
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)

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

# Natural voice: Piper code (not its training tools), espeak-ng data and libraries.
hiddenimports += collect_submodules("piper", filter=lambda name: not name.startswith("piper.train"))
datas += collect_data_files("piper", excludes=["**/train/**"])
binaries += collect_dynamic_libs("piper")

# Speech recognition.
datas += collect_data_files("faster_whisper")
binaries += collect_dynamic_libs("ctranslate2")

a = Analysis(
    ["run_locha.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "piper.train"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LoCha",
    icon="LoCha_app/LoCha_icon.ico",
    console=False,  # windowed app; output goes to %LOCALAPPDATA%\LoCha\locha.log
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="LoCha", upx=False)
