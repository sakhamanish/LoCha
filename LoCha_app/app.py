import sys
import os
from PySide6.QtGui import QIcon, QTextCursor, QPixmap
from PySide6.QtCore import Qt, Signal, QObject, QThread

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QLineEdit, QMessageBox, QHBoxLayout,
    QCheckBox
)

from LoCha_app.rag_engine import LoChaEngine
from LoCha_app import voice

# -------------------------------------------------
# Resource helper (works for dev + PyInstaller)
# -------------------------------------------------
def resource_path(relative_path):
    """Get absolute path to resource, works in dev and PyInstaller."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# Path to the icon
ICON_PATH = resource_path("LoCha_icon.ico")

if not os.path.exists(ICON_PATH):
    print("⚠️ ICON NOT FOUND:", ICON_PATH)

# -------------------------------------------------
# Engine
# -------------------------------------------------
engine = LoChaEngine()
recorder = voice.VoiceRecorder()
transcriber = voice.Transcriber()

# ---------------- Worker Classes ----------------
class LoadDocumentWorker(QObject):
    finished = Signal()
    status_update = Signal(str)
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.status_update.emit(
                f"<b style='font-size:16px; color:#2E7D32;'>Reading…</b> "
                f"<span style='font-size:14px; color:#555;'>"
                f"{os.path.basename(self.file_path)}</span>"
            )
            engine.load_document(self.file_path)
            transcriber.set_vocabulary(
                voice.extract_vocabulary(engine.document_text)
            )
            self.status_update.emit(
                "<b style='font-size:16px; color:#2E7D32;'>Read and indexed successfully, ready to take questions.</b>"
            )
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class AskQuestionWorker(QObject):
    finished = Signal()
    status_update = Signal(str)
    result_ready = Signal(str, str)
    error = Signal(str)

    def __init__(self, question):
        super().__init__()
        self.question = question

    def run(self):
        try:
            self.status_update.emit(
                "<b style='font-size:16px; color:#2E7D32;'>Contemplating question...</b>"
            )
            qa = engine.ask(self.question)

            question_html = (
                f'<span style="font-size:16px; font-weight:bold;">'
                f'Question: {qa["question"]}</span><br>'
            )

            answer_html = (
                f'<div style="background-color:#f2f2f2; padding:8px;">'
                f'<p style="font-size:16px;"><b>Answer:</b> {qa["answer"]}</p>'
                f'<p style="background-color:#b6f2a1; padding:4px;">'
                f'<b>References:</b> {qa["citations"] or "Not available"}</p>'
                f'<p><b>Confidence:</b> {qa["confidence"]}%</p>'
                f'</div>'
            )

            self.result_ready.emit(question_html, answer_html)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class TranscribeWorker(QObject):
    finished = Signal()
    text_ready = Signal(str)
    error = Signal(str)

    def __init__(self, audio):
        super().__init__()
        self.audio = audio

    def run(self):
        try:
            self.text_ready.emit(transcriber.transcribe(self.audio))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class SpeakWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.speaker = voice.Speaker()

    def run(self):
        try:
            self.speaker.speak(self.text)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


MIC_IDLE_STYLE = """
    QPushButton {
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #4A90E2;
        border-radius: 8px;
    }
    QPushButton:hover {
        background-color: #357ABD;
    }
    QPushButton:disabled {
        background-color: #A0A0A0;
    }
"""

MIC_RECORDING_STYLE = """
    QPushButton {
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #D32F2F;
        border-radius: 8px;
    }
    QPushButton:hover {
        background-color: #B71C1C;
    }
"""


# ---------------- Main GUI ----------------
class LoChaApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LoCha: Local Document Chat")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1300, 850)

        layout = QVBoxLayout(self)

        # ---------- Title ----------
        title_layout = QHBoxLayout()
        title_layout.addStretch()

        icon_label = QLabel()
        pixmap = QPixmap(ICON_PATH)
        pixmap = pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(pixmap)

        title_text = QLabel("LoCha")
        title_text.setStyleSheet("font-size: 30px; font-weight: bold;")

        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_text)
        title_layout.addStretch()

        layout.addLayout(title_layout)

        # ---------- Status ----------
        self.status = QLabel(
            "<b style='font-size:16px; color:#2E7D32;'>Load a document to get started...</b>"
        )
        layout.addWidget(self.status)

        # ---------- Buttons ----------
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load PDF / DOCX")
        self.save_btn = QPushButton("Save Conversation")

        for btn in (self.load_btn, self.save_btn):
            btn.setFixedSize(200, 50)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    font-weight: bold;
                    color: white;
                    background-color: #4A90E2;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #357ABD;
                }
            """)

        # ---------- Read-aloud controls ----------
        self.read_aloud_cb = QCheckBox("🔊 Read answers aloud")
        self.read_aloud_cb.setStyleSheet("font-size: 16px;")
        self.stop_speaking_btn = QPushButton("Stop speaking")
        self.stop_speaking_btn.setFixedSize(140, 40)
        self.stop_speaking_btn.setEnabled(False)
        if not voice.TTS_AVAILABLE:
            self.read_aloud_cb.setEnabled(False)
            self.read_aloud_cb.setToolTip(
                "Read-aloud unavailable. Install it with: pip install pyttsx3"
            )

        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.read_aloud_cb)
        btn_layout.addWidget(self.stop_speaking_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        # ---------- Question ----------
        question_layout = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about the document")
        self.question_input.setStyleSheet("font-size: 18px; padding: 12px;")
        question_layout.addWidget(self.question_input)

        self.mic_btn = QPushButton("🎤 Speak")
        self.mic_btn.setFixedSize(140, 50)
        self.mic_btn.setStyleSheet(MIC_IDLE_STYLE)
        if voice.STT_AVAILABLE:
            self.mic_btn.setToolTip("Click, ask your question out loud, then click Stop")
        else:
            self.mic_btn.setEnabled(False)
            self.mic_btn.setToolTip(
                "Voice input unavailable. Install it with: "
                "pip install sounddevice faster-whisper"
            )
        question_layout.addWidget(self.mic_btn)

        self.ask_btn = QPushButton("Ask")
        self.ask_btn.setFixedSize(100, 50)
        self.ask_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: #4A90E2;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        question_layout.addWidget(self.ask_btn)
        layout.addLayout(question_layout)

        # ---------- Answer ----------
        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setStyleSheet("font-size: 14px; background-color: #ffffff;")
        layout.addWidget(self.answer_box)

        # ---------- Signals ----------
        self.load_btn.clicked.connect(self.load_document)
        self.ask_btn.clicked.connect(self.ask_question)
        self.save_btn.clicked.connect(self.save_chat)
        self.question_input.returnPressed.connect(self.ask_question)
        self.mic_btn.clicked.connect(self.toggle_recording)
        self.stop_speaking_btn.clicked.connect(self.stop_speaking)

        self.voice_thread = None
        self.voice_worker = None
        self.tts_worker = None
        self.tts_jobs = []  # keeps (thread, worker) alive until speech ends
        self.asking = False

    # ---------------- Logic ----------------
    def load_document(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Documents (*.pdf *.docx)"
        )
        if not path:
            return

        self.answer_box.clear()
        engine.reset()

        self.thread = QThread()
        self.worker = LoadDocumentWorker(path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.update_status)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def ask_question(self):
        question = self.question_input.text().strip()
        if not question:
            return
        if not engine.index_ready:
            QMessageBox.warning(self, "Warning", "Document not loaded yet.")
            return

        self.thread = QThread()
        self.worker = AskQuestionWorker(question)
        self.worker.moveToThread(self.thread)

        self.asking = True
        self.update_mic_state()

        self.thread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.update_status)
        self.worker.result_ready.connect(self.display_answer)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_ask_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def display_answer(self, q_html, a_html):
        self.answer_box.moveCursor(QTextCursor.Start)
        self.answer_box.insertHtml(q_html + a_html + "<hr>")
        self.question_input.clear()
        self.status.setText(
            "<b style='font-size:16px; color:#2E7D32;'>Ready for another question.</b>"
        )
        if self.read_aloud_cb.isChecked() and engine.chat:
            self.speak(engine.chat[0]["answer"])

    def on_ask_finished(self):
        self.asking = False
        self.update_mic_state()

    # ---------------- Voice ----------------
    def update_mic_state(self):
        """Speak is unavailable while an answer is generated or read aloud,
        so the microphone never records LoCha's own voice."""
        if not voice.STT_AVAILABLE or recorder.is_recording:
            return
        if self.mic_btn.text() != "🎤 Speak":  # transcribing
            return
        if self.asking:
            self.mic_btn.setEnabled(False)
            self.mic_btn.setToolTip("Wait for the answer")
        elif self.tts_worker is not None:
            self.mic_btn.setEnabled(False)
            self.mic_btn.setToolTip("Wait for the answer to finish, or press Stop speaking")
        else:
            self.mic_btn.setEnabled(True)
            self.mic_btn.setToolTip("Click, ask your question out loud, then click Stop")

    def toggle_recording(self):
        if recorder.is_recording:
            self.finish_recording()
            return

        if not engine.index_ready:
            QMessageBox.warning(self, "Warning", "Document not loaded yet.")
            return

        self.stop_speaking()
        try:
            recorder.start()
        except Exception as e:
            self.show_error(f"Could not access the microphone:\n{e}")
            return

        self.mic_btn.setText("⏹ Stop")
        self.mic_btn.setStyleSheet(MIC_RECORDING_STYLE)
        self.status.setText(
            "<b style='font-size:16px; color:#D32F2F;'>Listening… "
            "ask your question, then click Stop.</b>"
        )

    def finish_recording(self):
        audio = recorder.stop()

        self.mic_btn.setText("Transcribing…")
        self.mic_btn.setStyleSheet(MIC_IDLE_STYLE)
        self.mic_btn.setEnabled(False)
        self.status.setText(
            "<b style='font-size:16px; color:#2E7D32;'>Transcribing your question…</b>"
        )

        self.voice_thread = QThread()
        self.voice_worker = TranscribeWorker(audio)
        self.voice_worker.moveToThread(self.voice_thread)

        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.text_ready.connect(self.on_transcribed)
        self.voice_worker.error.connect(self.on_transcribe_error)
        self.voice_worker.finished.connect(self.voice_thread.quit)
        self.voice_worker.finished.connect(self.voice_worker.deleteLater)
        self.voice_thread.finished.connect(self.voice_thread.deleteLater)

        self.voice_thread.start()

    def reset_mic_button(self):
        self.mic_btn.setText("🎤 Speak")
        self.mic_btn.setStyleSheet(MIC_IDLE_STYLE)
        self.update_mic_state()

    def on_transcribed(self, text):
        self.reset_mic_button()
        if not text:
            self.status.setText(
                "<b style='font-size:16px; color:#D32F2F;'>"
                "Didn't catch that. Click Speak and try again.</b>"
            )
            return
        # Send straight away; the transcript stays visible in the question
        # box and is shown above the answer.
        self.question_input.setText(text)
        self.ask_question()

    def on_transcribe_error(self, msg):
        self.reset_mic_button()
        self.show_error(f"Transcription failed:\n{msg}")

    def speak(self, text):
        self.stop_speaking()

        thread = QThread()
        worker = SpeakWorker(text)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.error.connect(self.show_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda: self.on_speaking_finished(thread, worker))

        self.tts_worker = worker
        self.tts_jobs.append((thread, worker))
        self.stop_speaking_btn.setEnabled(True)
        self.update_mic_state()
        thread.start()

    def stop_speaking(self):
        if self.tts_worker is not None:
            self.tts_worker.speaker.stop()
            self.stop_speaking_btn.setEnabled(False)

    def on_speaking_finished(self, thread, worker):
        self.tts_jobs.remove((thread, worker))
        worker.deleteLater()
        thread.deleteLater()
        if worker is self.tts_worker:
            self.tts_worker = None
            self.stop_speaking_btn.setEnabled(False)
            self.update_mic_state()

    def closeEvent(self, event):
        if recorder.is_recording:
            recorder.stop()
        self.stop_speaking()
        super().closeEvent(event)

    def update_status(self, msg):
        self.status.setText(msg)

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.status.setText("Error")

    def save_chat(self):
        if not engine.chat:
            QMessageBox.information(self, "Info", "No conversation to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Conversation", "LoCha_Conversation", "Text Files (*.txt)"
        )
        if path:
            engine.save_conversation(path)
            QMessageBox.information(self, "Saved", f"Saved to {path}")


# ---------------- Entry ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoChaApp()
    window.show()
    sys.exit(app.exec())