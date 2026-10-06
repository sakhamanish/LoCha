import sys
import os
import re
import html
import threading
from PySide6.QtGui import QIcon, QTextCursor, QPixmap, QTextCharFormat, QFont
from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QLineEdit, QMessageBox, QHBoxLayout,
    QCheckBox
)

from LoCha_app.rag_engine import LoChaEngine, check_ollama, format_match
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

def text_to_html(text):
    """Escapes model/user text for the rich-text answer box, keeping line
    breaks and **bold** (otherwise "a < b" vanishes and lists run together)."""
    escaped = html.escape(text or "").strip()
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return escaped.replace("\n", "<br>")


MATCH_COLORS = {"strong": "#2E7D32", "moderate": "#E65100", "weak": "#C62828"}

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
    answer_started = Signal(str)  # question HTML
    token = Signal(str)           # answer text as the model writes it
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
            question_html = (
                f'<span style="font-size:16px; font-weight:bold;">'
                f'Question: {text_to_html(self.question)}</span><br>'
            )
            self.answer_started.emit(question_html)
            qa = engine.ask(self.question, on_token=self.token.emit)

            match_color = MATCH_COLORS.get(qa["match"], "#555")
            answer_html = (
                f'<div style="background-color:#f2f2f2; padding:8px;">'
                f'<p style="font-size:16px;"><b>Answer:</b><br>{text_to_html(qa["answer"])}</p>'
                f'<p style="background-color:#b6f2a1; padding:4px;">'
                f'<b>References:</b> {text_to_html(qa["citations"]) or "Not available"}</p>'
                f'<p><b>Source match:</b> '
                f'<span style="color:{match_color};">{format_match(qa)}</span></p>'
                f'</div>'
            )

            self.result_ready.emit(question_html, answer_html)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class SummarizeWorker(QObject):
    finished = Signal()
    status_update = Signal(str)
    writing_started = Signal()  # a new summary is being written
    token = Signal(str)         # summary text as the model writes it
    result_ready = Signal(str)
    error = Signal(str)

    def run(self):
        try:
            if not engine.summary:  # cached summaries appear instantly
                self.status_update.emit(
                    "<b style='font-size:16px; color:#2E7D32;'>Summarizing the document…</b>"
                )
                self.writing_started.emit()
            self.result_ready.emit(engine.summarize(on_token=self.token.emit))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# Streamed text is shown at a steady reading pace instead of as fast as the
# model writes it; this is also about the voice's speaking rate, so with
# read-aloud on the text and voice stay roughly in step. Characters/second.
TEXT_SPEED = float(os.environ.get("LOCHA_TEXT_SPEED", "22"))
REVEAL_INTERVAL_MS = 40

SUMMARY_HEADER = '<span style="font-size:16px; font-weight:bold;">Document summary</span><br>'


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
        """text: a string, or a voice.LiveSpeech still receiving the answer."""
        super().__init__()
        self.text = text
        self.speaker = voice.Speaker()

    def run(self):
        try:
            if isinstance(self.text, voice.LiveSpeech):
                self.speaker.speak_live(self.text)
            else:
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
        self.summarize_btn = QPushButton("Summarize")
        self.save_btn = QPushButton("Save Conversation")

        for btn in (self.load_btn, self.summarize_btn, self.save_btn):
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
                QPushButton:disabled {
                    background-color: #A0A0A0;
                }
            """)

        # ---------- Read-aloud controls ----------
        self.read_aloud_cb = QCheckBox("🔊 Read answers aloud")
        self.read_aloud_cb.setStyleSheet("font-size: 16px;")
        self.stop_speaking_btn = QPushButton("Stop speaking")
        self.stop_speaking_btn.setFixedSize(140, 40)
        self.stop_speaking_btn.setEnabled(False)
        if not voice.READ_ALOUD_AVAILABLE:
            self.read_aloud_cb.setEnabled(False)
            self.read_aloud_cb.setToolTip(
                "Read-aloud unavailable. Install it with: pip install piper-tts pyttsx3"
            )

        self.summarize_btn.setToolTip("Get a brief overview of the whole document")

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.summarize_btn)
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
            QPushButton:disabled {
                background-color: #A0A0A0;
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
        self.summarize_btn.clicked.connect(self.summarize_document)
        self.question_input.returnPressed.connect(self.ask_question)
        self.mic_btn.clicked.connect(self.toggle_recording)
        self.stop_speaking_btn.clicked.connect(self.stop_speaking)

        self.voice_thread = None
        self.voice_worker = None
        self.tts_worker = None
        self.tts_jobs = []  # keeps (thread, worker) alive until speech ends
        self.asking = False  # loading, answering or summarizing
        self.live_cursor = None  # answer being streamed, if any
        self.live_has_text = False
        self.live_speech = None  # sentences being read aloud as they arrive
        self.live_status = ""
        self.live_result_received = False
        self.reveal_buffer = ""  # streamed text not yet shown
        self.reveal_credit = 0.0
        self.after_reveal = []
        self.reveal_timer = QTimer(self)
        self.reveal_timer.setInterval(REVEAL_INTERVAL_MS)
        self.reveal_timer.timeout.connect(self.reveal_tick)

        QTimer.singleShot(0, self.check_ollama_on_start)

    def check_ollama_on_start(self):
        try:
            check_ollama()
        except RuntimeError as e:
            first_line = html.escape(str(e).split("\n")[0])
            self.status.setText(
                f"<b style='font-size:16px; color:#C62828;'>{first_line}</b> "
                "<span style='font-size:14px; color:#555;'>Questions and summaries "
                "need it; you can still load a document.</span>"
            )

    def set_busy(self, busy):
        """Only one load/answer/summary at a time: starting another while a
        worker thread runs would destroy that running thread and crash."""
        self.asking = busy
        for btn in (self.load_btn, self.summarize_btn, self.ask_btn):
            btn.setEnabled(not busy)
        self.update_mic_state()

    # ---------------- Logic ----------------
    def load_document(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Documents (*.pdf *.docx)"
        )
        if not path or self.asking:
            return

        self.answer_box.clear()
        engine.reset()

        self.thread = QThread()
        self.worker = LoadDocumentWorker(path)
        self.worker.moveToThread(self.thread)

        self.set_busy(True)

        self.thread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.update_status)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda: self.set_busy(False))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def ask_question(self):
        question = self.question_input.text().strip()
        if not question or self.asking:
            return
        if not engine.index_ready:
            QMessageBox.warning(self, "Warning", "Document not loaded yet.")
            return

        self.thread = QThread()
        self.worker = AskQuestionWorker(question)
        self.worker.moveToThread(self.thread)

        self.set_busy(True)

        self.thread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.update_status)
        self.worker.answer_started.connect(self.start_live_answer)
        self.worker.token.connect(self.append_live_answer)
        self.worker.result_ready.connect(self.display_answer)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_ask_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    # ---------- Streaming answer ----------
    def start_live_answer(self, question_html):
        self.start_live(
            question_html
            + '<br><span style="font-size:16px; font-weight:bold;">Answer:</span><br>'
        )

    def start_live_summary(self):
        self.start_live(SUMMARY_HEADER, "Writing the summary…")

    def start_live(self, header_html, writing_status="Writing the answer…"):
        """Shows the header and an empty text at the top; tokens are
        appended as plain text until the formatted result replaces it.
        With read-aloud on, sentences are spoken as soon as they're complete."""
        cursor = QTextCursor(self.answer_box.document())
        cursor.movePosition(QTextCursor.Start)
        # Two empty blocks keep the live answer apart from older answers.
        cursor.insertBlock()
        cursor.insertBlock()
        cursor.movePosition(QTextCursor.Start)
        cursor.insertHtml(header_html)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(12)
        fmt.setFontWeight(QFont.Normal)
        cursor.setCharFormat(fmt)
        self.live_cursor = cursor
        self.live_has_text = False
        self.live_status = writing_status
        self.live_result_received = False
        self.reveal_buffer = ""
        self.reveal_credit = 0.0
        self.answer_box.moveCursor(QTextCursor.Start)
        if self.read_aloud_cb.isChecked() and voice.READ_ALOUD_AVAILABLE:
            self.live_speech = voice.LiveSpeech()
            self.speak(self.live_speech)

    def append_live_answer(self, token):
        if self.live_cursor is None:
            return
        if not self.live_has_text:
            self.live_has_text = True
            self.status.setText(
                f"<b style='font-size:16px; color:#2E7D32;'>{self.live_status}</b>"
            )
        # The voice gets text as soon as it's written; the screen shows it
        # at TEXT_SPEED (see reveal_tick).
        self.reveal_buffer += token
        if not self.reveal_timer.isActive():
            self.reveal_timer.start()
        if self.live_speech is not None:
            self.live_speech.feed(token)

    def reveal_tick(self):
        if self.live_cursor is not None and self.reveal_buffer:
            self.reveal_credit += TEXT_SPEED * REVEAL_INTERVAL_MS / 1000
            count = int(self.reveal_credit)
            if count:
                self.reveal_credit -= count
                self.live_cursor.insertText(self.reveal_buffer[:count])
                self.reveal_buffer = self.reveal_buffer[count:]
        if not self.reveal_buffer:
            self.reveal_timer.stop()
            callbacks, self.after_reveal = self.after_reveal, []
            for callback in callbacks:
                callback()

    def when_revealed(self, callback):
        """Runs callback once all streamed text is on screen."""
        if self.reveal_buffer:
            self.after_reveal.append(callback)
        else:
            callback()

    def finish_live_speech(self, full_text):
        """Speaks the rest once the text is complete (or all of it, if
        nothing was streamed). Returns False if live speech wasn't used."""
        if self.live_speech is None:
            return False
        if not self.live_has_text:
            self.live_speech.feed(full_text)
        self.live_speech.finish()
        self.live_speech = None
        return True

    def remove_live_answer(self):
        if self.live_cursor is None:
            return
        doc = self.answer_box.document()
        end = min(self.live_cursor.position() + 2, doc.characterCount() - 1)
        cursor = QTextCursor(doc)
        cursor.setPosition(0)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        self.live_cursor = None

    def display_answer(self, q_html, a_html):
        self.live_result_received = True
        answer = engine.chat[0]["answer"] if engine.chat else ""
        if not self.finish_live_speech(answer) and self.read_aloud_cb.isChecked() and answer:
            self.speak(answer)
        self.when_revealed(lambda: self.show_answer(q_html, a_html))

    def show_answer(self, q_html, a_html):
        self.remove_live_answer()
        self.answer_box.moveCursor(QTextCursor.Start)
        self.answer_box.insertHtml(q_html + a_html + "<hr>")
        self.question_input.clear()
        self.status.setText(
            "<b style='font-size:16px; color:#2E7D32;'>Ready for another question.</b>"
        )

    def summarize_document(self):
        if not engine.index_ready:
            QMessageBox.warning(self, "Warning", "Document not loaded yet.")
            return
        if self.asking:
            return

        self.set_busy(True)

        self.summary_thread = QThread()
        self.summary_worker = SummarizeWorker()
        self.summary_worker.moveToThread(self.summary_thread)

        self.summary_thread.started.connect(self.summary_worker.run)
        self.summary_worker.status_update.connect(self.update_status)
        self.summary_worker.writing_started.connect(self.start_live_summary)
        self.summary_worker.token.connect(self.append_live_answer)
        self.summary_worker.result_ready.connect(self.display_summary)
        self.summary_worker.error.connect(self.show_error)
        self.summary_worker.finished.connect(self.on_summary_finished)
        self.summary_worker.finished.connect(self.summary_thread.quit)
        self.summary_worker.finished.connect(self.summary_worker.deleteLater)
        self.summary_thread.finished.connect(self.summary_thread.deleteLater)

        self.summary_thread.start()

    def display_summary(self, summary):
        self.live_result_received = True
        if not self.finish_live_speech(summary) and self.read_aloud_cb.isChecked():
            self.speak(summary)
        self.when_revealed(lambda: self.show_summary(summary))

    def show_summary(self, summary):
        lines = []
        for line in summary.splitlines():
            line = line.strip()
            if not line:
                continue
            bullet = line[:2] in ("- ", "* ", "• ")
            line = text_to_html(line[2:] if bullet else line)
            lines.append(f"• {line}" if bullet else line)
        summary_html = (
            SUMMARY_HEADER
            + '<div style="background-color:#e8f0fe; padding:8px;">'
            f'<p style="font-size:16px;">{"<br>".join(lines)}</p>'
            '</div>'
        )
        self.remove_live_answer()
        self.answer_box.moveCursor(QTextCursor.Start)
        self.answer_box.insertHtml(summary_html + "<hr>")
        self.status.setText(
            "<b style='font-size:16px; color:#2E7D32;'>Summary ready. Ask a question for details.</b>"
        )

    def on_summary_finished(self):
        self.abort_live()
        self.when_revealed(lambda: self.set_busy(False))

    def abort_live(self):
        """A live text or live speech left over when its worker finishes
        without a result means it failed: remove the partial text and stop
        speaking."""
        if self.live_result_received:
            return
        self.reveal_buffer = ""
        self.after_reveal = []
        self.reveal_timer.stop()
        if self.live_speech is not None:
            self.live_speech.finish()
            self.live_speech = None
            self.stop_speaking()
        self.remove_live_answer()

    def on_ask_finished(self):
        self.abort_live()
        self.when_revealed(lambda: self.set_busy(False))

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
            self.mic_btn.setToolTip("Please wait until LoCha has finished")
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
        voice.shutdown()
        super().closeEvent(event)

    def update_status(self, msg):
        self.status.setText(msg)

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.status.setText("Error")

    def save_chat(self):
        if not engine.chat and not engine.summary:
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
    # Load the embedding model while the window opens, so the first
    # document (especially a cached one) loads quickly.
    threading.Thread(target=engine.warm_up, daemon=True).start()
    app = QApplication(sys.argv)
    window = LoChaApp()
    window.show()
    sys.exit(app.exec())