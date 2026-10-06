import os
import json
import hashlib
import logging
import threading
import urllib.request

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler

LLM_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Each document's search index, text and summary are cached here, keyed by
# a hash of the file's contents, so reopening it skips parsing and
# embedding. Changing the version, model or chunking invalidates the cache.
INDEX_CACHE_VERSION = f"v1-{EMBEDDING_MODEL}-{CHUNK_SIZE}-{CHUNK_OVERLAP}"
CACHE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"),
    "LoCha", "index",
)
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


def check_ollama(model=LLM_MODEL):
    """
    Raises RuntimeError with a plain-language fix if the Ollama server is
    not running or the model has not been downloaded.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            installed = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        raise RuntimeError(
            "Ollama isn't running.\n\nStart the Ollama app (or run 'ollama serve' "
            "in a terminal), then try again. If Ollama isn't installed, get it "
            "from https://ollama.com"
        )
    base = model.split(":")[0]
    if model not in installed and not (model.endswith(":latest") and base in installed):
        raise RuntimeError(
            f"The language model '{model}' isn't downloaded yet.\n\n"
            f"Run this in a terminal, then try again:\n  ollama pull {model}"
        )


from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ---------------- Prompts ----------------
# The LangChain default answer prompt tells the model to say "I don't know"
# whenever unsure, which small local models (llama3.2 3B) do for almost
# every question. This one asks it to use whatever the excerpts contain.
QA_PROMPT = PromptTemplate.from_template(
    """You are LoCha, an assistant that answers questions about the user's document.
Answer the question using the document excerpts below.
- Use the information in the excerpts even if it only partly answers the question, and say what is not covered.
- The question may have been spoken and transcribed, so it can contain small transcription errors; interpret it sensibly.
- Only if none of the excerpts relate to the question, reply: "I couldn't find that in the document."
- Do not invent facts that are not in the excerpts.

Document excerpts:
{context}

Question: {question}
Answer:"""
)

# Turns a follow-up ("what about the second one?") into a standalone
# question. Small models tend to wrap the rewrite in chatter ("Here is the
# rephrased question: ..."), hence the strict "question only" instruction.
CONDENSE_PROMPT = PromptTemplate.from_template(
    """Rewrite the follow-up question as a standalone question, using the conversation for context.
If it is already a standalone question, repeat it unchanged.
Reply with the question only, nothing else.

Conversation:
{chat_history}

Follow-up question: {question}
Standalone question:"""
)

# Summaries are written in a single model call (several sequential calls
# were slow on a CPU). Documents longer than SUMMARY_MAX_CHARS are
# represented by their opening passage plus the most typical passage of
# each topic (clusters of passage embeddings), in document order.
SUMMARY_MAX_CHARS = 9000   # about 2300 tokens
SUMMARY_TOPICS = 10

SUMMARY_PROMPT = PromptTemplate.from_template(
    """Below are {kind} a document.
Write a brief summary for someone who has not read it: first one short
paragraph giving the gist of the whole document, then 3 to 5 bullet points
with its most important points. Use only information from the text below.
Do not add a title.

{text}

Summary:"""
)

# Source match: cosine similarity thresholds for all-MiniLM-L6-v2, where
# closely related passages typically score above ~0.55 and unrelated
# ones below ~0.3.
MATCH_STRONG = 0.55
MATCH_MODERATE = 0.35

class _AnswerStreamer(BaseCallbackHandler):
    """Passes the answer's tokens to on_token as the model writes them.
    Only calls whose prompt starts with prompt_prefix are streamed: the
    chain also calls the model to rewrite follow-up questions."""

    def __init__(self, on_token, prompt_prefix="You are LoCha"):
        self.on_token = on_token
        self.prompt_prefix = prompt_prefix
        self.active = False

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.active = bool(prompts) and prompts[0].startswith(self.prompt_prefix)

    def on_llm_new_token(self, token, **kwargs):
        if self.active and token:
            self.on_token(token)

    def on_llm_end(self, response, **kwargs):
        self.active = False


def format_match(qa: dict) -> str:
    if qa.get("match_score") is None:
        return "not available"
    return f"{qa['match_score']:.2f} ({qa['match']})"


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LoChaEngine] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class LoChaEngine:
    def __init__(self):
        self._embeddings = None
        self._embeddings_lock = threading.Lock()
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.chat = []
        self.file_hash = None
        self.index_ready = False
        self.document_text = ""
        self.llm = None
        self.summary = None
        self.cache_path = None

    def reset(self):
        logger.info("Resetting engine state")
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.chat = []
        self.file_hash = None
        self.index_ready = False
        self.document_text = ""
        self.llm = None
        self.summary = None
        self.cache_path = None

    def warm_up(self):
        """Loads the embedding model ahead of time (call in the background
        at startup) so the first document loads quickly."""
        try:
            self._get_embeddings()
        except Exception as e:
            logger.warning(f"Embedding model warm-up failed: {e}")

    def _get_embeddings(self):
        with self._embeddings_lock:
            if self._embeddings is None:
                # Normalized vectors make FAISS's squared L2 distance map
                # directly to cosine similarity (used for the source match).
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                )
            return self._embeddings

    def load_document(self, file_path: str):
        logger.info(f"Loading document: {file_path}")

        with open(file_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()

        self.reset()
        self.file_hash = digest
        self.cache_path = os.path.join(CACHE_DIR, digest)
        embeddings = self._get_embeddings()

        if not self._load_from_cache(embeddings):
            suffix = os.path.splitext(file_path)[1].lower()
            loader = PyPDFLoader(file_path) if suffix == ".pdf" else Docx2txtLoader(file_path)

            docs = loader.load()
            for d in docs:
                d.metadata["source"] = os.path.basename(file_path)
            self.document_text = "\n".join(d.page_content for d in docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)
            logger.info(f"Document split into {len(chunks)} chunks")

            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            self._save_to_cache()

        # Keep only the last few exchanges: the full history is sent with
        # every follow-up and would otherwise overflow the context window.
        self.memory = ConversationBufferWindowMemory(
            k=4,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Ollama's default context window is small; 5 excerpts plus the
        # prompt can overflow it, silently cutting off the instructions.
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2, num_ctx=8192)
        self.llm = llm

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            condense_question_prompt=CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_generated_question=True,
            return_source_documents=True,
            output_key="answer"
        )

        self.index_ready = True
        logger.info("Document indexed successfully")

    # ---------------- Index cache ----------------
    def _load_from_cache(self, embeddings) -> bool:
        path = self.cache_path
        try:
            with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
                if json.load(f).get("version") != INDEX_CACHE_VERSION:
                    return False
            # The cache only ever contains files LoCha wrote itself.
            self.vectorstore = FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
            with open(os.path.join(path, "document.txt"), encoding="utf-8") as f:
                self.document_text = f.read()
            summary_file = os.path.join(path, "summary.txt")
            if os.path.exists(summary_file):
                with open(summary_file, encoding="utf-8") as f:
                    self.summary = f.read() or None
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"Ignoring unreadable index cache ({e}); re-indexing")
            self.vectorstore = None
            self.document_text = ""
            self.summary = None
            return False
        logger.info("Loaded document index from cache (skipped re-indexing)")
        return True

    def _save_to_cache(self):
        path = self.cache_path
        try:
            os.makedirs(path, exist_ok=True)
            self.vectorstore.save_local(path)
            with open(os.path.join(path, "document.txt"), "w", encoding="utf-8") as f:
                f.write(self.document_text)
            # Written last: a cache entry only counts once it is complete.
            with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"version": INDEX_CACHE_VERSION}, f)
        except Exception as e:
            logger.warning(f"Could not cache the document index: {e}")

    def ask(self, question: str, on_token=None) -> dict:
        """on_token(text), if given, receives the answer as it is written."""
        if not self.index_ready:
            raise RuntimeError("Document not indexed yet")

        logger.info(f"Question received: {question}")
        check_ollama()
        callbacks = [_AnswerStreamer(on_token)] if on_token else []
        result = self.qa_chain.invoke(
            {"question": question}, config={"callbacks": callbacks}
        )

        answer = result["answer"]
        sources = result.get("source_documents", [])
        searched = result.get("generated_question") or question
        logger.info(f"Searched document for: {searched}")
        logger.info(
            "Retrieved excerpts from: "
            + ", ".join(
                f"page {d.metadata['page'] + 1}" if "page" in d.metadata else d.metadata.get("source", "?")
                for d in sources
            )
        )

        # ---------------- Citations ----------------
        citations = {}
        for doc in sources:
            src = doc.metadata.get("source", "Unknown")
            suffix = os.path.splitext(src)[1].lower()
            if suffix == ".pdf":
                # For PDF, keep page number
                page = doc.metadata.get("page", 0) + 1
                citations.setdefault(src, set()).add(page)
            else:
                # For DOCX, take first 2-3 lines as reference snippet
                text_snippet = "\n".join(doc.page_content.splitlines()[:3]).strip()
                # truncate if too long
                if len(text_snippet) > 100:
                    text_snippet = text_snippet[:100] + "..."
                citations.setdefault(src, set()).add(f'"{text_snippet}"')

        # Format citation text
        citation_text = " ".join(
            f"[{src} {', '.join(map(str, sorted(pages))) if all(isinstance(p, int) for p in pages) else ', '.join(pages)}]"
            for src, pages in citations.items()
        )

        # ---------------- Source match ----------------
        # How closely the best passage matches the question that was
        # searched (cosine similarity of the normalized embeddings).
        match_score, match = self._source_match(searched) if sources else (0.0, "weak")

        qa = {
            "question": question,
            "answer": answer,
            "citations": citation_text,
            "match": match,
            "match_score": match_score,
        }

        self.chat.insert(0, qa)
        return qa

    def _source_match(self, query: str):
        """
        Returns (score, label): score is the cosine similarity (0.00-1.00)
        between the question and the best-matching passage; label puts it
        in words, as this model rarely scores even close matches above 0.75.
        """
        try:
            scored = self.vectorstore.similarity_search_with_score(query, k=1)
        except Exception as e:
            logger.warning(f"Source match estimation failed: {e}")
            return None, "unknown"
        if not scored:
            return 0.0, "weak"
        # For unit vectors, squared L2 distance d = 2 - 2 * cosine.
        cosine = max(0.0, min(1.0, 1 - float(scored[0][1]) / 2))
        logger.info(f"Best passage similarity: {cosine:.2f}")
        if cosine >= MATCH_STRONG:
            label = "strong"
        elif cosine >= MATCH_MODERATE:
            label = "moderate"
        else:
            label = "weak"
        return round(cosine, 2), label

    def summarize(self, on_token=None) -> str:
        """
        Returns a brief summary of the loaded document (cached per document).
        on_token(text), if given, receives the summary as it is written.
        """
        if not self.index_ready:
            raise RuntimeError("Document not indexed yet")
        if self.summary:
            return self.summary
        check_ollama()

        if len(self.document_text) <= SUMMARY_MAX_CHARS:
            kind, text = "the contents of", self.document_text
        else:
            kind, text = "representative excerpts (in order) from", self._representative_excerpts()
        if not text.strip():
            raise RuntimeError("The document has no readable text to summarize.")

        callbacks = [_AnswerStreamer(on_token, prompt_prefix="")] if on_token else []
        summary = self.llm.invoke(
            SUMMARY_PROMPT.format(kind=kind, text=text), config={"callbacks": callbacks}
        )

        self.summary = summary.strip()
        logger.info("Summary generated")
        if self.cache_path and os.path.isdir(self.cache_path):
            try:
                with open(os.path.join(self.cache_path, "summary.txt"), "w", encoding="utf-8") as f:
                    f.write(self.summary)
            except Exception as e:
                logger.warning(f"Could not cache the summary: {e}")
        return self.summary

    def _representative_excerpts(self) -> str:
        """The opening passage plus, for each of SUMMARY_TOPICS clusters of
        passage embeddings, the passage closest to the cluster's center,
        joined in document order within SUMMARY_MAX_CHARS."""
        store = self.vectorstore
        n = store.index.ntotal
        vectors = store.index.reconstruct_n(0, n)
        k = min(SUMMARY_TOPICS, n)

        # k-means (cosine), started from passages spread evenly through the
        # document so the result is deterministic.
        centers = vectors[np.linspace(0, n - 1, k).round().astype(int)]
        for _ in range(20):
            labels = np.argmax(vectors @ centers.T, axis=1)
            for c in range(k):
                members = vectors[labels == c]
                if len(members):
                    center = members.mean(axis=0)
                    centers[c] = center / (np.linalg.norm(center) or 1)
        labels = np.argmax(vectors @ centers.T, axis=1)

        picks = {0}  # the opening usually holds the title and introduction
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members):
                picks.add(int(members[np.argmax(vectors[members] @ centers[c])]))

        excerpts, total = [], 0
        for i in sorted(picks):
            doc = store.docstore.search(store.index_to_docstore_id[i])
            text = doc.page_content.strip()
            if total + len(text) > SUMMARY_MAX_CHARS:
                break
            excerpts.append(text)
            total += len(text)
        logger.info(f"Summarizing {len(excerpts)} representative passages of {n}")
        return "\n...\n".join(excerpts)

    def save_conversation(self, path: str):
        logger.info(f"Saving conversation to {path}")
        text = ""
        if self.summary:
            text += f"Document summary:\n{self.summary}\n" + "-" * 50 + "\n\n"
        for i, qa in enumerate(reversed(self.chat), 1):
            text += f"Question {i}: {qa['question']}\n"
            text += f"Answer: {qa['answer']}\n"
            text += f"References: {qa['citations']}\n"
            text += f"Source match: {format_match(qa)}\n"
            text += "-" * 50 + "\n\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
