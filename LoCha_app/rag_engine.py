import os
import hashlib
import logging

import subprocess

def check_ollama():
    try:
        subprocess.run(["ollama", "list"], check=True)
    except Exception:
        raise RuntimeError("Ollama not installed. Please install Ollama.")


from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LoChaEngine] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class LoChaEngine:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.chat = []
        self.file_hash = None
        self.index_ready = False
        self.document_text = ""

    def reset(self):
        logger.info("Resetting engine state")
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        self.chat = []
        self.file_hash = None
        self.index_ready = False
        self.document_text = ""

    def load_document(self, file_path: str):
        logger.info(f"Loading document: {file_path}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        new_hash = hashlib.md5(file_bytes).hexdigest()
        if new_hash == self.file_hash:
            logger.info("Same document detected, skipping re-index")
            return

        self.reset()
        self.file_hash = new_hash

        suffix = os.path.splitext(file_path)[1].lower()
        loader = PyPDFLoader(file_path) if suffix == ".pdf" else Docx2txtLoader(file_path)

        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(file_path)
        self.document_text = "\n".join(d.page_content for d in docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Document split into {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.from_documents(chunks, embeddings)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        llm = Ollama(model="llama3.2:latest", temperature=0.5)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )

        self.index_ready = True
        logger.info("Document indexed successfully")

    def ask(self, question: str) -> dict:
        if not self.index_ready:
            raise RuntimeError("Document not indexed yet")

        logger.info(f"Question received: {question}")
        result = self.qa_chain.invoke({"question": question})

        answer = result["answer"]
        sources = result.get("source_documents", [])

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

        # ---------------- Confidence (retrieval-score based) ----------------
        confidence = 50  # default fallback

        if sources:
            try:
                # Re-run retrieval to get similarity scores (FAISS returns distance; lower = better)
                scored_docs = self.vectorstore.similarity_search_with_score(question, k=5)

                distances = [score for _, score in scored_docs]

                if distances:
                    # Convert FAISS distance to similarity proxy in [0, 1]
                    similarities = [max(0.0, min(1.0, 1 - (d / 50)))  # 50 is a practical upper bound
                                    for d in distances
            ]

                    max_sim = max(similarities)
                    avg_sim = sum(similarities) / len(similarities)

                    # Core confidence calculation
                    confidence = round(
                        100 * (0.6 * max_sim + 0.4 * avg_sim),
                        1
                    )

                    # Clamp to realistic RAG bounds
                    confidence = int(max(45, min(confidence, 95)))
            except Exception as e:
                logger.warning(f"Confidence estimation failed, using fallback: {e}")
                confidence = 55


        qa = {
            "question": question,
            "answer": answer,
            "citations": citation_text,
            "confidence": confidence
        }

        self.chat.insert(0, qa)
        return qa

    def save_conversation(self, path: str):
        logger.info(f"Saving conversation to {path}")
        text = ""
        for i, qa in enumerate(reversed(self.chat), 1):
            text += f"Question {i}: {qa['question']}\n"
            text += f"Answer: {qa['answer']}\n"
            text += f"References: {qa['citations']}\n"
            text += f"Confidence: {qa['confidence']}%\n"
            text += "-" * 50 + "\n\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
