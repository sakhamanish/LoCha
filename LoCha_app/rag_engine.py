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

# Summaries: long documents are summarized in sections ("map"), then the
# section notes are combined into one brief summary ("reduce").
SUMMARY_SECTION_CHARS = 6000  # about 1500 tokens per call
SUMMARY_MAX_SECTIONS = 8      # longer documents are sampled evenly

SUMMARY_PART_PROMPT = PromptTemplate.from_template(
    """Below is part {index} of {total} of a document.
List the key points of this part in 2 to 4 short bullet points.
Use only information from the text. Reply with the bullet points only.

Text:
{text}

Key points:"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    """Below are {kind} a document.
Write a brief summary for someone who has not read it: first one short
paragraph giving the gist of the whole document, then 3 to 5 bullet points
with its most important points. Use only information from the text below.
Do not add a title.

{text}

Summary:"""
)

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
        self.llm = None
        self.summary = None

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
        llm = Ollama(model="llama3.2:latest", temperature=0.2, num_ctx=8192)
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

    def ask(self, question: str) -> dict:
        if not self.index_ready:
            raise RuntimeError("Document not indexed yet")

        logger.info(f"Question received: {question}")
        result = self.qa_chain.invoke({"question": question})

        answer = result["answer"]
        sources = result.get("source_documents", [])
        logger.info(f"Searched document for: {result.get('generated_question', question)}")
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

    def summarize(self, progress=None) -> str:
        """
        Returns a brief summary of the loaded document (cached per document).
        progress(step, total) is called before each model call.
        """
        if not self.index_ready:
            raise RuntimeError("Document not indexed yet")
        if self.summary:
            return self.summary

        sections = RecursiveCharacterTextSplitter(
            chunk_size=SUMMARY_SECTION_CHARS, chunk_overlap=0
        ).split_text(self.document_text)
        if not sections:
            raise RuntimeError("The document has no readable text to summarize.")
        if len(sections) > SUMMARY_MAX_SECTIONS:
            last = len(sections) - 1
            picks = sorted({
                round(i * last / (SUMMARY_MAX_SECTIONS - 1))
                for i in range(SUMMARY_MAX_SECTIONS)
            })
            logger.info(f"Summarizing {len(picks)} of {len(sections)} sections")
            sections = [sections[i] for i in picks]

        report = progress or (lambda step, total: None)
        if len(sections) == 1:
            report(1, 1)
            summary = self.llm.invoke(
                SUMMARY_PROMPT.format(kind="the contents of", text=sections[0])
            )
        else:
            total = len(sections) + 1
            notes = []
            for i, section in enumerate(sections, 1):
                report(i, total)
                part = self.llm.invoke(SUMMARY_PART_PROMPT.format(
                    index=i, total=len(sections), text=section
                ))
                notes.append(f"Part {i}:\n{part.strip()}")
            report(total, total)
            summary = self.llm.invoke(SUMMARY_PROMPT.format(
                kind="notes on each part of", text="\n\n".join(notes)
            ))

        self.summary = summary.strip()
        logger.info("Summary generated")
        return self.summary

    def save_conversation(self, path: str):
        logger.info(f"Saving conversation to {path}")
        text = ""
        if self.summary:
            text += f"Document summary:\n{self.summary}\n" + "-" * 50 + "\n\n"
        for i, qa in enumerate(reversed(self.chat), 1):
            text += f"Question {i}: {qa['question']}\n"
            text += f"Answer: {qa['answer']}\n"
            text += f"References: {qa['citations']}\n"
            text += f"Confidence: {qa['confidence']}%\n"
            text += "-" * 50 + "\n\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
