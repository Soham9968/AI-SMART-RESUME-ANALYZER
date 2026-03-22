# ============================================================
# RAG ENGINE — ResumeIQ
# Smart PDF handling without any external software
#
# PDF Strategy:
#   1. Try PyPDF (fast, works for most normal PDFs)
#   2. If garbage text detected → try pdfplumber (better layout)
#   3. If still garbage → return None so app shows paste option
#
# Flow: Load → Clean → Chunk → Embed → Store → Retrieve → Prompt
# ============================================================

import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ── LOAD EMBEDDING MODEL ONCE ────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")


# ════════════════════════════════════════════════════════════
# TEXT CLEANING
# ════════════════════════════════════════════════════════════

def clean_text(text):
    """
    Fix common PDF extraction problems.
    - Spaced characters: 'a i l . c o m' -> 'ail.com'
    - Removes weird symbols
    - Collapses extra spaces and newlines
    """
    if not text:
        return ""

    # Fix spaced out characters like 'a i l' -> 'ail'
    text = re.sub(r'(?<=[a-zA-Z]) (?=[a-zA-Z])', '', text)

    # Remove non printable characters
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def is_garbage(text):
    """
    Detect if PDF extraction produced garbled text.

    Signs of garbage:
    - Text too short for a resume
    - Too many single character tokens (spaced out letters)

    Returns True if garbage, False if text looks good.
    """
    if not text or len(text.strip()) < 100:
        return True

    words = text.split()

    if len(words) < 30:
        return True

    # Count single character words like 'a', 'i', 'l'
    single_chars = sum(1 for w in words if len(w) == 1)
    ratio = single_chars / len(words)

    # More than 35% single chars = spaced out garbage
    if ratio > 0.35:
        return True

    return False


# ════════════════════════════════════════════════════════════
# FILE LOADING WITH AUTO FALLBACK
# ════════════════════════════════════════════════════════════

def load_with_pypdf(file_path):
    """Method 1: PyPDF — fast, works for standard PDFs."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception:
        return ""


def load_with_pdfplumber(file_path):
    """
    Method 2: pdfplumber — handles complex layouts and columns.
    Better for designer resumes with multiple columns.
    """
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception:
        return ""


def load_resume(file_path):
    """
    Smart file loader with automatic fallback.

    PDF flow:
      Try PyPDF → check if garbage → try pdfplumber → check again
      If both fail → return None (app will show paste text option)

    TXT and DOCX:
      Direct read, always works.

    Returns tuple: (text, method_used, success)
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ── TXT ──────────────────────────────────────────────────
    if ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            text = clean_text(text)
            if len(text) > 50:
                return text, "TXT", True
            return None, "TXT", False
        except Exception:
            return None, "TXT", False

    # ── DOCX ─────────────────────────────────────────────────
    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([
                p.text for p in doc.paragraphs if p.text.strip()
            ])
            text = clean_text(text)
            if len(text) > 50:
                return text, "DOCX", True
            return None, "DOCX", False
        except Exception:
            return None, "DOCX", False

    # ── PDF — SMART FALLBACK ──────────────────────────────────
    elif ext == ".pdf":

        # Try Method 1: PyPDF
        text = load_with_pypdf(file_path)
        text = clean_text(text)
        if not is_garbage(text):
            return text, "PyPDF", True

        # Try Method 2: pdfplumber
        text = load_with_pdfplumber(file_path)
        text = clean_text(text)
        if not is_garbage(text):
            return text, "pdfplumber", True

        # Both failed — scanned or image based PDF
        # App will show the paste text option to user
        return None, "failed", False

    else:
        return None, "unsupported", False


# ════════════════════════════════════════════════════════════
# CHUNKING
# ════════════════════════════════════════════════════════════

def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split text into overlapping word based chunks.

    chunk_size=100 words works well for resumes because
    resume sections are short and dense.
    Small chunks = precise section level retrieval.

    overlap=20 means 20 words shared between adjacent chunks
    so context is not lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ════════════════════════════════════════════════════════════
# VECTOR STORE
# ════════════════════════════════════════════════════════════

def build_vector_store(chunks):
    """
    Embed chunks and store in FAISS index.
    Runs fully in memory, no external database needed.
    """
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, chunks


# ════════════════════════════════════════════════════════════
# RETRIEVAL
# ════════════════════════════════════════════════════════════

def retrieve_chunks(question, index, chunks, top_k=3):
    """
    Find top_k most relevant chunks for the question.
    Returns chunks with their relevance scores.
    """
    q_embed = embed_model.encode([question])
    q_embed = np.array(q_embed).astype("float32")
    faiss.normalize_L2(q_embed)

    scores, indices = index.search(q_embed, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[0][i])
            })

    return results


# ════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════

def build_qa_prompt(question, retrieved_chunks):
    """Prompt for resume Q&A using retrieved context."""
    context = "\n\n".join([r["chunk"] for r in retrieved_chunks])

    prompt = (
        "You are a helpful AI assistant analysing a resume.\n"
        "Answer the question using ONLY the resume content below.\n"
        "Keep your answer short and direct — maximum 4 sentences.\n\n"
        "Rules:\n"
        "- Only use information from the resume content\n"
        "- If not in resume say: This is not mentioned in the resume\n"
        "- Be professional and concise\n\n"
        "RESUME CONTENT:\n"
        + context +
        "\n\nQUESTION:\n"
        + question +
        "\n\nSHORT ANSWER:"
    )

    return prompt


def build_interview_prompt(full_resume_text):
    """Prompt for generating interview questions from resume."""
    resume_preview = full_resume_text[:2000]

    prompt = (
        "You are an expert technical interviewer.\n"
        "Based on the resume below generate exactly 10 interview questions.\n\n"
        "Instructions:\n"
        "- Number questions 1 to 10\n"
        "- Mix technical, project based and behavioural questions\n"
        "- Make questions specific to what is in the resume\n"
        "- Increase difficulty from question 1 to 10\n\n"
        "RESUME:\n"
        + resume_preview +
        "\n\n10 INTERVIEW QUESTIONS:"
    )

    return prompt