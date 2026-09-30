import os
import re
import time
import hashlib

import fitz
import numpy as np
import streamlit as st
from google import genai


# ============================================================
# BACKEND CONFIGURATION — UNCHANGED
# ============================================================

MODEL_NAME = "gemini-flash-lite-latest"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 60

SEMANTIC_TOP_K = 8
FINAL_TOP_K = 5

MIN_RELEVANCE_SCORE = 0.18


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ASTRA INTEL",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# GEMINI CLIENT — UNCHANGED
# ============================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# ============================================================
# GEMINI GENERATION WITH RETRIES — UNCHANGED
# ============================================================

def generate_with_retry(prompt):

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text

        except Exception as e:

            error_text = str(e)

            if "503" in error_text and attempt < 2:

                delay = 5 * (2 ** attempt)

                st.info(
                    f"Gemini is temporarily busy. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

            else:

                raise e


# ============================================================
# TEXT NORMALIZATION — UNCHANGED
# ============================================================

def normalize_text(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):

    text = normalize_text(text)

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were",
        "what", "why", "how", "when", "where", "which",
        "of", "to", "in", "on", "for", "and", "or",
        "with", "by", "from", "this", "that", "these",
        "those", "it", "its", "be", "as", "at"
    }

    return [
        word
        for word in words
        if word not in stopwords
    ]


# ============================================================
# PDF EXTRACTION — UNCHANGED
# ============================================================

def extract_pages(pdf_bytes):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        pages.append(
            {
                "page": page_number,
                "text": text.strip()
            }
        )

    document.close()

    return pages


# ============================================================
# CHUNKING — UNCHANGED
# ============================================================

def create_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []
    chunk_id = 0

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            ).strip()

            if chunk_text:

                chunks.append(
                    {
                        "id": chunk_id,
                        "page": page["page"],
                        "text": chunk_text
                    }
                )

                chunk_id += 1

            if end >= len(words):
                break

            start = max(
                0,
                end - overlap
            )

    return chunks


# ============================================================
# EMBEDDINGS — UNCHANGED
# ============================================================

def get_embedding(text):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )


# ============================================================
# BUILD INDEX — UNCHANGED
# ============================================================

def build_embedding_index(chunks):

    indexed_chunks = []

    progress = st.progress(
        0,
        text="Creating semantic index..."
    )

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        embedding = get_embedding(
            chunk["text"]
        )

        indexed_chunks.append(
            {
                "id": chunk["id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "embedding": embedding
            }
        )

        progress.progress(
            (i + 1) / total,
            text=(
                f"Preparing document "
                f"({i + 1}/{total})"
            )
        )

    progress.empty()

    return indexed_chunks


# ============================================================
# COSINE SIMILARITY — UNCHANGED
# ============================================================

def cosine_similarity(vector_a, vector_b):

    denominator = (
        np.linalg.norm(vector_a)
        *
        np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        /
        denominator
    )


# ============================================================
# KEYWORD SCORE — UNCHANGED
# ============================================================

def keyword_score(question, text):

    question_words = set(
        tokenize(question)
    )

    text_words = tokenize(text)

    if not question_words or not text_words:
        return 0.0

    text_word_set = set(text_words)

    overlap = (
        question_words
        &
        text_word_set
    )

    coverage = (
        len(overlap)
        /
        len(question_words)
    )

    normalized_question = normalize_text(
        question
    )

    normalized_text = normalize_text(
        text
    )

    phrase_bonus = 0.0

    if normalized_question in normalized_text:
        phrase_bonus = 0.5

    occurrence_bonus = 0.0

    for word in question_words:

        if len(word) >= 6:

            count = text_words.count(word)

            if count >= 2:
                occurrence_bonus += 0.05

    score = (
        0.75 * coverage
        +
        phrase_bonus
        +
        min(occurrence_bonus, 0.15)
    )

    return min(score, 1.0)


# ============================================================
# HYBRID RETRIEVAL — UNCHANGED
# ============================================================

def retrieve_chunks(
    indexed_chunks,
    question,
    top_k=FINAL_TOP_K
):

    question_embedding = get_embedding(
        question
    )

    scored_chunks = []

    for chunk in indexed_chunks:

        semantic = cosine_similarity(
            question_embedding,
            chunk["embedding"]
        )

        lexical = keyword_score(
            question,
            chunk["text"]
        )

        combined = (
            0.60 * semantic
            +
            0.40 * lexical
        )

        scored_chunks.append(
            {
                "id": chunk["id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "semantic_score": semantic,
                "keyword_score": lexical,
                "score": combined
            }
        )

    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []
    page_counts = {}

    for chunk in scored_chunks:

        page = chunk["page"]

        page_count = page_counts.get(
            page,
            0
        )

        if page_count >= 2:
            continue

        selected.append(chunk)

        page_counts[page] = page_count + 1

        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# SUMMARY — UNCHANGED
# ============================================================

def generate_summary(pages):

    document_text = "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"].strip()
    )

    prompt = f"""
You are ASTRA INTEL, a document intelligence assistant.

Summarize ONLY the uploaded document.

Rules:
- Use only the document.
- Do not use outside knowledge.
- Do not invent information.

Provide:

1. Main purpose
2. Key topics
3. Important concepts
4. Important conclusions

DOCUMENT:

{document_text}
"""

    return generate_with_retry(prompt)


# ============================================================
# ANSWER GENERATION — UNCHANGED
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    chat_history
):

    context = "\n\n".join(
        f"""
[PAGE {chunk['page']}]

{chunk['text']}
"""
        for chunk in retrieved_chunks
    )

    previous_conversation = ""

    if chat_history:

        previous_conversation = "\n\n".join(
            f"User: {item['question']}\n"
            f"ASTRA INTEL: {item['answer']}"
            for item in chat_history[-3:]
        )

    prompt = f"""
You are ASTRA INTEL.

You answer questions about an uploaded document.

IMPORTANT:
The document excerpts below are your ONLY factual source.

RULES:

1. Answer ONLY using information explicitly supported
   by the provided document excerpts.

2. Do NOT use your general knowledge.

3. Do NOT guess.

4. If the answer is clearly present, answer it directly.

5. If the answer is only partially supported, clearly say
   what the document supports and what is not available.

6. If the answer is not supported by the excerpts, respond:

   "The answer is not available in the uploaded document."

7. Every factual answer must include:
   "Source pages: Page X, Page Y"

8. Only cite pages that actually support the answer.

9. Do not cite pages merely because they contain related
   terminology.

10. Keep the answer concise but useful.

DOCUMENT EXCERPTS:

{context}

PREVIOUS CONVERSATION:

{previous_conversation}

CURRENT QUESTION:

{question}
"""

    return generate_with_retry(prompt)


# ============================================================
# SESSION STATE — UNCHANGED
# ============================================================

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# PREMIUM UI CSS
# ============================================================

st.markdown(
    """
<style>

/* ================================
   GLOBAL
================================ */

.stApp {
    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(99,102,241,0.10),
            transparent 24%
        ),
        radial-gradient(
            circle at 92% 18%,
            rgba(6,182,212,0.10),
            transparent 23%
        ),
        #f7f9fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

/* Hide default decoration */
[data-testid="stDecoration"] {
    display: none;
}

/* ================================
   ANIMATIONS
================================ */

@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-8px);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 0.5;
        transform: scale(0.85);
    }
    50% {
        opacity: 1;
        transform: scale(1.15);
    }
}

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

@keyframes reveal {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 15px rgba(99,102,241,0.10);
    }
    50% {
        box-shadow: 0 0 35px rgba(99,102,241,0.22);
    }
}

@keyframes scan {
    0% {
        transform: translateX(-120%);
    }
    100% {
        transform: translateX(120%);
    }
}

/* ================================
   HERO
================================ */

.astra-hero {
    position: relative;
    overflow: hidden;
    min-height: 390px;
    padding: 3.5rem 4rem;
    border-radius: 32px;
    color: white;
    background:
        linear-gradient(
            125deg,
            #0b1020,
            #111c3f,
            #172d69,
            #11152b,
            #0b1020
        );
    background-size: 400% 400%;
    animation:
        gradient 14s ease infinite,
        reveal 0.7s ease-out;

    box-shadow:
        0 30px 80px rgba(15,23,42,0.28);
}

.astra-hero::before {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    right: -150px;
    top: -190px;
    border-radius: 50%;
    background: rgba(59,130,246,0.22);
    filter: blur(55px);
}

.astra-hero::after {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    left: 35%;
    bottom: -260px;
    border-radius: 50%;
    background: rgba(139,92,246,0.20);
    filter: blur(70px);
}

.hero-grid {
    position: absolute;
    inset: 0;
    opacity: 0.08;
    background-image:
        linear-gradient(
            rgba(255,255,255,0.5) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255,255,255,0.5) 1px,
            transparent 1px
        );
    background-size: 45px 45px;
}

.hero-content {
    position: relative;
    z-index: 3;
    max-width: 800px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.82);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #67e8f9;
    animation: pulse 1.8s infinite;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 18px;
}

.hero-shield {
    font-size: 4rem;
    display: inline-block;
    animation: float 3.2s ease-in-out infinite;
    filter: drop-shadow(0 10px 20px rgba(59,130,246,0.3));
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.065em;
    margin: 0;
}

.hero-gradient-text {
    background:
        linear-gradient(
            90deg,
            #ffffff,
            #b9e8ff,
            #c4b5fd,
            #ffffff
        );
    background-size: 250% auto;
    animation: gradient 6s linear infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin-top: 1.5rem;
    max-width: 690px;
    color: rgba(255,255,255,0.72);
    font-size: 1.15rem;
    line-height: 1.7;
}

.hero-tech {
    margin-top: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.hero-tech span {
    padding: 7px 11px;
    border-radius: 9px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    color: rgba(255,255,255,0.65);
    font-size: 0.72rem;
}

/* ================================
   WORKFLOW
================================ */

.workflow {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0;
    animation: reveal 0.8s ease-out;
}

.workflow-card {
    flex: 1;
    min-height: 125px;
    padding: 1.2rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(148,163,184,0.20);
    box-shadow: 0 12px 30px rgba(15,23,42,0.055);
    transition: all 0.3s ease;
}

.workflow-card:hover {
    transform: translateY(-7px);
    border-color: rgba(99,102,241,0.30);
    box-shadow: 0 20px 45px rgba(15,23,42,0.11);
}

.workflow-num {
    color: #6366f1;
    font-size: 0.68rem;
    font-weight: 900;
    letter-spacing: 0.12em;
}

.workflow-title {
    margin-top: 8px;
    color: #111827;
    font-size: 1rem;
    font-weight: 850;
}

.workflow-description {
    margin-top: 6px;
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.5;
}

.workflow-arrow {
    color: #94a3b8;
    font-size: 1.5rem;
}

/* ================================
   SECTION
================================ */

.section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 2.2rem;
    margin-bottom: 0.35rem;
}

.section-icon {
    width: 43px;
    height: 43px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #ecfeff
        );
    font-size: 1.25rem;
    box-shadow: 0 8px 18px rgba(99,102,241,0.08);
}

.section-title {
    font-size: 1.35rem;
    font-weight: 850;
    color: #111827;
}

.section-description {
    margin-left: 55px;
    color: #64748b;
    font-size: 0.86rem;
    margin-bottom: 1rem;
}

/* ================================
   DOCUMENT CARD
================================ */

.document-card {
    position: relative;
    overflow: hidden;
    padding: 1.2rem 1.35rem;
    margin: 1rem 0;
    border-radius: 18px;
    background:
        linear-gradient(
            135deg,
            rgba(238,242,255,0.95),
            rgba(236,254,255,0.95)
        );
    border: 1px solid rgba(99,102,241,0.15);
    animation:
        reveal 0.5s ease-out,
        glow 4s ease-in-out infinite;
}

.document-card::after {
    content: "";
    position: absolute;
    width: 40%;
    height: 100%;
    top: 0;
    left: -50%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.7),
        transparent
    );
    transform: skewX(-20deg);
    animation: scan 5s ease-in-out infinite;
}

.document-name {
    position: relative;
    z-index: 2;
    color: #172033;
    font-size: 1rem;
    font-weight: 850;
}

.document-meta {
    position: relative;
    z-index: 2;
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 4px;
}

/* ================================
   QUESTION CARD
================================ */

.question-card {
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
    padding: 2rem;
    border-radius: 26px;
    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f0fdfa,
            #f5f3ff
        );
    background-size: 200% 200%;
    animation:
        gradient 9s ease infinite,
        reveal 0.6s ease-out;
    border: 1px solid rgba(99,102,241,0.15);
    box-shadow:
        0 20px 55px rgba(79,70,229,0.08);
}

.question-eyebrow {
    color: #6366f1;
    font-size: 0.7rem;
    font-weight: 900;
    letter-spacing: 0.12em;
}

.question-title {
    margin-top: 7px;
    font-size: 1.7rem;
    font-weight: 900;
    letter-spacing: -0.035em;
    color: #111827;
}

.question-description {
    color: #64748b;
    font-size: 0.86rem;
    margin-top: 5px;
}

/* ================================
   ANSWER
================================ */

.answer-header {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-top: 1.8rem;
    margin-bottom: 0.7rem;
}

.answer-icon {
    width: 39px;
    height: 39px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    color: white;
    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #0891b2
        );
    animation: glow 3s ease-in-out infinite;
}

.answer-title {
    font-size: 1.2rem;
    font-weight: 850;
    color: #111827;
}

.answer-card {
    padding: 1.4rem;
    border-radius: 20px;
    background: white;
    border: 1px solid rgba(99,102,241,0.14);
    box-shadow:
        0 15px 40px rgba(15,23,42,0.075);
    animation: reveal 0.5s ease-out;
}

/* ================================
   SOURCE CHIPS
================================ */

.sources-label {
    margin-top: 1.1rem;
    margin-bottom: 7px;
    color: #475569;
    font-size: 0.78rem;
    font-weight: 850;
}

.source-chip {
    display: inline-block;
    margin: 3px 4px 3px 0;
    padding: 7px 11px;
    border-radius: 999px;
    color: #4338ca;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    font-size: 0.74rem;
    font-weight: 800;
    transition: all 0.2s ease;
}

.source-chip:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 14px rgba(79,70,229,0.12);
}

/* ================================
   ARCHITECTURE
================================ */

.architecture {
    margin-top: 1.5rem;
    padding: 1.5rem;
    border-radius: 24px;
    background:
        radial-gradient(
            circle at 80% 20%,
            rgba(59,130,246,0.15),
            transparent 25%
        ),
        #0b1120;
    box-shadow: 0 20px 55px rgba(15,23,42,0.16);
}

.arch-title {
    color: white;
    font-size: 1rem;
    font-weight: 850;
    margin-bottom: 1.2rem;
}

.arch-flow {
    display: flex;
    align-items: center;
    gap: 8px;
}

.arch-node {
    flex: 1;
    padding: 1rem 0.4rem;
    text-align: center;
    border-radius: 15px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    transition: all 0.25s ease;
}

.arch-node:hover {
    transform: translateY(-5px);
    background: rgba(255,255,255,0.10);
    border-color: rgba(103,232,249,0.30);
}

.arch-icon {
    font-size: 1.35rem;
}

.arch-name {
    margin-top: 5px;
    color: rgba(255,255,255,0.78);
    font-size: 0.7rem;
    font-weight: 750;
}

.arch-arrow {
    color: #64748b;
    font-weight: 900;
}

/* ================================
   FOOTER
================================ */

.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    text-align: center;
    color: #94a3b8;
    font-size: 0.75rem;
}

.footer strong {
    color: #475569;
}

/* ================================
   STREAMLIT CONTROLS
================================ */

.stButton > button {
    border-radius: 13px;
    min-height: 45px;
    font-weight: 800;
    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(79,70,229,0.12);
}

.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #6366f1,
            #0891b2
        );
    background-size: 200% 100%;
    color: white;
    border: none;
    animation: gradient 5s ease infinite;
}

[data-testid="stFileUploader"] {
    border-radius: 18px;
}

[data-testid="stTextInput"] input {
    min-height: 50px;
    border-radius: 14px;
}

[data-testid="stMetric"] {
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.18);
    background: rgba(255,255,255,0.75);
    box-shadow: 0 8px 22px rgba(15,23,42,0.045);
}

[data-testid="stExpander"] {
    border-radius: 15px;
    border: 1px solid rgba(148,163,184,0.18);
}

/* ================================
   MOBILE
================================ */

@media (max-width: 800px) {

    .astra-hero {
        padding: 2.2rem 1.5rem;
    }

    .hero-title {
        font-size: 3rem;
    }

    .hero-shield {
        font-size: 3rem;
    }

    .workflow {
        flex-direction: column;
    }

    .workflow-card {
        width: 100%;
    }

    .workflow-arrow {
        transform: rotate(90deg);
    }

    .arch-flow {
        flex-direction: column;
    }

    .arch-node {
        width: 100%;
    }

    .arch-arrow {
        transform: rotate(90deg);
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
<div class="astra-hero">

    <div class="hero-grid"></div>

    <div class="hero-content">

        <div class="hero-badge">
            <span class="hero-dot"></span>
            DOCUMENT INTELLIGENCE SYSTEM
        </div>

        <div class="hero-brand">

            <div class="hero-shield">
                🛡️
            </div>

            <div class="hero-title">
                <span class="hero-gradient-text">
                    ASTRA INTEL
                </span>
            </div>

        </div>

        <div class="hero-subtitle">
            Turn static documents into interactive knowledge.
            Ask questions, retrieve relevant evidence, and get
            answers grounded in the document itself.
        </div>

        <div class="hero-tech">
            <span>Semantic Retrieval</span>
            <span>Hybrid Search</span>
            <span>Gemini AI</span>
            <span>Page-Level Sources</span>
            <span>Grounded Q&A</span>
        </div>

    </div>

</div>
"""
)


# ============================================================
# WORKFLOW
# ============================================================

st.html(
    """
<div class="workflow">

    <div class="workflow-card">

        <div class="workflow-num">
            01 / INGEST
        </div>

        <div class="workflow-title">
            📄 Upload
        </div>

        <div class="workflow-description">
            Give ASTRA a PDF and extract its contents.
        </div>

    </div>

    <div class="workflow-arrow">
        →
    </div>

    <div class="workflow-card">

        <div class="workflow-num">
            02 / INDEX
        </div>

        <div class="workflow-title">
            🧠 Prepare
        </div>

        <div class="workflow-description">
            Chunk and semantically index the document.
        </div>

    </div>

    <div class="workflow-arrow">
        →
    </div>

    <div class="workflow-card">

        <div class="workflow-num">
            03 / RETRIEVE
        </div>

        <div class="workflow-title">
            🔎 Search
        </div>

        <div class="workflow-description">
            Find the evidence most relevant to your question.
        </div>

    </div>

    <div class="workflow-arrow">
        →
    </div>

    <div class="workflow-card">

        <div class="workflow-num">
            04 / ANSWER
        </div>

        <div class="workflow-title">
            ✨ Ask
        </div>

        <div class="workflow-description">
            Generate a grounded answer with page sources.
        </div>

    </div>

</div>
"""
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.html(
    """
<div class="section-heading">

    <div class="section-icon">
        📄
    </div>

    <div class="section-title">
        Upload your document
    </div>

</div>

<div class="section-description">
    Start by giving ASTRA INTEL a PDF to understand.
</div>
"""
)


uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Upload the document you want ASTRA INTEL to analyze."
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    try:

        pdf_bytes = uploaded_file.getvalue()

        current_hash = hashlib.md5(
            pdf_bytes
        ).hexdigest()

        if (
            st.session_state.file_hash
            != current_hash
        ):

            st.session_state.file_hash = (
                current_hash
            )

            st.session_state.pages = (
                extract_pages(
                    pdf_bytes
                )
            )

            st.session_state.chunks = (
                create_chunks(
                    st.session_state.pages
                )
            )

            st.session_state.indexed_chunks = []

            st.session_state.chat_history = []

        pages = st.session_state.pages
        chunks = st.session_state.chunks


        # ====================================================
        # DOCUMENT CARD
        # ====================================================

        st.html(
            f"""
<div class="document-card">

    <div class="document-name">
        ✓ {uploaded_file.name}
    </div>

    <div class="document-meta">
        Document loaded successfully •
        {len(pages)} pages •
        {len(chunks)} searchable chunks
    </div>

</div>
"""
        )


        # ====================================================
        # METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "DOCUMENT PAGES",
                len(pages)
            )

        with col2:
            st.metric(
                "TEXT CHUNKS",
                len(chunks)
            )

        with col3:

            status = (
                "READY"
                if st.session_state.indexed_chunks
                else "WAITING"
            )

            st.metric(
                "INTELLIGENCE INDEX",
                status
            )


        # ====================================================
        # PREPARE
        # ====================================================

        st.html(
            """
<div class="section-heading">

    <div class="section-icon">
        🧠
    </div>

    <div class="section-title">
        Build the intelligence layer
    </div>

</div>

<div class="section-description">
    ASTRA converts the document into searchable semantic representations.
</div>
"""
        )


        if not st.session_state.indexed_chunks:

            if st.button(
                "⚡  Prepare Document",
                use_container_width=True
            ):

                try:

                    st.session_state.indexed_chunks = (
                        build_embedding_index(
                            chunks
                        )
                    )

                    st.success(
                        "✓ Document intelligence layer is ready."
                    )

                except Exception:

                    st.error(
                        "Could not prepare the document. "
                        "Please try again."
                    )

        else:

            st.success(
                "✓ Document prepared — ASTRA INTEL is ready."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.html(
            """
<div class="section-heading">

    <div class="section-icon">
        ✨
    </div>

    <div class="section-title">
        Document intelligence
    </div>

</div>

<div class="section-description">
    Get a concise overview generated strictly from the uploaded document.
</div>
"""
        )


        if st.button(
            "✨  Generate Summary",
            use_container_width=True
        ):

            with st.spinner(
                "ASTRA is analyzing the document..."
            ):

                try:

                    summary = generate_summary(
                        pages
                    )

                    st.html(
                        """
<div class="answer-card">
"""
                    )

                    st.markdown(summary)

                    st.html(
                        """
</div>
"""
                    )

                except Exception:

                    st.error(
                        "Unable to generate the summary."
                    )


        # ====================================================
        # EXTRACTED TEXT
        # ====================================================

        with st.expander(
            "📑  Inspect extracted document text"
        ):

            st.caption(
                "Raw text extracted from the uploaded PDF."
            )

            for page in pages:

                with st.expander(
                    f"Page {page['page']}"
                ):

                    if page["text"]:

                        st.write(
                            page["text"]
                        )

                    else:

                        st.warning(
                            "No readable text found on this page."
                        )


    except Exception:

        st.error(
            "Unable to process this PDF. "
            "Please make sure it is a valid PDF document."
        )


# ============================================================
# ASK SECTION
# ============================================================

st.divider()


st.html(
    """
<div class="question-card">

    <div class="question-eyebrow">
        STEP 03 / GROUNDED QUESTION ANSWERING
    </div>

    <div class="question-title">
        Ask your document anything.
    </div>

    <div class="question-description">
        ASTRA retrieves relevant evidence first,
        then generates an answer grounded in that evidence.
    </div>

</div>
"""
)


question = st.text_input(
    "Your question",
    placeholder="Try: What is an intrinsic semiconductor?",
    label_visibility="collapsed"
)


if st.button(
    "🚀  Ask ASTRA INTEL",
    use_container_width=True,
    type="primary"
):

    if uploaded_file is None:

        st.warning(
            "📄 Upload a PDF before asking a question."
        )

    elif not question.strip():

        st.warning(
            "💬 Enter a question first."
        )

    elif not st.session_state.indexed_chunks:

        st.warning(
            "🧠 Prepare the document before asking questions."
        )

    else:

        try:

            with st.spinner(
                "🔎 Retrieving relevant evidence..."
            ):

                retrieved_chunks = (
                    retrieve_chunks(
                        st.session_state.indexed_chunks,
                        question
                    )
                )


            with st.spinner(
                "🤖 Generating grounded answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    st.session_state.chat_history
                )


            # =================================================
            # ANSWER
            # =================================================

            st.html(
                """
<div class="answer-header">

    <div class="answer-icon">
        ✦
    </div>

    <div class="answer-title">
        ASTRA INTEL
    </div>

</div>
"""
            )


            st.html(
                """
<div class="answer-card">
"""
            )

            st.markdown(answer)

            st.html(
                """
</div>
"""
            )


            # =================================================
            # SOURCES
            # =================================================

            st.html(
                """
<div class="sources-label">
    📌 Retrieved source pages
</div>
"""
            )


            source_pages = sorted(
                set(
                    chunk["page"]
                    for chunk in retrieved_chunks
                )
            )


            source_html = ""

            for page in source_pages:

                source_html += (
                    f'<span class="source-chip">'
                    f'📄 Page {page}'
                    f'</span>'
                )


            st.html(
                source_html
            )


            # =================================================
            # EVIDENCE
            # =================================================

            with st.expander(
                "🔍  Inspect retrieved evidence"
            ):

                st.caption(
                    "These passages were retrieved and supplied to Gemini as context."
                )

                for index, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    with st.expander(
                        f"Evidence {index}  •  Page {chunk['page']}"
                    ):

                        st.write(
                            chunk["text"]
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.caption(
                                f"Combined: "
                                f"{chunk['score']:.3f}"
                            )

                        with col2:

                            st.caption(
                                f"Semantic: "
                                f"{chunk['semantic_score']:.3f}"
                            )

                        with col3:

                            st.caption(
                                f"Keyword: "
                                f"{chunk['keyword_score']:.3f}"
                            )


            # =================================================
            # SAVE CHAT
            # =================================================

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )


        except Exception:

            st.error(
                "Unable to process the question. "
                "Please try again."
            )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.chat_history:

    st.divider()


    st.html(
        """
<div class="section-heading">

    <div class="section-icon">
        💬
    </div>

    <div class="section-title">
        Conversation
    </div>

</div>

<div class="section-description">
    Your current document intelligence session.
</div>
"""
    )


    for number, item in enumerate(
        reversed(st.session_state.chat_history),
        start=1
    ):

        with st.expander(
            f"Question {number}  •  {item['question']}"
        ):

            st.markdown(
                "**You**"
            )

            st.write(
                item["question"]
            )

            st.markdown(
                "**ASTRA INTEL**"
            )

            st.info(
                item["answer"]
            )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()


with st.expander(
    "⚙️  Explore the ASTRA INTEL pipeline"
):

    st.html(
        """
<div class="architecture">

    <div class="arch-title">
        Retrieval-Augmented Generation Pipeline
    </div>

    <div class="arch-flow">

        <div class="arch-node">
            <div class="arch-icon">📄</div>
            <div class="arch-name">PDF</div>
        </div>

        <div class="arch-arrow">→</div>

        <div class="arch-node">
            <div class="arch-icon">✂️</div>
            <div class="arch-name">Chunking</div>
        </div>

        <div class="arch-arrow">→</div>

        <div class="arch-node">
            <div class="arch-icon">🧮</div>
            <div class="arch-name">Embeddings</div>
        </div>

        <div class="arch-arrow">→</div>

        <div class="arch-node">
            <div class="arch-icon">🔎</div>
            <div class="arch-name">Hybrid Retrieval</div>
        </div>

        <div class="arch-arrow">→</div>

        <div class="arch-node">
            <div class="arch-icon">🤖</div>
            <div class="arch-name">Gemini</div>
        </div>

        <div class="arch-arrow">→</div>

        <div class="arch-node">
            <div class="arch-icon">📌</div>
            <div class="arch-name">Grounded Answer</div>
        </div>

    </div>

</div>
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
<div class="footer">

    <strong>🛡️ ASTRA INTEL</strong>

    <br><br>

    Document Intelligence •
    Semantic Retrieval •
    Evidence-Grounded Generation

    <br>

    Built for the ASTRA 3-Day Build Challenge

</div>
"""
)