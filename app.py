# ============================================================
# ResumeIQ — FINAL STABLE VERSION
# ============================================================

import streamlit as st
import os
import tempfile

from rag_engine import (load_resume, chunk_text,
                        build_vector_store, retrieve_chunks,
                        build_interview_prompt)
from ollama_client import (generate_answer, get_available_models,
                            check_ollama_running)

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI SMART RESUME ANALYZER",
    page_icon="📄",
    layout="wide"
)

# ── SESSION STATE ────────────────────────────────────────────
for key in ["resumes_data", "last_files", "chat_history",
            "interview_questions"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "interview_questions" else None

# ── OLLAMA CHECK ─────────────────────────────────────────────
if not check_ollama_running():
    st.error("Run: ollama serve")
    st.stop()

# ── HEADER ───────────────────────────────────────────────────
st.title("📄 AI SMART RESUME ANALYZER")
st.caption("Upload resumes → Compare → Match JD → Generate Questions")
st.divider()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:

    st.header("Settings")

    installed_models = get_available_models()
    cloud_models = ["deepseek-v3.2:cloud"]
    extra_local = ["qwen3:0.6b"]

    all_models = installed_models + [
        m for m in extra_local if m not in installed_models
    ] + [
        m for m in cloud_models if m not in installed_models
    ]

    selected_model = st.selectbox(
        "Choose Model",
        all_models,
        format_func=lambda x: "☁️ " + x if x.endswith(":cloud") else "💻 " + x
    )

    st.divider()

    if st.button("Clear Everything", use_container_width=True):
        st.session_state.resumes_data = []
        st.session_state.last_files = []
        st.session_state.chat_history = []
        st.session_state.interview_questions = None
        st.rerun()

    st.divider()

    st.markdown("### How It Works")
    st.markdown(
        "1. Upload resumes\n"
        "2. Convert to embeddings\n"
        "3. Ask questions\n"
        "4. AI compares candidates\n"
        "5. Get ranked results"
    )

    st.divider()

    st.markdown("### Tech Stack")
    st.markdown(
        "- Ollama\n- FAISS\n- Sentence Transformers\n- Streamlit"
    )

# ════════════════════════════════════════════════════════════
# UPLOAD MULTIPLE RESUMES
# ════════════════════════════════════════════════════════════

st.subheader("Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload 5–10 resumes",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        file_key = file.name + str(file.size)

        if file_key in st.session_state.last_files:
            continue

        with st.spinner(f"Processing {file.name}..."):

            suffix = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            text, method, success = load_resume(tmp_path)
            os.unlink(tmp_path)

            if success:
                chunks = chunk_text(text)
                index, chunks = build_vector_store(chunks)

                st.session_state.resumes_data.append({
                    "name": file.name,
                    "text": text,
                    "chunks": chunks,
                    "index": index
                })

                st.session_state.last_files.append(file_key)

                st.success(f"{file.name} loaded using {method}")

            else:
                st.warning(f"{file.name} could not be processed")

# ── STATUS ───────────────────────────────────────────────────
if not st.session_state.resumes_data:
    st.info("Upload resumes to start")
    st.stop()

st.success(f"{len(st.session_state.resumes_data)} resumes loaded")

st.divider()

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["Ask & Compare", "Interview Questions"])

# ============================================================
# TAB 1 — CHAT
# ============================================================

with tab1:

    st.subheader("Ask Questions / Match JD")

    chat_container = st.container()

    # Show chat history
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    question = st.chat_input("Ask or paste JD (e.g., AI Engineer with Python, ML)")

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with chat_container:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):

                with st.spinner("Analyzing resumes..."):

                    all_results = []

                    for res in st.session_state.resumes_data:
                        chunks = retrieve_chunks(
                            question,
                            res["index"],
                            res["chunks"],
                            top_k=2
                        )

                        for c in chunks:
                            all_results.append({
                                "resume": res["name"],
                                "chunk": c["chunk"],
                                "score": c["score"]
                            })

                    all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
                    top_results = all_results[:6]

                    context = "\n\n".join([
                        f"Resume: {r['resume']}\n{r['chunk']}"
                        for r in top_results
                    ])

                    prompt = f"""
You are an expert recruiter.

If user gives a JOB DESCRIPTION:
- Match candidates
- Rank them
- Explain WHY each fits

If normal question:
- Compare candidates

Format:
1. Candidate — Reason

QUESTION:
{question}

DATA:
{context}

ANSWER:
"""

                    answer = generate_answer(
                        prompt,
                        model=selected_model,
                        temperature=0.4
                    )

                st.write(answer)

                # Show RAG chunks
                with st.expander("View Retrieval"):
                    for r in top_results:
                        st.markdown(f"**{r['resume']}** (Score: {round(r['score'], 3)})")
                        st.info(r["chunk"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

# ============================================================
# TAB 2 — INTERVIEW QUESTIONS (PER CANDIDATE)
# ============================================================

with tab2:

    st.subheader("Generate Interview Questions")

    candidate_names = [res["name"] for res in st.session_state.resumes_data]

    selected_candidate = st.selectbox(
        "Select Candidate",
        candidate_names
    )

    selected_text = ""
    for res in st.session_state.resumes_data:
        if res["name"] == selected_candidate:
            selected_text = res["text"]
            break

    if st.button("Generate Questions", use_container_width=True):

        with st.spinner(f"Generating for {selected_candidate}..."):

            prompt = build_interview_prompt(selected_text)

            questions = generate_answer(
                prompt,
                model=selected_model,
                temperature=0.7
            )

            st.session_state.interview_questions = questions

    if st.session_state.interview_questions:

        st.markdown(f"### Questions for {selected_candidate}")
        st.markdown(st.session_state.interview_questions)

        st.download_button(
            "Download Questions",
            data=st.session_state.interview_questions,
            file_name=f"{selected_candidate}_questions.txt"
        )