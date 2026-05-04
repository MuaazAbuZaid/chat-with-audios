# ============================================================
#  app.py  —  Chat-with-Audios · Streamlit UI
# ============================================================
#
#  Run with:  streamlit run app.py
#
# ============================================================

import streamlit as st
from pathlib import Path

import config as cfg
import pipeline as pl

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon=cfg.APP_ICON,
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .app-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.4rem;
        font-weight: 600;
        letter-spacing: -1px;
        margin: 0;
    }
    .app-header p {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 300;
    }

    /* ── Stage badges ── */
    .stage-badge {
        display: inline-block;
        background: #1a1a2e;
        color: #e0e0ff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid #3a3a6e;
        margin-bottom: 0.5rem;
    }

    /* ── Transcript box ── */
    .transcript-box {
        background: #f8f8f8;
        border-left: 4px solid #4f46e5;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        line-height: 1.7;
        max-height: 260px;
        overflow-y: auto;
        white-space: pre-wrap;
        color: #222;
    }

    /* ── Answer bubble ── */
    .answer-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1rem 1.3rem;
        border-radius: 12px;
        font-size: 0.95rem;
        line-height: 1.65;
        margin-top: 0.5rem;
    }

    /* ── Chat history item ── */
    .chat-q  { color: #555; font-size: 0.82rem; font-weight: 600; margin-top: 0.8rem; }
    .chat-a  { color: #222; font-size: 0.9rem;  line-height: 1.6; padding-left: 0.5rem; }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #eee; margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-header">
        <h1>{cfg.APP_ICON} {cfg.APP_TITLE}</h1>
        <p>{cfg.APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ───────────────────────────────────
if "transcript"   not in st.session_state: st.session_state.transcript   = ""
if "indexed"      not in st.session_state: st.session_state.indexed       = False
if "chat_history" not in st.session_state: st.session_state.chat_history  = []
if "audio_name"   not in st.session_state: st.session_state.audio_name    = ""

# ============================================================
#  STAGE 1 — UPLOAD
# ============================================================
st.markdown('<span class="stage-badge">① UPLOAD</span>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop an audio file",
    type=cfg.SUPPORTED_FORMATS,
    label_visibility="collapsed",
)

# ============================================================
#  STAGE 2 — TRANSCRIBE
# ============================================================
if uploaded:
    new_file = uploaded.name != st.session_state.audio_name

    if new_file:
        # Reset state for new upload
        st.session_state.transcript   = ""
        st.session_state.indexed       = False
        st.session_state.chat_history  = []
        st.session_state.audio_name    = uploaded.name

    st.audio(uploaded, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    st.markdown('<span class="stage-badge">② TRANSCRIBE</span>', unsafe_allow_html=True)

    if st.button("▶ Transcribe Audio", use_container_width=True):
        with st.spinner("Transcribing… this may take a moment."):
            try:
                audio_path = pl.save_uploaded_file(uploaded)
                st.session_state.transcript = pl.transcribe(audio_path)
                st.session_state.indexed    = False   # force re-index on new transcript
                st.session_state.chat_history = []
                st.success("Transcription complete!")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    if st.session_state.transcript:
        with st.expander("📄 View transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{st.session_state.transcript}</div>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "⬇ Download transcript (.txt)",
            data=st.session_state.transcript,
            file_name=f"{Path(uploaded.name).stem}_transcript.txt",
            mime="text/plain",
        )

# ============================================================
#  STAGE 3 — INDEX
# ============================================================
if st.session_state.transcript and not st.session_state.indexed:
    st.markdown(
        '<span class="stage-badge">③ INDEX TRANSCRIPT</span>',
        unsafe_allow_html=True,
    )

    if st.button("🗂 Build Vector Index", use_container_width=True):
        with st.spinner("Chunking, embedding and storing in ChromaDB…"):
            try:
                pl.build_index(
                    st.session_state.transcript,
                    doc_id=st.session_state.audio_name,
                )
                st.session_state.indexed = True
                st.success("Index ready — you can now ask questions!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# ============================================================
#  STAGE 4 — QA
# ============================================================
if st.session_state.indexed:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<span class="stage-badge">④ ASK A QUESTION</span>', unsafe_allow_html=True)

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the main topics discussed?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Thinking…"):
            try:
                answer = pl.query_index(question)
                st.session_state.chat_history.append((question, answer))

                # Optional TTS
                pl.speak(answer)
            except Exception as e:
                st.error(f"Query failed: {e}")

    # ── Latest answer ────────────────────────────────────────
    if st.session_state.chat_history:
        latest_q, latest_a = st.session_state.chat_history[-1]
        st.markdown(
            f'<div class="answer-bubble">{latest_a}</div>',
            unsafe_allow_html=True,
        )

        # ── History (all but last) ───────────────────────────
        if len(st.session_state.chat_history) > 1:
            with st.expander("💬 Previous Q&A", expanded=False):
                for q, a in reversed(st.session_state.chat_history[:-1]):
                    st.markdown(f'<p class="chat-q">Q: {q}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="chat-a">{a}</p>', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
cols = st.columns(3)
with cols[0]:
    model_label = cfg.WHISPER_MODEL_SIZE
    st.caption(f"🎙 Whisper · `{model_label}`")
with cols[1]:
    st.caption(f"🦙 Ollama · `{cfg.OLLAMA_MODEL}`")
with cols[2]:
    tts_status = "on" if cfg.TTS_ENABLED else "off"
    st.caption(f"🔊 TTS · `{tts_status}`")
