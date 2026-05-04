# ============================================================
#  pipeline.py  —  Chat-with-Audios · Core Logic
# ============================================================
#
#  Stages
#  ──────
#  1. transcribe(audio_path)          faster-whisper  → plain text
#  2. build_index(transcript, doc_id) LlamaIndex + ChromaDB → vector store
#  3. query_index(question)           Ollama (via LlamaIndex) → answer str
#  4. speak(text)                     pyttsx3 (optional) → audio output
#
# ============================================================

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional

# ── Config ──────────────────────────────────────────────────
import config as cfg


# ============================================================
#  1. TRANSCRIPTION  (faster-whisper)
# ============================================================

def transcribe(audio_path: str) -> str:
    """
    Convert an audio file to a transcript string.

    Parameters
    ----------
    audio_path : str
        Absolute or relative path to the audio file.

    Returns
    -------
    str
        Full transcript text joined from all detected segments.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        cfg.WHISPER_MODEL_SIZE,
        device=cfg.WHISPER_DEVICE,
        compute_type=cfg.WHISPER_COMPUTE_TYPE,
    )

    segments, _info = model.transcribe(
        audio_path,
        language=cfg.WHISPER_LANGUAGE,
        beam_size=5,
    )

    transcript = " ".join(seg.text.strip() for seg in segments)
    return transcript


# ============================================================
#  2. INDEXING  (LlamaIndex + ChromaDB)
# ============================================================

# Module-level index cache so we don't rebuild on every query
_query_engine = None


def build_index(transcript: str, doc_id: str = "audio_doc") -> None:
    """
    Chunk the transcript, embed it, and persist it in ChromaDB.
    Populates the module-level `_query_engine` ready for queries.

    Parameters
    ----------
    transcript : str
        Raw transcript text returned by `transcribe()`.
    doc_id : str
        Logical identifier for this document (e.g. the filename).
    """
    global _query_engine

    import chromadb
    from llama_index.core import (
        Document,
        Settings,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding

    # ── LLM & Embedding via Ollama ───────────────────────────
    Settings.llm = Ollama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=cfg.OLLAMA_EMBED_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )

    # ── ChromaDB persistent client ───────────────────────────
    chroma_client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
    chroma_col    = chroma_client.get_or_create_collection(cfg.CHROMA_COLLECTION)
    vector_store  = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)

    # ── Build index ──────────────────────────────────────────
    document = Document(text=transcript, doc_id=doc_id)
    index    = VectorStoreIndex.from_documents(
        [document],
        storage_context=storage_ctx,
    )

    _query_engine = index.as_query_engine(
        similarity_top_k=cfg.SIMILARITY_TOP_K,
    )


# ============================================================
#  3. QA  (Ollama via LlamaIndex query engine)
# ============================================================

def query_index(question: str) -> str:
    """
    Answer a question using the indexed transcript.

    Parameters
    ----------
    question : str
        User's natural-language question about the audio content.

    Returns
    -------
    str
        LLM-generated answer grounded in the transcript chunks.

    Raises
    ------
    RuntimeError
        If `build_index()` has not been called first.
    """
    if _query_engine is None:
        raise RuntimeError(
            "Index not built yet. Call build_index() before querying."
        )

    response = _query_engine.query(question)
    return str(response)


# ============================================================
#  4. TEXT-TO-SPEECH  (optional — pyttsx3)
# ============================================================

def speak(text: str) -> None:
    """
    Read `text` aloud using pyttsx3 (blocks until done).
    Only runs when cfg.TTS_ENABLED is True.

    Parameters
    ----------
    text : str
        The string to be synthesised and spoken.
    """
    if not cfg.TTS_ENABLED:
        return

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   cfg.TTS_RATE)
        engine.setProperty("volume", cfg.TTS_VOLUME)
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        print("[TTS] pyttsx3 not installed — skipping speech output.")
    except Exception as exc:
        print(f"[TTS] Error: {exc}")


# ============================================================
#  5. CONVENIENCE HELPER  (used by app.py)
# ============================================================

def save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temp path on disk.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns
    -------
    str
        Absolute path to the saved temp file.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
