# ============================================================
#  config.py  —  Chat-with-Audios · Project Settings
# ============================================================

# ── Whisper (faster-whisper) ────────────────────────────────
WHISPER_MODEL_SIZE   = "base"        # tiny | base | small | medium | large-v2
WHISPER_DEVICE       = "cpu"         # cpu | cuda
WHISPER_COMPUTE_TYPE = "int8"        # int8 (CPU-friendly) | float16 (GPU)
WHISPER_LANGUAGE     = None          # None = auto-detect, or e.g. "en", "ar"

# ── ChromaDB ────────────────────────────────────────────────
CHROMA_PERSIST_DIR   = "./chroma_db"
CHROMA_COLLECTION    = "audio_transcripts"

# ── LlamaIndex ──────────────────────────────────────────────
CHUNK_SIZE           = 512           # tokens per chunk
CHUNK_OVERLAP        = 64            # overlap between chunks
SIMILARITY_TOP_K     = 3             # retrieved chunks per query

# ── Ollama ──────────────────────────────────────────────────
OLLAMA_BASE_URL      = "http://localhost:11434"
OLLAMA_MODEL         = "qwen2.5:3b"      # any model pulled via `ollama pull <name>`
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ── Text-to-Speech (optional pyttsx3) ───────────────────────
TTS_ENABLED          = True         # flip to True to enable spoken answers
TTS_RATE             = 175           # words per minute
TTS_VOLUME           = 0.9           # 0.0 – 1.0

# ── Supported audio formats ─────────────────────────────────
SUPPORTED_FORMATS    = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]

# ── UI copy ─────────────────────────────────────────────────
APP_TITLE            = "Chat with Audios"
APP_SUBTITLE         = "Upload · Transcribe · Ask"
APP_ICON             = "🎙️"
