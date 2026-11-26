"""Central configuration for DevResolve (override via environment variables)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = Path(os.getenv("DEVRESOLVE_DB", BASE_DIR / "data" / "devresolve.db"))
VECTOR_DIR = Path(os.getenv("DEVRESOLVE_VECTOR_DIR", BASE_DIR / "vector_store"))

# Embeddings: "auto" tries Sentence Transformers, then falls back to an offline hashing embedder.
EMBEDDER = os.getenv("DEVRESOLVE_EMBEDDER", "auto")        # auto | sbert | hashing
SBERT_MODEL = os.getenv("DEVRESOLVE_SBERT_MODEL", "all-MiniLM-L6-v2")

# Local LLM through Ollama (optional - retrieval works without it)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

TOP_K = 5                 # experiences retrieved per query
CONTEXT_K = 4             # experiences passed to the LLM
MIN_SCORE = {"sbert": 0.35, "hashing": 0.15}   # similarity floor => "insufficient evidence"

TECHNOLOGIES = ["Python", "Java", "JavaScript", "React", "Node.js", "Git/GitHub",
                "Docker", "VS Code", "SQL/Database", "Linux", "Windows"]
OPERATING_SYSTEMS = ["Windows 11", "Windows 10", "macOS", "Ubuntu / Linux", "Cross-platform / Other"]
CATEGORIES = ["Environment / Package Error", "Syntax / Runtime Error", "Build / Dependency Error",
              "Configuration Error", "Network / Authentication Error", "Version Control Error",
              "Database / Query Error", "Tooling / IDE Error"]

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find sufficiently relevant community experiences to answer this confidently. "
    "Try adding the exact error message, technology and version, or submit a new experience."
)
