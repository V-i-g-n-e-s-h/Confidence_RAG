import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qa_collection")

QDRANT_TEXT_KEY = "text"
QDRANT_SOURCE_KEY = "source"
DOC_ID_KEY = "doc_id"
FILEPATH_KEY = "filepath"
CHUNK_INDEX_KEY = "chunk_index"
QDRANT_TOP_K = 5

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3-chatqa")

CONTEXT_MAX_CHARS = 4000

ENCODE_BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

CONF_THRESHOLD = 0.3
HIGH_T = 0.7
MED_T = 0.5

MAX_WORDS_PER_CHUNK = 200
CHUNK_WORD_OVERLAP = 40

MAX_TOKENS_PER_CHUNK = 256
