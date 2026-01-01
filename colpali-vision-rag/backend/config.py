import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
UPLOADS_DIR = DATA_DIR / "uploads"
PAGES_DIR = DATA_DIR / "pages"

# Create directories
for d in [DATA_DIR, INDEX_DIR, UPLOADS_DIR, PAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ColPali Configuration
COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.2")
COLPALI_DEVICE = os.getenv("COLPALI_DEVICE", "cpu")  # or "cuda"
INDEX_NAME = "documents"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Retrieval Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_IMAGES_IN_RESPONSE = int(os.getenv("MAX_IMAGES_IN_RESPONSE", "3"))
