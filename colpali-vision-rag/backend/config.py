import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PAGES_DIR = DATA_DIR / "pages"

# Create base directories
for d in [DATA_DIR, UPLOADS_DIR, PAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = os.getenv("COLPALI_DEVICE", "cpu")  # or "cuda"

# Dual Model Configuration
MODELS = {
    "fast": {
        "name": "vidore/colSmol-500M",
        "model_class": "ColIdefics3",
        "processor_class": "ColIdefics3Processor",
        "index_dir": DATA_DIR / "index_fast",
        "display_name": "Fast (ColSmol-500M)"
    },
    "deep": {
        "name": "vidore/colpali-v1.3",
        "model_class": "ColPali",
        "processor_class": "ColPaliProcessor",
        "index_dir": DATA_DIR / "index_deep",
        "display_name": "Deep (ColPali-v1.3)"
    }
}

# Create index directories for each model
for mode_config in MODELS.values():
    mode_config["index_dir"].mkdir(parents=True, exist_ok=True)

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
