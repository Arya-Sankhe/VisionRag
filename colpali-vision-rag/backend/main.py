from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from app.api.routes import documents, chat, health

app = FastAPI(
    title="ColPali Vision RAG API",
    description="Visual document retrieval using ColPali",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for VPS deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    print("🚀 Starting ColPali Vision RAG API")
    print(f"   Model: {config.COLPALI_MODEL}")
    print(f"   Device: {config.COLPALI_DEVICE}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
