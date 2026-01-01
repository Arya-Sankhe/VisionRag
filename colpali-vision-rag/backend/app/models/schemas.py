from pydantic import BaseModel
from typing import List, Optional, Literal

class HealthResponse(BaseModel):
    status: str
    models: dict  # Show status of each model

class DocumentInfo(BaseModel):
    id: str
    name: str
    page_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int

class UploadResponse(BaseModel):
    success: bool
    message: str
    documents_added: int
    indexing_status: dict  # Status for each model

class RetrievedPage(BaseModel):
    doc_id: int
    page_num: int
    score: float
    image_base64: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    mode: Literal["fast", "deep"] = "fast"  # Model mode selection
    include_images: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedPage]
    mode: str  # Which model was used

class ClearResponse(BaseModel):
    success: bool
    message: str
