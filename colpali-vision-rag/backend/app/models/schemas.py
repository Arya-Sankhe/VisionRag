from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

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

class RetrievedPage(BaseModel):
    doc_id: int
    page_num: int
    score: float
    image_base64: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    include_images: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedPage]

class ClearResponse(BaseModel):
    success: bool
    message: str
