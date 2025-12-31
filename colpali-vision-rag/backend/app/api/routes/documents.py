from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path

import config
from app.models.schemas import (
    DocumentListResponse, 
    DocumentInfo, 
    UploadResponse,
    ClearResponse
)
from app.core.retriever import retriever

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    docs = retriever.get_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs)
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and index PDF documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    indexed_count = 0
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        
        # Save uploaded file
        upload_path = config.UPLOADS_DIR / file.filename
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            # Index with ColPali
            doc_id, page_count = retriever.index_pdf(upload_path)
            indexed_count += 1
        except Exception as e:
            print(f"Error indexing {file.filename}: {e}")
            continue
    
    return UploadResponse(
        success=indexed_count > 0,
        message=f"Indexed {indexed_count} document(s)",
        documents_added=indexed_count
    )


@router.delete("/clear", response_model=ClearResponse)
async def clear_documents():
    """Clear all indexed documents."""
    try:
        retriever.clear_index()
        return ClearResponse(success=True, message="All documents cleared")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
