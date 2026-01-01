from fastapi import APIRouter, UploadFile, File, HTTPException, Query
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
async def list_documents(mode: str = Query("fast", description="Model mode: 'fast' or 'deep'")):
    """List all indexed documents for a specific mode."""
    docs = retriever.get_documents(mode)
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs)
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index PDF documents with BOTH models.
    First indexes with ColSmol-500M (fast), then ColPali-v1.3 (deep).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    indexed_count = 0
    indexing_status = {"fast": [], "deep": []}
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        
        # Save uploaded file
        upload_path = config.UPLOADS_DIR / file.filename
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            # Index with ALL models
            results = retriever.index_pdf(upload_path)
            indexed_count += 1
            
            # Track status per model
            for mode, result in results.items():
                if result:
                    indexing_status[mode].append({
                        "file": file.filename,
                        "status": "success",
                        "pages": result[1]
                    })
                else:
                    indexing_status[mode].append({
                        "file": file.filename,
                        "status": "failed"
                    })
                    
        except Exception as e:
            print(f"Error indexing {file.filename}: {e}")
            for mode in indexing_status:
                indexing_status[mode].append({
                    "file": file.filename,
                    "status": "error",
                    "error": str(e)
                })
            continue
    
    return UploadResponse(
        success=indexed_count > 0,
        message=f"Indexed {indexed_count} document(s) with both models",
        documents_added=indexed_count,
        indexing_status=indexing_status
    )


@router.delete("/clear", response_model=ClearResponse)
async def clear_documents():
    """Clear all indexed documents from both models."""
    try:
        retriever.clear_all_indexes()
        return ClearResponse(success=True, message="All documents cleared from both models")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
