from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.retriever import retriever

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=retriever.is_loaded
    )
