from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.retriever import retriever

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check with model status."""
    return HealthResponse(
        status="healthy",
        models={
            "fast": {
                "loaded": retriever.is_model_loaded("fast"),
                "name": "ColSmol-500M"
            },
            "deep": {
                "loaded": retriever.is_model_loaded("deep"),
                "name": "ColPali-v1.3"
            }
        }
    )
