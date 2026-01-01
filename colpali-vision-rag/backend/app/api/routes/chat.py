from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse, RetrievedPage
from app.core.retriever import retriever
from app.core.llm import llm_client

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Query the document collection and generate an answer.
    Uses the specified mode (fast/deep) for retrieval.
    """
    # Retrieve relevant pages using selected mode
    results = retriever.search(
        query=request.query,
        mode=request.mode  # "fast" or "deep"
    )
    
    # Generate answer using LLM with vision
    answer = llm_client.generate_answer(
        query=request.query,
        retrieved_pages=results,
        include_images=request.include_images
    )
    
    # Build response
    sources = [
        RetrievedPage(
            doc_id=r["doc_id"],
            page_num=r["page_num"],
            score=r["score"],
            image_base64=r.get("image_base64")
        )
        for r in results
    ]
    
    return ChatResponse(
        answer=answer, 
        sources=sources,
        mode=request.mode
    )
