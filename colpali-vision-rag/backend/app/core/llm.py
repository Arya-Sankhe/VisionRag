"""
LLM client for generating answers from retrieved context.
"""

from typing import List, Dict
from openai import OpenAI
import config

class LLMClient:
    """OpenAI client for answer generation."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
    
    def generate_answer(
        self,
        query: str,
        retrieved_pages: List[Dict],
        include_images: bool = True
    ) -> str:
        """
        Generate an answer using retrieved page images as context.
        
        Uses GPT-4o vision capabilities to understand page images.
        """
        if not retrieved_pages:
            return "I couldn't find any relevant information in the documents."
        
        # Build messages with images
        messages = [
            {
                "role": "system",
                "content": """You are a helpful document assistant. Answer questions based on the provided document pages.
                
Instructions:
- Answer based ONLY on the visible content in the provided page images
- Be specific and cite page numbers when relevant
- If the answer is not found in the pages, say so clearly
- Format your response with markdown for readability"""
            }
        ]
        
        # Add user message with images
        content = [{"type": "text", "text": f"Question: {query}\n\nHere are the relevant document pages:"}]
        
        # Add page images (limit to avoid token limits)
        pages_to_include = retrieved_pages[:config.MAX_IMAGES_IN_RESPONSE]
        
        for page in pages_to_include:
            if include_images and page.get("image_base64"):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page['image_base64']}",
                        "detail": "high"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"[Page {page['page_num']} from Document {page['doc_id']}, relevance: {page['score']:.2f}]"
                })
        
        messages.append({"role": "user", "content": content})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3
        )
        
        return response.choices[0].message.content


# Singleton instance
llm_client = LLMClient()
