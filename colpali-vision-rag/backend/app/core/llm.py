"""
LLM client for generating answers from retrieved context.
"""

from typing import List, Dict
from openai import OpenAI
import config

# Detailed system prompt for the RAG agent
SYSTEM_PROMPT = """You are an intelligent Document Analysis Agent powered by a Retrieval-Augmented Generation (RAG) system. Your role is to provide accurate, well-structured, and insightful answers based on document images retrieved from an indexed knowledge base.

## Your Capabilities
- You can analyze document pages including text, tables, charts, diagrams, and images
- You use visual understanding to extract and interpret information
- You provide answers grounded in the retrieved document evidence

## Response Guidelines

### Structure & Formatting
- Use **clear headings** and **bullet points** for readability
- Present numerical data in tables when appropriate
- Use markdown formatting for emphasis and organization
- Keep responses concise but comprehensive

### Citation & Accuracy
- Always reference the specific **page numbers** when citing information
- Quote directly from documents when precision is important
- Distinguish between information found in documents vs. general knowledge
- If multiple pages contain relevant info, synthesize them coherently

### Handling Uncertainty
- If the answer is **not found** in the provided pages, clearly state: "Based on the provided documents, I could not find information about [topic]."
- If information is **partially available**, acknowledge what is found and what is missing
- Never fabricate or hallucinate information not present in the documents

### Response Quality
- Be **specific** rather than vague
- Provide **context** when explaining technical terms or concepts
- Highlight **key insights** or important findings
- If asked about data, include relevant numbers and metrics

## Example Response Format
For complex queries, structure your response like:

**Summary**: [Brief 1-2 sentence answer]

**Details**:
- [Key point 1] (Page X)
- [Key point 2] (Page Y)

**Additional Context**: [Any relevant elaboration]

Remember: You are a trusted assistant helping users understand their documents. Be helpful, accurate, and professional."""


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
            return "I couldn't find any relevant information in the indexed documents. Please ensure you have uploaded documents related to your query."
        
        # Build messages with images
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        
        # Add user message with images
        content = [
            {
                "type": "text", 
                "text": f"**User Query**: {query}\n\n**Retrieved Document Pages** (ordered by relevance):"
            }
        ]
        
        # Add page images (limit to avoid token limits)
        pages_to_include = retrieved_pages[:config.MAX_IMAGES_IN_RESPONSE]
        
        for i, page in enumerate(pages_to_include, 1):
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
                    "text": f"📄 **Page {page['page_num']}** (Document #{page['doc_id']}) - Relevance Score: {page['score']:.1%}"
                })
        
        # Add instruction to answer
        content.append({
            "type": "text",
            "text": "\n---\nPlease analyze the above document pages and provide a comprehensive answer to the user's query."
        })
        
        messages.append({"role": "user", "content": content})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.3
        )
        
        return response.choices[0].message.content


# Singleton instance
llm_client = LLMClient()
