# ColPali Vision RAG

A visual document Q&A system using ColPali (colSmol-500M) for visual retrieval and GPT-4o for answer generation.

## Features

- 📄 **Visual Document Understanding**: Uses ColPali embeddings instead of OCR for native understanding of charts, tables, and layouts
- 🔍 **Semantic Page Retrieval**: Find relevant pages using visual similarity
- 💬 **Interactive Chat**: Ask questions and get answers with page references
- 🖼️ **Image Display**: View retrieved page images directly in chat

## Quick Start

1. **Clone and setup**:
   ```bash
   cd colpali-vision-rag
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY
   ```

2. **Run with Docker**:
   ```bash
   docker-compose up --build
   ```

3. **Use the application**:
   - Open http://localhost:3000
   - Upload PDF documents in the sidebar
   - Ask questions in the chat

## Tech Stack

- **Backend**: FastAPI + Python 3.11 + Byaldi + OpenAI
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **Retrieval**: ColPali (colSmol-500M via Byaldi)
- **LLM**: GPT-4o-mini (vision capabilities)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/documents` | GET | List indexed documents |
| `/api/v1/documents/upload` | POST | Upload and index PDFs |
| `/api/v1/documents/clear` | DELETE | Clear all documents |
| `/api/v1/chat` | POST | Query documents |

## Requirements

- Docker and Docker Compose
- OpenAI API key
- ~3GB RAM (for ColPali model on CPU)
