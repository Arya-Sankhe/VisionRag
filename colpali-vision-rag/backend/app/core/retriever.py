"""
ColPali/ColSmol Retriever using colpali-engine directly.
Handles document indexing and visual retrieval with support for ColSmol-500M.
"""

import base64
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from pdf2image import convert_from_path
from dataclasses import dataclass

import config

# Import from colpali-engine based on model type
def get_model_and_processor(model_name: str, device: str):
    """Load the appropriate model and processor based on model name."""
    model_name_lower = model_name.lower()
    
    if "colsmol" in model_name_lower or "smol" in model_name_lower:
        # ColSmol uses transformers directly
        from transformers import AutoProcessor, AutoModel
        print(f"📦 Loading ColSmol model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        return model, processor, "colsmol"
        
    elif "colqwen2" in model_name_lower:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        print(f"📦 Loading ColQwen2 model: {model_name}")
        
        processor = ColQwen2Processor.from_pretrained(model_name)
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
        return model, processor, "colqwen2"
        
    else:
        # Default to ColPali
        from colpali_engine.models import ColPali, ColPaliProcessor
        print(f"📦 Loading ColPali model: {model_name}")
        
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
        return model, processor, "colpali"


@dataclass
class SearchResult:
    """Search result container."""
    doc_id: int
    page_num: int
    score: float
    

class ColPaliRetriever:
    """Retriever supporting ColPali, ColQwen2, and ColSmol models."""
    
    _instance = None
    _model = None
    _processor = None
    _model_type = None
    _document_registry: Dict[str, Dict] = {}
    _embeddings: List[torch.Tensor] = []  # Store page embeddings
    _embed_to_doc: List[Dict] = []  # Map embedding index to doc/page
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._document_registry = {}
            cls._instance._embeddings = []
            cls._instance._embed_to_doc = []
        return cls._instance
    
    def _ensure_model_loaded(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        print(f"🔄 Loading model: {config.COLPALI_MODEL}")
        
        self._model, self._processor, self._model_type = get_model_and_processor(
            config.COLPALI_MODEL, 
            config.COLPALI_DEVICE
        )
        self._model = self._model.eval()
        
        if config.COLPALI_DEVICE != "cuda":
            self._model = self._model.to(config.COLPALI_DEVICE)
        
        # Load existing index if present
        self._load_index()
        
        print(f"✅ Model ready ({self._model_type})")
    
    def _load_index(self):
        """Load existing index from disk."""
        registry_path = config.INDEX_DIR / "registry.json"
        embeddings_path = config.INDEX_DIR / "embeddings.pt"
        mapping_path = config.INDEX_DIR / "embed_mapping.json"
        
        if registry_path.exists():
            self._document_registry = json.loads(registry_path.read_text())
            
        if embeddings_path.exists():
            self._embeddings = torch.load(embeddings_path, weights_only=False)
            
        if mapping_path.exists():
            self._embed_to_doc = json.loads(mapping_path.read_text())
    
    def _save_index(self):
        """Save index to disk."""
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        registry_path = config.INDEX_DIR / "registry.json"
        registry_path.write_text(json.dumps(self._document_registry, indent=2))
        
        if self._embeddings:
            embeddings_path = config.INDEX_DIR / "embeddings.pt"
            torch.save(self._embeddings, embeddings_path)
            
        mapping_path = config.INDEX_DIR / "embed_mapping.json"
        mapping_path.write_text(json.dumps(self._embed_to_doc, indent=2))
    
    def _embed_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Generate embeddings for a list of images."""
        embeddings = []
        
        with torch.no_grad():
            for img in images:
                if self._model_type == "colsmol":
                    # ColSmol uses different processing
                    inputs = self._processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                    outputs = self._model(**inputs)
                    # Get the image embeddings (last hidden state)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu()
                else:
                    # ColPali/ColQwen2 use process_images
                    batch = self._processor.process_images([img]).to(self._model.device)
                    emb = self._model(**batch).cpu()
                
                embeddings.append(emb)
        
        return embeddings
    
    def _embed_query(self, query: str) -> torch.Tensor:
        """Generate embedding for a query."""
        with torch.no_grad():
            if self._model_type == "colsmol":
                # ColSmol query embedding
                inputs = self._processor(text=query, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu()
            else:
                # ColPali/ColQwen2
                batch = self._processor.process_queries([query]).to(self._model.device)
                return self._model(**batch).cpu()
    
    def index_pdf(self, pdf_path: Path) -> Tuple[int, int]:
        """Index a PDF document."""
        self._ensure_model_loaded()
        
        doc_name = pdf_path.stem
        
        # Convert PDF to images
        print(f"📄 Converting PDF to images: {doc_name}")
        images = convert_from_path(str(pdf_path), dpi=150)
        page_count = len(images)
        
        # Save page images for later retrieval
        doc_pages_dir = config.PAGES_DIR / doc_name
        doc_pages_dir.mkdir(parents=True, exist_ok=True)
        
        pil_images = []
        for i, img in enumerate(images):
            img_path = doc_pages_dir / f"page_{i+1}.png"
            img.save(str(img_path), "PNG")
            pil_images.append(img)
        
        # Generate embeddings
        print(f"🔍 Indexing {page_count} pages...")
        doc_id = len(self._document_registry)
        
        page_embeddings = self._embed_images(pil_images)
        
        # Store embeddings and mapping
        start_idx = len(self._embeddings)
        for i, emb in enumerate(page_embeddings):
            self._embeddings.append(emb)
            self._embed_to_doc.append({
                "doc_id": doc_id,
                "page_num": i + 1
            })
        
        # Update registry
        self._document_registry[doc_name] = {
            "id": doc_id,
            "name": doc_name,
            "page_count": page_count,
            "path": str(doc_pages_dir),
            "embed_start": start_idx,
            "embed_end": start_idx + page_count
        }
        
        self._save_index()
        
        print(f"✅ Indexed {doc_name}: {page_count} pages")
        return doc_id, page_count
    
    def search(
        self, 
        query: str, 
        k: int = None,
        include_images: bool = True
    ) -> List[Dict]:
        """Search for relevant pages."""
        self._ensure_model_loaded()
        k = k or config.TOP_K_RESULTS
        
        if not self._embeddings:
            return []
        
        print(f"🔍 Searching: '{query[:50]}...'")
        
        # Get query embedding
        query_emb = self._embed_query(query)
        
        # Calculate similarities
        scores = []
        for i, page_emb in enumerate(self._embeddings):
            # Compute similarity (dot product or cosine)
            if self._model_type == "colsmol":
                # Simple cosine similarity for ColSmol
                sim = torch.nn.functional.cosine_similarity(
                    query_emb.flatten().unsqueeze(0),
                    page_emb.flatten().unsqueeze(0)
                ).item()
            else:
                # Late interaction scoring for ColPali/ColQwen2
                sim = self._processor.score_multi_vector(query_emb, page_emb)[0][0].item()
            
            scores.append((i, sim))
        
        # Sort by score and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:k]
        
        # Build results
        processed = []
        for idx, score in top_results:
            mapping = self._embed_to_doc[idx]
            item = {
                "doc_id": mapping["doc_id"],
                "page_num": mapping["page_num"],
                "score": float(score)
            }
            
            if include_images:
                image_b64 = self._get_page_image_base64(
                    mapping["doc_id"], 
                    mapping["page_num"]
                )
                if image_b64:
                    item["image_base64"] = image_b64
            
            processed.append(item)
        
        return processed
    
    def _get_page_image_base64(self, doc_id: int, page_num: int) -> Optional[str]:
        """Get base64 encoded page image."""
        for doc_name, doc_info in self._document_registry.items():
            if doc_info["id"] == doc_id:
                img_path = Path(doc_info["path"]) / f"page_{page_num}.png"
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
        return None
    
    def get_documents(self) -> List[Dict]:
        """Get list of indexed documents."""
        return [
            {
                "id": str(info["id"]),
                "name": info["name"],
                "page_count": info["page_count"]
            }
            for info in self._document_registry.values()
        ]
    
    def clear_index(self):
        """Clear all indexed documents."""
        import shutil
        
        for d in [config.INDEX_DIR, config.PAGES_DIR]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        self._document_registry = {}
        self._embeddings = []
        self._embed_to_doc = []
        self._model = None
        
        print("🗑️ Index cleared")
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Singleton instance
retriever = ColPaliRetriever()
