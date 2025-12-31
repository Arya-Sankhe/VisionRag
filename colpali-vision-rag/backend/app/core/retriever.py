"""
ColPali Retriever using Byaldi wrapper.
Handles document indexing and visual retrieval.
"""

import base64
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import io

from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path

import config

class ColPaliRetriever:
    """Singleton wrapper for Byaldi/ColPali model."""
    
    _instance = None
    _model = None
    _document_registry: Dict[str, Dict] = {}  # Track indexed documents
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _ensure_model_loaded(self):
        """Lazy load the ColPali model."""
        if self._model is not None:
            return
        
        print(f"🔄 Loading ColPali model: {config.COLPALI_MODEL}")
        
        # Check if index exists
        index_path = config.INDEX_DIR / config.INDEX_NAME
        if index_path.exists():
            print(f"📁 Loading existing index from {index_path}")
            self._model = RAGMultiModalModel.from_index(
                str(index_path),
                index_root=str(config.INDEX_DIR)
            )
            self._load_registry()
        else:
            print(f"🆕 Creating new model instance")
            self._model = RAGMultiModalModel.from_pretrained(
                config.COLPALI_MODEL,
                device=config.COLPALI_DEVICE
            )
        
        print(f"✅ ColPali model ready")
    
    def _load_registry(self):
        """Load document registry from disk."""
        registry_path = config.INDEX_DIR / "registry.json"
        if registry_path.exists():
            self._document_registry = json.loads(registry_path.read_text())
    
    def _save_registry(self):
        """Save document registry to disk."""
        registry_path = config.INDEX_DIR / "registry.json"
        registry_path.write_text(json.dumps(self._document_registry, indent=2))
    
    def index_pdf(self, pdf_path: Path) -> Tuple[int, int]:
        """
        Index a PDF document.
        
        Returns:
            Tuple of (doc_id, page_count)
        """
        self._ensure_model_loaded()
        
        doc_name = pdf_path.stem
        
        # Convert PDF to images
        print(f"📄 Converting PDF to images: {doc_name}")
        images = convert_from_path(str(pdf_path), dpi=150)
        page_count = len(images)
        
        # Save page images for later retrieval
        doc_pages_dir = config.PAGES_DIR / doc_name
        doc_pages_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            img_path = doc_pages_dir / f"page_{i+1}.png"
            img.save(str(img_path), "PNG")
        
        # Index with Byaldi
        print(f"🔍 Indexing {page_count} pages with ColPali...")
        
        # Check if this is first document or adding to existing
        if not self._document_registry:
            # First document - create new index
            self._model.index(
                input_path=str(doc_pages_dir),
                index_name=config.INDEX_NAME,
                store_collection_with_index=True,
                overwrite=True
            )
            doc_id = 0
        else:
            # Add to existing index
            doc_id = len(self._document_registry)
            self._model.add_to_index(
                input_path=str(doc_pages_dir),
                store_collection_with_index=True
            )
        
        # Update registry
        self._document_registry[doc_name] = {
            "id": doc_id,
            "name": doc_name,
            "page_count": page_count,
            "path": str(doc_pages_dir)
        }
        self._save_registry()
        
        print(f"✅ Indexed {doc_name}: {page_count} pages")
        return doc_id, page_count
    
    def search(
        self, 
        query: str, 
        k: int = None,
        include_images: bool = True
    ) -> List[Dict]:
        """
        Search for relevant pages.
        
        Returns:
            List of result dicts with doc_id, page_num, score, and optionally image_base64
        """
        self._ensure_model_loaded()
        k = k or config.TOP_K_RESULTS
        
        if not self._document_registry:
            return []
        
        print(f"🔍 Searching: '{query[:50]}...'")
        
        results = self._model.search(query, k=k)
        
        processed = []
        for result in results:
            item = {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": float(result.score)
            }
            
            if include_images:
                # Load and encode the page image
                image_b64 = self._get_page_image_base64(
                    result.doc_id, 
                    result.page_num
                )
                if image_b64:
                    item["image_base64"] = image_b64
            
            processed.append(item)
        
        return processed
    
    def _get_page_image_base64(self, doc_id: int, page_num: int) -> Optional[str]:
        """Get base64 encoded page image."""
        # Find document by ID
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
        
        # Clear directories
        for d in [config.INDEX_DIR, config.PAGES_DIR]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        
        # Reset state
        self._document_registry = {}
        self._model = None
        
        print("🗑️ Index cleared")
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Singleton instance
retriever = ColPaliRetriever()
