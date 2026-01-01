"""
Multi-Model ColPali Retriever.
Supports ColSmol-500M (Fast) and ColPali-v1.3 (Deep) with separate indexes.
"""

import base64
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from pdf2image import convert_from_path

import config


def load_model_and_processor(model_config: dict, device: str):
    """Load model and processor based on configuration."""
    model_class = model_config["model_class"]
    processor_class = model_config["processor_class"]
    model_name = model_config["name"]
    
    print(f"📦 Loading {model_name} ({model_class})...")
    
    if model_class == "ColIdefics3":
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        processor = ColIdefics3Processor.from_pretrained(model_name)
        model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            attn_implementation="eager"
        )
    elif model_class == "ColPali":
        from colpali_engine.models import ColPali, ColPaliProcessor
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
    elif model_class == "ColQwen2":
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        processor = ColQwen2Processor.from_pretrained(model_name)
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model = model.eval()
    if device != "cuda":
        model = model.to(device)
    
    print(f"✅ {model_name} loaded successfully")
    return model, processor


class ModelIndex:
    """Manages embeddings and index for a single model."""
    
    def __init__(self, mode: str, model_config: dict):
        self.mode = mode
        self.config = model_config
        self.index_dir = model_config["index_dir"]
        self.model = None
        self.processor = None
        self.embeddings: List[torch.Tensor] = []
        self.embed_to_doc: List[Dict] = []
        self.document_registry: Dict[str, Dict] = {}
        
        # Load existing index if present
        self._load_index()
    
    def _load_index(self):
        """Load existing index from disk."""
        registry_path = self.index_dir / "registry.json"
        embeddings_path = self.index_dir / "embeddings.pt"
        mapping_path = self.index_dir / "embed_mapping.json"
        
        if registry_path.exists():
            self.document_registry = json.loads(registry_path.read_text())
        if embeddings_path.exists():
            self.embeddings = torch.load(embeddings_path, weights_only=False)
        if mapping_path.exists():
            self.embed_to_doc = json.loads(mapping_path.read_text())
    
    def _save_index(self):
        """Save index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        registry_path = self.index_dir / "registry.json"
        registry_path.write_text(json.dumps(self.document_registry, indent=2))
        
        if self.embeddings:
            embeddings_path = self.index_dir / "embeddings.pt"
            torch.save(self.embeddings, embeddings_path)
        
        mapping_path = self.index_dir / "embed_mapping.json"
        mapping_path.write_text(json.dumps(self.embed_to_doc, indent=2))
    
    def ensure_model_loaded(self):
        """Lazy load the model."""
        if self.model is not None:
            return
        
        self.model, self.processor = load_model_and_processor(
            self.config, config.DEVICE
        )
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"🗑️ Unloaded {self.config['name']} to free memory")
    
    def index_images(self, doc_name: str, images: List[Image.Image], doc_pages_dir: Path) -> Tuple[int, int]:
        """Index images for this model."""
        self.ensure_model_loaded()
        
        page_count = len(images)
        doc_id = len(self.document_registry)
        
        print(f"🔍 [{self.mode.upper()}] Indexing {page_count} pages...")
        
        # Generate embeddings
        start_idx = len(self.embeddings)
        
        with torch.no_grad():
            for i, img in enumerate(images):
                batch = self.processor.process_images([img]).to(self.model.device)
                emb = self.model(**batch).cpu()
                self.embeddings.append(emb)
                self.embed_to_doc.append({
                    "doc_id": doc_id,
                    "page_num": i + 1
                })
        
        # Update registry
        self.document_registry[doc_name] = {
            "id": doc_id,
            "name": doc_name,
            "page_count": page_count,
            "path": str(doc_pages_dir),
            "embed_start": start_idx,
            "embed_end": start_idx + page_count
        }
        
        self._save_index()
        print(f"✅ [{self.mode.upper()}] Indexed {doc_name}: {page_count} pages")
        
        return doc_id, page_count
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """Search for relevant pages."""
        self.ensure_model_loaded()
        k = k or config.TOP_K_RESULTS
        
        if not self.embeddings:
            return []
        
        print(f"🔍 [{self.mode.upper()}] Searching: '{query[:50]}...'")
        
        # Get query embedding
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.model.device)
            query_emb = self.model(**batch).cpu()
        
        # Calculate similarities
        scores = []
        for i, page_emb in enumerate(self.embeddings):
            sim = self.processor.score_multi_vector(query_emb, page_emb)[0][0].item()
            scores.append((i, sim))
        
        # Sort by score and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:k]
        
        # Build results
        processed = []
        for idx, score in top_results:
            mapping = self.embed_to_doc[idx]
            item = {
                "doc_id": mapping["doc_id"],
                "page_num": mapping["page_num"],
                "score": float(score)
            }
            
            # Load image
            image_b64 = self._get_page_image_base64(mapping["doc_id"], mapping["page_num"])
            if image_b64:
                item["image_base64"] = image_b64
            
            processed.append(item)
        
        return processed
    
    def _get_page_image_base64(self, doc_id: int, page_num: int) -> Optional[str]:
        """Get base64 encoded page image."""
        for doc_name, doc_info in self.document_registry.items():
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
            for info in self.document_registry.values()
        ]
    
    def clear_index(self):
        """Clear index for this model."""
        import shutil
        
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.document_registry = {}
        self.embeddings = []
        self.embed_to_doc = []
        
        print(f"🗑️ [{self.mode.upper()}] Index cleared")


class MultiModelRetriever:
    """Manages multiple model indexes."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.indexes: Dict[str, ModelIndex] = {}
        for mode, model_config in config.MODELS.items():
            self.indexes[mode] = ModelIndex(mode, model_config)
        
        self._initialized = True
    
    def index_pdf(self, pdf_path: Path) -> Dict[str, Tuple[int, int]]:
        """
        Index a PDF with all models.
        Returns dict of mode -> (doc_id, page_count) for each model.
        """
        doc_name = pdf_path.stem
        
        # Convert PDF to images (shared across models)
        print(f"📄 Converting PDF to images: {doc_name}")
        images = convert_from_path(str(pdf_path), dpi=150)
        page_count = len(images)
        
        # Save page images
        doc_pages_dir = config.PAGES_DIR / doc_name
        doc_pages_dir.mkdir(parents=True, exist_ok=True)
        
        pil_images = []
        for i, img in enumerate(images):
            img_path = doc_pages_dir / f"page_{i+1}.png"
            img.save(str(img_path), "PNG")
            pil_images.append(img)
        
        # Index with each model sequentially
        results = {}
        for mode, index in self.indexes.items():
            try:
                doc_id, page_count = index.index_images(doc_name, pil_images, doc_pages_dir)
                results[mode] = (doc_id, page_count)
                # Unload model to free memory before loading next
                index.unload_model()
            except Exception as e:
                print(f"❌ [{mode.upper()}] Error indexing: {e}")
                results[mode] = None
        
        return results
    
    def search(self, query: str, mode: str = "fast", k: int = None) -> List[Dict]:
        """Search using the specified mode."""
        if mode not in self.indexes:
            raise ValueError(f"Unknown mode: {mode}. Use 'fast' or 'deep'.")
        
        return self.indexes[mode].search(query, k)
    
    def get_documents(self, mode: str = "fast") -> List[Dict]:
        """Get list of documents indexed for a specific mode."""
        if mode not in self.indexes:
            return []
        return self.indexes[mode].get_documents()
    
    def clear_all_indexes(self):
        """Clear indexes for all models."""
        import shutil
        
        # Also clear pages directory
        if config.PAGES_DIR.exists():
            shutil.rmtree(config.PAGES_DIR)
        config.PAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        for index in self.indexes.values():
            index.clear_index()
        
        print("🗑️ All indexes cleared")
    
    def is_model_loaded(self, mode: str) -> bool:
        """Check if a specific model is loaded."""
        if mode not in self.indexes:
            return False
        return self.indexes[mode].model is not None


# Singleton instance
retriever = MultiModelRetriever()
