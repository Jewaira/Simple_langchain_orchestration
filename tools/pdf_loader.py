"""
PDF Loader Utility - Loads and processes PDF content for RAG
"""
import os
from typing import List, Dict, Any
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


class PDFLoader:
    """Load and process PDF documents for RAG applications."""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.metadata = {}
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        """
        Load PDF and extract all text.
        Returns list of text pages.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            pages = []
            
            # Extract metadata
            if reader.metadata:
                self.metadata = {k: v for k, v in reader.metadata.items()}
            
            # Extract text from all pages
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(text)
            
            self.documents = pages
            return pages
        
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def split_into_chunks(self, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap for better context preservation.
        Returns list of chunks with metadata.
        """
        if not self.documents:
            return []
        
        chunks = []
        
        for page_num, page_text in enumerate(self.documents):
            # Split page into sentences/chunks
            words = page_text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num + 1,
                        "chunk_index": len(chunks),
                        "metadata": {
                            "source": "jewaira.pdf",
                            "page": page_num + 1,
                        }
                    })
        
        self.chunks = chunks
        return chunks
    
    def search_content(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search (can be enhanced with embeddings later).
        Returns top matching chunks.
        """
        if not self.chunks:
            return []
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            text = chunk["text"].lower()
            # Count matching words
            matches = sum(1 for word in query_words if word in text)
            
            if matches > 0:
                scored_chunks.append({
                    **chunk,
                    "score": matches / len(query_words) if query_words else 0
                })
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
    
    def get_all_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(self.documents)
    
    def get_page_content(self, page_num: int) -> str:
        """Get text from specific page (1-indexed)."""
        if 1 <= page_num <= len(self.documents):
            return self.documents[page_num - 1]
        return ""


# Global loader instance
_loader = None


def get_pdf_loader() -> PDFLoader:
    """Get or initialize the global PDF loader."""
    global _loader
    if _loader is None:
        _loader = PDFLoader()
        pdf_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "data", 
            "jewaira.pdf"
        )
        _loader.load_pdf(pdf_path)
        _loader.split_into_chunks(chunk_size=500, overlap=100)
    
    return _loader
