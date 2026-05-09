from .base_tool import BaseTool
from .pdf_loader import get_pdf_loader
import logging

logger = logging.getLogger("rag_tool")


class RAGTool(BaseTool):
    name = "rag_tool"

    def __init__(self):
        super().__init__()
        self.pdf_loader = get_pdf_loader()

    def retrieve(self, query: str) -> dict:
        """
        Retrieve relevant content from PDF based on query.
        Uses keyword matching to find similar chunks.
        """
        try:
            if not query or not query.strip():
                return {
                    "status": "error",
                    "error": "Empty query provided",
                    "coverage": "none",
                }
            
            # Search for relevant chunks
            results = self.pdf_loader.search_content(query, top_k=3)
            
            if not results:
                return {
                    "status": "success",
                    "query": query,
                    "coverage": "none",
                    "num_results": 0,
                    "message": "No matching content found for this query",
                }
            
            # Format results
            excerpts = []
            references = []
            
            for i, result in enumerate(results):
                excerpts.append({
                    "text": result["text"][:300],  # Truncate to 300 chars
                    "page": result["page"],
                    "relevance_score": result.get("score", 0),
                })
                
                references.append({
                    "doc": "jewaira.pdf",
                    "page": result["page"],
                    "chunk_id": result["chunk_index"],
                })
            
            return {
                "status": "success",
                "query": query,
                "coverage": "full" if len(results) >= 3 else "partial",
                "num_results": len(results),
                "excerpts": excerpts,
                "references": references,
                "context": "\n\n".join([r["text"] for r in results]),
            }
        
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "coverage": "none",
            }
    
    def retrieve_by_topic(self, topic: str, top_k: int = 5) -> dict:
        """
        Retrieve multiple chunks related to a topic.
        """
        try:
            results = self.pdf_loader.search_content(topic, top_k=top_k)
            
            return {
                "status": "success",
                "topic": topic,
                "num_results": len(results),
                "results": [
                    {
                        "text": r["text"],
                        "page": r["page"],
                        "score": r.get("score", 0),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error in topic retrieval: {e}")
            return {
                "status": "error",
                "topic": topic,
                "error": str(e),
            }
    
    def get_document_summary(self) -> dict:
        """
        Get overview of loaded document.
        """
        try:
            full_text = self.pdf_loader.get_all_text()
            
            return {
                "status": "success",
                "document": "jewaira.pdf",
                "pages": len(self.pdf_loader.documents),
                "total_characters": len(full_text),
                "total_words": len(full_text.split()),
                "chunks_available": len(self.pdf_loader.chunks),
            }
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
