from .base_tool import BaseTool
from .pdf_loader import get_pdf_loader
import logging

logger = logging.getLogger("document_tool")


class DocumentTool(BaseTool):
    name = "document_tool"

    def __init__(self):
        super().__init__()
        self.pdf_loader = get_pdf_loader()

    def analyze_document(self, document: dict = None) -> dict:
        """
        Analyze the loaded PDF document.
        Returns statistics and content summary.
        """
        try:
            full_text = self.pdf_loader.get_all_text()
            
            return {
                "status": "success",
                "length": len(full_text),
                "num_pages": len(self.pdf_loader.documents),
                "num_lines": full_text.count("\n"),
                "num_words": len(full_text.split()),
                "num_chunks": len(self.pdf_loader.chunks),
                "metadata": self.pdf_loader.metadata,
            }
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def summarize_document(self, document: dict = None) -> dict:
        """
        Return summary of document with actual content preview.
        """
        try:
            full_text = self.pdf_loader.get_all_text()
            
            return {
                "status": "success",
                "preview": full_text[:1000],
                "total_chars": len(full_text),
                "pages": len(self.pdf_loader.documents),
                "summary": f"PDF document with {len(self.pdf_loader.documents)} pages, {len(full_text)} characters",
            }
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def search_document(self, query: str) -> dict:
        """
        Search document for relevant chunks matching the query.
        """
        try:
            results = self.pdf_loader.search_content(query, top_k=3)
            
            return {
                "status": "success",
                "query": query,
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
            logger.error(f"Error searching document: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def get_page_content(self, page_num: int) -> dict:
        """
        Get content from a specific page.
        """
        try:
            content = self.pdf_loader.get_page_content(page_num)
            
            return {
                "status": "success",
                "page": page_num,
                "content": content,
                "length": len(content),
            }
        except Exception as e:
            logger.error(f"Error getting page content: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
