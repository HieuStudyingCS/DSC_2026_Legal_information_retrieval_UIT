from typing import List, Dict, Any
from retrieval.base import BaseRetriever

class SparseRetriever(BaseRetriever):
    """
    Truy vấn dựa trên Từ khóa BM25 (Sparse Keyword Search).
    """
    def __init__(self, bm25_path: str, doc_mapping_path: str = None):
        self.bm25_path = bm25_path
        self.doc_mapping_path = doc_mapping_path
        # TODO: Load BM25 Index từ file .pkl tại đây

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # TODO: Tách từ tiếng Việt & truy vấn từ khóa trên BM25 Index
        results = []
        return results
