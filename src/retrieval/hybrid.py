from typing import List, Dict, Any
from retrieval.base import BaseRetriever

class HybridRetriever(BaseRetriever):
    """
    Kết hợp Dense & Sparse Retrieval sử dụng thuật toán Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, dense_retriever: BaseRetriever, sparse_retriever: BaseRetriever, rrf_k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Gọi đồng thời Dense và Sparse Retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2)
        
        # 2. Tính điểm RRF (Reciprocal Rank Fusion): score = sum(1 / (k + rank))
        doc_scores = {}
        
        for rank, item in enumerate(dense_results):
            doc_id = item['doc_id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        for rank, item in enumerate(sparse_results):
            doc_id = item['doc_id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # 3. Sắp xếp lại danh sách kết quả theo điểm RRF giảm dần
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Trả về đúng Top K kết quả duy nhất (mặc định <= 5)
        hybrid_results = [{"doc_id": doc_id, "rrf_score": score} for doc_id, score in sorted_docs[:top_k]]
        return hybrid_results
