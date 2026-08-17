from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """
    Abstract Base Class định nghĩa Interface chuẩn cho tất cả các mô hình Retrieval (Strategy Pattern).
    """
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Phương thức truy vấn chính.
        :param query: Câu hỏi đầu vào.
        :param top_k: Số lượng tài liệu cần lấy ra (mặc định: 5 theo luật BTC).
        :return: Danh sách các dict chứa doc_id, score và metadata.
        """
        pass
