import torch

class PseudoRelevanceFeedback:
    """
    Lớp PseudoRelevanceFeedback thực hiện Pseudo-Relevance Feedback (PRF) để mở rộng truy vấn.
    PRF giúp cải thiện kết quả tìm kiếm bằng cách kết hợp vector truy vấn gốc với vector của các tài liệu liên quan.
    """
    
    def __init__(self, beta=0.4):
        """
        Khởi tạo thuật toán PRF.

        Args:
            beta (float): Hệ số điều chỉnh mức độ ảnh hưởng của tài liệu liên quan vào truy vấn mở rộng.
        """
        self.beta = beta  # Hệ số điều chỉnh trọng số của feedback từ tài liệu

    def apply_prf(self, query_embedding, doc_embeddings):
        """
        Áp dụng PRF để mở rộng vector truy vấn.

        Args:
            query_embedding (numpy.ndarray or torch.Tensor): Vector embedding của truy vấn ban đầu.
            doc_embeddings (list of numpy.ndarray or torch.Tensor): Danh sách các vector embedding của tài liệu liên quan.

        Returns:
            numpy.ndarray: Vector truy vấn đã mở rộng sau khi áp dụng PRF.
        """
        
        # Nếu query_embedding là một tensor PyTorch, chuyển nó sang numpy
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.numpy()

        # Sao chép query_embedding để không làm thay đổi dữ liệu gốc
        prf_embedding = query_embedding.copy()

        # Duyệt qua danh sách các tài liệu liên quan
        for idx in range(len(doc_embeddings)):
            doc_embedding = doc_embeddings[idx]
            
            # Nếu doc_embedding là tensor PyTorch, chuyển nó sang numpy
            if isinstance(doc_embedding, torch.Tensor):
                doc_embedding = doc_embedding.numpy()

            # Cập nhật vector truy vấn bằng cách cộng thêm trọng số của tài liệu
            prf_embedding += self.beta * doc_embedding  

        return prf_embedding  # Trả về truy vấn đã mở rộng

    
