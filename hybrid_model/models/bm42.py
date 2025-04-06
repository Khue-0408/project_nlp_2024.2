import math
import numpy as np
from models.encoder import TextEncoder  # Mô hình embedding để mã hóa văn bản
from retrieval.prf import PseudoRelevanceFeedback  # PRF giúp mở rộng truy vấn

class BM42:
    """
    BM42: Một biến thể của BM25, kết hợp vector embedding và Pseudo-Relevance Feedback (PRF).
    """
    
    def __init__(self, k1=1.2, b=0.75, beta=0.4):
        """
        Khởi tạo BM42 với các tham số:

        Args:
            k1 (float): Hệ số điều chỉnh trọng số TF (giống BM25).
            b (float): Hệ số điều chỉnh độ dài tài liệu (giống BM25).
            beta (float): Trọng số của Pseudo-Relevance Feedback (PRF).
        """
        self.k1 = k1  
        self.b = b  
        self.beta = beta  

        self.doc_lengths = {}  # Lưu độ dài của từng tài liệu
        self.avg_doc_length = 0  # Độ dài trung bình của tài liệu trong tập dữ liệu
        self.doc_freqs = {}  # Tần suất xuất hiện của từ trong toàn bộ corpus

        self.encoder = TextEncoder()  # Mô hình mã hóa văn bản thành vector embedding
        self.prf = PseudoRelevanceFeedback(beta=beta)  # Khởi tạo PRF với beta

    def index_documents(self, corpus):
        """
        Xây dựng index cho tập tài liệu.

        Args:
            corpus (dict): Dictionary chứa các tài liệu cần index, dạng {doc_id: doc_text}
        """
        total_length = 0  # Tổng độ dài của tất cả tài liệu
        self.embeddings = []  # Danh sách lưu vector embedding của tài liệu
        self.doc_ids = []  # Lưu danh sách doc_id để truy xuất

        print(f"Indexing {len(corpus)} documents...")  # Debug

        for doc_id, text in corpus.items():
            if not text:  # Kiểm tra tài liệu rỗng
                print(f"Document {doc_id} is empty; skipping.")
                continue

            words = text.split()  # Tách văn bản thành danh sách từ
            self.doc_lengths[doc_id] = len(words)  # Lưu độ dài tài liệu
            total_length += len(words)

            # Cập nhật tần suất xuất hiện của từ trong toàn bộ corpus
            for word in set(words):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1  

            # Mã hóa tài liệu thành vector embedding
            try:
                embedding = self.encoder.encode(text)  # Gọi mô hình để tạo vector embedding
                self.embeddings.append(embedding)
            except Exception as e:
                print(f"Error encoding document {doc_id}: {e}")  # Debug nếu lỗi
                continue

            self.doc_ids.append(doc_id)  # Thêm doc_id vào danh sách

        # Tính độ dài trung bình của tài liệu (tránh chia cho 0 nếu không có tài liệu)
        self.avg_doc_length = total_length / len(self.doc_ids) if self.doc_ids else 0

        # Xếp chồng các vector embedding thành một ma trận
        if self.embeddings:
            self.embeddings = np.vstack(self.embeddings)
        else:
            self.embeddings = np.empty((0, self.encoder.embedding_size))  # Tránh lỗi nếu không có tài liệu

        print(f"Document embeddings shape: {self.embeddings.shape}")
        print(f"Indexed {len(self.doc_ids)} documents with average length {self.avg_doc_length:.2f}.")  # Debugging

    def score(self, query, doc_id, doc_text, query_embedding):
        """
        Tính điểm BM42 cho một tài liệu.

        Args:
            query (str): Truy vấn đầu vào.
            doc_id (str): ID của tài liệu.
            doc_text (str): Nội dung của tài liệu.
            query_embedding (numpy.ndarray): Vector embedding của truy vấn.

        Returns:
            float: Điểm BM42 của tài liệu đối với truy vấn.
        """
        words = doc_text.split()  # Tách tài liệu thành danh sách từ
        bm42_score = 0  

        # Áp dụng Pseudo-Relevance Feedback (PRF) để mở rộng truy vấn
        refined_query_embedding = self.prf.apply_prf(query_embedding, self.embeddings)

        # Duyệt từng từ trong truy vấn
        for term in query.split():
            if term not in self.doc_freqs:
                continue  # Nếu từ không có trong tài liệu nào, bỏ qua

            df = self.doc_freqs[term]  # Số tài liệu chứa từ này
            idf = math.log((len(self.doc_lengths) - df + 0.5) / (df + 0.5) + 1)  # Tính IDF như trong BM25

            # Mã hóa từ thành vector embedding
            term_embedding = self.encoder.encode(term)

            # Tính độ tương đồng cosine giữa truy vấn và từ trong tài liệu
            term_similarity = np.dot(refined_query_embedding, term_embedding.T)

            # Cộng điểm vào tổng BM42 score
            bm42_score += idf * term_similarity  

        return bm42_score  # Trả về điểm BM42 của tài liệu so với truy vấn
