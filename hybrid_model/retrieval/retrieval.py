import numpy as np
import torch
from models.bm42 import BM42
from retrieval.prf import PseudoRelevanceFeedback
from tqdm import tqdm

class HybridRetrieval:
    """
    HybridRetrieval: Kết hợp giữa BM42 và tìm kiếm dựa trên embedding.
    """
    def __init__(self, corpus, beta=0.4, top_k=10):
        """
        Khởi tạo Hybrid Retrieval.

        Args:
            corpus (dict): Tập tài liệu để tìm kiếm, dạng {doc_id: doc_text}.
            beta (float): Trọng số của Pseudo-Relevance Feedback (PRF).
            top_k (int): Số tài liệu tối đa cần truy xuất.
        """
        self.corpus = corpus  # Lưu tập tài liệu
        self.bm42 = BM42()  # Khởi tạo mô hình BM42
        self.bm42.index_documents(corpus)  # Xây dựng index cho BM42
        self.prf = PseudoRelevanceFeedback(beta=beta)  # PRF giúp mở rộng truy vấn
        self.top_k = top_k  # Số tài liệu lấy ra sau tìm kiếm
        self.doc_embeddings = self.bm42.embeddings  # Lưu vector embedding của tài liệu

    def retrieve_bm42(self, query, top_k=20):
        """
        Tìm kiếm tài liệu dựa trên BM42.

        Args:
            query (str): Truy vấn đầu vào.
            top_k (int): Số tài liệu cần truy xuất.

        Returns:
            list: Danh sách doc_id của các tài liệu liên quan.
        """
        # Mã hóa truy vấn thành vector embedding
        query_embedding = self.bm42.encoder.encode(query)

        # Tính điểm BM42 cho từng tài liệu trong corpus
        scores = {doc_id: self.bm42.score(query, doc_id, doc_text, query_embedding)
                  for doc_id, doc_text in self.corpus.items()}

        # Trả về danh sách tài liệu được sắp xếp theo điểm BM42 (cao → thấp)
        return sorted(scores, key=scores.get, reverse=True)[:top_k]


    def retrieve_embedding_based(self, query, top_k=20):
        """
        Tìm kiếm tài liệu dựa trên độ tương đồng cosine giữa truy vấn và vector embedding.

        Args:
            query (str): Truy vấn đầu vào.
            top_k (int): Số tài liệu cần truy xuất.

        Returns:
            list: Danh sách doc_id của các tài liệu liên quan.
        """
        # Mã hóa truy vấn thành vector embedding
        query_embedding = self.bm42.encoder.encode(query)

        # Tính cosine similarity giữa truy vấn và từng tài liệu
        similarities = {doc_id: np.dot(query_embedding, self.doc_embeddings[idx].T)
                        for idx, doc_id in enumerate(self.bm42.doc_ids)}

        # Trả về danh sách tài liệu sắp xếp theo độ tương đồng (cao → thấp)
        return sorted(similarities, key=similarities.get, reverse=True)[:top_k]


    def combine_results(self, bm42_docs, embedding_docs):
        """
        Kết hợp kết quả từ BM42 và retrieval-based similarity search.

        Args:
            bm42_docs (list): Danh sách tài liệu từ BM42.
            embedding_docs (list): Danh sách tài liệu từ embedding-based retrieval.

        Returns:
            list: Danh sách tài liệu đã kết hợp, được sắp xếp theo mức độ liên quan.
        """
        combined = set(bm42_docs + embedding_docs)  # Hợp hai danh sách tài liệu
        combined_scores = {doc_id: 0 for doc_id in combined}

        # Cộng điểm cho tài liệu có mặt trong BM42
        for doc_id in bm42_docs:
            combined_scores[doc_id] += 1

        # Cộng điểm cho tài liệu có mặt trong retrieval bằng embedding
        for doc_id in embedding_docs:
            combined_scores[doc_id] += 1

        # Sắp xếp tài liệu theo tổng điểm xuất hiện
        return sorted(combined_scores, key=combined_scores.get, reverse=True)


    def rerank_with_bge(self, docs, query, tokenizer, model, top_k=10):
        """
        Xếp hạng lại kết quả tìm kiếm bằng BGE.

        Args:
            docs (list): Danh sách tài liệu cần xếp hạng lại.
            query (str): Truy vấn đầu vào.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer của mô hình BGE.
            model (torch.nn.Module): Mô hình transformer dùng để rerank.
            top_k (int): Số tài liệu cuối cùng cần lấy.

        Returns:
            list: Danh sách doc_id được rerank.
        """
        pairs = [(query, self.corpus[doc_id]) for doc_id in docs]  # Tạo danh sách (query, document)
        scores = self.rerank_scores(pairs, tokenizer, model)  # Tính điểm rerank
        reranked_docs = [doc_id for _, doc_id in sorted(zip(scores, docs), reverse=True)]
        return reranked_docs[:top_k]


    def rerank_scores(self, pairs, tokenizer, model):
        """
        Tính điểm rerank bằng mô hình Transformer.

        Args:
            pairs (list of tuples): Danh sách (query, document).
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer để chuyển đổi văn bản thành token.
            model (torch.nn.Module): Mô hình transformer.

        Returns:
            list: Danh sách điểm rerank.
        """
        with torch.no_grad():
            inputs = self.get_inputs(pairs, tokenizer)  # Tạo tensor từ tokenizer
            yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
            scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1).float()
        return scores.tolist()


    @staticmethod
    def get_inputs(pairs, tokenizer, max_length=512):
        """
        Prepares input tensors for query-document pairs.

        Args:
            pairs (list of tuples): List of (query, document) pairs.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to convert text to tokens.
            max_length (int): Maximum length of tokens for each input.

        Returns:
            dict: Tokenized and padded inputs for the model.
        """
        queries, documents = zip(*pairs)  # Separate queries and documents

        # Tokenize the pairs with padding and truncation
        inputs = tokenizer(
            list(queries), list(documents),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Move inputs to the correct device (e.g., GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        return inputs
