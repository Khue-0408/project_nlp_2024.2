# Import các thư viện cần thiết từ Hugging Face Transformers và PyTorch
from transformers import AutoTokenizer, AutoModel
import torch

class TextEncoder:
    """
    Lớp TextEncoder giúp mã hóa văn bản thành vector sử dụng mô hình pretrained từ Hugging Face.
    """
    def __init__(self, model_name='sentence-transformers/msmarco-bert-base-dot-v5'):
        """
        Khởi tạo bộ mã hóa văn bản với mô hình pretrained.

        Args:
            model_name (str): Tên mô hình được tải từ Hugging Face Model Hub.
        """
        # Load tokenizer để chuyển văn bản thành token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load mô hình Transformer được pretrained
        self.model = AutoModel.from_pretrained(model_name)
    
    def encode(self, text):
        """
        Mã hóa văn bản đầu vào thành vector nhúng.

        Args:
            text (str): Văn bản đầu vào cần mã hóa.

        Returns:
            np.array: Vector nhúng của văn bản.
        """
        # Tokenize văn bản: chuyển thành tensor, thêm padding và cắt bớt nếu quá dài
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Truyền dữ liệu qua mô hình để lấy đầu ra
        outputs = self.model(**inputs)

        # Trả về pooler output (vector nhúng cuối cùng của CLS token), và tách khỏi computational graph
        return outputs.pooler_output.detach().numpy()


    
# class TextEncoder:
#     def __init__(self, model_name='fine_tuned_ance'): 
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
    
#     def encode(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         outputs = self.model(**inputs)
#         return outputs.pooler_output.detach().numpy()


