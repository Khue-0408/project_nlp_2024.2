Những file ban đầu sẽ bao gồm
- corpus.csv: Tập các văn bản
- train.csv: Dữ liệu huấn luyện của cuộc thi
- private_test.csv: Bộ dữ liệu private test của cuộc thi
- full_emb.npy: Đây là file sử dụng mô hình BGE-M3 để encode tất cả các corpus. Do trong Docker em không thể làm chô mô hình BGE-M3 infer được bằng GPU nên em xin phép gửi link Kaggle tạo ra file này. Link Kaggle: https://www.kaggle.com/code/ndnmkaggle3/encoder-legal-corpus-embedding
- train_emb.npy: Đây là file sử dụng mô hình BGE-M3 để encode tất cả các câu hỏi trong bộ huấn luyện. Do trong Docker em không thể làm chô mô hình BGE-M3 infer được bằng GPU nên em xin phép gửi link Kaggle tạo ra file này. Link Kaggle:https://www.kaggle.com/code/ndnmkaggle3/encoder-legal-train-embedding
Docker để reproduce lại kết quả đã được nhóm up lên Dockerhub với link sau: https://hub.docker.com/repository/docker/minhndn/aio_plvc_bkai_legal/general
Để reproduce lại kết quả làm bước như sau
- Bước 1: Chạy lệnh docker run -it --name aio_plvc --runtime=nvidia --gpus all minhndn/aio_plvc_bkai_legal:latest /bin/bash/ dể vào Docker và chạy lệnh cd workdir
- Bước 2: Chạy file mine_hard.py bằng lệnh python mine_hard.py để cho ra file chứa các id các văn bản làm Hard Negative. Khi chạy file này sẽ tạo ra file hard_neg_pos_aware.csv
- Bước 3: Chạy file create_training_data.py để tạo ra dữ liệu huấn luyện. Lúc này sẽ tạo ra file cleaned_corpus.csv và full_hard_neg.csv
- Bước 4: Chạy file train_embedding.py để huấn luyện mô hình embedding.(Có thể skip nếu muốn infer luôn). Model sau khi train xong sẽ lưu vào đường dẫn ./e5_full_hard_neg/trained_model. Trong file đã có sẵn checkpoint có thể sử dụng để infer luôn
- Bước 5: Chạy lệnh python infer_embed.py để infer ra file predict.txt. Từ file này khi submit có thể đạt điểm số 0.7387 ở vòng private


Weight của mô hình Embedding ở trong đường dẫn /workdir/e5_full_hard_neg/trained_model trong Docker

