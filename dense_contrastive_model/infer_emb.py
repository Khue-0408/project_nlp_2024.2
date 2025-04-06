import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

def clean_text(text):
    cleaned_text = re.sub(r'[\xa0\xad]+', ' ', text)
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
    cleaned_text = re.sub(r'\…+', '', cleaned_text)
    cleaned_text = re.sub(r'[@#%^&*]+', '', cleaned_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    cleaned_text = re.sub(r'-{2,}', '', cleaned_text)
    cleaned_text = re.sub(r'_{4,}', '', cleaned_text)
    cleaned_text = re.sub(r':\.*', '', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

texts = pd.read_csv('cleaned_corpus.csv')['text'].values.tolist()
texts = [f"passage: {text}" for text in texts]
questions = pd.read_csv('private_test.csv')['question'].values.tolist()
questions = [f"query: {clean_text(question.lower())}" for question in questions]


model = SentenceTransformer('e5_full_hard_neg/trained_model')

# Encode corpus
print("Encoding corpus...")
emb = model.encode(texts, batch_size=8, show_progress_bar=True)
emb = np.array(corpus_embeddings)

# Encode questions
print("Encoding questions...")
ques = model.encode(questions, batch_size=8, show_progress_bar=True)
ques = np.array(question_embeddings)


corpus = pd.read_csv('corpus.csv')
question = pd.read_csv('private_test.csv')
cids = corpus['cid'].values.tolist()
qids = question['qid'].values.tolist()


indices = []
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Move embeddings to the GPU
emb_tensor = torch.Tensor(emb).to(device)

for idx, q in tqdm(enumerate(ques)):
    # Move the query tensor to the GPU
    q_tensor = torch.Tensor(q).unsqueeze(0).to(device)
    
    # Compute cosine similarity on the GPU
    cosine = F.cosine_similarity(q_tensor, emb_tensor)
    
    sorted_cids = [cids[i] for i in torch.argsort(cosine, descending=True)[:20].tolist()]
    
    indices.append(sorted_cids)

print(indices[0])


with open('predict.txt', 'w') as f:
    for id, index in enumerate(qids):
        f.writelines(f"{index} {' '.join(map(str, indices[id][:10]))}\n")
