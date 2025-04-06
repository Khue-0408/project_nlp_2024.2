from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import ast

train_emb = np.load('train_emb.npy')
full_emb = np.load('full_emb.npy')
# Reload data
train = pd.read_csv('train.csv')
corpus = pd.read_csv('corpus.csv')

# Create mappings for corpus
id2corpus = {cid: text for cid, text in zip(corpus['cid'].values, corpus['text'].values)}
id2index = {cid: idx for idx, cid in enumerate(corpus['cid'].values)}

# Corpus IDs as a tensor
corpus_ids = torch.tensor(corpus['cid'].values, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')

# Move embeddings to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_emb = torch.tensor(train_emb, device=device)
full_emb = torch.tensor(full_emb, device=device)

# Calculate hard negatives
hard_neg_index = []
hard_neg_scores = []

print("Calculating hard negatives...")
for emb in tqdm(train_emb, desc="Processing Train Embeddings"):
    emb = emb.unsqueeze(0)  # Add batch dimension
    
    # Compute cosine similarity
    dist = F.cosine_similarity(emb, full_emb, dim=1)
    
    # Sort in descending order
    sorted_values, sorted_indices = torch.sort(-dist)
    
    # Collect top 500 hard negatives
    hard_neg_index.append(corpus_ids[sorted_indices[:500]].cpu().numpy())
    hard_neg_scores.append(sorted_values[:500].cpu().numpy())

#Convert hard negatives to numpy arrays
hard_neg_index = np.array(hard_neg_index)
hard_neg_scores = np.array(hard_neg_scores)
# np.save('hard_neg_index.npy', hard_neg_index)
# np.save('hard_neg_scores.npy', hard_neg_scores)
# hard_neg_index = np.load('hard_neg_index.npy')
# hard_neg_scores = np.load('hard_neg_scores.npy')

# Process train CIDs to filter out missing data
index_no_cid = []
new_cids = []
for idx,cid in enumerate(train['cid']):
    cids = cid.strip().strip("][").split()
    local_cids = []
    for id in cids:
        if int(id) not in id2index:
            continue
        else:
            local_cids.append(int(id))
    if len(local_cids) == 0:
        index_no_cid.append(idx)
    new_cids.append(local_cids)

# Calculate thresholds
thresholds = []
print("Calculating thresholds...")
for idx, emb in tqdm(enumerate(train_emb), desc="Threshold Calculation"):
    if len(new_cids[idx]) == 0:
        thresholds.append(-100)
        continue
    else:
        emb = torch.Tensor(emb).unsqueeze(0)
        threshold = 1
        for id in new_cids[idx]:
            corpus_emb = full_emb[id2index[id]]
            corpus_emb = torch.Tensor(corpus_emb).unsqueeze(0)
            cosine = F.cosine_similarity(emb, corpus_emb)[0]
            threshold = min(cosine, threshold)
        thresholds.append(threshold * 0.95)

# Identify position-aware hard negatives
hard_neg_pos_aware = []
print("Filtering position-aware hard negatives...")
for idx, cids_list in tqdm(enumerate(new_cids), desc="Hard Negative Filtering"):
    if len(cids_list) == 0:
        hard_neg_pos_aware.append([])
    else:
        hards = []
        for hard, score in zip(hard_neg_index[idx], hard_neg_scores[idx]):
            if hard not in cids_list and (-score) < thresholds[idx]:
                hards.append(int(hard))
            if len(hards) == 8:  # Limit to 8 hard negatives
                break
        hard_neg_pos_aware.append(hards)

# Output the results
print("Hard negatives with position-aware filtering calculated successfully!")
pd.DataFrame({
    'text': train["question"].values,
    'pos': new_cids,
    'neg': hard_neg_pos_aware
}).to_csv('hard_neg_pos_aware.csv')
