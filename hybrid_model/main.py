import numpy as np
import pandas as pd
import torch
from retrieval.retrieval import HybridRetrieval
from evaluation.metrics import mrr_at_k
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

def load_data():
    corpus_df = pd.read_csv('data/corpus1.csv')
    train_df = pd.read_csv('data/train1.csv')
    corpus = {row['cid']: row['text'] for _, row in corpus_df.iterrows()}
    queries = {row['qid']: (row['question'], row['cid']) for _, row in train_df.iterrows()}
    return corpus, queries

def evaluate_retrieval(retrieval, queries, corpus, tokenizer, model):
    results = {}
    for qid, (question, relevant_cid) in tqdm(queries.items(), desc="Evaluating Retrieval", leave=False):
        # Step 1: Retrieve top 50 documents from both BM42 and embedding model
        bm42_docs = retrieval.retrieve_bm42(question, top_k=20)
        embedding_docs = retrieval.retrieve_embedding_based(question, top_k=20)
        
        # Step 2: Combine the top 50 results from both models
        combined_docs = retrieval.combine_results(bm42_docs, embedding_docs)
        
        # Step 3: Rerank the combined documents to get the top 20
        reranked_docs = retrieval.rerank_with_bge(combined_docs, question, tokenizer, model, top_k=10)
        
        # Step 4: Get final top 10 documents (simulating "prompting" stage)
        top_10_docs = reranked_docs[:10]
        
        results[question] = top_10_docs

    ground_truth = {question: relevant_cid for _, (question, relevant_cid) in queries.items()}
    mrr_score = mrr_at_k(results, ground_truth, k=10)
    return mrr_score

# def main():
#     corpus, queries = load_data()
#     config = AutoConfig.from_pretrained('BAAI/bge-reranker-v2-gemma')
#     config.hidden_activation = 'relu'

#     tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
#     model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma', config=config).eval()

#     retrieval = HybridRetrieval(corpus)
#     retrieval.bm42.k1 = 1.2
#     retrieval.bm42.b = 0.75
#     retrieval.bm42.beta = 0.4

#     mrr_score = evaluate_retrieval(retrieval, queries, corpus, tokenizer, model)
#     print(f"MRR@10: {mrr_score}")

if __name__ == "__main__":
    corpus, queries = load_data()
    print(corpus)
    # print(queries)
