def mrr_at_k(results, ground_truth, k=10):
    mrr_total = 0.0
    for query, retrieved_docs in results.items():
        for rank, doc_id in enumerate(retrieved_docs[:k], start=1):
            if str(doc_id) in ground_truth[query]:
                mrr_total += 1 / rank
                break
    return mrr_total / len(results)
