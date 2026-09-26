def retriever(vector,query,k=3):
    results = vector.max_marginal_relevance_search(query , k=k,fetch_k=10)
    return results


def retriever_with_scores(vector_store, query, k=3):

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results