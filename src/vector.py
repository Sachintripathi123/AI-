from langchain_community.vectorstores import FAISS


def vector_db(chunks,embeddings):
    vector_store = FAISS.from_documents(chunks,embeddings)
    return vector_store
    