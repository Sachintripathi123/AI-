from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings(model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

