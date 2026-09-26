from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(document):
    splliter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap=200 )
    chunks = splliter.split_documents(document)
    return chunks




