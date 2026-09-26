def create_context(documents):
    context = ""

    for doc in documents:
        context += doc.page_content + "\n\n"

    return context


def create_prompt(context, question):

    prompts = f"""
You are an AI Research Assistant.

Answer the question using ONLY the provided context.

Explain the answer in simple and easy-to-understand language.
Do not just copy sentences from the context.
First give a direct answer, then explain the important points.

If the answer is not present in the context, say:
"I don't have enough information in the provided paper."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompts


def get_source(documents):

    sources = []
    seen = set()

    for doc in documents:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get(
            "page_label",
            doc.metadata.get("page", "Unknown")
        )

        citation = f"{source} - page {page}"

        if citation not in seen:
            sources.append(citation)
            seen.add(citation)

    return sources