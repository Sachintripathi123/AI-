# from langchain_community.document_loaders import PyPDFLoader
# from pathlib import Path

# def load_pdf(filepath):
#     loader = PyPDFLoader(filepath)
#     document = loader.load()
#     return document


# def load_multiple_pdfs(folder_path):
#     folder = Path(folder_path)

#     all_documents = []

#     pdf_files = list(folder.glob("*.pdf"))

#     for pdf_file in pdf_files:
#        print(f"Loading: {pdf_file.name}")
#        loader = PyPDFLoader(pdf_file)
#        documents  = loader.load()

#        all_documents.extend(documents)


#     print(f"\nTotal PDFs: {len(pdf_files)}")
#     print(f"Total pages: {len(all_documents)}")

#     return all_documents

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(filepath):
    """
    Load a single PDF file and return LangChain documents.
    """

    filepath = Path(filepath)

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"PDF not found: {filepath}"
        )

    # Check extension
    if filepath.suffix.lower() != ".pdf":
        raise ValueError(
            f"Not a PDF file: {filepath}"
        )

    # Check file is not empty
    if filepath.stat().st_size == 0:
        raise ValueError(
            f"PDF file is empty: {filepath.name}"
        )

    print(f"Loading PDF: {filepath.name}")

    try:
        loader = PyPDFLoader(str(filepath))
        documents = loader.load()

    except Exception as e:
        raise RuntimeError(
            f"Failed to read PDF '{filepath.name}': {e}"
        ) from e

    if not documents:
        raise ValueError(
            f"No pages could be extracted from: {filepath.name}"
        )

    print(f"Loaded {len(documents)} pages")

    return documents


def load_multiple_pdfs(folder_path):
    """
    Load all PDF files from a folder.
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    if not folder.is_dir():
        raise ValueError(
            f"Not a folder: {folder}"
        )

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {folder}")
        return []

    all_documents = []

    for pdf_file in pdf_files:

        try:
            print(f"Loading: {pdf_file.name}")

            documents = load_pdf(pdf_file)

            all_documents.extend(documents)

            print(
                f"✓ {pdf_file.name}: "
                f"{len(documents)} pages loaded"
            )

        except Exception as e:

            print(
                f"✗ Failed: {pdf_file.name}"
            )

            print(
                f"  Error: {e}"
            )

    print("\n-------------------------")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Total pages: {len(all_documents)}")
    print("-------------------------")

    return all_documents