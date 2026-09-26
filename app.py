import os
import tempfile
import base64
import time

import streamlit as st

from src.document_loader import load_pdf
from src.chunking import chunk_document
from src.embedding import create_embeddings
from src.vector import vector_db
from src.retriever import retriever
from src.rag import create_prompt, create_context, get_source
from src.llm import create_llm


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "started" not in st.session_state:
    st.session_state.started = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def set_background(image_path):

    try:

        with open(image_path, "rb") as image_file:

            encoded = base64.b64encode(
                image_file.read()
            ).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image:
                    linear-gradient(
                        rgba(0, 0, 0, 0.45),
                        rgba(0, 0, 0, 0.45)
                    ),
                    url("data:image/webp;base64,{encoded}");

                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        print(
            "Background image error:",
            repr(e)
        )


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.started:

    set_background(
        "assest/robot_img_1.webp"
    )

    st.markdown(
        """
        <h1 style="
            text-align:center;
            color:white;
            font-size:55px;
            margin-top:180px;
        ">
            🧠 AI Research Assistant
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="
            text-align:center;
            color:white;
            font-size:22px;
        ">
            Explore research papers with AI
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        [1, 1, 1]
    )

    with col2:

        if st.button(
            "🚀 Start Question",
            use_container_width=True
        ):

            st.session_state.started = True

            st.rerun()

    st.stop()


# =========================================================
# MAIN PAGE
# =========================================================

st.title("🧠 AI Research Assistant")

st.write(
    "Ask questions from your research papers"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    if st.button("🗑️ Clear / Reset"):

        st.cache_resource.clear()

        st.session_state.messages = []

        st.session_state.started = False

        st.rerun()


# =========================================================
# CREATE VECTOR STORE
# =========================================================

@st.cache_resource
def create_vector_store(file_data):

    all_documents = []

    # =====================================================
    # LOAD PDFs
    # =====================================================

    for file_name, file_bytes in file_data:

        temp_path = None

        try:

            print(
                f"\nProcessing PDF: {file_name}"
            )

            print(
                f"File size: {len(file_bytes)} bytes"
            )

            # -------------------------------------------------
            # CREATE TEMPORARY PDF
            # -------------------------------------------------

            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(file_bytes)
                temp_file.flush()

                temp_path = temp_file.name

            print(
                f"Temporary file created: {temp_path}"
            )

            # -------------------------------------------------
            # CHECK FILE
            # -------------------------------------------------

            if not os.path.exists(temp_path):

                raise FileNotFoundError(
                    "Temporary PDF file was not created."
                )

            temp_size = os.path.getsize(
                temp_path
            )

            print(
                f"Temporary PDF size: "
                f"{temp_size} bytes"
            )

            if temp_size == 0:

                raise ValueError(
                    f"Uploaded PDF "
                    f"'{file_name}' is empty."
                )

            # -------------------------------------------------
            # LOAD PDF
            # -------------------------------------------------

            documents = load_pdf(
                temp_path
            )

            print(
                f"✓ {file_name}: "
                f"{len(documents)} pages loaded"
            )

            # -------------------------------------------------
            # SAVE ORIGINAL FILE NAME
            # -------------------------------------------------

            for doc in documents:

                doc.metadata["source"] = file_name

            # -------------------------------------------------
            # ADD DOCUMENTS
            # -------------------------------------------------

            all_documents.extend(
                documents
            )

        except Exception as e:

            print(
                f"\n❌ PDF processing failed: "
                f"{file_name}"
            )

            print(
                f"Error type: "
                f"{type(e).__name__}"
            )

            print(
                f"Error: {e}"
            )

            raise RuntimeError(
                f"Could not process "
                f"'{file_name}': {e}"
            ) from e

        finally:

            # -------------------------------------------------
            # DELETE TEMP FILE
            # -------------------------------------------------

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                try:

                    os.remove(
                        temp_path
                    )

                except Exception as cleanup_error:

                    print(
                        "Temporary file cleanup failed:",
                        cleanup_error
                    )

    # =====================================================
    # CHECK DOCUMENTS
    # =====================================================

    if not all_documents:

        raise ValueError(
            "No readable content was found "
            "in the uploaded PDF(s)."
        )

    print(
        f"\nTotal pages loaded: "
        f"{len(all_documents)}"
    )

    # =====================================================
    # CHUNKING
    # =====================================================

    chunks = chunk_document(
        all_documents
    )

    if not chunks:

        raise ValueError(
            "No chunks were created "
            "from the uploaded PDF(s)."
        )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    embeddings = create_embeddings(
        model_name=
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    # =====================================================
    # VECTOR DATABASE
    # =====================================================

    my_vector_store = vector_db(
        chunks,
        embeddings
    )

    print(
        "✓ Vector database created"
    )

    return my_vector_store


# =========================================================
# PROCESS PDFs
# =========================================================

my_vector_store = None


if uploaded_files:

    file_data = tuple(
        (
            file.name,
            file.getvalue()
        )
        for file in uploaded_files
    )

    try:

        with st.spinner(
            "📚 Processing research papers..."
        ):

            my_vector_store = (
                create_vector_store(
                    file_data
                )
            )

        st.sidebar.success(
            f"{len(uploaded_files)} "
            f"PDF(s) loaded successfully! ✅"
        )

    except Exception as e:

        my_vector_store = None

        st.sidebar.error(
            "❌ PDF processing failed."
        )

        st.sidebar.error(
            f"Error: {e}"
        )

        with st.expander(
            "🔍 Show PDF error details"
        ):

            st.exception(e)

        print(
            "PDF processing error:",
            repr(e)
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        # -------------------------------------------------
        # DISPLAY SOURCES
        # -------------------------------------------------

        if "sources" in message:

            sources = message["sources"]

            if sources:

                st.subheader(
                    "📚 Sources"
                )

                for source in sources:

                    st.write(
                        f"📄 {source}"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "Ask about your research papers..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if query:

    # =====================================================
    # USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.write(
            query
        )

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # =====================================================
    # CHECK VECTOR STORE
    # =====================================================

    if my_vector_store is None:

        with st.chat_message(
            "assistant"
        ):

            st.warning(
                "📚 Please upload at least "
                "one readable PDF first."
            )

    else:

        # =================================================
        # RETRIEVER
        # =================================================

        try:

            with st.spinner(
                "🔍 Searching research papers..."
            ):

                results = retriever(
                    my_vector_store,
                    query,
                    k=3
                )

        except Exception as e:

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    "❌ I couldn't search "
                    "the research papers."
                )

                st.exception(e)

            print(
                "Retriever error:",
                repr(e)
            )

            st.stop()

        # =================================================
        # CONTEXT
        # =================================================

        try:

            context = create_context(
                results
            )

        except Exception as e:

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    "❌ I couldn't create "
                    "the research context."
                )

                st.exception(e)

            print(
                "Context error:",
                repr(e)
            )

            st.stop()

        # =================================================
        # PROMPT
        # =================================================

        try:

            prompt = create_prompt(
                context,
                query
            )

        except Exception as e:

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    "❌ I couldn't prepare "
                    "the prompt."
                )

                st.exception(e)

            print(
                "Prompt error:",
                repr(e)
            )

            st.stop()

        # =================================================
        # GEMINI
        # =================================================

        try:

            with st.spinner(
                "🤖 Generating answer..."
            ):

                client = create_llm()

                answer = None

                # -------------------------------------------------
                # Gemini models
                #
                # Primary:
                # gemini-3.8-flash
                #
                # Fallback:
                # gemini-3.6-flash
                #
                # Final fallback:
                # gemini-3.5-flash-lite
                # -------------------------------------------------

                models = [
                    "gemini-3.8-flash",
                    "gemini-3.6-flash",
                    "gemini-3.5-flash-lite"
                ]

                last_error = None

                # -------------------------------------------------
                # TRY MODELS
                # -------------------------------------------------

                for model_name in models:

                    # Retry each model up to 3 times
                    for attempt in range(3):

                        try:

                            print(
                                f"\nTrying model: "
                                f"{model_name}"
                            )

                            print(
                                f"Attempt: "
                                f"{attempt + 1}/3"
                            )

                            response = (
                                client.models.generate_content(
                                    model=model_name,
                                    contents=prompt
                                )
                            )

                            # -------------------------------------
                            # CHECK RESPONSE
                            # -------------------------------------

                            if (
                                response
                                and response.text
                            ):

                                answer = (
                                    response.text
                                )

                                print(
                                    f"✓ Answer generated "
                                    f"using {model_name}"
                                )

                                break

                            raise ValueError(
                                "The model returned "
                                "an empty response."
                            )

                        except Exception as model_error:

                            last_error = model_error

                            error_text = str(
                                model_error
                            ).lower()

                            # -------------------------------------
                            # DETECT TEMPORARY ERRORS
                            # -------------------------------------

                            temporary_error = (
                                "503" in error_text
                                or
                                "unavailable" in error_text
                                or
                                "high demand" in error_text
                                or
                                "429" in error_text
                                or
                                "rate limit" in error_text
                                or
                                "too many requests" in error_text
                                or
                                "500" in error_text
                                or
                                "502" in error_text
                                or
                                "504" in error_text
                            )

                            print(
                                f"❌ {model_name} "
                                f"failed:"
                            )

                            print(
                                repr(model_error)
                            )

                            # -------------------------------------
                            # RETRY TEMPORARY ERROR
                            # -------------------------------------

                            if (
                                temporary_error
                                and attempt < 2
                            ):

                                wait_time = (
                                    2 ** attempt
                                )

                                print(
                                    f"Retrying in "
                                    f"{wait_time} seconds..."
                                )

                                time.sleep(
                                    wait_time
                                )

                                continue

                            # -------------------------------------
                            # Move to next model
                            # -------------------------------------

                            break

                    # ---------------------------------------------
                    # Stop model loop if answer generated
                    # ---------------------------------------------

                    if answer:

                        break

                # -------------------------------------------------
                # NO MODEL WORKED
                # -------------------------------------------------

                if not answer:

                    raise RuntimeError(
                        "Gemini could not generate "
                        "a response after retries. "
                        f"Last error: {last_error}"
                    )

        except Exception as e:

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    "❌ Unable to generate "
                    "an answer right now."
                )

                st.warning(
                    "The Gemini service may be "
                    "temporarily busy. "
                    "Please try the question again."
                )

                with st.expander(
                    "🔍 Show Gemini error"
                ):

                    st.exception(e)

            print(
                "LLM error:",
                repr(e)
            )

            st.stop()

        # =================================================
        # GET SOURCES
        # =================================================

        try:

            sources = get_source(
                results
            )

        except Exception as e:

            sources = []

            print(
                "Source error:",
                repr(e)
            )

        # =================================================
        # ASSISTANT RESPONSE
        # =================================================

        with st.chat_message(
            "assistant"
        ):

            st.write(
                answer
            )

            if sources:

                st.subheader(
                    "📚 Sources"
                )

                for source in sources:

                    st.write(
                        f"📄 {source}"
                    )

        # =================================================
        # SAVE ASSISTANT MESSAGE
        # =================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )