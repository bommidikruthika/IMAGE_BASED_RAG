import os
import io
import uuid
import hashlib
import cv2
import numpy as np
import pytesseract
import streamlit as st

from PIL import Image, ImageOps
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Image RAG | AI Document Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tesseract configuration for Windows and Streamlit Cloud
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.name == "nt":
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    import shutil

    tesseract_executable = shutil.which("tesseract")

    if tesseract_executable:
        pytesseract.pytesseract.tesseract_cmd = tesseract_executable
    else:
        raise RuntimeError(
            "Tesseract OCR is not installed. Add 'tesseract-ocr' "
            "to packages.txt in your Streamlit Cloud repository."
        )


# =========================================================
# 2. PROFESSIONAL PAGE DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f8fd;
    }

    [data-testid="stSidebar"] {
        background-color: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: #f3f7fc;
    }

    .hero {
        background: linear-gradient(120deg, #123c69, #1769aa);
        padding: 30px 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e5efff;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-card {
        background: white;
        padding: 20px;
        border: 1px solid #e1e9f2;
        border-radius: 14px;
        margin-bottom: 15px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #1769aa;
        border: none;
        border-radius: 9px;
        color: white;
        font-weight: 600;
        min-height: 44px;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #124d80;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e1e9f2;
        padding: 14px;
        border-radius: 12px;
    }

    h2, h3 {
        color: #163a5f;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>📘 Image-Based RAG Question Answering System</h1>
        <p>
            Upload multiple document images, extract text using OCR,
            retrieve relevant information, and ask questions using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="section-card">
        <b>How it works</b><br><br>
        📷 Multiple Images &nbsp; → &nbsp;
        🔎 OCR Extraction &nbsp; → &nbsp;
        🧠 Vector Search &nbsp; → &nbsp;
        💬 AI Answer
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(pil_image):
    """Prepare an image for OCR."""

    pil_image = ImageOps.exif_transpose(
        pil_image
    ).convert("RGB")

    rgb = np.array(pil_image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    # Enlarge smaller images to improve text recognition.
    if max(height, width) < 2200:
        scale = min(2.0, 2200 / max(height, width))

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    denoised = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return otsu


# =========================================================
# 5. OCR TEXT EXTRACTION
# =========================================================

def extract_text_from_image(pil_image):
    """Extract readable text from an image using Tesseract OCR."""

    processed_image = preprocess_image(pil_image)

    # Try multiple OCR configurations for better recognition.
    configurations = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 3"
    ]

    extracted_texts = []

    for config in configurations:
        try:
            text = pytesseract.image_to_string(
                processed_image,
                config=config
            )

            if text and text.strip():
                extracted_texts.append(text.strip())

        except pytesseract.TesseractError:
            continue

    if not extracted_texts:
        return ""

    # Use the result with the most readable content.
    best_text = max(
        extracted_texts,
        key=lambda item: len(item)
    )

    return best_text.strip()


# =========================================================
# 6. CLEAN EXTRACTED TEXT
# =========================================================

def clean_text(text):
    """Remove unnecessary whitespace from OCR output."""

    if not text:
        return ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


# =========================================================
# 7. SPLIT TEXT INTO CHUNKS
# =========================================================

def split_text_into_chunks(text, source_name):
    """Split extracted text into smaller searchable documents."""

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    documents = []

    for index, chunk in enumerate(chunks):
        if chunk.strip():
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "chunk": index + 1
                    }
                )
            )

    return documents
# =========================================================
# 8. CREATE VECTOR DATABASE
# =========================================================

def create_vector_database(documents, embeddings):
    """Create a ChromaDB vector database from text chunks."""

    if not documents:
        return None

    collection_name = "image_rag_" + uuid.uuid4().hex[:12]

    vector_database = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )

    return vector_database


# =========================================================
# 9. RETRIEVE RELEVANT DOCUMENTS
# =========================================================

def retrieve_relevant_documents(vector_database, question, k=4):
    """Find text chunks relevant to the user's question."""

    if vector_database is None:
        return []

    if not question or not question.strip():
        return []

    try:
        documents = vector_database.similarity_search(
            question,
            k=k
        )

        return documents

    except Exception as error:
        st.error(
            f"Unable to retrieve relevant text: {error}"
        )

        return []


# =========================================================
# 10. INITIALIZE GROQ LANGUAGE MODEL
# =========================================================

def load_groq_model():
    """Initialize the Groq chat model using an environment variable."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=1024
    )


# =========================================================
# 11. GENERATE ANSWERS USING RAG
# =========================================================

def generate_answer(question, retrieved_documents, llm):
    """
    Generate an answer using retrieved image text as context.
    The model is instructed not to invent unsupported facts.
    """

    if not retrieved_documents:
        return (
            "I could not find relevant information in the "
            "uploaded images. Please upload a clearer image "
            "or ask a question related to the extracted text."
        )

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "Unknown image"
        )

        context_parts.append(
            f"Source {index}: {source}\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant that answers questions about
        information extracted from uploaded images.

        Answer the user's question using only the context
        provided below.

        Instructions:
        1. Give a clear and direct answer.
        2. Use simple language.
        3. Do not invent facts that are not in the context.
        4. If the context does not contain the answer,
           clearly say that the uploaded images do not
           provide enough information.
        5. Use bullet points when they improve readability.

        Context:
        {context}

        User question:
        {question}

        Answer:
        """
    )

    try:
        chain = prompt | llm

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        return response.content

    except Exception as error:
        return (
            "Unable to generate an answer. Please check "
            "your Groq API key, model access, and internet "
            f"connection.\n\nDetails: {error}"
        )


# =========================================================
# 12. DISPLAY SOURCE REFERENCES
# =========================================================

def display_sources(documents):
    """Display the image sources used to answer a question."""

    if not documents:
        return

    st.markdown("### 📚 Retrieved Sources")

    displayed_sources = set()

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown image"
        )

        if source not in displayed_sources:
            displayed_sources.add(source)

            st.markdown(
                f"- **{source}**"
            )
            # =========================================================
# 13. INITIALIZE SESSION STATE
# =========================================================

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = {}

if "documents" not in st.session_state:
    st.session_state.documents = []

if "vector_database" not in st.session_state:
    st.session_state.vector_database = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# 14. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📘 Image RAG Assistant")

    st.write(
        "Upload document images, extract their text, "
        "and ask questions using AI."
    )

    st.divider()

    st.markdown("### 🔄 Workflow")

    st.write("1. Upload images")
    st.write("2. Extract text using OCR")
    st.write("3. Create a vector database")
    st.write("4. Ask questions")
    st.write("5. View AI answers and sources")

    st.divider()

    st.markdown("### 🛠️ Technologies")

    st.write("• Python and Streamlit")
    st.write("• OpenCV image preprocessing")
    st.write("• Tesseract OCR")
    st.write("• LangChain")
    st.write("• ChromaDB")
    st.write("• Groq LLM")

    if st.button(
        "🗑️ Clear Session",
        use_container_width=True
    ):
        for key in [
            "extracted_text",
            "documents",
            "vector_database",
            "processed_files",
            "chat_history"
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()


# =========================================================
# 15. IMAGE UPLOAD SECTION
# =========================================================

st.markdown("## 📤 Upload Your Images")

st.write(
    "Upload one or more images containing printed or "
    "clearly visible text."
)

uploaded_files = st.file_uploader(
    "Choose document images",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tif",
        "tiff"
    ],
    accept_multiple_files=True,
    help="You can upload multiple document images at once."
)


# =========================================================
# 16. DISPLAY IMAGE PREVIEWS
# =========================================================

if uploaded_files:
    st.markdown("### 🖼️ Image Previews")

    preview_columns = st.columns(
        min(len(uploaded_files), 3)
    )

    for index, uploaded_file in enumerate(uploaded_files):
        with preview_columns[index % len(preview_columns)]:
            image_bytes = uploaded_file.getvalue()

            try:
                preview_image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                st.image(
                    preview_image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            except Exception:
                st.warning(
                    f"Unable to preview {uploaded_file.name}."
                )


# =========================================================
# 17. PROCESS IMAGES AND BUILD VECTOR DATABASE
# =========================================================

if st.button(
    "🔍 Extract Text & Process Images",
    type="primary",
    use_container_width=True,
    disabled=not uploaded_files
):

    all_documents = []
    extracted_results = {}
    processing_errors = []

    try:
        with st.spinner(
            "Loading the embedding model..."
        ):
            embeddings = load_embeddings()

        progress_bar = st.progress(0)

        status_message = st.empty()

        total_files = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):

            status_message.info(
                f"Processing image {index + 1} of "
                f"{total_files}: {uploaded_file.name}"
            )

            try:
                image_bytes = uploaded_file.getvalue()

                pil_image = Image.open(
                    io.BytesIO(image_bytes)
                )

                # Extract text from the image.
                raw_text = extract_text_from_image(
                    pil_image
                )

                cleaned_text = clean_text(raw_text)

                extracted_results[
                    uploaded_file.name
                ] = cleaned_text

                if cleaned_text:
                    image_documents = split_text_into_chunks(
                        cleaned_text,
                        uploaded_file.name
                    )

                    all_documents.extend(
                        image_documents
                    )

            except Exception as error:
                processing_errors.append(
                    f"{uploaded_file.name}: {error}"
                )

            progress_bar.progress(
                (index + 1) / total_files
            )

        status_message.empty()
        progress_bar.empty()

        st.session_state.extracted_text = extracted_results
        st.session_state.documents = []
        st.session_state.vector_database = None
        st.session_state.processed_files = list(
            extracted_results.keys()
        )
        st.session_state.chat_history = []

        if all_documents:

            with st.spinner(
                "Creating the searchable vector database..."
            ):
                vector_database = create_vector_database(
                    all_documents,
                    embeddings
                )

            st.session_state.documents = all_documents

            st.session_state.vector_database = (
                vector_database
            )

            st.success(
                f"Successfully processed "
                f"{len(uploaded_files)} image(s) and created "
                f"{len(all_documents)} text chunk(s)."
            )

        else:
            st.warning(
                "No readable text was detected in the uploaded "
                "images. Try clearer images with larger, sharper "
                "text and good lighting."
            )

        if processing_errors:
            with st.expander(
                "⚠️ View image processing errors"
            ):
                for error_message in processing_errors:
                    st.write(f"- {error_message}")

    except Exception as error:
        st.error(
            "Image processing failed. Please check your "
            f"installation and try again.\n\nDetails: {error}"
        )


# =========================================================
# 18. DISPLAY PROCESSING SUMMARY
# =========================================================

if st.session_state.documents:

    st.divider()

    st.markdown("## 📊 Processing Summary")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Images Processed",
        len(st.session_state.processed_files)
    )

    metric2.metric(
        "Text Chunks",
        len(st.session_state.documents)
    )

    metric3.metric(
        "Vector Database",
        "Ready"
    )


# =========================================================
# 19. DISPLAY EXTRACTED TEXT
# =========================================================

if st.session_state.extracted_text:

    st.markdown("## 📝 Extracted Text")

    st.write(
        "Expand an image below to view the text extracted "
        "by the OCR system."
    )

    for filename, extracted_text in (
        st.session_state.extracted_text.items()
    ):

        with st.expander(
            f"📄 {filename}"
        ):

            if extracted_text:
                st.text_area(
                    "OCR output",
                    value=extracted_text,
                    height=200,
                    key="ocr_" + hashlib.md5(
                        filename.encode()
                    ).hexdigest()
                )

            else:
                st.warning(
                    "No readable text was extracted from this image."
                )


# =========================================================
# 20. QUESTION-ANSWERING SECTION
# =========================================================

st.divider()

st.markdown("## 💬 Ask Questions About Your Images")

st.write(
    "Ask a question about the content of your uploaded "
    "images. The system searches relevant text chunks "
    "before generating an answer."
)

if st.session_state.vector_database is None:

    st.info(
        "Upload your images and click "
        "'Extract Text & Process Images' before asking questions."
    )

else:

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What are the main points in the document?"
        ),
        key="question_input"
    )

    if st.button(
        "🤖 Generate Answer",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            # Support Streamlit Cloud secrets as well as .env.
            try:
                if (
                    not os.getenv("GROQ_API_KEY")
                    and st.secrets.get("GROQ_API_KEY")
                ):
                    os.environ["GROQ_API_KEY"] = (
                        st.secrets["GROQ_API_KEY"]
                    )
            except Exception:
                pass

            llm = load_groq_model()

            if llm is None:

                st.error(
                    "GROQ_API_KEY was not found. Add it to your "
                    "local .env file or Streamlit Cloud Secrets."
                )

            else:

                with st.spinner(
                    "Searching your documents and generating an answer..."
                ):

                    retrieved_documents = (
                        retrieve_relevant_documents(
                            st.session_state.vector_database,
                            question,
                            k=4
                        )
                    )

                    answer = generate_answer(
                        question,
                        retrieved_documents,
                        llm
                    )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": [
                            document.metadata.get(
                                "source",
                                "Unknown image"
                            )
                            for document in retrieved_documents
                        ]
                    }
                )


# =========================================================
# 21. DISPLAY ANSWERS AND SOURCES
# =========================================================

if st.session_state.chat_history:

    st.markdown("## 📚 Answers")

    for index, chat_item in enumerate(
        reversed(st.session_state.chat_history),
        start=1
    ):

        with st.container(border=True):

            st.markdown(
                f"### ❓ Question {index}"
            )

            st.write(
                chat_item["question"]
            )

            st.markdown("### ✅ AI Answer")

            st.markdown(
                chat_item["answer"]
            )

            unique_sources = list(
                dict.fromkeys(
                    chat_item["sources"]
                )
            )

            if unique_sources:

                st.markdown("### 📄 Sources Used")

                for source in unique_sources:
                    st.write(f"- {source}")

            else:

                st.caption(
                    "No relevant source chunks were retrieved."
                )


# =========================================================
# 22. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#60758a; padding:12px;">
        <p><b>Image-Based RAG Question Answering System</b></p>
        <p>Built with Python, OCR, LangChain, ChromaDB and Groq AI</p>
    </div>
    """,
    unsafe_allow_html=True
)