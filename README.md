# Image-Based RAG Question Answering System

## 📌 Project Overview

The **Image-Based Retrieval-Augmented Generation (RAG) Question Answering System** is a Python-based application that extracts text from images using Optical Character Recognition (OCR) and answers user questions based on the extracted content.

The system combines image processing, text extraction, semantic search, and Large Language Models (LLMs) to provide relevant, context-based answers from one or more uploaded images.

Users can upload images containing printed text, process them, and ask questions about the information found in those images through an interactive Streamlit web application.

## 🎯 Objectives

- Extract readable text from images using OCR.
- Improve text recognition through image preprocessing.
- Store extracted text in a vector database.
- Retrieve relevant information using semantic similarity.
- Generate context-based answers using a Large Language Model.
- Support question answering across multiple uploaded images.
- Provide a simple and user-friendly web interface.

## ✨ Features

- **Multiple Image Upload:** Upload multiple images in supported image formats.
- **Image Preprocessing:** Improve image quality using grayscale conversion, filtering, and thresholding.
- **Optical Character Recognition:** Extract text using Tesseract OCR.
- **Text Chunking:** Divide extracted text into smaller chunks for efficient retrieval.
- **Semantic Search:** Find relevant text using vector embeddings.
- **Vector Database:** Store and retrieve text chunks using ChromaDB.
- **AI-Powered Question Answering:** Generate answers using the Groq API and a supported language model.
- **Source-Aware Retrieval:** Associate extracted content with its source image.
- **Text Download:** Download the extracted OCR text.
- **Interactive Interface:** Use the application through a Streamlit web interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| OpenCV | Image preprocessing |
| Tesseract OCR | Text extraction from images |
| Pytesseract | Python interface for Tesseract |
| LangChain | RAG pipeline and document processing |
| Hugging Face | Text embedding model |
| Sentence Transformers | Semantic text embeddings |
| ChromaDB | Vector storage and similarity search |
| Groq API | Language model inference |
| Pillow | Image loading and orientation correction |
| NumPy | Image and numerical operations |

## ⚙️ System Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline.

1. **Image Upload:** The user uploads one or more images.
2. **Image Preprocessing:** Images are enhanced to improve OCR accuracy.
3. **Text Extraction:** Tesseract OCR extracts text from the processed images.
4. **Text Chunking:** The extracted text is divided into smaller, overlapping chunks.
5. **Embedding Generation:** A Hugging Face embedding model converts the text chunks into numerical vectors.
6. **Vector Storage:** ChromaDB stores the vectors and associated source information.
7. **Information Retrieval:** The system retrieves relevant chunks for the user's question.
8. **Answer Generation:** A Groq-hosted language model generates an answer using the retrieved context.
9. **Result Display:** The answer and relevant retrieved information are displayed in Streamlit.

## 🔄 Workflow

```text
User Uploads Images
        |
        v
Image Preprocessing
        |
        v
Text Extraction (OCR)
        |
        v
Text Chunking
        |
        v
Generate Embeddings
        |
        v
Store in ChromaDB
        |
        v
User Asks a Question
        |
        v
Retrieve Relevant Context
        |
        v
Groq Language Model
        |
        v
Display Answer in Streamlit
```

## 📂 Project Structure

```text
image_based_rag_qa/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore              # Files excluded from Git
├── .env                    # API key configuration (not uploaded)
└── .venv/                  # Python virtual environment (not uploaded)
```

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies.
- Tesseract OCR.
- A Groq API key.
- Internet access for downloading embedding models and accessing the Groq API.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/image-based-rag-qa.git
cd image-based-rag-qa
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR

Install Tesseract OCR and ensure that the executable is available to the application.

On Windows, the default installation path used by this project is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is installed in a different location, update the configuration in `app.py` accordingly.

### Step 5: Configure the Groq API Key

Create a `.env` file in the project root and add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key.

**Security:** Never upload your `.env` file or publish your API key on GitHub.

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser to use the application.

## 🧪 How to Use

1. Start the Streamlit application.
2. Upload one or more images containing readable text.
3. Allow the application to preprocess the images and extract their text.
4. Review the extracted text.
5. Enter a question related to the uploaded image content.
6. Submit the question.
7. View the answer generated from the retrieved context.
8. Download the extracted text if required.

**Example question:** “What does Artificial Intelligence enable computers to do?”

The system retrieves relevant text from the uploaded image and uses that context to generate an answer.

## 📚 RAG Approach

Retrieval-Augmented Generation combines information retrieval with language generation.

In this project:

- **Retrieval:** The embedding model and ChromaDB identify text chunks relevant to the question.
- **Augmentation:** The retrieved text is supplied as context to the language model.
- **Generation:** The language model produces an answer grounded in the provided context.

This approach helps the application answer questions using the content extracted from the user's images rather than relying only on the model's general knowledge.

## ✅ Advantages

- Converts text-containing images into searchable information.
- Supports questions across multiple uploaded images.
- Combines OCR with semantic retrieval and AI-based answers.
- Provides an interactive and easy-to-use interface.
- Preserves source-image information for retrieved text.
- Can reduce the need to manually read lengthy image content.

## ⚠️ Limitations

- OCR accuracy depends on image quality, font size, orientation, and layout.
- Handwritten text and complex tables may not be recognized accurately.
- The quality of answers depends on the extracted text and retrieved context.
- Groq API access requires a valid API key and internet connectivity.
- Embedding models and vector storage require memory and processing resources.
- The language model may occasionally produce incorrect or incomplete answers.

## 🚀 Applications

- Question answering from lecture notes and study materials.
- Extracting information from scanned documents.
- Searching text in screenshots and photographs.
- Understanding text in reports and printed pages.
- Educational document analysis.
- Information retrieval from collections of image-based documents.

## 🔮 Future Enhancements

- Add support for PDF documents.
- Improve OCR for handwritten text and complex layouts.
- Add multilingual text extraction.
- Display image regions associated with retrieved answers.
- Support persistent vector storage across application sessions.
- Deploy the application to a cloud platform.
- Add user authentication and document management.

## 🏁 Conclusion

The Image-Based RAG Question Answering System demonstrates how image processing, OCR, vector databases, semantic search, and Large Language Models can be integrated into a single application. It enables users to extract information from images and ask questions in natural language, making image-based content easier to search and understand.

## 👩‍💻 Author

**Bommidi Kruthika**

Computer Science and Engineering Student

GitHub: [github.com/bommidikruthika](https://github.com/bommidikruthika)

---

*Developed as a Python mini project demonstrating OCR, Retrieval-Augmented Generation, and AI-powered question answering.*
