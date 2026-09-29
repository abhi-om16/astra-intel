# astra-intel
AI-powered defence document intelligence system for PDF summarization, grounded question answering, and page-level source citations using RAG.
# ASTRA INTEL

### AI-Powered Defence Document Intelligence System

ASTRA INTEL is an AI-powered document intelligence system that allows users to upload defence-related PDF documents, generate concise summaries, and ask questions about their contents.

The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant sections from the uploaded document before generating an answer. This helps keep responses grounded in the provided document rather than relying solely on the language model's general knowledge.

> Built as part of the **ASTRA Software Team 3-Day Build Challenge 2026–27** at BMS Institute of Technology & Management.

---

## 🚀 Features

### Core Features

* 📄 Upload PDF documents
* 🔍 Extract text while preserving page information
* ✂️ Split documents into searchable chunks
* 🧠 Generate embeddings for document chunks
* 🗂️ Store and search embeddings using a vector database
* 💬 Ask questions about the uploaded document
* 📝 Generate concise document summaries
* 📑 Display source pages for generated answers
* 🛡️ Ground answers in the uploaded document
* ⚠️ Clearly indicate when the document does not contain enough information
* 💬 Support multi-turn document conversations
* ❌ Handle invalid files and processing errors

---

## 🧠 How It Works

ASTRA INTEL follows a Retrieval-Augmented Generation pipeline:

```text
                 PDF Upload
                     │
                     ▼
              Text Extraction
                     │
                     ▼
             Text Chunking
                     │
                     ▼
               Embeddings
                     │
                     ▼
              Vector Store
                     │
              ┌──────┴──────┐
              │             │
          User Query     Summary
              │
              ▼
       Similarity Retrieval
              │
              ▼
       Relevant Document
            Context
              │
              ▼
             LLM
              │
              ▼
      Grounded Response
              │
              ▼
       Source / Page Citations
```

---

## 🏗️ Architecture

The application is divided into the following components:

### 1. Document Processing

The uploaded PDF is processed page by page. Text is extracted while maintaining page numbers so that retrieved information can later be traced back to its original location.

### 2. Text Chunking

The extracted document text is divided into smaller chunks. This allows the retrieval system to search for relevant portions of the document instead of processing the entire document for every question.

### 3. Embedding Generation

Each text chunk is converted into a numerical vector representation using an embedding model.

### 4. Vector Search

The embeddings are stored in a vector index. When the user asks a question, the system retrieves the chunks that are semantically most relevant to the query.

### 5. Retrieval-Augmented Generation

The retrieved chunks are provided to the language model as context.

The model is instructed to answer using the provided document context and avoid introducing information that cannot be supported by the document.

### 6. Source Attribution

The system keeps the original page information associated with each chunk and displays relevant source pages alongside the generated answer.

---

## 🛠️ Tech Stack

| Component       | Technology        |
| --------------- | ----------------- |
| Frontend        | Streamlit         |
| Language        | Python            |
| PDF Processing  | PyMuPDF           |
| Embeddings      | OpenAI Embeddings |
| Vector Search   | FAISS             |
| LLM             | OpenAI API        |
| Retrieval       | RAG               |
| Version Control | Git & GitHub      |

---

## 📂 Project Structure

```text
astra-intel/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm.py
│   └── prompts.py
│
└── tests/
    └── test_cases.md
```

> The exact structure may change as the implementation evolves.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/abhi-om16/astra-intel.git
cd astra-intel
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
OPENAI_API_KEY=your_api_key_here
```

Never commit API keys, passwords, tokens, or other secrets to the repository.

### 5. Run the application

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit server.

---

## 🔐 Environment Variables

| Variable         | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `OPENAI_API_KEY` | API key used for embeddings and/or language model access |

See `.env.example` for the required configuration.

---

## 🧪 Testing

The system was tested against different types of inputs and queries, including:

* Valid PDF documents
* Empty or text-poor PDFs
* Questions directly answered by the document
* Questions requiring information from different pages
* Questions unrelated to the document
* Ambiguous questions
* Repeated questions
* Missing API key / configuration errors
* Invalid file uploads

### Grounding Test

A key test is asking a question whose answer is **not present in the uploaded document**.

Expected behavior:

```text
The provided document does not contain enough information
to answer this question.
```

This helps prevent the model from confidently generating unsupported information.

---

## 🤖 AI Usage

AI-assisted development was used during the project for:

* Understanding and exploring implementation approaches
* Generating initial code structures
* Debugging errors
* Improving prompts
* Exploring RAG implementation techniques
* Reviewing and refining parts of the code

AI-generated code was **tested, modified, and validated** during development rather than being used without verification.

The final implementation, architecture decisions, integration, testing, and debugging were reviewed as part of the development process.

---

## ⚠️ Limitations

* The quality of answers depends on the quality and structure of the uploaded document.
* Scanned PDFs may require OCR for reliable text extraction.
* Very large documents may require additional optimization.
* Retrieval quality depends on chunking, embedding, and similarity-search parameters.
* The system is designed to answer based on the uploaded document and should not be treated as an independent source of defence intelligence.
* API-based models introduce external service and API dependency.

---

## 🔮 Future Improvements

Potential improvements include:

* 📚 Support for multiple documents
* 🔎 Improved semantic search
* 📊 Document comparison
* 🖼️ OCR support for scanned documents
* 🧠 Local/open-source language models
* 💾 Persistent conversation history
* 🔗 Improved source highlighting
* 🛡️ More advanced hallucination/grounding detection
* ⚡ Retrieval and processing optimizations

---

## 📌 Challenge Requirements

ASTRA INTEL was developed as a solution for the **ASTRA 3-Day Build Challenge 2026–27**.

The project focuses on the required document upload, processing, summarization, grounded question answering, and source/page attribution functionality specified in the challenge.

The challenge emphasizes a working product, reliability, testing, documentation, and the ability to explain technical decisions rather than simply producing a large amount of code.

---

## 👨‍💻 Author

**Abhinav Om**

B.Tech CSE
BMS Institute of Technology & Management

GitHub: `https://github.com/abhi-om16`

---

## 📄 License

This project is developed as part of an academic/student technical build challenge.
