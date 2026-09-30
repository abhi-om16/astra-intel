# 🛡️ ASTRA INTEL
[![🚀 Live Demo](https://img.shields.io/badge/🚀_Live_Demo-ASTRA_INTEL-blue?style=for-the-badge)](https://astra-intel-hyfjmqlm2n2qdtme9yozft.streamlit.app/)
### Document Intelligence • Hybrid Retrieval • Grounded Q&A

ASTRA INTEL is a document intelligence application that allows users to upload a PDF, process its contents, and ask questions about the document.

Instead of allowing the language model to answer purely from general knowledge, ASTRA INTEL first retrieves relevant passages from the uploaded document and provides them to the language model as context.

The goal is to generate answers that are grounded in the uploaded document and accompanied by relevant source pages.

---

## 🚀 Features

- 📄 PDF document upload
- 🔍 Automatic PDF text extraction
- ✂️ Text chunking with overlapping chunks
- 🧠 Semantic embeddings
- 🔎 Hybrid retrieval using:
  - Semantic similarity
  - Keyword matching
- 🤖 Gemini-powered question answering
- 📌 Page-level source references
- 💬 Multi-turn conversation history
- ✨ Document summarization
- 🔬 Retrieved evidence inspection
- ⚠️ Basic error handling
- 🎨 Interactive web interface
- 🔐 API key handled through environment variables/secrets

---

# 🎯 Problem Statement

Large documents can contain hundreds of pages, making it difficult to quickly locate specific information.

Traditional document reading requires users to manually search through pages and understand the surrounding context.

ASTRA INTEL addresses this problem by allowing users to ask natural-language questions about an uploaded document.

The system:

1. Processes the uploaded document.
2. Breaks the document into searchable chunks.
3. Creates semantic representations of those chunks.
4. Retrieves relevant passages for a question.
5. Sends the retrieved passages to Gemini.
6. Generates an answer using the retrieved document context.
7. Displays the relevant source pages and evidence.

---

# 🧠 How ASTRA INTEL Works

ASTRA INTEL follows a Retrieval-Augmented Generation (RAG) approach.

```text
                    ┌─────────────────┐
                    │   PDF Upload    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ PyMuPDF Text    │
                    │ Extraction      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Chunking   │
                    │ 250 words       │
                    │ 60 word overlap │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Gemini          │
                    │ Embeddings      │
                    └────────┬────────┘
                             │
                             ▼
             ┌──────────────────────────────┐
             │      Hybrid Retrieval       │
             │                              │
             │  Semantic Similarity  60%    │
             │  Keyword Matching     40%    │
             └──────────────┬───────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Relevant Chunks │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Gemini LLM      │
                   │ Grounded Answer │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Answer + Pages  │
                   └─────────────────┘


                   🔎 Retrieval System

ASTRA INTEL does not rely solely on semantic similarity.

For every document chunk, two relevance signals are calculated.

1. Semantic Similarity

The user's question is converted into an embedding using:

gemini-embedding-001

Each document chunk also has an embedding.

Cosine similarity is then used to measure how semantically related the question and document chunk are.

2. Keyword Matching

The system also checks for overlap between meaningful words in the question and the document chunk.

This provides a lexical signal that can help retrieve passages containing important terms from the question.

3. Combined Score

The final retrieval score is:

Combined Score =
    0.60 × Semantic Score
  + 0.40 × Keyword Score

The highest-scoring relevant chunks are selected while limiting excessive retrieval from the same page.

This hybrid approach combines semantic understanding with direct keyword overlap.

🤖 Grounded Question Answering

Retrieved document passages are provided to Gemini as context.

The answering instructions require the model to:

Use only the supplied document excerpts
Avoid relying on outside knowledge
Avoid guessing
Clearly state when the requested information is not available
Provide supporting source pages for factual answers

This helps reduce unsupported answers and makes the connection between the generated answer and the uploaded document more transparent.

📌 Source Attribution

ASTRA INTEL keeps track of the page associated with every retrieved chunk.

After answering a question, the interface displays the retrieved source pages.

Users can also expand the evidence section to inspect the passages that were retrieved for the question.

This allows users to verify the context used by the system instead of blindly trusting the generated response.

💬 Multi-Turn Conversation

ASTRA INTEL maintains conversation history during the current document session.

Previous questions and answers can be used as conversational context for subsequent questions.

This allows users to ask follow-up questions instead of treating every question as a completely isolated interaction.

✨ Document Summarization

ASTRA INTEL can generate a concise summary of the uploaded document.

The summarization process is instructed to use the uploaded document as its source rather than relying on unrelated outside information.

🛠️ Tech Stack
Technology	Purpose
Python	Application logic
Streamlit	Web application interface
PyMuPDF	PDF text extraction
NumPy	Vector operations and cosine similarity
Google Gemini	Embeddings and language generation
Gemini Embedding Model	Semantic document representations
GitHub	Source code and version control
GitHub Codespaces	Browser-based development
📁 Project Structure
astra-intel/
│
├── app.py
├── index.html
├── README.md
└── requirements.txt
⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/abhi-om16/astra-intel.git
cd astra-intel
2. Install dependencies
pip install -r requirements.txt
3. Configure Gemini API Key

ASTRA INTEL requires a Gemini API key.

The application reads the key from the environment variable:

GEMINI_API_KEY

Do not hardcode the API key inside app.py.

Do not commit API keys, passwords, tokens, or .env files containing secrets to GitHub.

4. Run the application
streamlit run app.py

The application will then be available through the Streamlit local/forwarded port.

🔐 Security

The Gemini API key is accessed through:

os.environ["GEMINI_API_KEY"]

The key is not stored directly inside the source code.

For deployment environments, the API key should be configured using the platform's secret-management system.

🧪 Testing

The application was tested using questions whose answers were explicitly present in the uploaded document, as well as questions designed to test the grounding behavior of the system.

Test Case 1 — Direct factual question
What is an intrinsic semiconductor?

Expected behavior:

Retrieve the relevant section.
Generate an answer using the document.
Display the supporting page.

Observed behavior:

An intrinsic semiconductor is a pure semiconductor.

Source:

Page 15
Test Case 2 — Relationship question
What is the relationship between electron and hole concentrations in an intrinsic semiconductor?

Expected behavior:

Retrieve relevant document passages.
Generate the answer from the retrieved context.
Display supporting source pages.
Test Case 3 — Conceptual question
What is the valence band?

Expected behavior:

Retrieve the relevant document passage.
Generate an answer based on the document.
Display the relevant source page.
Test Case 4 — Conceptual question
What is the conduction band?

Expected behavior:

Retrieve the relevant document passage.
Generate an answer based on the document.
Display the relevant source page.
Test Case 5 — Out-of-document question
What is a pizza recipe?

Expected behavior:

The system should not use general knowledge to answer an unrelated question.

If the document does not contain the requested information, the system should indicate that the answer is not available in the uploaded document.

⚠️ Current Limitations
PDF Extraction

The current pipeline primarily relies on text extraction from PDFs.

Image-only or heavily scanned PDFs may not provide useful extracted text.

Retrieval

The current retrieval system uses in-memory embeddings and NumPy cosine similarity rather than a dedicated persistent vector database.

Session Persistence

Conversation history and document indexing are maintained during the current application session rather than being stored in a persistent database.

Document Scope

The current interface is primarily designed around processing an uploaded PDF document.

Retrieval Accuracy

Semantic similarity and keyword matching can sometimes retrieve related passages that do not directly contain the exact answer.

The system therefore relies on grounded prompting and evidence inspection to reduce unsupported responses.

🔮 Future Improvements

Potential future improvements include:

OCR support for scanned documents
Multiple document upload
Cross-document comparison
Persistent conversation history
Persistent vector database
Improved citation precision
Hallucination detection
Better retrieval reranking
Local embedding models
More advanced document parsing
Support for additional document formats
🤖 AI Usage Disclosure

AI tools were used during the development of ASTRA INTEL.

AI assistance was used for:

Exploring implementation approaches
Generating and refining portions of code
Debugging development issues
Improving the user interface
Reviewing implementation ideas
Iterating on prompts
Improving retrieval behavior

The generated code and suggestions were reviewed, modified, tested, and validated during development.

The final implementation was tested against document-based questions to verify the retrieval and grounding pipeline.

🧩 Design Philosophy

ASTRA INTEL was developed with the following priority:

Core Functionality
       ↓
Reliability
       ↓
Testing
       ↓
Documentation
       ↓
Advanced Features

The main focus was to build a working document intelligence pipeline rather than adding unnecessary complexity.

The engineering focus is the complete path from document ingestion to evidence retrieval and grounded answer generation.

🏗️ Architecture

The main components of ASTRA INTEL are:

1. Document Processing

PyMuPDF extracts text from the uploaded PDF.

2. Chunking

The extracted text is divided into overlapping chunks of approximately 250 words with a 60-word overlap.

3. Embedding Generation

Gemini's embedding model converts the document chunks and user queries into vector representations.

4. Hybrid Retrieval

The system combines semantic similarity and keyword matching to identify relevant chunks.

5. Context Construction

The highest-ranked chunks are provided to Gemini as document context.

6. Grounded Generation

Gemini generates the final response using the retrieved document context.

7. Source Display

The application displays the pages associated with the retrieved evidence.

🌐 Deployment

The application is designed to run as a Streamlit application.

For deployment, the Gemini API key should be configured using the hosting platform's secret-management system rather than committing the key to the repository.

👨‍💻 Author

Abhinav Om

B.Tech Computer Science & Engineering

BMS Institute of Technology & Management

GitHub:

https://github.com/abhi-om16

🛡️ ASTRA INTEL

Ask your documents. Get grounded answers.

