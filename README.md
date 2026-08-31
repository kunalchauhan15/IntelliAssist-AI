# 🤖 IntelliAssist AI

### AI-Powered Document Intelligence & RAG Assistant

**Developed by:** Kunal Chauhan
**Project:** IntelliAssist AI
**Role:** AI / Generative AI Developer
**Technology:** Python, Streamlit, RAG, Gemini API, FAISS
**Project Type:** AI-powered Document Question Answering System

---

## 📌 About Me

Hi, I'm **Kunal Chauhan**, a passionate and aspiring **AI / Generative AI Developer** interested in building intelligent applications using modern AI technologies.

My technical interests include:

* Java
* Python
* C / C++
* Data Structures & Algorithms
* SQL / MySQL
* HTML / CSS / JavaScript
* Git & GitHub
* API Integration & API Testing
* Artificial Intelligence
* Generative AI
* Agentic AI
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)

---

# 🚀 About IntelliAssist AI

**IntelliAssist AI** is an AI-powered document intelligence application that allows users to upload documents and interact with them using natural-language questions.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate intelligent answers using **Google Gemini**.

The goal of this project is to build a practical AI assistant that can understand documents, retrieve relevant information, answer questions, summarize content, and maintain conversation context.

---

# ✨ Features

### 📄 Document Processing

* Upload PDF documents
* Upload DOCX documents
* Upload TXT documents
* Extract text from documents
* Process documents page by page
* Clean and normalize extracted text

### 🧠 RAG Pipeline

* Text chunking
* Semantic embeddings
* FAISS vector search
* Relevant context retrieval
* Context-aware answer generation
* Configurable retrieval settings

### 🤖 Gemini AI

* Google Gemini API integration
* AI-powered question answering
* Document-based responses
* Configurable Gemini model
* Temperature control

### 💬 Chat System

* Interactive chat interface
* Conversation history
* Multiple conversations
* Current chat tracking
* Clear chat functionality
* Chat history management

### 📝 Document Summarization

* Generate document summaries
* AI-powered summarization
* Summary displayed inside the application

### 🔍 Semantic Search

* Search inside uploaded documents
* Retrieve semantically relevant content
* Display matching document information

### 🎯 AI Analysis

* Sentiment analysis
* Intent analysis
* Context-aware analysis

### ⚙️ Settings

Users can configure:

* Gemini model
* Temperature
* Top-K retrieval
* Distance threshold
* Chat history
* Sentiment analysis
* Intent analysis
* Citations
* Semantic search

---

# 🏗️ Project Architecture

```text
IntelliAssist-AI/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── chat_history.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── intent.py
│   ├── rag_pipeline.py
│   ├── sentiment.py
│   ├── summarizer.py
│   ├── text_processor.py
│   └── vector_store.py
│
├── utils/
│   └── helpers.py
│
├── data/
│   └── documents/
│
├── vectorstore/
│
├── test_gemini.py
├── test_summary.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

---

# 🔄 How IntelliAssist AI Works

```text
             ┌─────────────────────┐
             │   Upload Document   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Document Extraction │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Text Cleaning       │
             │ & Chunking          │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Generate Embeddings │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   FAISS Vector DB   │
             └──────────┬──────────┘
                        │
                  User Question
                        │
                        ▼
             ┌─────────────────────┐
             │ Semantic Retrieval  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Relevant Context    │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Gemini AI Model   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   AI Answer         │
             └─────────────────────┘
```

---

# 🛠️ Technologies Used

| Technology               | Purpose                   |
| ------------------------ | ------------------------- |
| Python                   | Core programming language |
| Streamlit                | Web application interface |
| Google Gemini            | AI answer generation      |
| Google GenAI SDK         | Gemini API integration    |
| Sentence Transformers    | Text embeddings           |
| FAISS                    | Vector similarity search  |
| PyPDF                    | PDF processing            |
| python-docx              | DOCX processing           |
| LangChain Text Splitters | Text chunking             |
| NumPy                    | Numerical operations      |
| python-dotenv            | Environment configuration |
| Git                      | Version control           |
| GitHub                   | Source code hosting       |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd IntelliAssist-AI
```

## 2. Create Virtual Environment

```powershell
python -m venv venv
```

## 3. Activate Virtual Environment

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

You should see:

```text
(venv)
```

at the beginning of your terminal.

---

# 📥 Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

# 🔑 Configure Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

⚠️ **Never upload your ****`.env`**** file or API key to GitHub.**

The `.gitignore` file already excludes:

```text
.env
```

---

# ▶️ Run the Application

Start IntelliAssist AI using:

```powershell
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 🧪 Testing

The project contains test files for validating important functionality.

### Gemini API Test

```powershell
python test_gemini.py
```

### Summary Test

```powershell
python test_summary.py
```

---

# 📂 Supported Documents

IntelliAssist AI currently supports:

* `.pdf`
* `.docx`
* `.txt`

---

# 🔐 Security

Sensitive configuration is stored using environment variables.

The following files and directories should not be committed:

```text
.env
venv/
.venv/
__pycache__/
vectorstore/faiss_index/
```

---

# 📈 Future Improvements

Planned improvements may include:

* Persistent chat history database
* User authentication
* Multiple document collections
* Improved citation system
* Advanced conversation memory
* Streaming AI responses
* Document comparison
* OCR support
* More file formats
* Cloud deployment
* Advanced Agentic AI capabilities
* Improved UI/UX
* Production-ready database integration

---

# 🎯 Project Objective

The main objective of **IntelliAssist AI** is to demonstrate how modern Generative AI, embeddings, vector databases, and RAG architecture can be combined to build a practical document intelligence system.

This project is also designed as a learning and portfolio project demonstrating practical implementation of:

**Python → AI → Generative AI → LLM → Embeddings → Vector Search → RAG → AI Application**

---

# 👨‍💻 Developer

**Kunal Chauhan**

Aspiring **AI / Generative AI Developer**

Interested in:

```text
Artificial Intelligence
Generative AI
Agentic AI
Large Language Models
RAG
Python
Java
SQL
API Integration
Software Development
```

---

# 📫 Professional Profiles

Replace the following placeholders with your actual professional profiles before publishing the repository:

* **GitHub:** https://github.com/kunalchauhan15
* **LinkedIn:** https://www.linkedin.com/in/kunalchauhan15/
* **Email:** [kunalchauhan0205@gamil.com](mailto:kunalchauhan0205@gamil.com)

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is created for educational, learning, and portfolio purposes.
