# GitRAG: Repository-powered RAG with LangChain and Zephyr

**GitRAG** is a simple Retrieval-Augmented Generation (RAG) application that leverages Hugging Face's [Zephyr model](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) and LangChain to answer questions based on the contents of a GitHub repository.

This project is based on Hugging Face's tutorial:  
🔗 https://huggingface.co/learn/cookbook/rag_zephyr_langchain

---

## 🚀 Features

- Load and parse documents directly from a GitHub repository
- Split and embed documents using HuggingFace BGE embeddings
- Retrieve relevant context with FAISS
- Generate answers using a quantized Zephyr model (or any other Hugging Face-supported model)
- Built using LangChain and Hugging Face Transformers

---

## 📁 Project Structure

.
├── main.py # Main script for running RAG pipeline
├── requirements.txt # Python dependencies
├── .env # Environment variables (see below)
└── README.md # You're reading this!

---

## ⚙️ Setup

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/gitrag.git
cd gitrag
```

### 2. Create and activate a Conda environment

```bash
conda create -n gitrag python=3.10
conda activate gitrag
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup your `.env` file

Create a .env file in the root directory with the following variables:

```bash
GITHUB_TOKEN=ghp_yourgithubtokenhere
GITHUB_REPO=owner/repo-name
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
```

## 🧠 How It Works

- Document loading: Uses GitHubRepositoryLoader to fetch code/docs from a repo.
- Splitting: Breaks documents into manageable chunks using RecursiveCharacterTextSplitter.
- Embedding + Retrieval: Uses FAISS and BGE embeddings to retrieve relevant chunks.
- Generation: Passes the retrieved context to a Zephyr-based language model via LangChain.


## 🧪 Running the Script

```bash
python main.py
```

The script will:
- Prompt you for the GitHub access token (unless stored in .env)
- Load, chunk, and embed documents
- Run a simple RAG query: `"How do you combine multiple adapters?"`
- Print out responses from both:
    - The LLM without context
    - The RAG-enhanced LLM with GitHub context


## 📝 Notes

- Ensure you have a valid GitHub token with repo access.
- The model is loaded with 4-bit quantization using BitsAndBytesConfig for memory efficiency.

## 📚 References

- [Hugging Face Cookbook – RAG with Zephyr & LangChain](https://huggingface.co/learn/cookbook/rag_zephyr_langchain)
- [LangChain Documentation](https://www.langchain.com/)
- [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

