# 🧠 CONFIDENCE-RAG:

## 📌 About the Project

**CONFIDENCE-RAG** is a novel framework designed to reduce hallucinations in large language models by introducing **confidence-weighted retrieval and generation**. Traditional Retrieval-Augmented Generation (RAG) methods treat all retrieved evidence as equally trustworthy, leading to inaccuracies and overconfident responses. CONFIDENCE-RAG addresses this by learning **calibrated passage-level confidence scores**, which are then injected into the retrieval, prompt construction, and generation phases of the pipeline.

This system integrates:
- A **dual-encoder retriever** using SentenceTransformers.
- A lightweight **confidence scorer** that uses retrieval signals and document metadata.
- A modified **LLM generation** process that scales attention based on passage trustworthiness.
- A front-end interface that visually communicates the system’s confidence in each answer.

The framework is designed to work **efficiently on local hardware**, using tools like Qdrant for vector search, Streamlit for UI, and Ollama to run the LLM. It is especially applicable in domains where **accuracy and transparency** are critical (e.g., healthcare, legal, finance).

---

## 🗂️ Vector Database Setup

As part of CONFIDENCE-RAG, a **vector database** is essential for storing and retrieving high-dimensional document embeddings. This explains how to set up **Qdrant**, an open-source vector database, using **Docker** - a key infrastructure component of the system.

---

## 🚀 Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- Basic understanding of Docker commands.
- A directory to store persistent Qdrant data.

---

## ⚙️ Setting Up Qdrant with Docker

### 1. Install Docker Desktop

Get Docker for your OS from [docker.com](https://www.docker.com/products/docker-desktop/) and ensure it’s running.

---

### 2. Pull the Qdrant Image

```bash
docker pull qdrant/qdrant:latest
```

---

### 3. Create a Local Storage Directory

```bash
mkdir C:\Your_Project_Location\Confidence_RAG\qdrant\storage
```

This ensures that Qdrant data persists between container runs.

---

### 4. Start the Qdrant Container

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v C:\Your_Project_Location\Confidence_RAG\qdrant\storage:/qdrant/storage qdrant/qdrant:latest
```

---

### 5. Verify the Qdrant Server

```bash
curl http://localhost:6333/
```

You should receive a JSON response indicating that Qdrant is up and running.

---

## ✅ Qdrant Is Ready!

---

## 🦙 Ollama Installation and Model Setup

### 📦 Step 1: Install Ollama
Download and install [Ollama](https://ollama.com/download) for your operating system

---

### ✅ Step 2: Verify Installation
After installation, open your terminal (or command prompt) and run:
```bash
ollama --version
```
If installed correctly, this will display the current Ollama version.

---

### 🤖 Step 3: Download the Model
Use Ollama to pull the `llama3-chatqa` model to your local system:
```bash
ollama pull llama3-chatqa:latest
```

---

### 🚀 Step 4: Run the Model
Start the model with:
```bash
ollama run llama3-chatqa:latest
```
You can now interact with the model directly in your terminal.

---

## ✅ Llama Model Is Ready!

---

## 🔗 Additional Resources

- 📄 [Qdrant Docs](https://qdrant.tech/documentation/quickstart/)
- 🤖 [Ollama Model Docs](https://ollama.com/library/llama3-chatqa)

