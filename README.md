# Building a RAG Pipeline with Hugging Face, ChromaDB, and Railway

## Local Deployment

To deploy the application locally, follow these steps:

### Start the Backend API

Setup `.env`:

``
HF_API_KEY=<HuggingFace API KEY>
``

Run the following command to start the FastAPI server:

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Launch the UI

Run the UI application using:

```sh
python app_ui.py
```

### Access the Application

Once deployed, you can access the UI through:

- **Localhost URL:** [http://127.0.0.1:7860](http://127.0.0.1:7860)

## How the Application Works

The application allows users to upload PDFs, which are processed and stored in a vector database. Users can then ask questions about the content, and the system retrieves relevant text from the documents to provide an informed response using a language model.

### Steps:

1. **Upload PDFs** – The system extracts and stores text from the uploaded files.
2. **Ask a Question** – Users enter a question related to the uploaded documents.
3. **Retrieve Information** – The system finds relevant document sections based on the question.
4. **Generate Response** – A language model processes the retrieved text and generates an answer.

## Introduction to Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances language model responses by retrieving relevant information from external data sources. Unlike standard language models that rely solely on their training data, RAG dynamically retrieves context from a knowledge base, improving accuracy and reducing hallucinations.

### How RAG Works

1. **Query Embedding:** Convert user queries into vector representations using an embedding model.
2. **Retrieval:** Search for the most relevant documents in a vector database based on semantic similarity.
3. **Contextual Generation:** Provide retrieved documents as context to a language model to generate informed responses.

### Our RAG Tech Stack

For our implementation, we use the following components:

- **Embedding Model:** Hugging Face's Sentence-Transformers for vector embeddings.
- **Vector Database:** ChromaDB for efficient document retrieval.
- **LLM:** Hugging Face API for text generation.
- **Framework:** FastAPI for backend API.
- **Deployment:** Railway for cloud deployment.
- **Frontend:** Gradio for an interactive UI.

## Breakdown of Backend Code

Our backend handles document ingestion, retrieval, and question-answering.

### 1. Ingestion

We use SentenceTransformers to convert text into vector embeddings and store them in ChromaDB, enabling efficient semantic search.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Load and process PDF
loader = PyPDFLoader("example.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Embed and store chunks
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
for i, chunk in enumerate(chunk_texts):
    collection.add(ids=[f"doc_{i}"], embeddings=[chunk_embeddings[i].tolist()], documents=[chunk])
```

### 2. Retrieval

We retrieve the most relevant chunks from ChromaDB using the encoded query.

```python
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results else ["No relevant context found."]
```

### 3. Question-Answering

We use a language model via Hugging Face API to generate responses based on retrieved context.

```python
import requests

def generate_response(query, context):
    prompt = f"""
    Based on the following document excerpts, answer the question as accurately as possible.

    === DOCUMENT EXCERPTS ===
    {context}
    =========================

    QUESTION: {query}

    If the answer is not found in the document, explicitly say "I cannot find the answer in the provided text."

    ANSWER:
    """
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, json=payload, headers=headers)
    output = response.json()
    if isinstance(output, list) and "generated_text" in output[0]:
        return output[0]["generated_text"].strip()
    return "Error in generating response."
```

### 4. Using FastAPI

FastAPI is used to create API endpoints for querying and updating the database.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    retrieved_chunks = retrieve_relevant_chunks(request.query)
    context = "
".join(retrieved_chunks)
    response = generate_response(request.query, context)
    return {"query": request.query, "response": response}
```

This improved version includes clear explanations, the usage of SentenceTransformers for embeddings, and ChromaDB for retrieval, while also detailing how the FastAPI framework integrates the components.

Our backend handles document ingestion, retrieval, and question-answering.

## Breakdown of Frontend Code (app\_ui.py)

Our frontend provides a user interface to interact with the RAG system using Gradio.

### 1. API Endpoints

We define URLs to communicate with the backend API hosted on Railway.

```python
import gradio as gr
import requests
import os

RAILWAY_API_URL = os.getenv("RAILWAY_API_URL", "https://your-railway-url")
API_URL_ASK = f"{RAILWAY_API_URL}/ask"
```

### 2. Query Function

This function sends a query request to the backend API and retrieves the answer.

```python
def ask_question(query):
    response = requests.post(API_URL_ASK, json={"query": query})
    return response.json().get("response", "Error fetching response.")
```

### 3. UI Construction with Gradio

We define a Gradio interface with a text input and a button to submit queries.

```python
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 RAG-based AI Assistant")
    question_input = gr.Textbox(label="Ask a question")
    ask_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_button.click(ask_question, inputs=[question_input], outputs=[answer_output])

demo.launch(server_name="0.0.0.0", server_port=7860)
```

### .env Configuration

Create a `.env` file to store environment variables locally before deployment:

```sh
HF_API_KEY=your_huggingface_api_key
RAILWAY_API_URL=https://your-railway-url
```

## Railway Deployment

### Procfile

To ensure Railway runs the application correctly, create a `Procfile` in the root directory with the following content:

```sh
web: uvicorn app:app --host 0.0.0.0 --port $PORT
ui: python app_ui.py
```

### .env Configuration

Create a `.env` file to store environment variables locally before deployment:

```sh
HF_API_KEY=your_huggingface_api_key
RAILWAY_API_URL=https://your-railway-url
```

## Railway Deployment

To deploy the application on Railway, follow these steps:

### 1. Set Up a Railway Project

- Go to [Railway.app](https://railway.app/)
- Create a new project and connect it to your GitHub repository containing the application code.

### 2. Configure Environment Variables

- Add necessary environment variables like `HF_API_KEY` and `RAILWAY_API_URL`.

### 3. Deploy the Application

- Use the Railway dashboard to deploy your application.
- Monitor logs and ensure that the API and UI are running properly.

### 4. Access the Deployed Application

Once deployed, you can access your RAG-based AI assistant through the provided Railway URL.

## Conclusion

We have broken down the backend and frontend code, explaining how each component plays a role in our RAG pipeline. The backend handles document retrieval and response generation, while the frontend provides an easy-to-use interface for users to interact with the system.

