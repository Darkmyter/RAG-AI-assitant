import base64
from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
import os
import tempfile

# Initialize FastAPI
app = FastAPI()

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load ChromaDB
chroma_db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=chroma_db_path)
collection = chroma_client.get_or_create_collection(name="documents")

# Load GPT4ALL Model
model_path = "c:/Users/bader/AppData/Local/nomic.ai/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf"
gpt4all_model = GPT4All(model_path)

# Define API Request Schema
class QueryRequest(BaseModel):
    query: str

class Base64PDFUpload(BaseModel):
    filename: str
    file_data: str  # Base64 encoded string

class Base64PDFUploadList(BaseModel):
    files: list[Base64PDFUpload]

### 📌 ROUTE 1: Retrieve and Answer Queries ###
@app.post("/ask")
def ask_question(request: QueryRequest):
    retrieved_chunks = retrieve_relevant_chunks(request.query)
    context = "\n".join(retrieved_chunks)
    response = generate_response(request.query, context)
    return {"query": request.query, "response": response}


### 📌 ROUTE 2: Update the Database with Base64 PDFs ###
@app.post("/update_db")
async def update_database(request: Base64PDFUploadList):
    global collection

    # Clear the existing database before inserting new files (optional)
    print("Clearing old database...")
    chroma_client.delete_collection(name="documents")
    collection = chroma_client.get_or_create_collection(name="documents")

    new_documents = []

    for file in request.files:


        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file_path = temp_file.name  # Get the temp file path

            # Decode Base64 content and save as a PDF
            pdf_bytes = base64.b64decode(file.file_data)
            temp_file.write(pdf_bytes)  # Write to temp file

        # Load and extract text from PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Embed chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

        # Store in ChromaDB
        for i, chunk in enumerate(chunk_texts):
            collection.add(
                ids=[f"{file.filename}_{i}"],  # Unique ID per chunk
                embeddings=[chunk_embeddings[i].tolist()],
                documents=[chunk]
            )

        new_documents.append(file.filename)

        # Ensure the temp file is deleted after processing
        try:
            os.remove(file_path)
        except PermissionError:
            print(f"Warning: Could not delete {file_path}. It may still be in use.")

    return {"message": "Database updated successfully!", "processed_files": new_documents}


### 📌 Utility Functions ###
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results else ["No relevant context found."]


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
    
    with gpt4all_model.chat_session():
        response = gpt4all_model.generate(prompt, max_tokens=250)
    
    return response


### 📌 ROUTE 3: Home Route ###
@app.get("/")
def home():
    return {"message": "Welcome to the RAG API. Use /ask to query and /update_db to add PDFs via Base64."}
