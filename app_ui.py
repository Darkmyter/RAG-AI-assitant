import gradio as gr
import requests
import base64
import os
LOCAL = True

RAILWAY_API_URL = os.getenv("RAILWAY_API_URL", "https://rag-ai-assitant-production.up.railway.app")

# API Endpoints
if LOCAL:
    API_URL_ASK = "http://127.0.0.1:8000/ask"
    API_URL_UPDATE = "http://127.0.0.1:8000/update_db"
else:
    API_URL_ASK = f"{RAILWAY_API_URL}/ask"
    API_URL_UPDATE = f"{RAILWAY_API_URL}/update_db"

### 📌 Function: Convert PDF (Binary) to Base64 ###
def encode_pdf_to_base64(file_binary):
    return base64.b64encode(file_binary).decode("utf-8")

### 📌 Function: Upload PDFs via Base64 ###
def upload_pdfs(files):
    if not files:
        return "❌ No files uploaded."

    base64_files = []

    # Handle multiple files correctly
    if isinstance(files, list):  # If multiple files
        for file in files:
            file_data = encode_pdf_to_base64(file)  # Convert binary content to Base64
            base64_files.append({"filename": "uploaded_file.pdf", "file_data": file_data})  # No name available, assign generic
    else:  # If a single file
        file_data = encode_pdf_to_base64(files)  # Convert binary content
        base64_files.append({"filename": "uploaded_file.pdf", "file_data": file_data})  # Assign generic name

    # Send Base64 PDFs to API
    response = requests.post(API_URL_UPDATE, json={"files": base64_files})
    
    if response.status_code == 200:
        return "✅ Database updated successfully!"
    else:
        return "❌ Error updating the database."

### 📌 Function: Ask a Question ###
def ask_question(query):
    response = requests.post(API_URL_ASK, json={"query": query})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Failed to fetch response."

### 📌 Build Gradio UI ###
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 RAG-based AI Assistant")
    
    # File Upload
    with gr.Row():
        file_input = gr.File(label="Upload PDFs", file_types=[".pdf"], interactive=True, type="binary")
        upload_button = gr.Button("Update Database")
        upload_output = gr.Textbox(label="Upload Status", interactive=False)

    upload_button.click(upload_pdfs, inputs=[file_input], outputs=[upload_output])

    # Ask a Question
    question_input = gr.Textbox(label="Ask a question")
    ask_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", interactive=False)

    ask_button.click(ask_question, inputs=[question_input], outputs=[answer_output])

# Run the Gradio UI
if LOCAL:
    demo.launch(server_name="127.0.0.1", server_port=7860)
else:
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
