from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import fitz  # PyMuPDF
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load Sentence Transformer for semantic search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Storage for documents and their embeddings
documents = []
document_embeddings = []

def parse_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save the uploaded PDF file
    file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Parse the document
    document_text = parse_pdf(file_path)
    
    # Store the document and its embedding
    documents.append(document_text)
    document_embedding = embedding_model.encode(document_text, convert_to_tensor=True)
    document_embeddings.append(document_embedding)

    return {"filename": file.filename, "status": "Uploaded successfully"}

@app.post("/query")
async def query_document(question: str):
    if not documents:
        return JSONResponse(content={"answer": "No documents uploaded."}, status_code=400)

    # Embed the question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(question_embedding, document_embeddings)
    top_results = np.argpartition(-similarities.cpu().numpy(), kth=3)[:3]

    # Retrieve the top 3 relevant documents
    relevant_chunks = [documents[i] for i in top_results]

    # Generate an answer based on the relevant chunks
    answer_input = f"Context: {' '.join(relevant_chunks)}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(answer_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
