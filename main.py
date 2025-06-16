from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, shutil, uuid
from PDF_service import load_pdf
from RAG_model import create_retriever_from_docs, build_rag_chain

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever/rag_chain cache
retriever = None
rag_chain = None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global retriever, rag_chain

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_pdf(file_path)
    retriever = create_retriever_from_docs(docs)
    rag_chain = build_rag_chain(retriever)

    return {"message": "File processed and RAG chain created"}

@app.post("/query")
async def query_api(query: dict):
    global rag_chain
    if rag_chain is None:
        return JSONResponse(content={"reply": "Please upload a PDF first."}, status_code=400)

    question = query.get("query", "")
    if not question.strip():
        return JSONResponse(content={"reply": "Query cannot be empty."}, status_code=400)

    try:
        reply = rag_chain.invoke(question)
        return {"reply": reply}
    except Exception as e:
        return JSONResponse(content={"reply": str(e)}, status_code=500)
