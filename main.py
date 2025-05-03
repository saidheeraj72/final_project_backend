# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
import os
import json
import pickle
from ultralytics import YOLO
import torchvision
import torch.nn as nn
from torchvision import transforms
from typing import List, Dict
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate

app = FastAPI(title="Cattle Muzzle Identifier + Chatbot")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== SETTINGS ==================
BASE_DIR = 'filtered_cattle_dataset'
YOLO_WEIGHTS = 'model.pt'
FEAT_WEIGHTS = 'feature_extractor.pth'
PKL_FILE = 'cattle_embs.pkl'
META_FILE = 'cattle_meta.json'
CONF_THRES = 0.15
PADDING = 10
SIM_THRESHOLD = 0.85
TOP_K = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================== CATTLE IDENTIFICATION ==================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet50(weights=None)
        self.features = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        return self.features(x).flatten(1)

detector = YOLO(YOLO_WEIGHTS)
fe = FeatureExtractor().to(DEVICE)
fe.load_state_dict(torch.load(FEAT_WEIGHTS, map_location=DEVICE))
fe.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def load_database():
    if not (os.path.exists(PKL_FILE) and os.path.exists(META_FILE)):
        return None, None, None
    with open(PKL_FILE, 'rb') as f: arr = pickle.load(f)
    with open(META_FILE, 'r') as f: meta = json.load(f)
    embs = torch.from_numpy(arr)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs, meta['ids'], meta['paths']

@app.get("/next-id")
async def get_next_id():
    try:
        if os.path.exists(BASE_DIR):
            existing_ids = [d for d in os.listdir(BASE_DIR) if d.startswith("cattle_")]
            if existing_ids:
                last_num = max([int(d.split("_")[1]) for d in existing_ids])
                return {"next_id": f"cattle_{last_num + 1}"}
        return {"next_id": "cattle_1"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify")
async def identify_cattle(file: UploadFile = File(...)):
    contents = await file.read()
    embs, ids, paths = load_database()
    if embs is None:
        raise HTTPException(status_code=400, detail="Database empty")
    file_bytes = np.asarray(bytearray(contents), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = detector(rgb, conf=CONF_THRES, verbose=False)
    if len(results[0].boxes) == 0:
        return {"error": "No muzzle detected"}
    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    crop = rgb[y1-PADDING:y2+PADDING, x1-PADDING:x2+PADDING]
    with torch.no_grad():
        img = Image.fromarray(crop)
        emb = fe(tfm(img).unsqueeze(0)).squeeze()
        emb = torch.nn.functional.normalize(emb, p=2, dim=0)
    sims = torch.matmul(embs, emb)
    best_idx = sims.argmax()
    best_id = ids[best_idx]
    similarity = float(sims[best_idx])
    return {
        "cattle_id": best_id,
        "similarity": similarity,
        "match": similarity >= SIM_THRESHOLD
    }

def crop_first_muzzle(img_bytes, pad=PADDING, conf=CONF_THRES):
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = detector(rgb, conf=conf, verbose=False)
    if len(res[0].boxes) == 0:
        return rgb, None, None
    i = res[0].boxes.conf.argmax()
    x1, y1, x2, y2 = res[0].boxes.xyxy[i].cpu().numpy().astype(int)
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = rgb[y1:y2, x1:x2]
    return rgb, crop, (x1, y1, x2, y2)

@app.post("/add")
async def add_cattle(
    cattle_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        embs, ids, paths = load_database()
        new_embs, new_ids, new_paths = [], [], []
        for file in files:
            contents = await file.read()
            full, crop, _ = crop_first_muzzle(contents)
            if crop is None:
                continue
            with torch.no_grad():
                img = Image.fromarray(crop)
                emb = fe(tfm(img).unsqueeze(0)).squeeze()
                emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            os.makedirs(os.path.join(BASE_DIR, cattle_id), exist_ok=True)
            dst_path = os.path.join(BASE_DIR, cattle_id, file.filename)
            with open(dst_path, "wb") as f:
                f.write(contents)
            new_embs.append(emb)
            new_ids.append(cattle_id)
            new_paths.append(dst_path)
        if not new_embs:
            raise HTTPException(
                status_code=400,
                detail="No valid muzzles detected in uploaded images."
            )
        if embs is not None:
            all_embs = torch.cat([embs, torch.stack(new_embs)])
            all_ids = ids + new_ids
            all_paths = paths + new_paths
        else:
            all_embs, all_ids, all_paths = new_embs, new_ids, new_paths
        arr = all_embs.numpy().astype(np.float32)
        with open(PKL_FILE, 'wb') as f: pickle.dump(arr, f)
        with open(META_FILE, 'w') as f: 
            json.dump({'ids': all_ids, 'paths': all_paths}, f)
        return {
            "message": f"Added {len(new_embs)} images for {cattle_id}",
            "cattle_id": cattle_id,
            "total_embeddings": len(all_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== CHATBOT FUNCTIONALITY ==================
GROQ_API_KEY = "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"
FAISS_DB_DIR = "faiss_dbs"
os.makedirs(FAISS_DB_DIR, exist_ok=True)

def extract_all_content_as_documents(pdf_path):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"page": page_number}))
    return documents

def total_json_data():
    MAIN_DIR = 'jsons/'
    files = os.listdir(MAIN_DIR)
    documents = []
    for file in files:
        file_path = os.path.join(MAIN_DIR, file)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        documents.append(Document(page_content=str(data), metadata={"source": file_path}))
    return documents

def load_or_create_db():
    db_path = os.path.join(FAISS_DB_DIR, "cattle_diseases_main_db")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    pdf_path = "pdfs/cattle-diseases-farmers-guide.pdf"
    pdf_docs = extract_all_content_as_documents(pdf_path)
    json_docs = total_json_data()
    all_docs = pdf_docs + json_docs
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(db_path)
    return db

def setup_chain():
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.05)
    prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in cattle disease detection and treatment.
        Your goal is to provide precise, research-backed information on cattle diseases, including:
        - Symptoms
        - Causes
        - Prevention
        - Treatment
        - Best veterinary practices
        Always provide detailed, actionable responses based on the available disease knowledge base.
        Chat History:
        {chat_history}
        Context:
        {context}
        Question: {input}
        Please provide an accurate and detailed response related to the cattle disease mentioned in the query.
        """)
    db = load_or_create_db()
    retriever = db.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return retrieval_chain, memory

retrieval_chain, memory = setup_chain()

@app.post("/chat")
async def chat(request: Dict[str, str]):
    user_input = request.get("message", "").strip().lower()
    if user_input in ["hi", "hello", "hey", "hola", "howdy", "greetings"]:
        return {"response": "Hello, how may I assist you today?"}
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    ai_response = "\n".join(response['answer'].splitlines())
    memory.save_context({"input": user_input}, {"answer": ai_response})
    return {"response": ai_response}

# ================== RUN SERVER ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)