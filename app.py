import os
import re
import json
import base64
import pdfplumber
import warnings
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate
from flask_cors import CORS

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API")

# Initialize Flask app
app = Flask(__name__)
CORS(app) 
app.secret_key = "super_secret_key"

FAISS_DB_DIR = "faiss_dbs"
os.makedirs(FAISS_DB_DIR, exist_ok=True)

GREETING_RESPONSES = ["hi", "hello", "hey", "hola", "howdy", "greetings"]

# -------------------- PDF & JSON Handling --------------------

def extract_all_content_as_documents(pdf_path):
    """Extract all text from a PDF as a list of documents."""
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                documents.append(Document(
                    page_content=page_text,
                    metadata={"page": page_number}
                ))
    return documents

def extract_all_content_from_json(file_path):
    """Extract structured content from a JSON file and convert it into document format."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        data = json.load(file)

    title = data.get("title", "")
    content_list = data.get("content", [])
    tables = data.get("tables", [])

    chunks = []
    current_chunk = {"title": "", "content": []}

    for item in content_list:
        if not item.strip():
            continue

        if item.startswith(("What", "Diagnosing", "Preventing", "Summary")):
            if current_chunk["title"] and current_chunk["content"]:
                chunks.append(current_chunk.copy())
            current_chunk = {"title": item, "content": []}
        elif item.startswith(("Fig", "Table")):
            current_chunk["content"].append(item)
        else:
            if not current_chunk["title"] and item.startswith("Part"):
                current_chunk["title"] = item
            else:
                current_chunk["content"].append(item)

    if current_chunk["title"] and current_chunk["content"]:
        chunks.append(current_chunk)

    if tables:
        table_chunk = {"title": "Tables", "content": []}
        for table in tables:
            formatted_table = [" | ".join(cell for cell in row if cell) for row in table]
            table_chunk["content"].extend(formatted_table)
        chunks.append(table_chunk)

    return chunks

def total_json_data():
    """Extract and process all JSON data from the 'jsons/' directory."""
    MAIN_DIR = 'jsons/'
    files = os.listdir(MAIN_DIR)
    chunks = []
    for file in files:
        file_path = os.path.join(MAIN_DIR, file)
        chunk = extract_all_content_from_json(file_path)
        chunks.extend(chunk)
    return chunks

# -------------------- FAISS Database --------------------

def load_or_create_db():
    """Load or create a FAISS vector store from PDF and JSON documents."""
    db_path = os.path.join(FAISS_DB_DIR, "cattle_diseases_main_db")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Extract PDF content
    pdf_path = "pdfs/cattle-diseases-farmers-guide.pdf"
    pdf_documents = extract_all_content_as_documents(pdf_path)

    # Extract JSON content
    json_data = total_json_data()
    json_documents = [
        Document(page_content=" ".join(d["content"]) if isinstance(d["content"], list) else str(d["content"]),
                metadata={"title": d["title"]})
        for d in json_data
    ]

    all_documents = pdf_documents + json_documents

    # Create FAISS DB
    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(db_path)
    return db

# -------------------- AI Model & Retrieval Chain --------------------

def setup_chain():
    """Setup the AI retrieval chain."""
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.05
    )

    prompt = ChatPromptTemplate.from_template(
    """
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
    """
    )

    db = load_or_create_db()
    retriever = db.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    return retrieval_chain, memory

retrieval_chain, memory = setup_chain()

# -------------------- Flask Routes --------------------

@app.route('/')
def index():
    """Serve the HTML UI."""
    Greeting='Hello from chat app'
    return Greeting

@app.route('/chat', methods=['POST'])
def query():
    """Handle user queries and return AI responses."""
    user_input = request.json.get("message", "").strip().lower()

    if user_input in GREETING_RESPONSES:
        return jsonify({"response": "Hello, how may I assist you today?"})

    chat_history = session.get("chat_history", [])

    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    ai_response = "\n\n".join(response['answer'].splitlines())

    # Update session memory
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": ai_response})
    session["chat_history"] = chat_history

    # Save conversation to memory
    memory.save_context({"input": user_input}, {"answer": ai_response})

    return jsonify({"response": ai_response})

# -------------------- Run Flask App --------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
