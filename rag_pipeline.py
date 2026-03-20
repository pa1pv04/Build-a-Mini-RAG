import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ------------------ CONFIG ------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4o-mini"
TOP_K = 3
CHUNK_SIZE = 300
OVERLAP = 50

# ------------------ INIT ------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

llm_client = OpenAI(api_key=api_key)

# ------------------ DOCUMENT LOADING ------------------

def load_documents(folder_path: str) -> str:
    """Load all PDF documents and extract text"""
    text_data = ""

    if not os.path.exists(folder_path):
        print("Data folder not found")
        return text_data

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text_data += content + "\n"
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return text_data


# ------------------ CHUNKING ------------------

def chunk_text(text: str) -> list:
    """Split text into overlapping chunks"""
    if not text.strip():
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)

    return chunks


# ------------------ EMBEDDINGS ------------------

def embed_chunks(chunks: list) -> np.ndarray:
    """Convert text chunks to embeddings"""
    if not chunks:
        return np.array([])

    return embedding_model.encode(chunks)


# ------------------ INDEXING ------------------

def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index"""
    if embeddings.size == 0:
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


# ------------------ RETRIEVAL ------------------

def retrieve_chunks(query: str, index, chunks: list, top_k=TOP_K) -> list:
    """Retrieve top-k relevant chunks"""
    if index is None:
        return []

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results


# ------------------ GENERATION ------------------

def generate_response(query: str, retrieved_chunks: list) -> str:
    """Generate grounded answer using LLM"""

    if not retrieved_chunks:
        return "No relevant information found in the documents."

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI assistant.

Rules:
- Answer only using the context below.
- Do not use outside knowledge.
- If answer is not present, say "Not found".

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"


# ------------------ FULL PIPELINE ------------------

class RAGPipeline:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.chunks = []
        self.index = None

    def setup(self):
        """Prepare pipeline (load → chunk → embed → index)"""
        text = load_documents(self.data_path)
        self.chunks = chunk_text(text)
        embeddings = embed_chunks(self.chunks)
        self.index = build_faiss_index(embeddings)

    def query(self, user_query: str):
        """End-to-end query processing"""
        retrieved = retrieve_chunks(user_query, self.index, self.chunks)
        answer = generate_response(user_query, retrieved)

        return {
            "retrieved_chunks": retrieved,
            "answer": answer
        }
