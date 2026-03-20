from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os
from openai import OpenAI

# Load API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ LOAD PDFs ------------------
def load_all_pdfs(folder_path):
    text = ""

    if not os.path.exists(folder_path):
        return ""

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    return text


# ------------------ CHUNKING ------------------
def chunk_text(text, chunk_size=300, overlap=50):
    if not text:
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ------------------ EMBEDDINGS ------------------
def create_embeddings(chunks):
    if len(chunks) == 0:
        return np.array([])

    return model.encode(chunks)


# ------------------ INDEX ------------------
def build_index(embeddings):
    if len(embeddings) == 0:
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


# ------------------ RETRIEVE ------------------
def retrieve(query, index, chunks, k=3):
    if index is None:
        return []

    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)

    return [chunks[i] for i in I[0]]


# ------------------ LLM ------------------
def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information found in documents."

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an AI assistant.

STRICT RULES:
- Answer ONLY using the context below
- Do NOT use outside knowledge
- If answer is not in context, say "Not found"

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
