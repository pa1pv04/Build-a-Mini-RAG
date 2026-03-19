from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ DOCUMENT PROCESSING ------------------
def load_all_pdfs(folder_path):
    text = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def create_embeddings(chunks):
    return model.encode(chunks)


# ------------------ VECTOR INDEX ------------------
def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def retrieve(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]


# ------------------ LLM GENERATION ------------------
def generate_answer(query, retrieved_chunks):

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an AI assistant.

STRICT RULES:
- Answer ONLY using the context below
- Do NOT use outside knowledge
- If answer is not in context, say "Not found"
- Do NOT hallucinate

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
