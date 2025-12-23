# cli_app.py
import os
import fitz
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PDF → Text
pdf_path = "data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf"
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text()

# Chunking
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(full_text)

# Embeddings + FAISS
vectors = []
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
    vectors.extend([e.embedding for e in resp.data])

dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# 🔹 Ye function UI ke liye
def get_answer(query):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([query_embedding]), k=5)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
You are a medical assistant.
Use ONLY the information below.
Explain in simple words.
If not found, say: "I don't know based on the given information."

Information:
{context}

Question:
{query}
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content
