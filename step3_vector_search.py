import time
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from step2_read_pdf import extract_text_from_pdf, chunk_text

load_dotenv()
gemini_client = genai.Client()

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        while True:
            try:
                response = gemini_client.models.embed_content(
                    model=self.model_name,
                    contents=list(input),
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                if "429" in str(e):
                    print("  Rate limit hit—pausing 62s and retrying automatically...")
                    time.sleep(62)
                else:
                    raise

# 1. Auto-detect the active embedding model on your API key
embed_model = "gemini-embedding-001"
for m in gemini_client.models.list():
    if "embedContent" in (m.supported_actions or []):
        embed_model = m.name.replace("models/", "")
        break
print(f"Using embedding model: {embed_model}")

# 2. Initialize persistent ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="book_knowledge",
    embedding_function=GeminiEmbeddingFunction(model_name=embed_model)
)

# 3. Build the index ONLY if empty (~365 chunks, 25 per batch = ~18k tokens/min)
if collection.count() == 0:
    print("Reading PDF and creating chunks...")
    raw_text = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(raw_text, chunk_size=2500, overlap=250)

    total = len(chunks)
    print(f"Created {total} chunks. Indexing in safe batches of 25 (~18k tokens/min)...")

    batch_size = 25
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        collection.add(documents=batch, ids=ids)

        done = min(i + batch_size, total)
        print(f"  Indexed {done}/{total} chunks...")

        if done < total:
            time.sleep(62)

    print("Database built and saved permanently to ./chroma_db!\n")
else:
    print(f"Loaded existing database with {collection.count()} chunks!\n")

# 4. Test semantic search!
question = "What is the main topic of Chapter 7 Jnana?"
results = collection.query(
    query_texts=[question],
    n_results=3
)

print(f"Question: {question}\n")
for idx, doc in enumerate(results["documents"][0]):
    print(f"--- Match #{idx + 1} ---")
    print(doc[:400] + "...\n")