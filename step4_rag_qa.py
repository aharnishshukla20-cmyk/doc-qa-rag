import time
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini_client = genai.Client(http_options=types.HttpOptions(timeout=30000))

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        for attempt in range(3):
            try:
                response = gemini_client.models.embed_content(
                    model=self.model_name,
                    contents=list(input),
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                if ("429" in str(e) or "503" in str(e)) and attempt < 2:
                    print("  Embedding API busy, retrying in 5s...")
                    time.sleep(5)
                else:
                    raise

# 1. Connect to your existing persistent ChromaDB index
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="book_knowledge",
    embedding_function=GeminiEmbeddingFunction(model_name="gemini-embedding-001"),
)

# 2. Fetch ALL active text-generation models (including Pro & Gemma pools as backup)
active_models = []
for m in gemini_client.models.list():
    if "generateContent" in (m.supported_actions or []):
        name = m.name.replace("models/", "")
        if any(tag in name for tag in ["gemini", "gemma"]) and not any(
            skip in name for skip in ["image", "audio", "tts", "robotics", "computer"]
        ):
            active_models.append(name)

# Prioritize: flash-lite -> flash -> pro -> gemma
def model_priority(name: str) -> int:
    if "flash-lite" in name:
        return 0
    if "flash" in name:
        return 1
    if "pro" in name:
        return 2
    return 3

active_models.sort(key=model_priority)

print(f"Connected to knowledge base ({collection.count()} chunks ready).")
print(f"Fallback chain ({len(active_models)} models): {active_models[:6]}")
print("Ask any question about your PDF (or type 'exit' to quit).\n")

def ask_document(question: str) -> str:
    # Step A: RETRIEVE top 3 relevant passages
    print("  [1/2] Searching vector database...")
    results = collection.query(query_texts=[question], n_results=3)
    retrieved_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Step B: AUGMENT prompt with strict grounding rules
    prompt = f"""You are a helpful document Q&A assistant.
Answer the user's question using ONLY the provided context excerpts from the document.
If the answer is not present in the context, clearly say: "I couldn't find that information in the provided document."
Mention the page numbers if they appear in the context.

CONTEXT:
{context}

QUESTION:
{question}
"""

    # Step C: GENERATE grounded answer across diverse model pools with backoff
    last_error = None
    for round_num in range(2):
        for model_name in active_models[:8]:
            try:
                print(f"  [2/2] Generating answer with {model_name}...")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                print(f"  -> {model_name} busy/unavailable, switching...")
        if round_num == 0:
            print("  All primary models spiked—waiting 5s before retry round...")
            time.sleep(5)

    raise last_error

# 3. Interactive Q&A loop
while True:
    user_q = input("You: ").strip()
    if not user_q:
        continue
    if user_q.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        break

    print()
    try:
        answer = ask_document(user_q)
        print(f"\nAI:\n{answer}\n")
    except Exception as e:
        print(f"\nError: {e}\n")
    print("-" * 60 + "\n")