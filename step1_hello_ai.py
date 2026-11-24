from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Explain what an API is to a 10-year-old in 2 simple sentences."
)

print("\nAI Response:\n")
print(response.text)