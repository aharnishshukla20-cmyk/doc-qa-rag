from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """Opens a PDF and combines text from all pages."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    return full_text

def chunk_text(text, chunk_size=500, overlap=100):
    """Slices giant text into overlapping index cards (chunks)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so cards slightly overlap
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    # 1. Read the PDF
    raw_text = extract_text_from_pdf("sample.pdf")
    print(f"Total characters extracted: {len(raw_text)}")

    # 2. Chop into overlapping chunks
    cards = chunk_text(raw_text, chunk_size=500, overlap=100)
    print(f"Total index cards (chunks) created: {len(cards)}\n")

    # 3. Preview the first 2 cards to see the overlap in action!
    print("=== CARD 1 ===")
    print(cards[0])
    print("\n=== CARD 2 ===")
    print(cards[1])