from src.document_loader import load_document
from src.text_processor import split_pages
from src.rag_pipeline import summarize_document


pages = load_document("data/documents/odos_report.pdf")

chunks = split_pages(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

print("\nGenerating document summary...\n")

summary = summarize_document(chunks, batch_size=5)

print("--- DOCUMENT SUMMARY ---")
print(summary)