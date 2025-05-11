import os
import argparse
from dotenv import load_dotenv
import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def load_pdf_elements(pdf_path: str):
    """
    Load PDF as individual text blocks using PyMuPDF get_text("blocks").
    Returns a list of langchain.schema.Document with primitive metadata.
    """
    docs = []
    pdf = fitz.open(pdf_path)
    for page in pdf:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            bbox_str = ",".join(f"{coord:.2f}" for coord in b[:4])
            meta = {
                "page_number": page.number + 1,
                "bbox": bbox_str,
                "block_no": b[5] if len(b) > 5 else None
            }
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def ingest(pdf_path: str, persist_directory: str = "db"):
    docs = load_pdf_elements(pdf_path)

    print("📄 Preview extracted elements:")
    for i, doc in enumerate(docs[:5]):
        m = doc.metadata
        preview = doc.page_content.replace("\n", " ")[:100]
        print(f"[{i}] Page={m['page_number']} BBox={m['bbox']} BlockNo={m['block_no']} '{preview}...' ")
    print(f"... Total {len(docs)} elements extracted.\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    docs_split = splitter.split_documents(docs)

    print("✂️ Preview chunks:")
    for i, chunk in enumerate(docs_split[:3]):
        m = chunk.metadata
        snippet = chunk.page_content.replace("\n", " ")[:150]
        print(f"[{i}] Page={m['page_number']} Length={len(chunk.page_content)} -> '{snippet}...'\n")
    print(f"... Total {len(docs_split)} chunks created.\n")

    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Ingested {len(docs_split)} document chunks into '{persist_directory}' using OpenAI Embeddings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into a Chroma vector database.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    parser.add_argument("--persist_dir", default="db", help="Directory to persist the vector store.")
    args = parser.parse_args()
    ingest(args.pdf_path, args.persist_dir)
