# ingest.py
import os, glob, json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
OpenAIEmbeddings(openai_api_key=api_key)

DATA_DIR = "data"
INDEX_DIR = "storage/faiss_index"

def load_docs():
    docs = []
    # PDFs (slides)
    for pdf in glob.glob(os.path.join(DATA_DIR, "slides", "*.pdf")):
        loader = PyPDFLoader(pdf)
        for d in loader.load():
            # keep slide/page metadata
            d.metadata["source"] = os.path.basename(pdf)
            d.metadata["page"] = d.metadata.get("page", None)
            docs.append(d)

    # OCR’d text (for images or any extra notes)
    # for txt in glob.glob(os.path.join(DATA_DIR, "text", "*.text")):
    #     loader = TextLoader(txt, encoding="utf-8")
    #     for d in loader.load():
    #         d.metadata["source"] = os.path.basename(txt)
    #         d.metadata["page"] = None
    #         docs.append(d)
    # return docs

    # Word .text files
    for docx in glob.glob(os.path.join(DATA_DIR, "text", "*.docx")):
        try:
            loader = Docx2txtLoader(docx)
            doc_list = loader.load()
            if doc_list is None:
                print(f"[WARN] {docx} returned no docs.")
            else:
                for d in doc_list:
                    d.metadata["source"] = os.path.basename(docx)
                    d.metadata["page"] = None
                    docs.append(d)
        except Exception as e:
            print(f"[WARN] Failed to read {docx}: {e}")
    return docs


def main():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env/secrets
    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR}. Chunks: {len(chunks)}")

if __name__ == "__main__":
    main()
