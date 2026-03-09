import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()



def load_documents(docs_path="docs"):
  print(f"Loading documents from {docs_path}...")

  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"Directory '{docs_path}' does not exist. Please create it and add some text files to ingest.")

  loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)

  documents = loader.load()

  if len(documents) == 0:
    raise FileNotFoundError(f"No .txt files found in '{docs_path}'. Please add some .txt files to ingest.")

  for i, doc in enumerate(documents[:2]):
    print(f"\nDocument {i+1}:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content length: {len(doc.page_content)} characters")
    print(f"  Content preview: {doc.page_content[:100]}...")
    print(f"  Metadata: {doc.metadata}")

  return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
  print("Splitting documents into smaller chunks...")

  text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

  chunks = text_splitter.split_documents(documents)

  if chunks:
    for i, chunk in enumerate(chunks[:5]):
      print(f"\nChunk {i+1}:")
      print(f"  Source: {chunk.metadata['source']}")
      print(f"  Content length: {len(chunk.page_content)} characters")
      print(f"  Content preview: {chunk.page_content[:100]}...")
      print(f"  Metadata: {chunk.metadata}")
      print("-" * 40)

    if len(chunks) > 5:
      print(f"... and {len(chunks) - 5} more chunks.")
  
  return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
  print("Creating vector store and embedding chunks...")

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  print("--- Creating vector store ---")
  vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
  )
  print("--- Finished creating vector store ---")

  print(f"Vector store created and saved to '{persist_directory}'.")
  return vectorstore

  


def main():
  # 1. Load documents from the "docs" directory
  documents = load_documents(docs_path="docs")

  # 2. Split documents into smaller chunks
  chuncks = split_documents(documents)

  # 3. Embedding and storing the chunks in ChromaDB
  vector_store = create_vector_store(chuncks)


  


if __name__ == "__main__":
    main()
