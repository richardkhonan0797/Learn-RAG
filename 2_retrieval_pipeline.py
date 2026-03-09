from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=embedding_model,
  collection_metadata={"hnsw:space": "cosine"}
)

query = "Which island does SpaceX lease for its launches in the Pacific?"

retriever = db.as_retriever(search_kwargs={"k": 3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("--- Context ---")

if __name__ == "__main__":
    main()