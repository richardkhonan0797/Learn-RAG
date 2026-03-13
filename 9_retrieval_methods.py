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

query = "How much did Microsoft pay to acquire GitHub?"
print(f"🔍 Query: {query}")

# -----------------------------------------------
# METHOD 1: Basic Similarity Search
# Returns the top k most similar documents.
# -----------------------------------------------

# print("=== METHOD 1: Basic Similarity Search (k=3) ===")
# retriever = db.as_retriever(search_kwargs={"k": 3})

# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents:\n")
# for i, doc in enumerate(docs,1):
#   print(f"Document {i}:")
#   print(f"Content: {doc.page_content[:200]}...")
#   print("-" * 40)

# -------------------------------------------------------------------
# METHOD 2: Similarity Search with Score Threshold
# Returns documents with similarity score above a certain threshold.
# -------------------------------------------------------------------

# print("\n=== METHOD 2: Similarity Search with Score Threshold (threshold=0.3) ===")
# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 3})

# docs= retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents:\n")
# for i, doc in enumerate(docs,1):
#   print(f"Document {i}:")
#   print(f"Content: {doc.page_content[:200]}...")
#   print(f"Similarity Score: {doc.metadata['similarity_score']:.4f}")
#   print("-" * 40)


# -------------------------------------------------------------------
# METHOD 3: Maximal Marginal Relevance (MMR) Search
# Balances relevance and diversity - avoids redundant results.
# -------------------------------------------------------------------

print("\n=== METHOD 3: Maximal Marginal Relevance (MMR) Search (k=3, lambda=0.5) ===")
retriever = db.as_retriever(
  search_type="mmr",
  search_kwargs={
    "k": 3, # Final number of documents to return
    "fetch_k": 10, # Number of top candidates to consider for MMR
    "lambda_mult": 0.5 # 0 = max diversity, 1 = max relevance
  }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:\n")
for i, doc in enumerate(docs,1):
  print(f"Document {i}:")
  print(f"Content: {doc.page_content[:200]}...")
  print(f"Similarity Score: {doc.metadata['similarity_score']:.4f}")
  print("-" * 40)