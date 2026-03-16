from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

persistent_directory = "db/chroma_db"
llm = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=llm,
  collection_metadata={"hnsw:space": "cosine"}
)

class QueryVariations(BaseModel):
  queries: List[str]

original_query = "How does Tesla make money?"
print(f"🔍 Original Query: {original_query}")

# -----------------------------------------------
# STEP 1: Generate Multiple Query Variations
# -----------------------------------------------

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents.

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("\nGenerated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("-" * 50)

# -----------------------------------------------------------
# STEP 2: Search with each query variation and store results
# -----------------------------------------------------------

retriever = db.as_retriever(search_kwargs={"k": 5}) # Get more docs for better RRF
all_retrieval_results = [] # Store all results for RRF

for i, query in enumerate(query_variations, 1):
  print(f"\n🔍 Searching with Query Variation {i}: {query}")

  docs = retriever.invoke(query)
  all_retrieval_results.append(docs)

  print(f"Retrieved {len(docs)} documents for Query Variation {i}:\n")

  for j, doc in enumerate(docs, 1):
    print(f"Document {j}:")
    print(f"Content: {doc.page_content[:200]}...")
      
      
  print("-" * 40)

print("\n" + "=" * 50)
print("Multi-Query retrieval complete!")
