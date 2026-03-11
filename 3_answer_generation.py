from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=embedding_model,
  collection_metadata={"hnsw:space": "cosine"}
)

query = "In what year did Tesla begin production of the Roadster?"

retriever = db.as_retriever(
  search_type="similarity_score_threshold",
  search_kwargs={"k": 5, "score_threshold": 0.3}
)

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("-" * 40)
