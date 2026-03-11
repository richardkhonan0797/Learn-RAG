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


combined_input = f"""Based on the following retrieved documents, answer the question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a concise and accurate answer to the question using the information from the documents.
If you can't find the answer in the documents, please say "I can't find the answer based on the retrieved documents."
"""

model = ChatOpenAI(model="gpt-4o")

messages = [
  SystemMessage(content="You are a helpful assistant that answers questions based on retrieved documents."),
  HumanMessage(content=combined_input)
]

result = model.invoke(messages)


print("\n--- Generated Answer ---")
print("Content only:")
print(result.content)



# Synthetic Questions:

# 1. "What was NVIDIA's first product and when was it released?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product and when was it released?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
