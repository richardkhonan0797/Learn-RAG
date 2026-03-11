from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

persistent_directory = "db/chroma_db"
db = Chroma(
  persist_directory=persistent_directory,
  embedding_function=embeddings,
  collection_metadata={"hnsw:space": "cosine"}
)

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
  if chat_history:
    messages = [
      SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
    ] + chat_history + [
      HumanMessage(content=f"New question: {user_question}")
    ]

    result = model.invoke(messages)
    search_question = result.content.strip()
    print(f"Searching for: {search_question}")
  else:
    search_question = user_question

  retriever = db.as_retriever(search_kwargs={"k": 3})
  docs = retriever.invoke(search_question)

  # print(f"Found {len(docs)} relevant documents:")
  # for i, doc in enumerate(docs, 1):
  #   lines = doc.page_content.split("\n")[:2]
  #   preview = "\n".join(lines)
  #   print(f" Doc {i}: {preview}...")

  combined_input = f"""Based on the following documents, please answer this question: {user_question}

  Documents:
  {chr(10).join([f"- {doc.page_content}" for doc in docs])}

  Please provide a clear and concise answer to the question using the information from the documents. If you can't find the answer in the documents, please say "I can't find the answer based on the retrieved documents."
  """

  messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on retrieved documents."),
    HumanMessage(content=combined_input)
  ]

  result = model.invoke(messages)
  answer = result.content

  chat_history.append(HumanMessage(content=user_question))
  chat_history.append(SystemMessage(content=answer))

  print("\n--- Answer ---")
  print(answer)


def start_chat():
  print("Ask me questions! Type 'quit' to exit")

  while True:
    question = input('\nYour Question: ')

    if question.lower() == 'quit':
      print("Goodbye!")
      break

    ask_question(question)


if __name__ == "__main__":
    start_chat()
