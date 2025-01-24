import os  # Import the os module
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Initialize API keys
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Set Google API key as environment variable
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize ChatGroq with the provided Groq API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Define the chat prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant for Basrah Gas Company (BGC). Your task is to answer questions based on the provided context about BGC. Follow these rules strictly:

    1. **Language Handling:**
       - If the question is in English, answer in English.
       - If the question is in Arabic, answer in Arabic.
       - If the user explicitly asks for a response in a specific language, respond in that language.

    2. **Contextual Answers:**
       - Provide accurate and concise answers based on the context provided.
       - Do not explicitly mention the source of information unless asked.

    3. **Handling Unclear or Unanswerable Questions:**
       - If the question is unclear or lacks sufficient context, respond with:
         - In English: "I'm sorry, I couldn't understand your question. Could you please provide more details?"
         - In Arabic: "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك تقديم المزيد من التفاصيل؟"
       - If the question cannot be answered based on the provided context, respond with:
         - In English: "I'm sorry, I don't have enough information to answer that question."
         - In Arabic: "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."

    4. **User Interface Language:**
       - If the user has selected Arabic as the interface language, prioritize Arabic in your responses unless the question is explicitly in English.
       - If the user has selected English as the interface language, prioritize English in your responses unless the question is explicitly in Arabic.

    5. **Professional Tone:**
       - Maintain a professional and respectful tone in all responses.
       - Avoid making assumptions or providing speculative answers.
    """),
    MessagesPlaceholder(variable_name="history"),  # Add chat history to the prompt
    ("human", "{input}"),
    ("system", "Context: {context}"),
])

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load existing FAISS index with safe deserialization
embeddings_path = "embeddings"  # Path to your embeddings folder
try:
    vectors = FAISS.load_local(
        embeddings_path,
        embeddings,
        allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
    )
except Exception as e:
    print(f"Error loading embeddings: {str(e)}")
    vectors = None

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

# Create and configure the document chain and retriever
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Example usage of the retrieval chain
def get_response(query):
    """
    Get a response from the model based on the query.
    """
    response = retrieval_chain.invoke({
        "input": query,
        "context": retriever.get_relevant_documents(query),
        "history": memory.chat_memory.messages  # Include chat history
    })
    return response["answer"]

# Example query
query = "What is the main activity of Basrah Gas Company?"
response = get_response(query)
print("Response:", response)
