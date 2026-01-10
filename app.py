import os
import sqlite3
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- Database Setup ---
DB_NAME = "chat_history.db"

def setup_sqlite_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor() 
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT,
                agent_response TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print('Successfully initialized database')
    except Exception as e:
        print(f"Error initializing database: {e}")

def store_message(session_id, user_query, agent_response):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO chat_history (session_id, timestamp, user_query, agent_response) 
            VALUES (?, ?, ?, ?)
        ''', (session_id, timestamp, user_query, agent_response))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error storing message: {e}")

# --- Initialize Resources ---
file_path = "jewaira.pdf" 
documents = []
try:
    if os.path.exists(file_path):
        loader = PyPDFLoader(file_path) if file_path.endswith('.pdf') else TextLoader(file_path)
        documents = loader.load()
    else:
        print(f"Warning: {file_path} not found. Skipping document loading.")
except Exception as e:
    print(f"Error loading documents: {e}")

if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
else:
    retriever = None

model = ChatOpenAI(model="gpt-4o-mini")

# --- Initialize Chain ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the context provided to answer."),
    ("human", "Context: {context}\n\nQuestion: {input}")
])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def _get_query_input(x):
    if isinstance(x, dict):
        # Handle various input formats (dict vs string)
        return x.get("input", x.get("analysis", x.get("input", "")))
    return x

if retriever:
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(_get_query_input(x)))
        )
        | prompt
        | model
        | StrOutputParser()
    )
else:
    chain = (
        RunnablePassthrough.assign(context=lambda x: "No context available.")
        | prompt
        | model
        | StrOutputParser()
    )