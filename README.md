# RAG Agent Chatbot

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot built with **Chainlit** and **LangChain**. It leverages a multi-agent orchestration pipeline to process user queries, extract key facts, reason about the information, and provide concise summaries based on the knowledge base.

## Features
- **Multi-Agent Orchestrator**: A sophisticated pipeline that moves through Extraction, Retrieval, Reasoning, and Summarization steps.
- **RAG Architecture**: Uses FAISS for vector storage and OpenAI embeddings to retrieve relevant context.
- **Persistent History**: Stores chat sessions and interactions in a local SQLite database (`chat_history.db`).
- **Interactive UI**: Clean and responsive chat interface provided by Chainlit.

## Prerequisites
- Python 3.9+
- OpenAI API Key

## Installation

1. **Clone the repository** (if applicable)

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your_api_key_here
   ```

## Usage

Start the application using the Chainlit CLI:

```bash
chainlit run main.py -w
```
The `-w` flag enables auto-reloading during development.

The application will launch in your default browser at `http://localhost:8000`.

## Project Structure

- **`main.py`**: The entry point for the Chainlit application. Handles the UI and passes messages to the orchestrator.
- **`app.py`**: Handles infrastructure setup (SQLite database, Vector Store initialization, RAG chain creation).
- **`agent/`**: Contains the logic for the different agents:
  - `orchestrator.py`: Manages the flow between agents.
  - `simple_agent.py`: Extracts key facts.
  - `reasoning_agent.py`: Analyzes the retrieved information.
  - `summary_agent.py`: Creates the final executive summary.
- **`jewaira.pdf`**: The source document for the knowledge base.

## Database
The application uses a local SQLite database `chat_history.db` to store:
- Session IDs
- Timestamps
- User queries
- Agent responses
