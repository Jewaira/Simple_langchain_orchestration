import chainlit as cl
from app1 import setup_sqlite_db, store_message
from agents.orchestrator import orchestrator


setup_sqlite_db()

@cl.on_chat_start
async def on_chat_start():
    pass

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.context.session.id
  
    try:
        answer = orchestrator(input_text=message.content)
        if not isinstance(answer, str):
            answer = str(answer)
            
    except Exception as e:
        answer = f"I encountered an error processing your request: {str(e)}"
    
    # Send response only ONCE
    await cl.Message(content=answer).send()
    
    # Store in DB
    store_message(session_id, message.content, answer)
