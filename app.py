import chainlit as cl
from cli_app import get_answer

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "difficulty breathing",
    "severe bleeding", "unconscious", "stroke",
    "paralysis", "seizure", "fainting"
]

@cl.on_chat_start
async def start():
    await cl.Message(content="🩺 Welcome to MediBot! Ask a medical question.").send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content.lower()

    if any(word in query for word in EMERGENCY_KEYWORDS):
        await cl.Message(content="🚨 Medical Emergency Detected! Call emergency services.").send()
        return

    answer = get_answer(query)
    await cl.Message(content=answer).send()
