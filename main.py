import os
import google.generativeai as genai
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
genai.configure(api_key=API_KEY)

app = FastAPI()

class MessagePart(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    parts: list[MessagePart]

class InputData(BaseModel):
    messages: list[Message]

@app.post("/healthAssistant")
async def healthAssistant(req: Request):
    body = await req.json()

    # auto detect similar to your TS
    input_data = body.get("data") or body.get("input") or body

    # parse to schema
    messages = InputData(**input_data).messages

    # build history string
    conversation = ""
    for m in messages:
        conversation += f"{m.role}: {' '.join([p.text for p in m.parts])}\n"

    prompt = f"""
You are a helpful AI health companion. Provide friendly, informative responses about general health and wellness.
Always remind users to consult healthcare professionals for medical advice.
Keep responses concise and supportive.

Conversation so far:
{conversation}
"""

    model = genai.GenerativeModel("gemini-2.0-pro-exp")
    result = model.generate_content(prompt)

    text = result.text if result else "Sorry, I could not generate response"

    return {"text": text}
