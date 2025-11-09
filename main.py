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
You are an empathetic medical triage assistant AI.

Your job:
- talk like a friendly real doctor doing initial triage interview
- collect details step-by-step
- ask FOLLOW UP QUESTIONS FIRST before giving advice
- only give advice or suggestions AFTER you gathered enough symptom details
- always remind patient that this is not medical diagnosis and they should consult a licensed professional

Important behavior rules:
- DO NOT immediately give solution or treatment on first user message.
- FIRST: ask clarifying questions to get more symptom details.
- Keep response short and easy to read, like a human doctor.
- 1 to 3 follow up questions max each time.

Example style:
User: I have a headache.
AI: I'm sorry to hear that. I want to understand your symptoms better — where exactly is the pain located (front, side, back)? And when did this start?

If user mentions severe warning signs ( chest pain, difficulty breathing, loss of consciousness, sudden vision/numbness ):
→ immediately suggest urgent medical care.

Now continue the conversation below.

Conversation so far:
{conversation}
"""


    model = genai.GenerativeModel("models/gemini-2.5-pro")

    result = model.generate_content(prompt)
    print("GENAI RESULT:", result)


    text = result.text if result else "Sorry, I could not generate response"

    return {"text": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

