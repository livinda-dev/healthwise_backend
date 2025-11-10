import os
import json
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# ----- Load ENV -----
load_dotenv()

API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_GENAI_API_KEY")

genai.configure(api_key=API_KEY)

# ----- Firestore (via service account JSON in env) -----
from google.oauth2 import service_account
from google.cloud import firestore

FIREBASE_SERVICE_KEY = os.getenv("FIREBASE_SERVICE_KEY")
if not FIREBASE_SERVICE_KEY:
    raise RuntimeError("Missing FIREBASE_SERVICE_KEY (paste full service account JSON)")

creds_dict = json.loads(FIREBASE_SERVICE_KEY)
creds = service_account.Credentials.from_service_account_info(creds_dict)
db = firestore.Client(credentials=creds, project=creds.project_id)

COLLECTION = "patient_states"  # per your choice


# ----- FastAPI App -----
app = FastAPI()


# ----- Schemas from your client -----
class MessagePart(BaseModel):
    text: str

class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    parts: List[MessagePart]

class InputData(BaseModel):
    userEmail: str = Field(..., description="Firebase Auth user.email")
    messages: List[Message]


# ----- Helpers -----
BAD_TERMS = [
    "fuck", "shit", "bitch", "suck my", "dick", "asshole",
    "motherfucker", "f*ck", "f**k"
]

EMERGENCY_TERMS = [
    "chest pain", "pressure on chest", "shortness of breath",
    "trouble breathing", "difficulty breathing",
    "fainting", "passed out", "loss of consciousness",
    "sudden weakness", "sudden numbness", "one-sided weakness",
    "slurred speech", "sudden vision loss", "seizure", "seizures",
    "severe allergic reaction", "anaphylaxis",
    "suicidal", "suicide", "kill myself", "kill himself", "kill herself",
    "overdose", "bleeding that won't stop", "uncontrolled bleeding"
]

def get_last_user_text(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return " ".join([p.text for p in m.parts]).strip()
    return ""

def is_direct_abuse(text: str) -> bool:
    lower = text.lower()
    short = len(lower.split()) <= 6  # short + profanity → likely directed abuse
    contains_bad = any(term in lower for term in BAD_TERMS)
    return contains_bad and short

def has_emergency_signal(text: str) -> Optional[str]:
    lower = text.lower()
    for key in EMERGENCY_TERMS:
        if key in lower:
            return key
    return None

def normalize_severity(value: Any) -> Optional[int]:
    """
    Try to coerce severity to an int 1-10 if possible.
    Accept forms like "7", "7/10", "eight", etc. (simple heuristics).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = int(round(float(value)))
        return min(10, max(1, val))
    s = str(value).strip().lower()
    # e.g., "7/10"
    m = re.search(r"(\d{1,2})\s*/\s*10", s)
    if m:
        try:
            val = int(m.group(1))
            return min(10, max(1, val))
        except:
            pass
    # bare number in string
    m2 = re.search(r"\b(\d{1,2})\b", s)
    if m2:
        try:
            val = int(m2.group(1))
            return min(10, max(1, val))
        except:
            pass
    return None

def _safe_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


# ----- Core Endpoint -----
@app.post("/healthAssistant")
async def healthAssistant(req: Request):
    body = await req.json()

    # Support your previous shape {data|input|raw}
    raw_input = body.get("data") or body.get("input") or body

    # Validate payload
    try:
        payload = InputData(**raw_input)
    except ValidationError as ve:
        return {"text": f"Invalid input: {ve.errors()}"}

    userEmail = payload.userEmail.strip() if payload.userEmail else ""
    if not userEmail:
        return {"text": "Missing user email"}

    messages = payload.messages
    last_user = get_last_user_text(messages)

    # ---------- Direct abuse filter ----------
    if last_user and is_direct_abuse(last_user):
        return {"text": "Please speak respectfully. I'm here to help you with your health."}

    # ---------- Load or init state ----------
    doc_ref = db.collection(COLLECTION).document(userEmail)
    snap = doc_ref.get()
    if snap.exists:
        state = snap.to_dict()
    else:
        state = {
            "stage": "intake",
            "symptoms": {
                "location": None,
                "duration": None,
                "severity": None,
                "otherSymptoms": []
            }
        }

    # ---------- Emergency keyword check (before main model) ----------
    danger_key = has_emergency_signal(last_user)
    if danger_key:
        state["stage"] = "emergency"
        # Save state immediately (we detected risk)
        doc_ref.set(state)
        urgent_text = (
            "⚠️ Your message suggests a potentially serious symptom "
            f'("{danger_key}"). Please seek urgent medical care now (local emergency number or nearest ER). '
            "If you’re alone, consider contacting someone who can help you get medical attention."
        )
        return {"text": urgent_text}

    # ---------- Compose conversation string ----------
    conversation = ""
    for m in messages:
        conversation += f"{m.role}: {' '.join([p.text for p in m.parts])}\n"

    # ---------- Build dynamic triage guidance from missing fields ----------
    symp = state.get("symptoms", {})
    missing = []
    if not symp.get("location"):
        missing.append("location of the problem (e.g., left temple, lower right abdomen)")
    if not symp.get("duration"):
        missing.append("duration (how long this has been going on)")
    if not symp.get("severity"):
        missing.append("severity on a 1–10 scale")
    # otherSymptoms is a list; we treat it as optional follow-up bucket

    missing_hint = ""
    if missing:
        missing_hint = "- Ask about: " + "; ".join(missing) + ".\n"

    # ---------- Main triage assistant prompt ----------
    triage_prompt = f"""
You are an empathetic medical triage assistant AI.

Your job:
- talk like a friendly real doctor during initial triage.
- collect details step-by-step.
- ask FOLLOW-UP QUESTIONS FIRST before giving advice.
- only give suggestions AFTER gathering enough info.
- always remind the patient this is not a diagnosis and they should consult a licensed professional.

Behavior:
- Do NOT immediately give treatment on the first user message.
- Keep responses short and easy to read.
- Ask 1–3 clarifying questions max per turn.
- If the patient uses profanity inside a symptom description, ignore it and continue professionally.
- Only if the user directly insults you (e.g., "fuck you"), ask them to be respectful.

Current triage stage: {state.get("stage", "intake")}
Known symptom details so far: {json.dumps(symp, ensure_ascii=False)}
{missing_hint}
If warning signs like chest pain, trouble breathing, sudden weakness/numbness, vision loss, fainting, severe allergic reaction arise, advise urgent medical care.

Now continue the conversation.

Conversation so far:
{conversation}
"""

    # ---------- Generate the patient-facing reply ----------
    try:
        main_model = genai.GenerativeModel("models/gemini-2.5-pro")
        main_result = main_model.generate_content(triage_prompt)
        reply_text = main_result.text or "I'm here to help. Could you share a bit more detail?"
    except Exception as e:
        reply_text = f"Sorry, I had trouble generating a response. ({e})"

    # ---------- Extraction pass: pull structured fields from last user message ----------
    # We only try to extract from the *latest* user input to update state incrementally.
    if last_user:
        extract_prompt = f"""
Extract the following structured health fields from the user's message.

Fields to extract (JSON keys):
- location: string or null
- duration: string or null
- severity: number 1-10 or null
- otherSymptoms: array of short strings (symptom keywords). If none, return [].

Rules:
- If a field is not mentioned, return null (or [] for otherSymptoms).
- Return STRICT JSON ONLY. No markdown. No additional text.

User message:
\"\"\"{last_user}\"\"\"
"""
        try:
            extract_model = genai.GenerativeModel("models/gemini-2.0-flash")
            extract_result = extract_model.generate_content(extract_prompt)
            raw_json = extract_result.text or "{}"

            # Some models may wrap JSON in code fences or add stray text; try to sanitize.
            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
                cleaned = re.sub(r"```$", "", cleaned).strip()

            extracted = json.loads(cleaned)  # may raise
            # Merge into state carefully
            loc = extracted.get("location")
            dur = extracted.get("duration")
            sev = normalize_severity(extracted.get("severity"))
            oth = _safe_list(extracted.get("otherSymptoms"))

            if loc:
                state["symptoms"]["location"] = loc
            if dur:
                state["symptoms"]["duration"] = dur
            if sev:
                state["symptoms"]["severity"] = sev
            if oth:
                # merge unique
                current = set(state["symptoms"].get("otherSymptoms", []))
                for item in oth:
                    if item not in current:
                        current.add(item)
                state["symptoms"]["otherSymptoms"] = list(current)

        except Exception as e:
            # Extraction failure should not break the chat
            pass

    # ---------- Advance stage if enough info ----------
    symp = state["symptoms"]
    if symp.get("location") and symp.get("duration") and symp.get("severity"):
        # We have core triage details; move to analysis mode
        state["stage"] = "analysis"

    # ---------- Save state ----------
    doc_ref.set(state)

    # ---------- Return to client ----------
    return {"text": reply_text}


# ----- Local dev entrypoint -----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
