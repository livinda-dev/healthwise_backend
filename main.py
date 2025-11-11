import os, json, re, hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# ---------- ENV ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_GENAI_API_KEY")
genai.configure(api_key=API_KEY)

FIREBASE_SERVICE_KEY = os.getenv("FIREBASE_SERVICE_KEY")
if not FIREBASE_SERVICE_KEY:
    raise RuntimeError("Missing FIREBASE_SERVICE_KEY")

# ---------- Firestore ----------
from google.oauth2 import service_account
from google.cloud import firestore

creds_dict = json.loads(FIREBASE_SERVICE_KEY)
creds = service_account.Credentials.from_service_account_info(creds_dict)
db = firestore.Client(credentials=creds, project=creds.project_id)

# ---------- Firebase Admin (FCM) ----------
import firebase_admin
from firebase_admin import credentials as fa_credentials, messaging

if not firebase_admin._apps:
    firebase_admin.initialize_app(fa_credentials.Certificate(creds_dict))

# ---------- App ----------
app = FastAPI()

COLL = "patient_states"
TOKENS = "fcm_tokens"
ANON_ARCHIVE = "anonymous_archive"

# ---------- Schemas ----------
class MessagePart(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    parts: List[MessagePart]

class InputData(BaseModel):
    userEmail: str
    messages: List[Message]

class RegisterTokenBody(BaseModel):
    userEmail: str
    fcmToken: str

class ResolveBody(BaseModel):
    userEmail: str
    note: Optional[str] = None

class DeleteHistoryBody(BaseModel):
    userEmail: str
    historyIndex: int

# ---------- Helpers ----------
BAD_TERMS = [
    "fuck","shit","bitch","suck my","dick","asshole","motherfucker","f*ck","f**k"
]
EMERGENCY_TERMS = [
    "chest pain","pressure on chest","shortness of breath",
    "trouble breathing","difficulty breathing","fainting","passed out",
    "loss of consciousness","sudden weakness","sudden numbness","one-sided weakness",
    "slurred speech","sudden vision loss","seizure","seizures",
    "severe allergic reaction","anaphylaxis","suicidal","suicide",
    "kill myself","overdose","bleeding that won't stop","uncontrolled bleeding"
]
UTC = timezone.utc

def now_utc() -> datetime:
    return datetime.now(tz=UTC)

def last_user_text(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return " ".join(p.text for p in m.parts).strip()
    return ""

def is_direct_abuse(text: str) -> bool:
    lower = text.lower()
    short = len(lower.split()) <= 6
    return any(t in lower for t in BAD_TERMS) and short

def has_emergency(text: str) -> Optional[str]:
    lower = text.lower()
    for k in EMERGENCY_TERMS:
        if k in lower:
            return k
    return None

def normalize_severity(value: Any) -> Optional[int]:
    if value is None: return None
    if isinstance(value, (int, float)):
        v = int(round(float(value))); return max(1, min(10, v))
    s = str(value).lower()
    m = re.search(r"(\d{1,2})\s*/\s*10", s) or re.search(r"\b(\d{1,2})\b", s)
    if m:
        try:
            v = int(m.group(1)); return max(1, min(10, v))
        except: pass
    return None

def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []

def greeting(text: str) -> bool:
    return bool(re.search(r"\b(hi|hello|hey|good\s+morning|good\s+evening)\b", text.lower()))

# ---------- Reminder plan rules (simple; expand later) ----------
def build_reminder_plan(condition: str) -> List[Dict[str, Any]]:
    t = now_utc()
    # Example plan for headache; defaults otherwise
    if "headache" in condition.lower():
        return [
            {"type": "drink_water", "title":"💧 Hydration break", "body":"Drink a glass of water.", "next_at": (t + timedelta(hours=2)).isoformat(), "every_hours": 2},
            {"type": "eye_rest", "title":"👀 Eye rest", "body":"Rest your eyes for 3–5 minutes.", "next_at": (t + timedelta(hours=3)).isoformat(), "every_hours": 3},
        ]
    # default gentle plan
    return [
        {"type": "check_in", "title":"🩺 Quick check-in", "body":"How are you feeling now?", "next_at": (t + timedelta(hours=4)).isoformat(), "every_hours": 4}
    ]

def set_active_condition(state: Dict[str, Any], condition: str, extracted: Dict[str, Any]):
    state["active_condition"] = {
        "condition": condition,
        "since": now_utc().isoformat(),
        "tips": ["Hydration", "Rest", "Avoid triggers"],
        "extracted": extracted
    }
    state["reminders"] = build_reminder_plan(condition)
    state["stage"] = "reminder"  # after analysis, enter reminder loop

# ---------- Core Chat Endpoint ----------
@app.post("/healthAssistant")
async def healthAssistant(req: Request):
    body = await req.json()
    raw = body.get("data") or body.get("input") or body
    try:
        payload = InputData(**raw)
    except ValidationError as ve:
        return {"text": f"Invalid input: {ve.errors()}"}

    userEmail = payload.userEmail.strip()
    if not userEmail:
        return {"text": "Missing user email"}

    messages = payload.messages
    last_user = last_user_text(messages)

    if last_user and is_direct_abuse(last_user):
        return {"text":"Please speak respectfully. I'm here to help you with your health."}

    doc_ref = db.collection(COLL).document(userEmail)
    snap = doc_ref.get()
    if snap.exists:
        state = snap.to_dict()
    else:
        state = {
            "stage":"intake",
            "symptoms":{
                "location":None,"duration":None,"severity":None,"otherSymptoms":[]
            },
            "history":[],
            "recycle_bin":[]
        }

    # Auto-reset if user greets and we previously finished
    if state.get("stage") == "analysis" and greeting(last_user):
        state["stage"]="intake"
        state["symptoms"]={"location":None,"duration":None,"severity":None,"otherSymptoms":[]}
        state.pop("active_condition", None)
        state.pop("reminders", None)
        doc_ref.set(state)

    # Emergency shortcut
    danger = has_emergency(last_user)
    if danger:
        state["stage"]="emergency"
        doc_ref.set(state)
        return {"text": f"⚠️ Your message suggests a serious symptom (\"{danger}\"). Please seek urgent medical care now or call your local emergency number."}

    # Build conversation text
    convo = ""
    for m in messages:
        convo += f"{m.role}: {' '.join(p.text for p in m.parts)}\n"

    # Missing triage fields hint
    symp = state.get("symptoms", {})
    missing = []
    if not symp.get("location"): missing.append("location (e.g., left temple)")
    if not symp.get("duration"): missing.append("duration")
    if not symp.get("severity"): missing.append("severity (1–10)")
    missing_hint = "- Ask about: "+"; ".join(missing)+"." if missing else ""

    # Main reply
    triage_prompt = f"""
You are an empathetic medical triage assistant AI.

Your job:
- act like a doctor doing initial triage.
- ask 1–3 follow-up questions FIRST; only give suggestions after enough info.
- keep messages short and clear.
- ignore profanity inside symptom description; only ask for respectful tone if directly insulted.
- always remind this isn't a medical diagnosis.

Triage stage: {state.get("stage","intake")}
Known symptoms: {json.dumps(symp, ensure_ascii=False)}
{missing_hint}

If red flags (chest pain, trouble breathing, sudden weakness/numbness, vision loss, fainting, severe allergy) appear, advise urgent care.

Now continue the conversation.

Conversation so far:
{convo}
"""
    try:
        main_model = genai.GenerativeModel("models/gemini-2.5-pro")
        main_result = main_model.generate_content(triage_prompt)
        reply_text = main_result.text or "I'm here to help. Could you share a bit more detail?"
    except Exception as e:
        reply_text = f"Sorry, I had trouble generating a response. ({e})"

    # Extraction pass (adds: condition too)
    condition_label = None
    if last_user:
        extract_prompt = f"""
Extract structured fields from the user's message.
Return STRICT JSON with keys:
- location: string or null
- duration: string or null
- severity: number 1-10 or null
- otherSymptoms: array of short strings (may be empty)
- condition: a SHORT condition label if obvious (e.g., "headache", "cough", "fever") else null

User message:
\"\"\"{last_user}\"\"\"
"""
        try:
            fast_model = genai.GenerativeModel("models/gemini-2.0-flash")
            er = fast_model.generate_content(extract_prompt)
            cleaned = (er.text or "{}").strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
                cleaned = re.sub(r"```$", "", cleaned).strip()
            extracted = json.loads(cleaned)

            loc = extracted.get("location")
            dur = extracted.get("duration")
            sev = normalize_severity(extracted.get("severity"))
            oth = safe_list(extracted.get("otherSymptoms"))
            condition_label = (extracted.get("condition") or "")[:64]

            if loc: state["symptoms"]["location"]=loc
            if dur: state["symptoms"]["duration"]=dur
            if sev: state["symptoms"]["severity"]=sev
            if oth:
                cur = set(state["symptoms"].get("otherSymptoms",[]))
                for x in oth:
                    if x not in cur: cur.add(x)
                state["symptoms"]["otherSymptoms"]=list(cur)
        except Exception:
            pass

    # Move to analysis when core fields are known
    symp = state["symptoms"]
    if symp.get("location") and symp.get("duration") and symp.get("severity"):
        state["stage"]="analysis"

    # If we have a condition label and just reached analysis, start reminders
    if state["stage"]=="analysis" and condition_label and not state.get("active_condition"):
        set_active_condition(state, condition_label, state["symptoms"])

    # Save state
    doc_ref.set(state)
    return {"text": reply_text}

# ---------- Register FCM token ----------
@app.post("/registerToken")
async def register_token(body: RegisterTokenBody):
    userId = body.userEmail.strip()
    if not userId: return {"ok": False, "error":"Missing email"}
    db.collection(TOKENS).document(userId).set({"token": body.fcmToken, "updated_at": now_utc().isoformat()})
    return {"ok": True}

# ---------- Resolve current condition (user says I'm better now) ----------
@app.post("/conditions/resolve")
async def resolve_condition(body: ResolveBody):
    user = body.userEmail.strip()
    ref = db.collection(COLL).document(user)
    snap = ref.get()
    if not snap.exists: return {"ok": True}

    state = snap.to_dict()
    ac = state.get("active_condition")
    if ac:
        item = {
            "condition": ac.get("condition"),
            "resolved": now_utc().isoformat(),
            "duration": state["symptoms"].get("duration"),
            "severity": state["symptoms"].get("severity"),
            "note": body.note or "",
        }
        hist = state.get("history", [])
        hist.append(item)
        state["history"]=hist
        # clear active + reset stage/symptoms
        state.pop("active_condition", None)
        state.pop("reminders", None)
        state["stage"]="intake"
        state["symptoms"]={"location":None,"duration":None,"severity":None,"otherSymptoms":[]}
        ref.set(state)
    return {"ok": True}

# ---------- Delete a history item -> recycle_bin ----------
@app.post("/conditions/delete")
async def delete_history_item(body: DeleteHistoryBody):
    user = body.userEmail.strip()
    ref = db.collection(COLL).document(user)
    snap = ref.get()
    if not snap.exists: return {"ok": False, "error":"no user state"}

    state = snap.to_dict()
    hist = state.get("history", [])
    if body.historyIndex < 0 or body.historyIndex >= len(hist):
        return {"ok": False, "error":"index out of range"}

    item = hist.pop(body.historyIndex)
    rb = state.get("recycle_bin", [])
    item["deleted_at"] = now_utc().isoformat()
    rb.append(item)
    state["history"]=hist
    state["recycle_bin"]=rb
    ref.set(state)
    return {"ok": True}

# ---------- Reset all (dev helper) ----------
@app.post("/conditions/reset")
async def reset_all(body: ResolveBody):
    user = body.userEmail.strip()
    ref = db.collection(COLL).document(user)
    ref.set({
        "stage":"intake",
        "symptoms":{"location":None,"duration":None,"severity":None,"otherSymptoms":[]},
        "history":[],
        "recycle_bin":[]
    })
    return {"ok": True}

# ---------- DEBUG: get state ----------
@app.get("/state")
async def get_state(userEmail: str = Query(...)):
    snap = db.collection(COLL).document(userEmail).get()
    return snap.to_dict() if snap.exists else {}

# ---------- CRON: send due reminders ----------
@app.get("/cron/dispatchReminders")
async def cron_dispatch():
    sent = 0
    now = now_utc()
    users = db.collection(COLL).stream()
    for doc in users:
        uid = doc.id
        st = doc.to_dict()
        ac = st.get("active_condition")
        rems = st.get("reminders", [])
        if not ac or not rems: continue

        tok_snap = db.collection(TOKENS).document(uid).get()
        if not tok_snap.exists: continue
        token = tok_snap.to_dict().get("token")
        if not token: continue

        updated = False
        for r in rems:
            due = datetime.fromisoformat(r["next_at"])
            if due <= now:
                # send FCM
                try:
                    messaging.send(messaging.Message(
                        notification=messaging.Notification(title=r.get("title","Health reminder"), body=r.get("body","How are you feeling?")),
                        token=token
                    ))
                    sent += 1
                except Exception:
                    pass
                # schedule next
                interval = int(r.get("every_hours", 4))
                r["next_at"] = (now + timedelta(hours=interval)).isoformat()
                updated = True
        if updated:
            db.collection(COLL).document(uid).update({"reminders": rems})
    return {"ok": True, "sent": sent}

# ---------- CRON: cleanup recycle bin -> anonymous archive after 30d ----------
@app.get("/cron/cleanupRecycleBin")
async def cron_cleanup():
    moved = 0
    cutoff = now_utc() - timedelta(days=30)
    for doc in db.collection(COLL).stream():
        uid = doc.id
        st = doc.to_dict()
        rb = st.get("recycle_bin", [])
        keep = []
        for item in rb:
            try:
                dt = datetime.fromisoformat(item["deleted_at"])
            except Exception:
                keep.append(item); continue
            if dt <= cutoff:
                # anonymize + archive
                anon_id = hashlib.sha256(f"{uid}:{item.get('condition','')}:{item.get('resolved','')}".encode()).hexdigest()
                db.collection(ANON_ARCHIVE).document(anon_id).set({
                    "condition": item.get("condition"),
                    "resolved": item.get("resolved"),
                    "severity": item.get("severity"),
                    "duration": item.get("duration"),
                    "archived_at": now_utc().isoformat()
                })
                moved += 1
            else:
                keep.append(item)
        if len(keep) != len(rb):
            db.collection(COLL).document(uid).update({"recycle_bin": keep})
    return {"ok": True, "archived": moved}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
