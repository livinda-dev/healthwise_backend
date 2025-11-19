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

UTC = timezone.utc
def now_utc() -> datetime:
    return datetime.now(tz=UTC)

# ---------- Toxicity Check ----------
BAD_TERMS = [
    "fuck", "shit", "bitch", "suck my", "dick",
    "asshole", "motherfucker", "f*ck", "f**k"
]

EMERGENCY_TERMS = [
    "chest pain","pressure on chest","shortness of breath",
    "difficulty breathing","fainting","passed out",
    "sudden weakness","sudden numbness","slurred speech",
    "vision loss","seizure","seizures","suicidal",
    "suicide","kill myself","overdose",
    "bleeding that won't stop","uncontrolled bleeding"
]

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

def greeting(text: str) -> bool:
    return bool(re.search(r"\b(hi|hello|hey|good morning|good evening)\b", text.lower()))

# ---------- Helpers ----------
def last_user_text(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return " ".join(p.text for p in m.parts).strip()
    return ""

def normalize_severity(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = int(val)
        return max(1, min(10, v))
    m = re.search(r"(\d{1,2})", str(val))
    if m:
        return max(1, min(10, int(m.group(1))))
    return None

def safe_list(x):
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []

# ---------- Reminder Generator ----------
def build_reminder_plan(condition: str):
    t = now_utc()
    if "headache" in condition.lower():
        return [
            {
                "type": "water",
                "title": "💧 Hydration check",
                "body": "Drink some water.",
                "next_at": (t + timedelta(hours=2)).isoformat(),
                "every_hours": 2
            }
        ]
    return [
        {
            "type": "check_in",
            "title": "🩺 Health check-in",
            "body": "How are you feeling now?",
            "next_at": (t + timedelta(hours=4)).isoformat(),
            "every_hours": 4
        }
    ]

def set_active_condition(state, condition, extracted):
    state["active_condition"] = {
        "condition": condition,
        "since": now_utc().isoformat(),
        "extracted": extracted
    }
    state["reminders"] = build_reminder_plan(condition)

# ===================================================================
# ⚡️ MAIN CHAT ENDPOINT
# ===================================================================

@app.post("/healthAssistant")
async def healthAssistant(req: Request):
    body = await req.json()
    raw = body.get("data") or body.get("input") or body
    payload = InputData(**raw)

    userEmail = payload.userEmail
    messages = payload.messages
    last_user = last_user_text(messages)

    # ---------- Abuse Check ----------
    if is_direct_abuse(last_user):
        return {"text": "Please be respectful. I’m here to help you with your health."}

    # ---------- Emergency Rule ----------
    danger = has_emergency(last_user)
    if danger:
        return {"text": f"⚠️ This seems serious (“{danger}”). Please seek emergency care immediately."}

    # ---------- Load State ----------
    doc_ref = db.collection(COLL).document(userEmail)
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
            },
            "history": [],
            "recycle_bin": []
        }

    # ---------- If user says hi → reset triage but keep history ----------
    if greeting(last_user) and state.get("stage") == "analysis":
        state["stage"] = "intake"
        state["symptoms"] = {"location": None, "duration": None, "severity": None, "otherSymptoms": []}
        state.pop("active_condition", None)
        state.pop("reminders", None)
        doc_ref.set(state)

    # ---------- AI Triage Prompt ----------
    symp = state["symptoms"]
    convo = "\n".join([f"{m.role}: {' '.join(p.text for p in m.parts)}" for m in messages])

    prompt = f"""
You are a medical triage assistant. Ask 1–3 short follow-up questions first.
Keep responses short. No diagnosis. Give advice only after enough info.

Current triage stage: {state.get("stage")}
Known symptoms: {json.dumps(symp)}

Conversation:
{convo}
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        result = model.generate_content(prompt)
        reply = result.text
    except:
        reply = "I'm here to help. Could you tell me more?"

    # ---------- Extraction Pass ----------
    extracted = {}
    condition_label = None
    if last_user:
        eprompt = f"""
Extract JSON with:
location, duration, severity (1–10), otherSymptoms[], condition

User: \"{last_user}\"
"""
        try:
            m2 = genai.GenerativeModel("models/gemini-2.0-flash")
            raw = m2.generate_content(eprompt).text.strip()
            raw = raw.replace("```json", "").replace("```", "")
            extracted = json.loads(raw)

            if extracted.get("location"):
                state["symptoms"]["location"] = extracted["location"]
            if extracted.get("duration"):
                state["symptoms"]["duration"] = extracted["duration"]
            if extracted.get("severity") is not None:
                state["symptoms"]["severity"] = normalize_severity(extracted["severity"])
            if extracted.get("otherSymptoms"):
                exist = set(state["symptoms"].get("otherSymptoms", []))
                for s in extracted["otherSymptoms"]:
                    exist.add(s)
                state["symptoms"]["otherSymptoms"] = list(exist)

            condition_label = extracted.get("condition")
        except:
            pass

    # ---------- Move to analysis ----------
    s = state["symptoms"]
    if s["location"] and s["duration"] and s["severity"]:
        state["stage"] = "analysis"

    # ---------- Attach condition ----------
    if state["stage"] == "analysis" and condition_label and not state.get("active_condition"):
        set_active_condition(state, condition_label, extracted)

    # ---------- Save ----------
    doc_ref.set(state)

    return {"text": reply}


    # ---------- ROADMAP ENDPOINT ----------

@app.get("/roadmap")
async def get_roadmap(userEmail: str):
    """
    Return AI-generated roadmap + existing progress.
    {
      "condition": {...} or null,
      "roadmap": [...],
      "progress": {...}
    }
    """

    doc = db.collection(COLL).document(userEmail).get()
    if not doc.exists:
        return {"condition": None, "roadmap": [], "progress": {}}

    state = doc.to_dict()
    active = state.get("active_condition")
    symptoms = state.get("symptoms", {})
    progress = state.get("roadmap_progress", {})

    # If no active condition → no roadmap
    if not active:
        return {"condition": None, "roadmap": [], "progress": progress}

    condition = (active.get("condition") or "health issue").lower()
    severity = symptoms.get("severity")
    duration = symptoms.get("duration") or ""
    other_symptoms = symptoms.get("otherSymptoms", [])

    # ---------------------------
    # Build AI prompt
    # ---------------------------
    prompt = f"""
You are a careful medical triage assistant.

Create a SIMPLE 1–3 day self-care plan ("care roadmap") for this user.
Focus only on lifestyle and home-care: hydration, rest, screen time, posture, stretching, sleep, gentle activity etc.
NEVER prescribe medication by name.

User Info:
- Condition: {condition}
- Severity (1–10): {severity}
- Duration: {duration}
- Other symptoms: {json.dumps(other_symptoms, ensure_ascii=False)}

Rules:
- Assume this is NOT an emergency.
- Mild cases: 1–2 days.
- Moderate cases: 2–3 days.
- If severity ≥ 7 OR duration > 3 days → include short warning.
- Each day: 3–6 short actions.
- Encourage hydration, rest, screen moderation, posture, stress reduction.
- Always remind this is NOT a medical diagnosis.

OUTPUT STRICT JSON ONLY:
{
  "roadmap": [
    {
      "title": "Day 1 — Short title",
      "actions": [
        "Action 1.",
        "Action 2."
      ],
      "warning": "Short warning or null"
    }
  ]
}
"""

    # ---------------------------
    # Call Gemini
    # ---------------------------
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        res = model.generate_content(prompt)
        text = (res.text or "").strip()

        # remove ```json wrappers
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        parsed = json.loads(text)
        roadmap = parsed.get("roadmap", [])

        if not isinstance(roadmap, list):
            raise ValueError("roadmap must be a list")

    except Exception as e:
        print("ROADMAP AI ERROR:", e)
        roadmap = [
            {
                "title": "Day 1 — Basic Care",
                "actions": [
                    "Drink water regularly.",
                    "Reduce screen time today.",
                    "Get enough sleep tonight."
                ],
                "warning": "If symptoms do not improve in a few days, consider seeing a doctor."
            }
        ]

    # ---------------------------
    # Build progress structure
    # ---------------------------
    for day in roadmap:
        title = day["title"]
        progress.setdefault(title, {})
        for act in day["actions"]:
            progress[title].setdefault(act, False)

    # Save updated progress
    db.collection(COLL).document(userEmail).update({
        "roadmap_progress": progress
    })

    return {
        "condition": active,
        "roadmap": roadmap,
        "progress": progress
    }


@app.get("/cron/roadmapReminders")
async def cron_roadmap_reminders():
    now = now_utc()
    users = db.collection(COLL).stream()
    sent = 0

    for doc in users:
        email = doc.id
        state = doc.to_dict()
        active = state.get("active_condition")
        progress = state.get("roadmap_progress", {})

        if not active:
            continue

        # find unfinished actions
        for day, acts in progress.items():
            for act, done in acts.items():
                if not done:  # action incomplete
                    # send reminder
                    tok_snap = db.collection(TOKENS).document(email).get()
                    if not tok_snap.exists:
                        continue
                    token = tok_snap.to_dict().get("token")

                    try:
                        messaging.send(messaging.Message(
                            token=token,
                            notification=messaging.Notification(
                                title=f"Care step remaining",
                                body=f"Don't forget: {act}",
                            )
                        ))
                        sent += 1
                    except Exception:
                        pass

    return {"ok": True, "sent": sent}




# ===================================================================
# 🔥 USER CONDITION STATUS (New for your UI!)
# ===================================================================

@app.get("/conditions/status/{email}")
async def condition_status(email: str):
    snap = db.collection(COLL).document(email).get()
    if not snap.exists:
        return {"active": False}

    st = snap.to_dict()
    return {"active": bool(st.get("active_condition"))}

# ===================================================================
# FCM TOKEN REGISTER
# ===================================================================

@app.post("/registerToken")
async def register_token(body: RegisterTokenBody):
    db.collection(TOKENS).document(body.userEmail).set({
        "token": body.fcmToken,
        "updated_at": now_utc().isoformat()
    })
    return {"ok": True}

# ===================================================================
# USER SAYS "I'M BETTER NOW"
# ===================================================================

@app.post("/conditions/resolve")
async def resolve_condition(body: ResolveBody):
    user = body.userEmail
    ref = db.collection(COLL).document(user)
    snap = ref.get()

    if not snap.exists:
        return {"ok": True}

    st = snap.to_dict()
    ac = st.get("active_condition")
    if not ac:
        return {"ok": True}

    # store in history
    item = {
        "condition": ac.get("condition"),
        "resolved": now_utc().isoformat(),
        "severity": st["symptoms"].get("severity"),
        "duration": st["symptoms"].get("duration"),
        "note": body.note or ""
    }

    hist = st.get("history", [])
    hist.append(item)
    st["history"] = hist

    # reset triage
    st.pop("active_condition", None)
    st.pop("reminders", None)

    st["stage"] = "intake"
    st["symptoms"] = {"location": None, "duration": None, "severity": None, "otherSymptoms": []}

    ref.set(st)

    return {"ok": True}

# ===================================================================
# DELETE HISTORY (Soft delete → recycle_bin)
# ===================================================================

@app.post("/conditions/delete")
async def delete_history_item(body: DeleteHistoryBody):
    user = body.userEmail
    ref = db.collection(COLL).document(user)
    snap = ref.get()

    if not snap.exists:
        return {"ok": False}

    st = snap.to_dict()
    hist = st.get("history", [])
    if body.historyIndex < 0 or body.historyIndex >= len(hist):
        return {"ok": False}

    item = hist.pop(body.historyIndex)
    item["deleted_at"] = now_utc().isoformat()

    rb = st.get("recycle_bin", [])
    rb.append(item)

    st["history"] = hist
    st["recycle_bin"] = rb

    ref.set(st)
    return {"ok": True}

# ===================================================================
# DEBUG: GET RAW STATE
# ===================================================================

@app.get("/state")
async def get_state(userEmail: str = Query(...)):
    snap = db.collection(COLL).document(userEmail).get()
    return snap.to_dict() if snap.exists else {}

# ===================================================================
# CRON JOBS
# ===================================================================

@app.get("/cron/dispatchReminders")
async def cron_dispatch():
    now = now_utc()
    sent = 0

    users = db.collection(COLL).stream()
    for doc in users:
        uid = doc.id
        st = doc.to_dict()

        if "active_condition" not in st:
            continue

        rems = st.get("reminders", [])
        if not rems:
            continue

        tok = db.collection(TOKENS).document(uid).get()
        if not tok.exists:
            continue

        token = tok.to_dict().get("token")
        if not token:
            continue

        changed = False
        for r in rems:
            due = datetime.fromisoformat(r["next_at"])
            if due <= now:
                try:
                    messaging.send(messaging.Message(
                        notification=messaging.Notification(
                            title=r["title"],
                            body=r["body"]
                        ),
                        token=token
                    ))
                    sent += 1
                except:
                    pass

                r["next_at"] = (now + timedelta(hours=r["every_hours"])).isoformat()
                changed = True

        if changed:
            db.collection(COLL).document(uid).update({"reminders": rems})

    return {"ok": True, "sent": sent}

@app.get("/cron/cleanupRecycleBin")
async def cleanup():
    cutoff = now_utc() - timedelta(days=30)
    moved = 0

    for doc in db.collection(COLL).stream():
        uid = doc.id
        st = doc.to_dict()

        rb = st.get("recycle_bin", [])
        keep = []

        for item in rb:
            dt = datetime.fromisoformat(item["deleted_at"])
            if dt <= cutoff:
                anon_id = hashlib.sha256(f"{uid}:{item.get('condition')}".encode()).hexdigest()
                db.collection(ANON_ARCHIVE).document(anon_id).set({
                    "condition": item.get("condition"),
                    "severity": item.get("severity"),
                    "duration": item.get("duration"),
                    "resolved": item.get("resolved"),
                    "archived_at": now_utc().isoformat(),
                })
                moved += 1
            else:
                keep.append(item)

        if len(keep) != len(rb):
            db.collection(COLL).document(uid).update({"recycle_bin": keep})

    return {"ok": True, "archived": moved}

@app.post("/conditions/restore")
async def restore_condition(body: DeleteHistoryBody):
    user = body.userEmail.strip()
    index = body.historyIndex
    ref = db.collection(COLL).document(user)
    snap = ref.get()
    if not snap.exists:
        return {"ok": False, "error": "User not found"}

    state = snap.to_dict()
    rb = state.get("recycle_bin", [])
    hist = state.get("history", [])

    if index < 0 or index >= len(rb):
        return {"ok": False, "error": "Index out of range"}

    item = rb.pop(index)
    hist.append(item)

    state["history"] = hist
    state["recycle_bin"] = rb
    ref.set(state)

    return {"ok": True}


@app.post("/conditions/deleteForever")
async def delete_forever(body: DeleteHistoryBody):
    user = body.userEmail.strip()
    index = body.historyIndex
    ref = db.collection(COLL).document(user)
    snap = ref.get()
    if not snap.exists:
        return {"ok": False, "error": "User not found"}

    state = snap.to_dict()
    rb = state.get("recycle_bin", [])

    if index < 0 or index >= len(rb):
        return {"ok": False, "error": "Index out of range"}

    rb.pop(index)
    state["recycle_bin"] = rb
    ref.set(state)

    return {"ok": True}


# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
