import os, json, re, hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, ValidationError
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

class RoadmapUpdateBody(BaseModel):
    email: str
    day: str
    action: str
    value: bool

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
    """
    Ensure severity is an int 1–10, accepting:
    - raw numbers (from Gemini)
    - strings like "7/10" or "pain level 7"
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(round(float(value)))
        return max(1, min(10, v))

    s = str(value).lower()
    # try "7/10" or "7"
    m = re.search(r"(\d{1,2})\s*/\s*10", s) or re.search(r"\b(\d{1,2})\b", s)
    if m:
        try:
            v = int(m.group(1))
            return max(1, min(10, v))
        except Exception:
            pass
    return None

def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []

def greeting(text: str) -> bool:
    # simple greeting detection
    return bool(re.search(r"\b(hi|hello|hey|good\s+morning|good\s+evening)\b", text.lower()))

# ---------- Reminder plan rules ----------
def build_reminder_plan(condition: str) -> List[Dict[str, Any]]:
    t = now_utc()
    cond_lower = condition.lower()

    if "headache" in cond_lower:
        return [
            {
                "type": "drink_water",
                "title": "💧 Hydration break",
                "body": "Drink a glass of water.",
                "next_at": (t + timedelta(hours=2)).isoformat(),
                "every_hours": 2,
            },
            {
                "type": "eye_rest",
                "title": "👀 Eye rest",
                "body": "Rest your eyes from screens for a few minutes.",
                "next_at": (t + timedelta(hours=3)).isoformat(),
                "every_hours": 3,
            },
        ]

    # default gentle plan
    return [
        {
            "type": "check_in",
            "title": "🩺 Quick check-in",
            "body": "How are you feeling now?",
            "next_at": (t + timedelta(hours=4)).isoformat(),
            "every_hours": 4,
        }
    ]

def set_active_condition(state: Dict[str, Any], condition: str, extracted: Dict[str, Any]):
    state["active_condition"] = {
        "condition": condition,
        "since": now_utc().isoformat(),
        "tips": ["Hydration", "Rest", "Avoid triggers"],
        "extracted": extracted,
    }
    state["reminders"] = build_reminder_plan(condition)
    state["stage"] = "reminder"  # after analysis, enter reminder loop

# ---------- Core Chat Endpoint ----------
@app.post("/healthAssistant")
async def healthAssistant(req: Request):
    body = await req.json()
    raw = body.get("data") or body.get("input") or body

    # parse + validate
    try:
        payload = InputData(**raw)
    except ValidationError as ve:
        return {"text": f"Invalid input: {ve.errors()}"}

    userEmail = payload.userEmail.strip()
    if not userEmail:
        return {"text": "Missing user email"}

    messages = payload.messages
    last_user = last_user_text(messages)

    # profanity rule
    if last_user and is_direct_abuse(last_user):
        return {
            "text": "Please speak respectfully. I'm here to help you with your health."
        }

    # load or create state
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
                "otherSymptoms": [],
            },
            "history": [],
            "recycle_bin": [],
            "roadmap_progress": {},
        }

    # reset if user greets while stage is finished/analysis
    if state.get("stage") in ("analysis", "reminder") and greeting(last_user):
        state["stage"] = "intake"
        state["symptoms"] = {
            "location": None,
            "duration": None,
            "severity": None,
            "otherSymptoms": [],
        }
        state.pop("active_condition", None)
        state.pop("reminders", None)
        doc_ref.set(state)

    # emergency shortcut
    danger = has_emergency(last_user)
    if danger:
        state["stage"] = "emergency"
        doc_ref.set(state)
        return {
            "text": f"⚠️ Your message suggests a serious symptom (\"{danger}\"). "
                    "Please seek urgent medical care now or call your local emergency number."
        }

    # build conversation string
    convo = ""
    for m in messages:
        convo += f"{m.role}: {' '.join(p.text for p in m.parts)}\n"

    symp = state.get("symptoms", {})
    missing = []
    if not symp.get("location"):
        missing.append("location (e.g., left side of head, chest, etc.)")
    if not symp.get("duration"):
        missing.append("duration (e.g., since yesterday, 2 hours)")
    if not symp.get("severity"):
        missing.append("severity (1–10)")

    missing_hint = ""
    if missing:
        missing_hint = "- Ask about: " + "; ".join(missing) + "."

    # main triage prompt
    triage_prompt = f"""
You are an empathetic medical triage assistant AI.

Your job:
- Act like a doctor doing initial triage.
- Ask 1–3 follow-up questions FIRST; only give suggestions after enough info.
- Keep messages short and clear.
- Ignore mild profanity inside symptom description; only ask for respectful tone if user directly insults you.
- Always remind this is NOT a formal medical diagnosis.

Triage stage: {state.get("stage", "intake")}
Known symptoms (may be incomplete): {json.dumps(symp, ensure_ascii=False)}
{missing_hint}

If red flags (chest pain, trouble breathing, sudden weakness/numbness, vision loss, fainting, severe allergy, suicidal thoughts) appear, advise urgent care.

Continue the conversation in the user's language.

Conversation so far:
{convo}
"""

    try:
        main_model = genai.GenerativeModel("models/gemini-2.5-flash")
        main_result = main_model.generate_content(triage_prompt)
        reply_text = main_result.text or "I'm here to help. Could you share a bit more detail?"
    except Exception as e:
        reply_text = f"Sorry, I had trouble generating a response. ({e})"

    # ---------- MULTILINGUAL EXTRACTION PASS ----------
    condition_label: Optional[str] = None

    if last_user:
        extract_prompt = f"""
You extract health triage information from ANY LANGUAGE (including Khmer).

Always return STRICT JSON ONLY, like:
{{
  "location": string or null,
  "duration": string or null,
  "severity": number or null,
  "otherSymptoms": [string, ...],
  "condition": string or null
}}

Interpret severity even if described in non-English words. Map to 1–10:
- very mild / "ឈឺតិចៗ" ≈ 2
- mild / "ឈឺតិច" ≈ 3
- moderate / "មធ្យម" ≈ 5
- quite strong / "ឈឺខ្លាំង" ≈ 7
- very strong / "ខ្លាំងណាស់" ≈ 8–9

If the text clearly describes strong pain, choose a higher number. If unsure, guess a reasonable 1–10.

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
            condition_label = (extracted.get("condition") or "").strip()[:64]

            # update state
            if loc:
                state["symptoms"]["location"] = loc
            if dur:
                state["symptoms"]["duration"] = dur
            if sev:
                state["symptoms"]["severity"] = sev
            if oth:
                cur = set(state["symptoms"].get("otherSymptoms", []))
                for x in oth:
                    if x not in cur:
                        cur.add(x)
                state["symptoms"]["otherSymptoms"] = list(cur)
        except Exception:
            # extraction failure is non-fatal
            pass

    # move to analysis when core fields are known
    symp = state["symptoms"]
    if symp.get("location") and symp.get("duration") and symp.get("severity"):
        state["stage"] = "analysis"

    # start reminders when we have a condition and just reached analysis
    if state.get("stage") == "analysis" and condition_label and not state.get("active_condition"):
        set_active_condition(state, condition_label, state["symptoms"])

    # save state
    doc_ref.set(state)
    return {"text": reply_text}

# ---------- Register FCM token ----------
@app.post("/registerToken")
async def register_token(body: RegisterTokenBody):
    user_id = body.userEmail.strip()
    if not user_id:
        return {"ok": False, "error": "Missing email"}

    db.collection(TOKENS).document(user_id).set(
        {"token": body.fcmToken, "updated_at": now_utc().isoformat()}
    )
    return {"ok": True}

# ---------- Resolve current condition ----------
@app.post("/conditions/resolve")
async def resolve_condition(body: ResolveBody):
    user = body.userEmail.strip()
    ref = db.collection(COLL).document(user)
    snap = ref.get()
    if not snap.exists:
        return {"ok": True}

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
        state["history"] = hist

        # clear active condition + reminders + symptoms
        state.pop("active_condition", None)
        state.pop("reminders", None)
        state["stage"] = "intake"
        state["symptoms"] = {
            "location": None,
            "duration": None,
            "severity": None,
            "otherSymptoms": [],
        }
        ref.set(state)

    return {"ok": True}

# ---------- Delete a history item -> recycle_bin ----------
@app.post("/conditions/delete")
async def delete_history_item(body: DeleteHistoryBody):
    user = body.userEmail.strip()
    ref = db.collection(COLL).document(user)
    snap = ref.get()
    if not snap.exists:
        return {"ok": False, "error": "no user state"}

    state = snap.to_dict()
    hist = state.get("history", [])
    if body.historyIndex < 0 or body.historyIndex >= len(hist):
        return {"ok": False, "error": "index out of range"}

    item = hist.pop(body.historyIndex)
    rb = state.get("recycle_bin", [])
    item["deleted_at"] = now_utc().isoformat()
    rb.append(item)

    state["history"] = hist
    state["recycle_bin"] = rb
    ref.set(state)
    return {"ok": True}

# ---------- Reset all (dev helper) ----------
@app.post("/conditions/reset")
async def reset_all(body: ResolveBody):
    user = body.userEmail.strip()
    ref = db.collection(COLL).document(user)
    ref.set(
        {
            "stage": "intake",
            "symptoms": {
                "location": None,
                "duration": None,
                "severity": None,
                "otherSymptoms": [],
            },
            "history": [],
            "recycle_bin": [],
            "roadmap_progress": {},
        }
    )
    return {"ok": True}

# ---------- DEBUG: get state ----------
@app.get("/state")
async def get_state(userEmail: str = Query(...)):
    snap = db.collection(COLL).document(userEmail).get()
    return snap.to_dict() if snap.exists else {}

@app.get("/roadmap")
async def get_roadmap(userEmail: str):
    """Return cached roadmap or generate and save it once."""
    doc_ref = db.collection(COLL).document(userEmail)
    snap = doc_ref.get()

    if not snap.exists:
        return {"condition": None, "roadmap": [], "progress": {}}

    state = snap.to_dict()
    active = state.get("active_condition")
    symptoms = state.get("symptoms", {})
    progress = state.get("roadmap_progress", {})

    # No active condition → No roadmap
    if not active:
        return {"condition": None, "roadmap": [], "progress": progress}

    # If roadmap already cached → return it
    saved = state.get("saved_roadmap")
    if saved:
        return {"condition": active, "roadmap": saved, "progress": progress}

    # ---- Otherwise generate NEW roadmap once ----
    condition = (active.get("condition") or "health issue").lower()
    severity = symptoms.get("severity")
    duration = symptoms.get("duration") or ""
    other_symptoms = symptoms.get("otherSymptoms", [])

    prompt = f"""
Create a 1–3 day care roadmap for:
- condition: {condition}
- severity: {severity}
- duration: {duration}
- other symptoms: {json.dumps(other_symptoms, ensure_ascii=False)}

Rules:
- 3–6 steps per day
- No medical prescriptions (generic terms ok)
- Include short warning for high severity
- JSON only!
Format:
{{
  "roadmap":[
    {{
      "title":"Day 1 — Short title",
      "actions":["Drink water","Rest eyes"],
      "warning":null
    }}
  ]
}}
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        res = model.generate_content(prompt)
        text = (res.text or "").strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        parsed = json.loads(text)
        roadmap = parsed.get("roadmap", [])

    except Exception as e:
        print("ROADMAP ERROR:", e)
        roadmap = [{
            "title": "Day 1 — Basic care",
            "actions": [
                "Drink water regularly",
                "Rest from screens",
                "Sleep earlier tonight"
            ],
            "warning": "Seek a doctor if symptoms worsen."
        }]

    # Save roadmap ONCE
    doc_ref.update({"saved_roadmap": roadmap})

    return {
        "condition": active,
        "roadmap": roadmap,
        "progress": progress
    }


@app.post("/roadmap/update")
async def update_roadmap(body: dict):
    """Update checked/un-checked actions in Firestore."""
    email = body.get("email")
    day = body.get("day")
    action = body.get("action")
    value = body.get("value")

    ref = db.collection(COLL).document(email)
    snap = ref.get()

    if not snap.exists:
        return {"ok": False, "error": "user not found"}

    state = snap.to_dict()
    rp = state.get("roadmap_progress", {})

    if day not in rp:
        rp[day] = {}

    rp[day][action] = value

    ref.update({"roadmap_progress": rp})
    return {"ok": True}


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
                keep.append(item)
                continue

            if dt <= cutoff:
                anon_id = hashlib.sha256(
                    f"{uid}:{item.get('condition','')}:{item.get('resolved','')}".encode()
                ).hexdigest()
                db.collection(ANON_ARCHIVE).document(anon_id).set(
                    {
                        "condition": item.get("condition"),
                        "resolved": item.get("resolved"),
                        "severity": item.get("severity"),
                        "duration": item.get("duration"),
                        "archived_at": now_utc().isoformat(),
                    }
                )
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
