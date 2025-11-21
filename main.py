import os
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId

from database import db, create_document, get_documents

app = FastAPI(title="Decision Time Machine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility helpers
# -----------------------------

def oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")

import hashlib
import secrets

def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"

def verify_password(password: str, hashed: str) -> bool:
    try:
        salt, digest = hashed.split("$")
    except ValueError:
        return False
    return hashlib.sha256((salt + password).encode()).hexdigest() == digest

# -----------------------------
# Pydantic Models
# -----------------------------

class ScenarioIn(BaseModel):
    type: str
    timeline: List[str]
    risk: str
    opportunity: str
    probability: int = Field(..., ge=0, le=100)
    impactScore: int = Field(..., ge=0, le=100)
    effortRequired: int = Field(..., ge=0, le=100)

class DecisionIn(BaseModel):
    decisionTitle: str
    description: str
    timeframe: str
    importanceLevel: int = Field(..., ge=1, le=5)
    emotionalState: str
    effortLevel: int = Field(..., ge=1, le=5)
    userId: str

class DecisionOut(BaseModel):
    id: str
    decisionTitle: str
    description: str
    timeframe: str
    importanceLevel: int
    emotionalState: str
    effortLevel: int
    scenarios: List[ScenarioIn]
    decisionClarityScore: int
    riskScore: int
    outcomeStabilityScore: int
    createdAt: datetime
    userId: str

class RegisterIn(BaseModel):
    username: str
    password: str

class LoginIn(BaseModel):
    username: str
    password: str

# -----------------------------
# Decision Logic
# -----------------------------

def generate_scenarios(payload: DecisionIn) -> List[ScenarioIn]:
    importance = payload.importanceLevel
    effort_level = payload.effortLevel
    mood = payload.emotionalState.lower()

    mood_bias = 0
    if any(k in mood for k in ["anx", "fear", "worri", "stress"]):
        mood_bias = -10
    elif any(k in mood for k in ["calm", "optim", "confid", "excited"]):
        mood_bias = 10

    base_prob = 50 + (importance - 3) * 5 + mood_bias // 2

    best_prob = min(95, max(10, base_prob + 20 - effort_level))
    typical_prob = min(85, max(10, base_prob))
    worst_prob = 100 - best_prob - typical_prob
    worst_prob = max(5, min(80, worst_prob))

    def mk_timeline(label: str) -> List[str]:
        tf = payload.timeframe
        return [
            f"Week 1: Define goals and constraints for {payload.decisionTitle}",
            f"{tf} mid-point: Evaluate progress and adjust plan",
            f"{tf} end: Consolidate outcomes and reflect",
        ]

    best = ScenarioIn(
        type="best",
        timeline=mk_timeline("best"),
        risk="Overconfidence, scope creep",
        opportunity="High leverage gains, learning, recognition",
        probability=best_prob,
        impactScore=min(100, 70 + importance * 6),
        effortRequired=min(100, 40 + effort_level * 8),
    )

    typical = ScenarioIn(
        type="typical",
        timeline=mk_timeline("typical"),
        risk="Execution variance, minor blockers",
        opportunity="Steady progress, moderate gains",
        probability=typical_prob,
        impactScore=min(100, 50 + importance * 5),
        effortRequired=min(100, 50 + effort_level * 6),
    )

    worst = ScenarioIn(
        type="worst",
        timeline=mk_timeline("worst"),
        risk="Delays, burnout, external shocks",
        opportunity="Lessons learned, pivots identified",
        probability=worst_prob,
        impactScore=max(10, 30 + (importance - 3) * 5),
        effortRequired=min(100, 60 + effort_level * 7),
    )

    return [best, typical, worst]


def compute_scores(scenarios: List[ScenarioIn], payload: DecisionIn) -> dict:
    # Risk Score: weighted by probability and inverse of impact for negative outcome
    worst = next((s for s in scenarios if s.type == "worst"), scenarios[-1])
    typical = next((s for s in scenarios if s.type == "typical"), scenarios[1])
    best = next((s for s in scenarios if s.type == "best"), scenarios[0])

    risk_score = min(100, max(0, int(0.6 * worst.probability + 0.4 * (worst.effortRequired / 100 * 50))))

    # Outcome Stability: lower spread between impact and probability indicates stability
    prob_spread = max(best.probability, typical.probability, worst.probability) - min(best.probability, typical.probability, worst.probability)
    impact_spread = max(best.impactScore, typical.impactScore, worst.impactScore) - min(best.impactScore, typical.impactScore, worst.impactScore)
    stability = 100 - int(0.6 * prob_spread + 0.4 * (impact_spread / 100 * 100))
    stability = max(0, min(100, stability))

    # Clarity Score: importance alignment + mood bias proxy from typical probability
    clarity = int(0.5 * (payload.importanceLevel / 5 * 100) + 0.5 * typical.probability)
    clarity = max(0, min(100, clarity))

    return {
        "decisionClarityScore": clarity,
        "riskScore": risk_score,
        "outcomeStabilityScore": stability,
    }

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def read_root():
    return {"message": "Decision Time Machine API"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {e}"[:120]
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {e}"[:120]
    return response

@app.post("/api/user/register")
def register_user(payload: RegisterIn):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    users = db["user"]
    if users.find_one({"username": payload.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = hash_password(payload.password)
    doc = {
        "username": payload.username,
        "password": hashed,
        "createdAt": datetime.now(timezone.utc),
    }
    res = users.insert_one(doc)
    return {"userId": str(res.inserted_id), "username": payload.username}

@app.post("/api/user/login")
def login_user(payload: LoginIn):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    user = db["user"].find_one({"username": payload.username})
    if not user or not verify_password(payload.password, user.get("password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"userId": str(user["_id"]), "username": user["username"]}

@app.post("/api/decision/generate")
def generate_decision(payload: DecisionIn):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    scenarios = generate_scenarios(payload)
    scores = compute_scores(scenarios, payload)

    doc = {
        "decisionTitle": payload.decisionTitle,
        "description": payload.description,
        "timeframe": payload.timeframe,
        "importanceLevel": payload.importanceLevel,
        "emotionalState": payload.emotionalState,
        "effortLevel": payload.effortLevel,
        "scenarios": [s.model_dump() for s in scenarios],
        **scores,
        "createdAt": datetime.now(timezone.utc),
        "userId": payload.userId,
    }
    res_id = db["decision"].insert_one(doc).inserted_id

    out = {
        "id": str(res_id),
        **{k: doc[k] for k in [
            "decisionTitle","description","timeframe","importanceLevel","emotionalState","effortLevel","scenarios",
            "decisionClarityScore","riskScore","outcomeStabilityScore","createdAt","userId"
        ]}
    }
    return out

@app.get("/api/decision/{id}")
def get_decision(id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    doc = db["decision"].find_one({"_id": oid(id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Decision not found")
    doc["id"] = str(doc.pop("_id"))
    return doc

@app.get("/api/decision/history/{userId}")
def get_history(userId: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    cursor = db["decision"].find({"userId": userId}).sort("createdAt", -1)
    out = []
    for d in cursor:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
