"""
Database Schemas for Decision Time Machine

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Decision -> "decision" collection
- Scenario -> embedded in Decision documents
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Scenario(BaseModel):
    """Embedded scenario schema inside a Decision"""
    type: str = Field(..., description="best | typical | worst")
    timeline: List[str] = Field(default_factory=list, description="Ordered events describing the scenario timeline")
    risk: str = Field(..., description="Key risks in this scenario")
    opportunity: str = Field(..., description="Key opportunities in this scenario")
    probability: int = Field(..., ge=0, le=100, description="Expected probability (0-100)")
    impactScore: int = Field(..., ge=0, le=100, description="Impact score (0-100)")
    effortRequired: int = Field(..., ge=0, le=100, description="Effort required (0-100)")

class Decision(BaseModel):
    decisionTitle: str
    description: str
    timeframe: str
    importanceLevel: int = Field(..., ge=1, le=5)
    emotionalState: str
    effortLevel: int = Field(..., ge=1, le=5)
    scenarios: List[Scenario] = Field(default_factory=list)
    decisionClarityScore: int = Field(0, ge=0, le=100)
    riskScore: int = Field(0, ge=0, le=100)
    outcomeStabilityScore: int = Field(0, ge=0, le=100)
    createdAt: Optional[datetime] = None
    userId: str

class User(BaseModel):
    username: str
    password: str  # hashed value stored in DB
    createdAt: Optional[datetime] = None
