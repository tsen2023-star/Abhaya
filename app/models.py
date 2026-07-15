from pydantic import BaseModel, Field
from datetime import datetime, timezone

class SOSAlert(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    trigger_method: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class IncidentReport(BaseModel):
    latitude: float
    longitude: float
    description: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    name: str
    phone_number: str
    emergency_contact: str
    password: str

class UserLogin(BaseModel):
    phone_number: str
    password: str

class UserResponse(BaseModel):
    user_id: str
    name: str
    phone_number: str
    emergency_contact: str