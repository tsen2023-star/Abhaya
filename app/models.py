from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

class SOSAlert(Base):
    __tablename__ = "sos_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    trigger_method = Column(String, nullable=False) 
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

# New Table: Stores anonymous crowd-sourced safety reports
class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(String, nullable=False) # e.g., "No streetlights and suspicious group"
    timestamp = Column(DateTime(timezone=True), server_default=func.now())