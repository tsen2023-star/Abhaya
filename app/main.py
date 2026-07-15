from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
from .database import get_db
from . import models
from .services import alerts, routing, audio_inference, nlp_clustering

app = FastAPI(title="Abhaya API", description="Backend for the Abhaya safety companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_location(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.get("/")
def health_check():
    return {"status": "Abhaya Backend is active and fearless"}

@app.post("/api/sos/trigger")
def trigger_sos(user_id: str, lat: float, lon: float, trigger_method: str, db = Depends(get_db)):
    new_alert = models.SOSAlert(
        user_id=user_id, latitude=lat, longitude=lon, trigger_method=trigger_method
    )
    result = db.sos_alerts.insert_one(new_alert.model_dump())
    alerts.send_emergency_sms(user_id=user_id, lat=lat, lon=lon)
    return {"status": "SOS Logged and Alerts Sent", "alert_id": str(result.inserted_id)}

@app.post("/api/sos/fake-call")
def trigger_fake_call(user_id: str):
    alerts.trigger_fake_call_logic(user_id=user_id)
    return {"status": "Fake call initiated"}

@app.get("/api/route/score")
def get_area_safety_score(lat: float, lon: float):
    score = routing.get_safety_score(lat=lat, lon=lon)
    status = "Safe" if score >= 75 else "Moderate" if score >= 50 else "High Risk"
    return {"latitude": lat, "longitude": lon, "safety_score": score, "status": status}

@app.post("/api/sos/audio-detect")
async def detect_audio_distress(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    return audio_inference.analyze_audio_chunk(audio_bytes)

# --- NEW: NLP Crowd-Sourced Alerts ---

@app.post("/api/reports/submit")
def submit_incident_report(lat: float, lon: float, description: str, db = Depends(get_db)):
    new_report = models.IncidentReport(latitude=lat, longitude=lon, description=description)
    result = db.incident_reports.insert_one(new_report.model_dump())
    return {"status": "Report submitted anonymously", "report_id": str(result.inserted_id)}

@app.get("/api/reports/hotspots")
def get_verified_hotspots(db = Depends(get_db)):
    # Fetch all reports (In production, you'd filter this to only the last 24-48 hours)
    all_reports = list(db.incident_reports.find())
    
    # Convert DB objects to standard dictionaries for the ML model
    reports_data = [
        {"id": str(r["_id"]), "description": r["description"], "latitude": r["latitude"], "longitude": r["longitude"]} 
        for r in all_reports
    ]
    
    # Run the NLP clustering
    clusters = nlp_clustering.cluster_incident_reports(reports_data)
    return {"verified_threat_clusters": clusters}

# --- WEBSOCKET ---

@app.websocket("/ws/sos/{user_id}")
async def sos_location_stream(websocket: WebSocket, user_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            location_payload = {"user_id": user_id, "type": "live_location", "data": json.loads(data)}
            await manager.broadcast_location(location_payload)
    except WebSocketDisconnect:
        manager.disconnect(websocket)