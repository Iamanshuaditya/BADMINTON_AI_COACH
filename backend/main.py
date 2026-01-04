"""
FastAPI Application - ShuttleSense Video Coach API
"""

import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.pipeline import AnalysisPipeline
from core.grounded_chat import GroundedChat, StubChat

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ShuttleSense Video Coach API",
    description="AI-powered badminton footwork analysis",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pipeline = AnalysisPipeline(output_dir="./data/sessions")

# Use stub chat if no API key, otherwise real chat
api_key = os.getenv("GOOGLE_API_KEY")
chat_engine = GroundedChat(api_key) if api_key else StubChat()


# Request/Response models
class AnalyzeRequest(BaseModel):
    drill_type: str = "unknown"


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: list
    grounded: bool


# Health check
@app.get("/")
async def root():
    return {"status": "ok", "service": "ShuttleSense Video Coach API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Upload endpoint
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for analysis."""
    
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_types}")
    
    # Generate unique filename
    ext = Path(file.filename).suffix or ".mp4"
    video_id = str(uuid.uuid4())[:8]
    filename = f"{video_id}{ext}"
    filepath = UPLOAD_DIR / filename
    
    # Save file
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, "Failed to save video")
    
    return {
        "video_id": video_id,
        "filename": filename,
        "path": str(filepath),
        "size": filepath.stat().st_size
    }


# Analyze endpoint
@app.post("/api/analyze")
async def analyze_video(
    video_id: str = Form(...),
    drill_type: str = Form("unknown")
):
    """Analyze an uploaded video."""
    
    # Find video file
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(404, f"Video not found: {video_id}")
    
    video_path = video_files[0]
    
    try:
        result = pipeline.analyze(
            video_path=str(video_path),
            drill_type=drill_type,
            session_id=video_id,
            save_poses=True
        )
        
        if "error" in result:
            raise HTTPException(422, result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# Combined upload + analyze
@app.post("/api/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    drill_type: str = Form("unknown")
):
    """Upload and analyze in one step."""
    
    # Upload
    upload_result = await upload_video(file)
    video_id = upload_result["video_id"]
    
    # Analyze
    try:
        result = pipeline.analyze(
            video_path=upload_result["path"],
            drill_type=drill_type,
            session_id=video_id,
            save_poses=True
        )
        
        if "error" in result:
            raise HTTPException(422, result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# Get report
@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Get analysis report for a session."""
    
    session = pipeline.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    return session


# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about a session (grounded in evidence)."""
    
    session = pipeline.get_session(request.session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {request.session_id}")
    
    chunks = session.get("evidence_chunks", [])
    if not chunks:
        raise HTTPException(422, "No evidence available for this session")
    
    result = chat_engine.chat(
        question=request.question,
        evidence_chunks=chunks,
        session_summary=session.get("report", {}).get("metrics_summary")
    )
    
    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        grounded=result["grounded"]
    )


# List sessions
@app.get("/api/sessions")
async def list_sessions():
    """List all analysis sessions."""
    return {"sessions": pipeline.list_sessions()}


# Delete session
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    session_dir = Path(f"./data/sessions/{session_id}")
    if session_dir.exists():
        shutil.rmtree(session_dir)
    
    # Also delete video if exists
    for f in UPLOAD_DIR.glob(f"{session_id}.*"):
        f.unlink()
    
    return {"deleted": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
