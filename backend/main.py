"""
FastAPI Application - ShuttleSense Video Coach API
Production-ready with security, rate limiting, and proper error handling.
"""

import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Internal imports
from config.settings import get_settings, Settings
from core.pipeline import AnalysisPipeline
from core.grounded_chat import GroundedChat
from exceptions import (
    ShuttleSenseException,
    SessionNotFound,
    VideoNotFound,
    InvalidVideoFormat,
    FileTooLarge,
    InsufficientDiskSpace,
    VideoProcessingError,
    InsufficientPoseData
)
from middleware.error_handler import setup_error_handlers
from middleware.performance import PerformanceMiddleware
from middleware.rate_limiter import limiter, setup_rate_limiting

# Auth imports (optional - can be enabled per-endpoint)
from auth.jwt_handler import get_current_user, get_current_user_optional, UserInfo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="AI-powered badminton footwork and stroke analysis",
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None
    )
    
    # Setup CORS with proper security
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],
        max_age=3600,  # Cache preflight for 1 hour
    )
    
    # Add performance monitoring middleware
    app.add_middleware(PerformanceMiddleware)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Setup rate limiting
    setup_rate_limiting(app)
    
    return app


# Create app instance
app = create_app()

# =============================================================================
# Global State
# =============================================================================

# Allowed video types with extensions
ALLOWED_VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm"
}

# Create directories
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize pipeline
pipeline = AnalysisPipeline(output_dir=settings.SESSIONS_DIR)

# Initialize chat engine (evidence-only fallback if no API key)
chat_engine = GroundedChat(settings.GOOGLE_API_KEY, data_dir=settings.SESSIONS_DIR)

# =============================================================================
# Request/Response Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    drill_type: str = Field(default="unknown", description="Type of drill being performed")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to query")
    question: str = Field(..., min_length=1, max_length=500, description="Question to ask")
    include_debug: bool = Field(default=False, description="Include debug info in response")


class Citation(BaseModel):
    timestamp: float
    type: str = "evidence_reference"


class MissingEvidence(BaseModel):
    reason: str
    max_similarity: float
    threshold: float
    suggested_questions: list = []


class ChatDebug(BaseModel):
    top_similarity: float
    selected_chunk_types: list = []
    retrieval_count: int = 0
    llm_model: str = ""


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    grounded: bool
    debug: Optional[ChatDebug] = None
    missing_evidence: Optional[MissingEvidence] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    size: int


# =============================================================================
# File Validation Helpers
# =============================================================================

def validate_video_file(file: UploadFile) -> str:
    """
    Validate uploaded video file.
    
    Returns:
        Generated safe filename
    
    Raises:
        InvalidVideoFormat: If file type not allowed
        FileTooLarge: If file exceeds size limit
    """
    # Check MIME type
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise InvalidVideoFormat(
            content_type=file.content_type or "unknown",
            allowed_types=list(ALLOWED_VIDEO_TYPES.keys())
        )
    
    # Check file size if available
    if file.size and file.size > settings.max_video_size_bytes:
        raise FileTooLarge(
            file_size_mb=file.size / (1024 * 1024),
            max_size_mb=settings.MAX_VIDEO_SIZE_MB
        )
    
    # Generate safe filename
    ext = ALLOWED_VIDEO_TYPES.get(file.content_type, ".mp4")
    return f"{uuid.uuid4()}{ext}"


def check_disk_space(required_bytes: int) -> None:
    """
    Check if sufficient disk space is available.
    
    Raises:
        InsufficientDiskSpace: If not enough space
    """
    try:
        disk_usage = shutil.disk_usage(UPLOAD_DIR)
        buffer = 100 * 1024 * 1024  # 100MB buffer
        
        if disk_usage.free < required_bytes + buffer:
            raise InsufficientDiskSpace(
                required_mb=(required_bytes + buffer) / (1024 * 1024),
                available_mb=disk_usage.free / (1024 * 1024)
            )
    except OSError as e:
        logger.error(f"Disk space check failed: {e}")
        # Continue if check fails - don't block upload


def cleanup_file(filepath: Path) -> None:
    """Clean up file on failure."""
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup file {filepath}: {e}")


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - basic health check."""
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION
    )


@app.get("/health", tags=["Health"])
async def health():
    """Simple health check for load balancers."""
    return {"status": "healthy"}


@app.get("/health/live", tags=["Health"])
async def health_live():
    """Liveness probe - is service responding?"""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def health_ready():
    """Readiness probe - is service ready for traffic?"""
    try:
        # Check disk space
        disk = shutil.disk_usage(UPLOAD_DIR)
        if disk.free < 100 * 1024 * 1024:  # Less than 100MB
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "low_disk_space"}
            )
        
        # Check upload directory accessible
        if not UPLOAD_DIR.exists():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "upload_dir_missing"}
            )
        
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(e)}
        )


@app.get("/metrics", tags=["Health"])
async def metrics():
    """Basic metrics endpoint for monitoring."""
    import psutil
    
    try:
        disk = shutil.disk_usage(UPLOAD_DIR)
        process = psutil.Process(os.getpid())
        
        # Calculate upload directory size
        upload_size = sum(f.stat().st_size for f in UPLOAD_DIR.rglob("*") if f.is_file())
        
        return {
            "uptime_seconds": process.create_time(),
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "disk_free_gb": round(disk.free / 1024**3, 2),
            "disk_used_gb": round(disk.used / 1024**3, 2),
            "sessions_count": len(pipeline.list_sessions()),
            "upload_dir_size_mb": round(upload_size / 1024**2, 2)
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e)}


# =============================================================================
# Upload Endpoint
# =============================================================================

@app.post("/api/upload", response_model=UploadResponse, tags=["Video"])
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_video(
    request: Request,
    file: UploadFile = File(...)
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """
    Upload a video file for analysis.
    
    - **file**: Video file (MP4, MOV, AVI, WebM)
    - Max size: 500MB
    """
    # Validate file
    safe_filename = validate_video_file(file)
    video_id = safe_filename.split(".")[0]
    filepath = UPLOAD_DIR / safe_filename
    
    # Check disk space (estimate from content-length header)
    content_length = request.headers.get("content-length")
    if content_length:
        check_disk_space(int(content_length))
    
    # Save file
    try:
        with open(filepath, "wb") as buffer:
            # Read in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            total_size = 0
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                total_size += len(chunk)
                
                # Check size limit during upload
                if total_size > settings.max_video_size_bytes:
                    cleanup_file(filepath)
                    raise FileTooLarge(
                        file_size_mb=total_size / (1024 * 1024),
                        max_size_mb=settings.MAX_VIDEO_SIZE_MB
                    )
                
                buffer.write(chunk)
        
        logger.info(f"Video uploaded: {video_id} ({total_size / 1024 / 1024:.1f}MB)")
        
        return UploadResponse(
            video_id=video_id,
            filename=safe_filename,
            size=filepath.stat().st_size
        )
        
    except ShuttleSenseException:
        raise
    except Exception as e:
        cleanup_file(filepath)
        logger.error(f"Upload failed: {e}")
        raise VideoProcessingError(f"Failed to save video: {str(e)}", stage="upload")


# =============================================================================
# Analyze Endpoint
# =============================================================================

@app.post("/api/analyze", tags=["Analysis"])
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def analyze_video(
    request: Request,
    video_id: str = Form(...),
    drill_type: str = Form("unknown")
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """
    Analyze an uploaded video.
    
    - **video_id**: ID from upload response
    - **drill_type**: Type of drill (footwork, overhead-shadow, etc.)
    """
    # Find video file
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise VideoNotFound(video_id)
    
    video_path = video_files[0]
    
    try:
        result = pipeline.analyze(
            video_path=str(video_path),
            drill_type=drill_type,
            session_id=video_id,
            save_poses=True
        )
        
        if "error" in result:
            raise InsufficientPoseData(result["error"])
        
        logger.info(f"Analysis complete: {video_id}")
        return result
        
    except ShuttleSenseException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {video_id}: {e}", exc_info=True)
        raise VideoProcessingError(f"Analysis failed: {str(e)}", stage="analysis")


# =============================================================================
# Combined Upload + Analyze Endpoint
# =============================================================================

@app.post("/api/upload-and-analyze", tags=["Analysis"])
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def upload_and_analyze(
    request: Request,
    file: UploadFile = File(...),
    drill_type: str = Form("unknown")
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """
    Upload and analyze in one step.
    
    - **file**: Video file (MP4, MOV, AVI, WebM)
    - **drill_type**: Type of drill being performed
    """
    # Validate and upload
    safe_filename = validate_video_file(file)
    video_id = safe_filename.split(".")[0]
    filepath = UPLOAD_DIR / safe_filename
    
    try:
        # Save file
        with open(filepath, "wb") as buffer:
            content = await file.read()
            
            # Final size check
            if len(content) > settings.max_video_size_bytes:
                raise FileTooLarge(
                    file_size_mb=len(content) / (1024 * 1024),
                    max_size_mb=settings.MAX_VIDEO_SIZE_MB
                )
            
            buffer.write(content)
        
        logger.info(f"Video uploaded for analysis: {video_id}")
        
        # Analyze
        result = pipeline.analyze(
            video_path=str(filepath),
            drill_type=drill_type,
            session_id=video_id,
            save_poses=True
        )
        
        if "error" in result:
            raise InsufficientPoseData(result["error"])
        
        logger.info(f"Analysis complete: {video_id}")
        return result
        
    except ShuttleSenseException:
        raise
    except Exception as e:
        logger.error(f"Upload and analyze failed: {e}", exc_info=True)
        raise VideoProcessingError(f"Analysis failed: {str(e)}", stage="upload_and_analyze")


# =============================================================================
# Report Endpoint
# =============================================================================

@app.get("/api/report/{session_id}", tags=["Analysis"])
async def get_report(
    session_id: str
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """
    Get analysis report for a session.
    
    - **session_id**: Session ID from analysis
    """
    session = pipeline.get_session(session_id)
    if not session:
        raise SessionNotFound(session_id)
    
    return session


# =============================================================================
# Chat Endpoint
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    chat_request: ChatRequest
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """
    Ask a question about a session (grounded in evidence).
    
    - **session_id**: Session to query
    - **question**: Question to ask the AI coach
    - **include_debug**: Include debug info in response (similarity scores, etc.)
    """
    session = pipeline.get_session(chat_request.session_id)
    if not session:
        raise SessionNotFound(chat_request.session_id)
    
    chunks = session.get("evidence_chunks", [])
    if not chunks:
        raise VideoProcessingError(
            "No evidence available for this session",
            stage="chat"
        )
    
    try:
        result = chat_engine.chat(
            question=chat_request.question,
            evidence_chunks=chunks,
            session_id=chat_request.session_id,
            session_summary=session.get("report", {}).get("metrics_summary"),
            include_debug=chat_request.include_debug
        )
        
        # Build response with proper models
        citations = [
            Citation(timestamp=c.get("timestamp", 0), type=c.get("type", "evidence"))
            for c in result.get("citations", [])
        ]
        
        debug = None
        if result.get("debug"):
            d = result["debug"]
            debug = ChatDebug(
                top_similarity=d.get("top_similarity", 0),
                selected_chunk_types=d.get("selected_chunk_types", []),
                retrieval_count=d.get("retrieval_count", 0),
                llm_model=d.get("llm_model", "")
            )
        
        missing_evidence = None
        if result.get("missing_evidence"):
            me = result["missing_evidence"]
            missing_evidence = MissingEvidence(
                reason=me.get("reason", ""),
                max_similarity=me.get("max_similarity", 0),
                threshold=me.get("threshold", 0),
                suggested_questions=me.get("suggested_questions", [])
            )
        
        return ChatResponse(
            answer=result["answer"],
            citations=citations,
            grounded=result["grounded"],
            debug=debug,
            missing_evidence=missing_evidence
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise VideoProcessingError(f"Chat failed: {str(e)}", stage="chat")


# =============================================================================
# Streaming Chat Endpoint
# =============================================================================

@app.post("/api/chat/stream", tags=["Chat"])
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat_stream(
    request: Request,
    chat_request: ChatRequest
):
    """
    Stream chat response - returns text as it's generated.
    Uses Server-Sent Events (SSE) format.
    """
    session = pipeline.get_session(chat_request.session_id)
    if not session:
        raise SessionNotFound(chat_request.session_id)
    
    chunks = session.get("evidence_chunks", [])
    if not chunks:
        raise VideoProcessingError(
            "No evidence available for this session",
            stage="chat"
        )
    
    def generate():
        try:
            for text_chunk in chat_engine.chat_stream(
                question=chat_request.question,
                evidence_chunks=chunks,
                session_id=chat_request.session_id,
                session_summary=session.get("report", {}).get("metrics_summary")
            ):
                # SSE format: data: <content>\n\n
                yield f"data: {text_chunk}\n\n"
            
            # Signal end of stream
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Stream chat failed: {e}")
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# =============================================================================
# Session Management Endpoints
# =============================================================================

@app.get("/api/sessions", tags=["Sessions"])
async def list_sessions(
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """List all analysis sessions."""
    return {"sessions": pipeline.list_sessions()}


@app.delete("/api/session/{session_id}", tags=["Sessions"])
async def delete_session(
    session_id: str
    # Uncomment to require auth: user: UserInfo = Depends(get_current_user)
):
    """
    Delete a session and its data.
    
    - **session_id**: Session to delete
    """
    session_dir = Path(settings.SESSIONS_DIR) / session_id
    
    # Delete session data
    if session_dir.exists():
        shutil.rmtree(session_dir)
    
    # Delete video if exists
    for f in UPLOAD_DIR.glob(f"{session_id}.*"):
        f.unlink()
    
    logger.info(f"Session deleted: {session_id}")
    return {"deleted": session_id}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
