# ShuttleSense API Documentation

## Overview

ShuttleSense is an AI-powered badminton footwork and stroke analysis platform. This document describes the REST API endpoints.

**Base URL**: `http://localhost:8000` (development) or your production URL

**API Version**: 2.0.0

---

## Authentication

> **Note**: Authentication is currently optional but recommended for production use.

When enabled, all protected endpoints require a JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Auth Endpoints (When Enabled)

#### POST /api/auth/login
```json
{
  "email": "user@example.com",
  "password": "your-password"
}
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

---

## Endpoints

### Health & Status

#### GET /
Root endpoint - service information.

**Response 200**:
```json
{
  "status": "ok",
  "service": "ShuttleSense Video Coach API",
  "version": "2.0.0"
}
```

#### GET /health
Basic health check.

**Response 200**:
```json
{
  "status": "healthy"
}
```

#### GET /health/ready
Readiness probe for load balancers.

**Response 200**:
```json
{
  "status": "ready"
}
```

**Response 503** (Service Not Ready):
```json
{
  "status": "not_ready",
  "reason": "low_disk_space"
}
```

#### GET /metrics
System metrics for monitoring.

**Response 200**:
```json
{
  "uptime_seconds": 1234567890,
  "memory_mb": 512.5,
  "cpu_percent": 25.0,
  "disk_free_gb": 50.5,
  "disk_used_gb": 10.2,
  "sessions_count": 42,
  "upload_dir_size_mb": 1500.0
}
```

---

### Video Upload

#### POST /api/upload
Upload a video file for later analysis.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` - Video file (MP4, MOV, AVI, WebM)
- Max Size: 500MB

**Response 200**:
```json
{
  "video_id": "a1b2c3d4",
  "filename": "a1b2c3d4.mp4",
  "size": 15000000
}
```

**Response 413** (File Too Large):
```json
{
  "error": "FILE_TOO_LARGE",
  "detail": "File too large: 600.0MB (max 500MB)",
  "path": "/api/upload"
}
```

**Response 415** (Invalid Format):
```json
{
  "error": "INVALID_VIDEO_FORMAT",
  "detail": "Invalid video format: image/jpeg. Allowed: video/mp4, video/quicktime, video/x-msvideo, video/webm",
  "path": "/api/upload"
}
```

---

### Video Analysis

#### POST /api/analyze
Analyze a previously uploaded video.

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `video_id` (required): ID from upload response
  - `drill_type` (optional): Type of drill

**Drill Types**:
| Value | Description |
|-------|-------------|
| `unknown` | Auto-detect (default) |
| `6-corner-shadow` | 6-corner footwork drill |
| `side-to-side` | Side-to-side defensive |
| `front-back` | Front-back movement |
| `overhead-shadow` | Overhead stroke shadow |
| `overhead-clear` | Clear shadow practice |
| `overhead-smash` | Smash shadow practice |

**Response 200**:
```json
{
  "session_id": "a1b2c3d4",
  "success": true,
  "processing_time_sec": 15.5,
  "report": {
    "session_id": "a1b2c3d4",
    "video_duration": 30.0,
    "fps": 30,
    "drill_type": "6-corner-shadow",
    "events": [...],
    "mistakes": [...],
    "top_mistakes": [...],
    "fix_first_plan": {...},
    "metrics_summary": {...}
  },
  "evidence_chunks": [...]
}
```

**Response 404** (Video Not Found):
```json
{
  "error": "VIDEO_NOT_FOUND",
  "detail": "Video not found: xyz123",
  "path": "/api/analyze"
}
```

#### POST /api/upload-and-analyze
Combined upload and analysis in one request.

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): Video file
  - `drill_type` (optional): Type of drill

**Response**: Same as `/api/analyze`

---

### Reports

#### GET /api/report/{session_id}
Get analysis report for a session.

**Response 200**:
```json
{
  "session_id": "a1b2c3d4",
  "report": {
    "session_id": "a1b2c3d4",
    "created_at": "2024-01-06T10:00:00Z",
    "video_duration": 30.0,
    "fps": 30,
    "resolution": {"width": 1920, "height": 1080},
    "drill_type": "6-corner-shadow",
    "events": [
      {
        "type": "split_step",
        "timestamp": 5.3,
        "end_timestamp": 5.7,
        "duration": 0.4,
        "confidence": 0.92
      }
    ],
    "mistakes": [
      {
        "type": "knee_collapse",
        "timestamp": 12.1,
        "duration": 0.8,
        "severity": 0.7,
        "confidence": 0.85,
        "cue": "Knee out",
        "description": "Knee collapsing inward during lunge",
        "evidence": [11.9, 12.1, 12.5]
      }
    ],
    "top_mistakes": [
      {"type": "knee_collapse", "count": 3, "cue": "Knee out"}
    ],
    "fix_first_plan": {
      "primary_issue": "knee_collapse",
      "occurrences": 3,
      "cue": "Knee out",
      "focus_drill": "Single-leg squats - knee over toes",
      "key_timestamps": [12.1, 24.5, 38.2]
    },
    "metrics_summary": {
      "avg_stance_width_ratio": 1.05,
      "avg_pose_confidence": 0.87,
      "total_events": 15,
      "total_mistakes": 5
    }
  },
  "evidence_chunks": [...]
}
```

**Response 404**:
```json
{
  "error": "SESSION_NOT_FOUND",
  "detail": "Session not found: xyz123",
  "path": "/api/report/xyz123"
}
```

---

### AI Coach Chat

#### POST /api/chat
Ask the AI coach questions about a session.

**Request**:
```json
{
  "session_id": "a1b2c3d4",
  "question": "What was my biggest issue?"
}
```

**Response 200**:
```json
{
  "answer": "Your biggest issue was knee collapse, which occurred 3 times during lunges. Focus on keeping your knee aligned with your toes during lateral movements.",
  "citations": ["mistake:knee_collapse:12.1", "mistake:knee_collapse:24.5"],
  "grounded": true
}
```

---

### Session Management

#### GET /api/sessions
List all analysis sessions.

**Response 200**:
```json
{
  "sessions": ["a1b2c3d4", "e5f6g7h8", "i9j0k1l2"]
}
```

#### DELETE /api/session/{session_id}
Delete a session and its data.

**Response 200**:
```json
{
  "deleted": "a1b2c3d4"
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": "ERROR_CODE",
  "detail": "Human-readable message",
  "path": "/api/endpoint",
  "details": {}
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | No auth token provided |
| `TOKEN_EXPIRED` | 401 | JWT token expired |
| `INVALID_TOKEN` | 401 | JWT token invalid |
| `PERMISSION_DENIED` | 403 | User lacks permission |
| `SESSION_NOT_FOUND` | 404 | Session doesn't exist |
| `VIDEO_NOT_FOUND` | 404 | Video file not found |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `INVALID_VIDEO_FORMAT` | 415 | Unsupported video format |
| `FILE_TOO_LARGE` | 413 | Upload exceeds size limit |
| `INSUFFICIENT_POSE_DATA` | 422 | Not enough poses detected |
| `VIDEO_PROCESSING_ERROR` | 422 | Video processing failed |
| `POSE_EXTRACTION_ERROR` | 422 | Pose extraction failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INSUFFICIENT_DISK_SPACE` | 507 | Server disk full |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/upload` | 10 requests/hour |
| `/api/analyze` | 5 requests/hour |
| `/api/upload-and-analyze` | 5 requests/hour |
| `/api/chat` | 30 requests/hour |
| Global | 100 requests/hour per IP |

Rate limit info is returned in response headers:
```
X-RateLimit-Limit: 10/hour
Retry-After: 3600
```

---

## Correlation IDs

All requests are assigned a correlation ID for tracing. It's returned in response headers:

```
X-Correlation-ID: abc12345
X-Process-Time-Ms: 150.25
```

You can also provide your own correlation ID in the request:
```
X-Correlation-ID: your-trace-id
```
