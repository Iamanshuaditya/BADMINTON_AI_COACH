# ShuttleSense - AI Badminton Footwork Coach

Real-time badminton footwork and stroke analysis from video using pose estimation and AI-powered coaching.

![Version](https://img.shields.io/badge/version-2.0.0-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

- **Video Analysis**: Upload badminton practice videos for AI-powered analysis
- **Pose Detection**: MediaPipe-based pose extraction with keypoint smoothing
- **Footwork Analysis**: Detect split steps, lunges, recovery patterns
- **Stroke Analysis**: Overhead stroke mechanics evaluation
- **Mistake Detection**: Identify common issues (knee collapse, poor stance, etc.)
- **Fix-First Plan**: Prioritized improvement recommendations
- **AI Coach Chat**: Ask questions about your session, grounded in evidence

## 🏗️ Project Structure

```
BADMINTON_AI_COACH/
├── backend/                    # FastAPI backend
│   ├── auth/                   # JWT authentication
│   ├── config/                 # Settings & thresholds
│   ├── core/                   # Processing pipeline
│   │   ├── pose_extractor.py   # MediaPipe pose extraction
│   │   ├── feature_computer.py # Biomechanical features
│   │   ├── event_fsm.py        # Event detection FSM
│   │   ├── mistake_detector.py # Mistake detection rules
│   │   ├── stroke_analyzer.py  # Overhead stroke analysis
│   │   ├── report_generator.py # Coach report generation
│   │   ├── grounded_chat.py    # RAG-based chat
│   │   └── pipeline.py         # Full analysis orchestration
│   ├── middleware/             # Error handling, rate limiting
│   ├── tests/                  # Unit & integration tests
│   ├── main.py                 # FastAPI app
│   ├── Dockerfile              # Production Docker image
│   └── requirements.txt        # Python dependencies
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx             # Main application
│   │   └── App.css             # Styling
│   ├── Dockerfile              # Production Docker image
│   └── nginx.conf              # Nginx configuration
├── docs/                       # Documentation
│   ├── API.md                  # API reference
│   └── DEPLOYMENT.md           # Deployment guide
├── .github/workflows/          # CI/CD pipelines
└── docker-compose.yml          # Docker orchestration
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd BADMINTON_AI_COACH

# Copy environment file
cp backend/.env.example backend/.env
# Edit .env and set GOOGLE_API_KEY (optional, for AI chat)

# Build and run
docker-compose up --build

# Access the app at http://localhost:80
```

### Option 2: Local Development

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start server
python main.py
# or with hot reload:
uvicorn main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### Access Points

| Service | Development | Docker |
|---------|-------------|--------|
| Frontend | http://localhost:5173 | http://localhost:80 |
| Backend API | http://localhost:8000 | http://localhost:8000 |
| API Docs | http://localhost:8000/docs | (disabled in prod) |

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload video file |
| `/api/analyze` | POST | Analyze uploaded video |
| `/api/upload-and-analyze` | POST | Upload + analyze in one step |
| `/api/report/{session_id}` | GET | Get session report |
| `/api/chat` | POST | Ask grounded questions |
| `/api/sessions` | GET | List all sessions |
| `/api/session/{session_id}` | DELETE | Delete a session |
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness probe |
| `/metrics` | GET | System metrics |

See [docs/API.md](docs/API.md) for full API documentation.

## 🎯 Coach Report Schema

```json
{
  "session_id": "abc12345",
  "video_duration": 45.2,
  "fps": 30,
  "drill_type": "6-corner-shadow",
  "events": [
    {"type": "split_step", "timestamp": 5.3, "confidence": 0.92}
  ],
  "mistakes": [
    {"type": "knee_collapse", "timestamp": 12.1, "severity": 0.7, "cue": "Knee out"}
  ],
  "top_mistakes": [
    {"type": "knee_collapse", "count": 3, "cue": "Knee out"}
  ],
  "fix_first_plan": {
    "primary_issue": "knee_collapse",
    "cue": "Knee out",
    "focus_drill": "Single-leg squats - knee over toes"
  }
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | No | Gemini API for AI chat |
| `JWT_SECRET_KEY` | Yes (prod) | Secret for JWT auth |
| `ENVIRONMENT` | No | `development` or `production` |
| `CORS_ORIGINS` | No | Allowed origins (comma-separated) |
| `RATE_LIMIT_ENABLED` | No | Enable rate limiting (default: true) |

### Thresholds

All analysis thresholds are in `backend/config/thresholds.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_step.window_ms` | 400 | Split step detection window |
| `stance.min_ratio` | 0.7 | Minimum stance width ratio |
| `knee.valgus_threshold` | 8.0 | Knee collapse angle threshold |

## 🧪 Testing

```bash
cd backend

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov-report=html

# Integration tests only
pytest tests/test_integration.py -v
```

## 🐳 Docker

```bash
# Build images
docker-compose build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## 📹 Video Requirements

- **Format**: MP4, MOV, WebM
- **Max Size**: 500MB
- **Camera Angle**: Behind or 45° back-side (preferred)
- **Visibility**: Full body (head to feet)
- **Setup**: Phone on tripod, static camera
- **Frame Rate**: 30 FPS or higher

## 🔒 Security Features

- **JWT Authentication** (optional, easily enabled)
- **Rate Limiting**: Per-IP and per-user limits
- **CORS Configuration**: Configurable allowed origins
- **File Validation**: Size limits, MIME type checking
- **Error Handling**: Structured error responses

## 📊 Monitoring

- **Health Endpoints**: `/health`, `/health/live`, `/health/ready`
- **Metrics**: `/metrics` for CPU, memory, disk usage
- **Structured Logging**: JSON format for log aggregation
- **Correlation IDs**: Request tracing support

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Pose Detection**: MediaPipe Pose
- **Computer Vision**: OpenCV
- **AI Chat**: Google Gemini API
- **Auth**: python-jose (JWT)

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: CSS with CSS Variables

### Infrastructure
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **CI/CD**: GitHub Actions
- **Web Server**: Nginx (production)

## 📖 Documentation

- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🔄 Development Workflow

1. Create feature branch from `develop`
2. Make changes with tests
3. Run `pytest` and fix any failures
4. Create PR to `develop`
5. After review, merge to `develop`
6. Release: merge `develop` to `main`

## 📄 License

MIT License - see LICENSE file for details.

---

Built with ❤️ for badminton improvement | v2.0.0
