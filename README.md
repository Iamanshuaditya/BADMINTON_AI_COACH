# ShuttleSense - AI Badminton Footwork Coach

Real-time badminton footwork analysis from video using pose estimation and rules-based coaching.

## 🏗️ Project Structure

```
BADMINTON_AI_COACH/
├── backend/                 # FastAPI backend
│   ├── config/             # Configurable thresholds
│   │   └── thresholds.py   # All tunable parameters
│   ├── core/               # Processing pipeline
│   │   ├── one_euro_filter.py   # Keypoint smoothing
│   │   ├── pose_extractor.py    # MediaPipe pose extraction
│   │   ├── feature_computer.py  # Biomechanical features
│   │   ├── event_fsm.py         # Event detection FSM
│   │   ├── mistake_detector.py  # Mistake detection rules
│   │   ├── report_generator.py  # Coach report generation
│   │   ├── grounded_chat.py     # RAG-based chat
│   │   └── pipeline.py          # Full analysis orchestration
│   ├── tests/              # Unit tests
│   ├── data/               # Session data (created on run)
│   │   ├── uploads/        # Uploaded videos
│   │   └── sessions/       # Analysis results
│   ├── main.py             # FastAPI app
│   └── requirements.txt    # Python dependencies
├── frontend/               # React + Vite frontend
│   └── src/
│       ├── App.jsx         # Main application
│       └── App.css         # Styling
└── docs/                   # Research & planning
```

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set Gemini API key for chat
export GOOGLE_API_KEY=your_api_key_here

# Start server
python main.py
# or
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### 3. Use the App

1. Open http://localhost:3000
2. Upload a badminton footwork video
3. Select drill type
4. Click "Analyze Video"
5. View report and chat with the coach

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload video file |
| `/api/analyze` | POST | Analyze uploaded video |
| `/api/upload-and-analyze` | POST | Upload + analyze in one step |
| `/api/report/{session_id}` | GET | Get session report |
| `/api/chat` | POST | Ask grounded questions |
| `/api/sessions` | GET | List all sessions |

## 🎯 Coach Report JSON Schema

```json
{
  "session_id": "abc12345",
  "created_at": "2026-01-04T15:30:00Z",
  "video_duration": 45.2,
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
  },
  "confidence_notes": []
}
```

## ⚙️ Configuration

All thresholds are in `backend/config/thresholds.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_step.window_ms` | 400 | Split step detection window |
| `stance.min_ratio` | 0.7 | Minimum stance width ratio |
| `knee.valgus_threshold` | 8.0 | Knee collapse angle threshold |
| `recovery.max_time_sec` | 2.0 | Max recovery time to base |
| `fsm.cue_cooldown_sec` | 1.5 | Cooldown between same cue |

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

## 📹 Video Requirements

- **Format**: MP4, MOV, WebM
- **Camera angle**: Behind or 45° back-side (preferred)
- **Visibility**: Full body (head to feet)
- **Setup**: Phone on tripod, static camera
- **Frame rate**: 30 FPS or higher

## 🔒 Privacy

- By default, only pose JSON and reports are stored
- Raw video is optional (for playback only)
- All processing is local (no cloud required for V1)
- Delete sessions via API or delete `data/sessions/` folder

## 🛠️ Tech Stack

- **Backend**: FastAPI, MediaPipe Pose, OpenCV
- **Frontend**: React, Vite
- **Smoothing**: OneEuro Filter
- **Chat**: Gemini API (optional) or stub mode

---

Built with ❤️ for badminton improvement
