# ShuttleSense Deployment Guide

This guide covers deploying ShuttleSense to various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Environment Variables](#environment-variables)
6. [Health Checks](#health-checks)
7. [Monitoring](#monitoring)
8. [Rollback Procedure](#rollback-procedure)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **CPU**: 2+ cores recommended (MediaPipe is CPU-intensive)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 10GB for application, additional for video storage
- **OS**: Linux (Ubuntu 22.04+), macOS, or Windows with WSL2

### Software Requirements

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (for containerized deployment)
- Git

---

## Local Development

### Backend Setup

```bash
# Clone repository
git clone <repository-url>
cd BADMINTON_AI_COACH

# Create virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and set your configuration
# At minimum, set GOOGLE_API_KEY for AI chat functionality

# Start development server
python main.py
# or with hot reload:
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access Points

- **Frontend**: http://localhost:5173 (Vite dev server)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## Docker Deployment

### Quick Start

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Points

- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000

### Configuration

Create a `.env` file in the project root:

```env
# Required for production
JWT_SECRET_KEY=your-secure-random-key-here
GOOGLE_API_KEY=your-gemini-api-key

# Optional overrides
ENVIRONMENT=production
DEBUG=false
RATE_LIMIT_ENABLED=true
```

### Building Individual Images

```bash
# Backend
docker build -t shuttlesense-backend:latest ./backend

# Frontend
docker build -t shuttlesense-frontend:latest ./frontend
```

---

## Production Deployment

### Pre-deployment Checklist

- [ ] Generate secure JWT_SECRET_KEY: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- [ ] Set ENVIRONMENT=production
- [ ] Set DEBUG=false
- [ ] Configure proper CORS_ORIGINS
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Set up database backups (if using PostgreSQL)
- [ ] Configure SSL/TLS certificates
- [ ] Set up rate limiting storage (Redis for production)

### Kubernetes Deployment

```yaml
# Example deployment (see k8s/ directory for full configs)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shuttlesense-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shuttlesense-backend
  template:
    spec:
      containers:
      - name: backend
        image: shuttlesense-backend:latest
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
```

### Cloud Platform Guides

#### AWS ECS

1. Push images to ECR
2. Create ECS cluster
3. Define task definitions
4. Create services with ALB

#### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/shuttlesense-backend ./backend
gcloud builds submit --tag gcr.io/PROJECT_ID/shuttlesense-frontend ./frontend

# Deploy
gcloud run deploy shuttlesense-backend \
  --image gcr.io/PROJECT_ID/shuttlesense-backend \
  --platform managed \
  --memory 4Gi \
  --cpu 2
```

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `JWT_SECRET_KEY` | Secret for JWT signing | `abc123...` (32+ chars) |
| `GOOGLE_API_KEY` | Gemini API key for chat | `AIza...` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` |
| `DEBUG` | `false` | Enable debug mode |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `MAX_VIDEO_SIZE_MB` | `500` | Max upload size |

---

## Health Checks

### Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Basic health | `{"status": "healthy"}` |
| `/health/live` | Liveness probe | `{"status": "alive"}` |
| `/health/ready` | Readiness probe | `{"status": "ready"}` |
| `/metrics` | System metrics | JSON with CPU, memory, disk stats |

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## Monitoring

### Metrics to Watch

1. **API Latency**: P50, P95, P99 response times
2. **Error Rate**: 4xx and 5xx responses per minute
3. **Upload Volume**: Videos uploaded per hour
4. **Analysis Duration**: Time to complete video analysis
5. **Disk Usage**: Upload directory size
6. **Memory Usage**: Process memory consumption

### Log Aggregation

In production, logs are output in JSON format. Configure your log aggregator to parse:

```json
{
  "timestamp": "2024-01-06T10:00:00Z",
  "level": "INFO",
  "logger": "main",
  "message": "Request completed",
  "correlation_id": "abc123",
  "duration_ms": 150
}
```

### Alerting Rules

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Error rate | > 5% for 5 min | Critical |
| P99 latency | > 30s for 5 min | Warning |
| Disk usage | > 80% | Warning |
| Memory usage | > 90% | Critical |

---

## Rollback Procedure

### Docker Rollback

```bash
# List previous images
docker images shuttlesense-backend

# Rollback to specific version
docker-compose down
docker tag shuttlesense-backend:previous shuttlesense-backend:latest
docker-compose up -d
```

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/shuttlesense-backend

# Rollback to previous version
kubectl rollout undo deployment/shuttlesense-backend

# Rollback to specific revision
kubectl rollout undo deployment/shuttlesense-backend --to-revision=2
```

---

## Troubleshooting

### Common Issues

#### 1. MediaPipe Import Error

```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution**: Install OpenGL dependencies:
```bash
apt-get install -y libgl1-mesa-glx
```

#### 2. Video Processing Timeout

**Symptoms**: Analysis takes too long or times out

**Solutions**:
- Increase timeout in nginx config
- Check video codec compatibility
- Reduce video resolution/length

#### 3. Rate Limit Exceeded

**Symptoms**: 429 Too Many Requests

**Solutions**:
- Wait for rate limit window to reset
- Increase rate limits in configuration
- Use authenticated requests for higher limits

#### 4. Disk Space Full

**Symptoms**: Upload failures, 507 errors

**Solutions**:
- Clean old sessions: `DELETE /api/session/{id}`
- Increase disk quota
- Enable automatic cleanup

### Debug Mode

Enable debug mode for detailed error messages:

```env
DEBUG=true
ENVIRONMENT=development
```

### Logs

```bash
# View backend logs
docker-compose logs -f backend

# View specific error
docker-compose logs backend | grep ERROR
```
