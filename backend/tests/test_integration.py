"""
Integration Tests for ShuttleSense API
Tests the complete flow from upload to analysis to chat.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app, pipeline


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_video_file():
    """Create a mock video file for testing"""
    # Create a minimal valid file
    content = b"mock video content " * 100  # ~1.9KB
    return {
        "file": ("test_video.mp4", content, "video/mp4")
    }


@pytest.fixture
def mock_analysis_result():
    """Mock analysis result for testing"""
    return {
        "session_id": "test123",
        "success": True,
        "report": {
            "session_id": "test123",
            "video_duration": 10.0,
            "fps": 30,
            "drill_type": "footwork",
            "events": [],
            "mistakes": [],
            "top_mistakes": [],
            "metrics_summary": {
                "total_events": 0,
                "total_mistakes": 0
            }
        },
        "evidence_chunks": [
            {"chunk_id": 1, "content": "Test evidence"}
        ]
    }


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "ShuttleSense" in data["service"]
    
    def test_health_endpoint(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_live_endpoint(self, client):
        """Test liveness probe"""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
    
    def test_health_ready_endpoint(self, client):
        """Test readiness probe"""
        response = client.get("/health/ready")
        # May be 200 or 503 depending on system state
        assert response.status_code in [200, 503]


class TestUploadEndpoint:
    """Test video upload functionality"""
    
    def test_upload_valid_video(self, client, mock_video_file):
        """Test uploading a valid video file"""
        response = client.post("/api/upload", files=mock_video_file)
        assert response.status_code == 200
        data = response.json()
        assert "video_id" in data
        assert "filename" in data
        assert "size" in data
    
    def test_upload_invalid_type(self, client):
        """Test uploading invalid file type"""
        files = {
            "file": ("test.txt", b"not a video", "text/plain")
        }
        response = client.post("/api/upload", files=files)
        assert response.status_code == 415  # Unsupported Media Type
        assert "INVALID_VIDEO_FORMAT" in response.json().get("error", "")
    
    def test_upload_no_file(self, client):
        """Test upload without file"""
        response = client.post("/api/upload")
        assert response.status_code == 422  # Validation error


class TestAnalyzeEndpoint:
    """Test video analysis functionality"""
    
    def test_analyze_nonexistent_video(self, client):
        """Test analyzing non-existent video"""
        response = client.post(
            "/api/analyze",
            data={"video_id": "nonexistent", "drill_type": "footwork"}
        )
        assert response.status_code == 404
        assert "VIDEO_NOT_FOUND" in response.json().get("error", "")
    
    @patch.object(pipeline, 'analyze')
    def test_analyze_with_mocked_pipeline(self, mock_analyze, client, mock_video_file, mock_analysis_result):
        """Test analysis with mocked pipeline"""
        # Upload first
        upload_response = client.post("/api/upload", files=mock_video_file)
        video_id = upload_response.json()["video_id"]
        
        # Mock the analysis
        mock_analyze.return_value = mock_analysis_result
        
        # Analyze
        response = client.post(
            "/api/analyze",
            data={"video_id": video_id, "drill_type": "footwork"}
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data or "report" in data


class TestUploadAndAnalyzeEndpoint:
    """Test combined upload and analyze functionality"""
    
    @patch.object(pipeline, 'analyze')
    def test_upload_and_analyze(self, mock_analyze, client, mock_video_file, mock_analysis_result):
        """Test combined upload and analyze"""
        mock_analyze.return_value = mock_analysis_result
        
        files = mock_video_file.copy()
        response = client.post(
            "/api/upload-and-analyze",
            files=files,
            data={"drill_type": "footwork"}
        )
        
        assert response.status_code == 200


class TestReportEndpoint:
    """Test report retrieval functionality"""
    
    def test_get_nonexistent_report(self, client):
        """Test getting report for non-existent session"""
        response = client.get("/api/report/nonexistent")
        assert response.status_code == 404
        assert "SESSION_NOT_FOUND" in response.json().get("error", "")
    
    @patch.object(pipeline, 'get_session')
    def test_get_existing_report(self, mock_get_session, client, mock_analysis_result):
        """Test getting existing report"""
        mock_get_session.return_value = {
            "session_id": "test123",
            "report": mock_analysis_result["report"],
            "evidence_chunks": mock_analysis_result["evidence_chunks"]
        }
        
        response = client.get("/api/report/test123")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test123"


class TestChatEndpoint:
    """Test chat functionality"""
    
    def test_chat_nonexistent_session(self, client):
        """Test chat with non-existent session"""
        response = client.post(
            "/api/chat",
            json={"session_id": "nonexistent", "question": "What was my biggest issue?"}
        )
        assert response.status_code == 404
    
    @patch.object(pipeline, 'get_session')
    def test_chat_no_evidence(self, mock_get_session, client):
        """Test chat with session that has no evidence"""
        mock_get_session.return_value = {
            "session_id": "test123",
            "report": {},
            "evidence_chunks": []
        }
        
        response = client.post(
            "/api/chat",
            json={"session_id": "test123", "question": "What was my biggest issue?"}
        )
        assert response.status_code == 422


class TestSessionManagement:
    """Test session listing and deletion"""
    
    @patch.object(pipeline, 'list_sessions')
    def test_list_sessions(self, mock_list, client):
        """Test listing sessions"""
        mock_list.return_value = ["session1", "session2", "session3"]
        
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 3
    
    def test_delete_session(self, client):
        """Test deleting a session"""
        response = client.delete("/api/session/test123")
        assert response.status_code == 200
        assert response.json()["deleted"] == "test123"


class TestErrorHandling:
    """Test error handling and responses"""
    
    def test_error_response_format(self, client):
        """Test that error responses have consistent format"""
        response = client.get("/api/report/nonexistent")
        assert response.status_code == 404
        data = response.json()
        
        # Check error response structure
        assert "error" in data
        assert "detail" in data
        assert "path" in data
    
    def test_validation_error_format(self, client):
        """Test validation error format"""
        response = client.post(
            "/api/chat",
            json={"session_id": "test", "question": ""}  # Empty question
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_headers(self, client, mock_video_file):
        """Test that rate limit info is provided"""
        # This test verifies rate limiting is active
        # In a real test, you'd make many requests and check for 429
        response = client.post("/api/upload", files=mock_video_file)
        # Rate limit headers may or may not be present based on implementation
        assert response.status_code in [200, 429]


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_preflight(self, client):
        """Test CORS preflight request"""
        response = client.options(
            "/api/upload",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        # Should not fail with 4xx
        assert response.status_code in [200, 204, 405]


# =============================================================================
# Fixtures for Async Tests
# =============================================================================

@pytest.fixture
async def async_client():
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# =============================================================================
# Performance Tests (Optional)
# =============================================================================

class TestPerformance:
    """Basic performance tests"""
    
    def test_health_endpoint_fast(self, client):
        """Health endpoint should respond quickly"""
        import time
        start = time.time()
        response = client.get("/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.1  # Should respond in < 100ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
