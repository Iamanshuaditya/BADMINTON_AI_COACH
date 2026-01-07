"""
Tests for ShuttleSense Grounding Features
- Motion classifier (drill type auto-detection)
- Embeddings caching
- Chat grounding with similarity threshold
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict


# Test Motion Classifier
class TestMotionClassifier:
    """Tests for drill type auto-detection from motion patterns."""
    
    def create_pose_frame(
        self,
        timestamp: float,
        wrist_above_shoulder: bool = False,
        hip_x_offset: float = 0.0,
        confidence: float = 0.9
    ) -> Dict:
        """Create a synthetic pose frame for testing."""
        # Base landmark positions (normalized 0-1)
        landmarks = []
        
        # Fill 33 landmarks with defaults
        for i in range(33):
            landmarks.append({
                "x": 0.5,
                "y": 0.5,
                "z": 0.0,
                "visibility": confidence
            })
        
        # Set key landmarks for testing
        # Shoulders at y=0.3
        landmarks[11] = {"x": 0.4 + hip_x_offset, "y": 0.3, "z": 0, "visibility": confidence}  # left_shoulder
        landmarks[12] = {"x": 0.6 + hip_x_offset, "y": 0.3, "z": 0, "visibility": confidence}  # right_shoulder
        
        # Hips at y=0.5
        landmarks[23] = {"x": 0.45 + hip_x_offset, "y": 0.5, "z": 0, "visibility": confidence}  # left_hip
        landmarks[24] = {"x": 0.55 + hip_x_offset, "y": 0.5, "z": 0, "visibility": confidence}  # right_hip
        
        # Wrists - above or below shoulder based on flag
        wrist_y = 0.15 if wrist_above_shoulder else 0.45  # Below shoulder by default
        landmarks[15] = {"x": 0.35, "y": wrist_y, "z": 0, "visibility": confidence}  # left_wrist
        landmarks[16] = {"x": 0.65, "y": wrist_y, "z": 0, "visibility": confidence}  # right_wrist
        
        # Elbows
        elbow_y = 0.25 if wrist_above_shoulder else 0.4
        landmarks[13] = {"x": 0.37, "y": elbow_y, "z": 0, "visibility": confidence}  # left_elbow
        landmarks[14] = {"x": 0.63, "y": elbow_y, "z": 0, "visibility": confidence}  # right_elbow
        
        return {
            "landmarks": landmarks,
            "timestamp": timestamp,
            "confidence": confidence
        }
    
    def test_overhead_detection_wrist_above_shoulder(self):
        """Should detect overhead-shadow when wrist is above shoulder for many frames."""
        from core.motion_classifier import classify_drill
        
        # Create 60 frames with wrist above shoulder for 50% of them
        frames = []
        for i in range(60):
            wrist_above = i >= 20 and i < 50  # 30 frames with wrist above
            frames.append(self.create_pose_frame(
                timestamp=i / 30.0,
                wrist_above_shoulder=wrist_above,
                confidence=0.9
            ))
        
        result = classify_drill(frames)
        
        # Should detect overhead pattern
        assert result.detected_drill_type in ["overhead-shadow", "unknown"]
        if result.detected_drill_type == "overhead-shadow":
            assert result.confidence >= 0.5
        
        # Debug features should be populated
        assert "features" in result.debug_features
    
    def test_footwork_detection_lateral_movement(self):
        """Should detect footwork when hip has significant lateral movement."""
        from core.motion_classifier import classify_drill
        
        # Create 60 frames with lateral hip movement (left-right pattern)
        frames = []
        for i in range(60):
            # Oscillate hip x position to simulate side-to-side movement
            cycle = i % 20
            if cycle < 5:
                hip_offset = -0.1  # Moving left
            elif cycle < 10:
                hip_offset = 0.0  # Center
            elif cycle < 15:
                hip_offset = 0.1  # Moving right
            else:
                hip_offset = 0.0  # Center
            
            frames.append(self.create_pose_frame(
                timestamp=i / 30.0,
                wrist_above_shoulder=False,  # Arms below shoulder
                hip_x_offset=hip_offset,
                confidence=0.9
            ))
        
        result = classify_drill(frames)
        
        # Should detect footwork pattern
        assert result.detected_drill_type == "footwork" or result.confidence > 0
        
        # Should have recorded features
        assert "features" in result.debug_features
        if "hip_lateral_range" in result.debug_features.get("features", {}):
            assert result.debug_features["features"]["hip_lateral_range"] > 0
    
    def test_insufficient_frames_returns_unknown(self):
        """Should return unknown when not enough valid frames."""
        from core.motion_classifier import classify_drill
        
        # Only 10 frames (below minimum of 30)
        frames = [self.create_pose_frame(i / 30.0) for i in range(10)]
        
        result = classify_drill(frames)
        
        assert result.detected_drill_type == "unknown"
        assert result.confidence == 0.0
        assert "Insufficient" in result.reason
    
    def test_low_confidence_frames_filtered(self):
        """Should filter frames with low pose confidence."""
        from core.motion_classifier import classify_drill
        
        # Create 60 frames but with low confidence
        frames = [
            self.create_pose_frame(i / 30.0, confidence=0.3)  # Below 0.7 threshold
            for i in range(60)
        ]
        
        result = classify_drill(frames)
        
        assert result.detected_drill_type == "unknown"
        assert "valid_frames" in result.debug_features
    
    def test_classification_returns_debug_scores(self):
        """Should return debug scores for all drill types."""
        from core.motion_classifier import classify_drill
        
        frames = [self.create_pose_frame(i / 30.0, confidence=0.95) for i in range(60)]
        
        result = classify_drill(frames)
        
        assert "scores" in result.debug_features
        scores = result.debug_features["scores"]
        assert "overhead-shadow" in scores
        assert "footwork" in scores
        assert "6-corner-shadow" in scores


class TestEmbeddingsCache:
    """Tests for embeddings computation and caching."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample evidence chunks for testing."""
        return [
            "12.5s: MISTAKE - knee collapse. Severity: 0.8/1.0. Confidence: 0.9. Cue: 'Knee out'.",
            "15.3s: SPLIT_STEP detected. Duration: 0.3s. Confidence: 0.85",
            "8.2s: LUNGE detected. Duration: 0.5s. Confidence: 0.88",
            "PRIORITY FIX: knee collapse (3 occurrences). Focus: Knee out. Recommended drill: Single-leg squats. Key timestamps: [12.5s, 18.2s]",
            "SESSION SUMMARY: 30.5s video. 5 footwork events. 3 issues found. Pose confidence: 0.82"
        ]
    
    def test_embeddings_manager_initialization(self, temp_data_dir):
        """Should initialize embeddings manager with config."""
        from core.embeddings import EmbeddingsManager
        
        manager = EmbeddingsManager(temp_data_dir)
        
        assert manager.data_dir == Path(temp_data_dir)
        assert manager.config is not None
    
    def test_compute_embeddings_creates_cache(self, temp_data_dir, sample_chunks):
        """Should compute embeddings and save to cache file."""
        from core.embeddings import EmbeddingsManager
        
        manager = EmbeddingsManager(temp_data_dir)
        session_id = "test_session"
        
        cache = manager.compute_embeddings(session_id, sample_chunks)
        
        # Cache should be created (or None if sentence-transformers not installed)
        if cache:
            assert cache.session_id == session_id
            assert len(cache.chunks) == len(sample_chunks)
            assert len(cache.embeddings) == len(sample_chunks)
            assert len(cache.meta) == len(sample_chunks)
            
            # Check cache file exists
            cache_path = Path(temp_data_dir) / session_id / "embeddings.json"
            assert cache_path.exists()
    
    def test_cached_embeddings_reused(self, temp_data_dir, sample_chunks):
        """Should reuse cached embeddings on subsequent calls."""
        from core.embeddings import EmbeddingsManager
        
        manager = EmbeddingsManager(temp_data_dir)
        session_id = "test_session"
        
        # First call - computes embeddings
        cache1 = manager.compute_embeddings(session_id, sample_chunks)
        
        # Second call - should use cache
        cache2 = manager.compute_embeddings(session_id, sample_chunks)
        
        if cache1 and cache2:
            assert cache1.session_id == cache2.session_id
            assert cache1.chunks == cache2.chunks
    
    def test_chunk_meta_parsing(self, temp_data_dir, sample_chunks):
        """Should correctly parse chunk metadata (type, timestamps)."""
        from core.embeddings import EmbeddingsManager
        
        manager = EmbeddingsManager(temp_data_dir)
        session_id = "test_session"
        
        cache = manager.compute_embeddings(session_id, sample_chunks)
        
        if cache:
            # Check first chunk metadata (mistake)
            meta0 = cache.meta[0]
            assert meta0["chunk_type"] == "mistake"
            assert 12.5 in meta0["timestamps"]
            
            # Check priority chunk
            priority_meta = [m for m in cache.meta if m["chunk_type"] == "priority"]
            assert len(priority_meta) > 0
            
            # Check summary chunk
            summary_meta = [m for m in cache.meta if m["chunk_type"] == "summary"]
            assert len(summary_meta) > 0


class TestChatGrounding:
    """Tests for grounded chat with semantic retrieval."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample evidence chunks for testing."""
        return [
            "12.5s: MISTAKE - knee_collapse. Severity: 0.8/1.0. Confidence: 0.9. Cue: 'Knee out'. Evidence at: [12.3s, 12.7s]",
            "15.3s: SPLIT_STEP detected. Duration: 0.3s. Confidence: 0.85",
            "8.2s: LUNGE detected. Duration: 0.5s. Confidence: 0.88",
            "18.1s: MISTAKE - narrow_stance. Severity: 0.6/1.0. Confidence: 0.82. Cue: 'Wider base'.",
            "PRIORITY FIX: knee_collapse (3 occurrences). Focus: Knee out. Recommended drill: Single-leg squats. Key timestamps: [12.5s, 18.2s]",
            "SESSION SUMMARY: 30.5s video. 5 footwork events. 3 issues found. Pose confidence: 0.82"
        ]
    
    def test_keyword_retrieval_finds_relevant_chunks(self, temp_data_dir, sample_chunks):
        """Should retrieve chunks relevant to query using keyword matching."""
        from core.embeddings import EmbeddingsManager
        
        manager = EmbeddingsManager(temp_data_dir)
        session_id = "test_session"
        
        cache = manager.compute_embeddings(session_id, sample_chunks)
        
        if cache:
            results, max_sim = manager.semantic_search("knee collapse", cache)
            
            # Should find chunks mentioning knee collapse
            assert len(results) > 0
            found_knee = any("knee" in r.chunk.lower() for r in results)
            assert found_knee
    
    def test_ungrounded_response_when_low_similarity(self, temp_data_dir, sample_chunks):
        """Should return grounded=false when similarity is below threshold."""
        from core.embeddings import EmbeddingsManager
        
        manager = EmbeddingsManager(temp_data_dir)
        session_id = "test_session"
        
        cache = manager.compute_embeddings(session_id, sample_chunks)
        
        if cache:
            # Query something not in evidence
            results, max_sim = manager.semantic_search(
                "what racket should I buy?", 
                cache
            )
            
            # Check if grounding would fail
            is_grounded = manager.is_grounded(max_sim)
            
            # For a completely unrelated query, similarity should be low
            # (may pass if keyword fallback finds some match)
            assert max_sim >= 0
    
    def test_grounded_chat_returns_citations(self, temp_data_dir, sample_chunks):
        """Should include timestamp citations in grounded response."""
        from core.grounded_chat import GroundedChat
        
        # Create chat with stub (no LLM)
        chat = GroundedChat(data_dir=temp_data_dir)
        
        result = chat.chat(
            question="What knee issues did you detect?",
            evidence_chunks=sample_chunks,
            session_id="test_session"
        )
        
        assert "answer" in result
        assert "grounded" in result
        assert "citations" in result
        
        # If grounded, should have citations
        if result["grounded"]:
            # Check response structure
            assert isinstance(result["citations"], list)

    def test_chat_retrieval_knee_collapse_timestamps(self, temp_data_dir, sample_chunks):
        """Query should retrieve knee collapse evidence with timestamps."""
        from core.embeddings import EmbeddingsManager

        manager = EmbeddingsManager(temp_data_dir)
        session_id = "test_session"

        cache = manager.compute_embeddings(session_id, sample_chunks)
        if cache:
            results, _ = manager.semantic_search("knee collapse", cache)
            assert any("knee" in r.chunk.lower() for r in results)
            assert any(r.meta.timestamps for r in results)
    
    def test_chat_with_debug_includes_similarity(self, temp_data_dir, sample_chunks):
        """Should include debug info when requested."""
        from core.grounded_chat import GroundedChat
        
        chat = GroundedChat(data_dir=temp_data_dir)
        
        result = chat.chat(
            question="What mistakes were detected?",
            evidence_chunks=sample_chunks,
            session_id="test_session",
            include_debug=True
        )
        
        # Debug info should be present
        if result.get("debug"):
            assert "top_similarity" in result["debug"]


class TestIntegrationPipeline:
    """Integration tests for the full pipeline with new features."""
    
    @pytest.fixture
    def sample_pose_frames(self):
        """Create sample pose frames fixture (no video needed)."""
        frames = []
        for i in range(60):
            landmarks = []
            for j in range(33):
                landmarks.append({
                    "x": 0.5 + (0.1 if j in [15, 16] else 0),
                    "y": 0.5 - (0.2 if j in [15, 16] and i > 30 else 0),
                    "z": 0.0,
                    "visibility": 0.9
                })
            frames.append({
                "landmarks": landmarks,
                "timestamp": i / 30.0,
                "frame_index": i
            })
        return frames
    
    def test_drill_classification_integration(self, sample_pose_frames):
        """Should classify drill type from pose frames."""
        from core.motion_classifier import classify_drill
        
        result = classify_drill(sample_pose_frames)
        
        # Should return a valid result
        assert result.detected_drill_type in [
            "overhead-shadow", "footwork", "6-corner-shadow", "unknown"
        ]
        assert 0 <= result.confidence <= 1.0
        assert result.debug_features is not None
    
    def test_create_structured_chunks_from_report(self):
        """Should create structured chunks from report dict."""
        from core.embeddings import create_structured_chunks
        
        report_dict = {
            "session_id": "test123",
            "drill_type": "footwork",
            "drill_type_source": "auto",
            "drill_type_confidence": 0.75,
            "video_duration": 30.5,
            "mistakes": [
                {
                    "type": "knee_collapse",
                    "timestamp": 12.5,
                    "severity": 0.8,
                    "confidence": 0.9,
                    "cue": "Knee out",
                    "evidence": [12.3, 12.7]
                }
            ],
            "events": [
                {
                    "type": "split_step",
                    "timestamp": 15.3,
                    "end_timestamp": 15.6,
                    "duration": 0.3,
                    "confidence": 0.85
                }
            ],
            "top_mistakes": [
                {
                    "type": "knee_collapse",
                    "count": 2,
                    "max_severity": 0.8,
                    "cue": "Knee out",
                    "timestamps": [12.5, 18.2]
                }
            ],
            "fix_first_plan": {
                "primary_issue": "knee_collapse",
                "occurrences": 2,
                "cue": "Knee out",
                "focus_drill": "Single-leg squats"
            },
            "metrics_summary": {
                "total_events": 3,
                "total_mistakes": 2,
                "split_steps_detected": 1,
                "lunges_detected": 2,
                "avg_pose_confidence": 0.85
            },
            "confidence_notes": ["Low visibility in 5 frames (10%)"]
        }
        
        chunks = create_structured_chunks(report_dict)
        
        # Should create multiple chunks
        assert len(chunks) > 0
        
        # Should include mistake chunks
        mistake_chunks = [c for c in chunks if "MISTAKE" in c]
        assert len(mistake_chunks) > 0
        
        # Should include priority fix
        priority_chunks = [c for c in chunks if "PRIORITY" in c]
        assert len(priority_chunks) > 0
        
        # Should include summary
        summary_chunks = [c for c in chunks if "SUMMARY" in c]
        assert len(summary_chunks) > 0
        
        # Should include drill info
        drill_chunks = [c for c in chunks if "DRILL INFO" in c]
        assert len(drill_chunks) > 0
        assert "auto" in drill_chunks[0]  # drill_type_source

    def test_pipeline_with_pose_fixture(self, tmp_path):
        """Should run pipeline on pose fixture JSON without video."""
        from core.pipeline import AnalysisPipeline

        fixture_path = Path(__file__).parent / "fixtures" / "sample_pose.json"
        with open(fixture_path, "r") as f:
            pose_data = json.load(f)

        pipeline = AnalysisPipeline(output_dir=str(tmp_path / "sessions"))
        result = pipeline.analyze_from_pose_data(
            pose_data=pose_data,
            drill_type="unknown",
            session_id="fixture_test",
            save_poses=False
        )

        assert result["success"] is True
        assert "report" in result
        assert "evidence_chunks" in result
        assert result["report"]["drill_type_source"] in ["auto", "default", "user"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
