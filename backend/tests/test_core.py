"""
Unit Tests for ShuttleSense Core Components
"""

import pytest
import math
from typing import List, Dict


# Test OneEuro Filter
class TestOneEuroFilter:
    
    def test_filter_stability(self):
        """Filter should stabilize noisy signal"""
        from core.one_euro_filter import OneEuroFilter
        
        f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.007)
        
        # Simulate noisy signal around 0.5
        import random
        random.seed(42)
        
        noisy = [0.5 + random.uniform(-0.05, 0.05) for _ in range(100)]
        filtered = [f.filter(v) for v in noisy]
        
        # Filtered should have less variance
        noisy_var = sum((v - 0.5)**2 for v in noisy) / len(noisy)
        filtered_var = sum((v - 0.5)**2 for v in filtered[10:]) / len(filtered[10:])
        
        assert filtered_var < noisy_var, "Filtered variance should be less than input"
    
    def test_filter_tracks_step(self):
        """Filter should track step changes"""
        from core.one_euro_filter import OneEuroFilter
        
        f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.5)  # Higher beta for faster tracking
        
        # Step from 0 to 1
        signal = [0.0] * 30 + [1.0] * 30
        filtered = [f.filter(v) for v in signal]
        
        # After step, should reach close to 1.0
        assert filtered[-1] > 0.9, "Should track to new value"
        # Before step, should be close to 0
        assert filtered[25] < 0.1, "Should stay at old value before step"
    
    def test_filter_reset(self):
        """Filter reset should clear state"""
        from core.one_euro_filter import OneEuroFilter
        
        f = OneEuroFilter()
        f.filter(1.0)
        f.filter(1.0)
        f.reset()
        
        # After reset, first value should pass through
        result = f.filter(0.5)
        assert result == 0.5, "First value after reset should pass through"


class TestKeypointSmoother:
    
    def test_smoother_handles_landmarks(self):
        """Smoother should handle list of landmarks"""
        from core.one_euro_filter import KeypointSmoother
        
        smoother = KeypointSmoother(num_landmarks=33)
        
        landmarks = [{"x": 0.5, "y": 0.5, "visibility": 0.9} for _ in range(33)]
        
        smoothed = smoother.smooth(landmarks, timestamp=0.0)
        
        assert len(smoothed) == 33
        assert all(s["smoothed"] for s in smoothed)
    
    def test_low_visibility_not_smoothed(self):
        """Low visibility landmarks should not be aggressively smoothed"""
        from core.one_euro_filter import KeypointSmoother
        
        smoother = KeypointSmoother(num_landmarks=5)
        
        landmarks = [
            {"x": 0.5, "y": 0.5, "visibility": 0.9},
            {"x": 0.5, "y": 0.5, "visibility": 0.3},  # Low visibility
            {"x": 0.5, "y": 0.5, "visibility": 0.8},
            {"x": 0.5, "y": 0.5, "visibility": 0.2},  # Low visibility
            {"x": 0.5, "y": 0.5, "visibility": 0.7},
        ]
        
        smoothed = smoother.smooth(landmarks, visibility_threshold=0.5)
        
        assert smoothed[0]["reliable"] == True
        assert smoothed[1]["reliable"] == False
        assert smoothed[3]["reliable"] == False


# Test Feature Computer
class TestFeatureComputer:
    
    def test_stance_width_ratio(self):
        """Stance width ratio = ankle_dist / shoulder_dist"""
        from core.feature_computer import FeatureComputer, LANDMARKS
        
        fc = FeatureComputer()
        
        # Create mock landmarks
        landmarks = [{"x": 0, "y": 0, "visibility": 0.9} for _ in range(33)]
        
        # Set shoulders 0.2 apart
        landmarks[LANDMARKS["left_shoulder"]] = {"x": 0.4, "y": 0.3, "visibility": 0.9}
        landmarks[LANDMARKS["right_shoulder"]] = {"x": 0.6, "y": 0.3, "visibility": 0.9}
        
        # Set ankles 0.2 apart (same as shoulders = ratio 1.0)
        landmarks[LANDMARKS["left_ankle"]] = {"x": 0.4, "y": 0.8, "visibility": 0.9}
        landmarks[LANDMARKS["right_ankle"]] = {"x": 0.6, "y": 0.8, "visibility": 0.9}
        
        # Set hips and knees
        landmarks[LANDMARKS["left_hip"]] = {"x": 0.45, "y": 0.5, "visibility": 0.9}
        landmarks[LANDMARKS["right_hip"]] = {"x": 0.55, "y": 0.5, "visibility": 0.9}
        landmarks[LANDMARKS["left_knee"]] = {"x": 0.42, "y": 0.65, "visibility": 0.9}
        landmarks[LANDMARKS["right_knee"]] = {"x": 0.58, "y": 0.65, "visibility": 0.9}
        
        frame = {"landmarks": landmarks, "timestamp": 0, "frame_index": 0}
        feat = fc.compute(frame)
        
        assert feat is not None
        assert abs(feat.stance_width_ratio - 1.0) < 0.01, "Ratio should be ~1.0"
    
    def test_narrow_stance_detection(self):
        """Narrow stance should have ratio < 1"""
        from core.feature_computer import FeatureComputer, LANDMARKS
        
        fc = FeatureComputer()
        
        landmarks = [{"x": 0, "y": 0, "visibility": 0.9} for _ in range(33)]
        
        # Shoulders 0.3 apart
        landmarks[LANDMARKS["left_shoulder"]] = {"x": 0.35, "y": 0.3, "visibility": 0.9}
        landmarks[LANDMARKS["right_shoulder"]] = {"x": 0.65, "y": 0.3, "visibility": 0.9}
        
        # Ankles only 0.1 apart (narrow!)
        landmarks[LANDMARKS["left_ankle"]] = {"x": 0.45, "y": 0.8, "visibility": 0.9}
        landmarks[LANDMARKS["right_ankle"]] = {"x": 0.55, "y": 0.8, "visibility": 0.9}
        
        # Hips and knees
        landmarks[LANDMARKS["left_hip"]] = {"x": 0.45, "y": 0.5, "visibility": 0.9}
        landmarks[LANDMARKS["right_hip"]] = {"x": 0.55, "y": 0.5, "visibility": 0.9}
        landmarks[LANDMARKS["left_knee"]] = {"x": 0.45, "y": 0.65, "visibility": 0.9}
        landmarks[LANDMARKS["right_knee"]] = {"x": 0.55, "y": 0.65, "visibility": 0.9}
        
        frame = {"landmarks": landmarks, "timestamp": 0, "frame_index": 0}
        feat = fc.compute(frame)
        
        assert feat.stance_width_ratio < 0.5, "Narrow stance should have low ratio"


# Test FSM
class TestFSM:
    
    def test_split_step_detection(self):
        """FSM should detect split step from vertical velocity pattern"""
        from core.event_fsm import FootworkFSM
        from core.feature_computer import FrameFeatures
        
        fsm = FootworkFSM()
        
        # Simulate split step: up velocity then down
        features = []
        windowed = []
        
        for i in range(30):
            t = i / 30.0
            
            # Simulate hop: up for 5 frames, down for 5 frames
            if 10 <= i < 15:
                v_vel = 0.05  # Going up
            elif 15 <= i < 20:
                v_vel = -0.03  # Coming down
            else:
                v_vel = 0.0
            
            feat = FrameFeatures(
                timestamp=t, frame_index=i,
                ankle_distance=0.2, shoulder_distance=0.2, stance_width_ratio=1.0,
                hip_center=(0.5, 0.5), shoulder_center=(0.5, 0.3), com_proxy=(0.5, 0.5),
                vertical_velocity=v_vel, horizontal_velocity=0,
                com_velocity_magnitude=abs(v_vel),
                left_knee_angle=160, right_knee_angle=160,
                torso_lean_angle=10,
                left_knee_valgus=2, right_knee_valgus=2,
                distance_from_base=0.02,
                lower_body_confidence=0.9, is_reliable=True
            )
            features.append(feat)
            windowed.append({"avg_stance_ratio": 1.0, "avg_confidence": 0.9,
                           "horizontal_velocity_std": 0.01, "avg_left_valgus": 2,
                           "avg_right_valgus": 2, "avg_left_knee_angle": 160,
                           "avg_right_knee_angle": 160, "avg_vertical_velocity": v_vel,
                           "avg_horizontal_velocity": 0, "vertical_velocity_std": 0.01})
        
        events = fsm.process_all(features, windowed)
        
        split_steps = [e for e in events if e.event_type == "split_step"]
        assert len(split_steps) >= 1, "Should detect at least one split step"


class TestMistakeDetector:
    
    def test_narrow_stance_mistake(self):
        """Should detect sustained narrow stance as mistake"""
        from core.mistake_detector import MistakeDetector
        from core.feature_computer import FrameFeatures
        
        detector = MistakeDetector()
        
        features = []
        windowed = []
        
        # 15 frames of narrow stance
        for i in range(15):
            feat = FrameFeatures(
                timestamp=i/30.0, frame_index=i,
                ankle_distance=0.1, shoulder_distance=0.2, stance_width_ratio=0.5,
                hip_center=(0.5, 0.5), shoulder_center=(0.5, 0.3), com_proxy=(0.5, 0.5),
                vertical_velocity=0, horizontal_velocity=0, com_velocity_magnitude=0,
                left_knee_angle=160, right_knee_angle=160, torso_lean_angle=10,
                left_knee_valgus=2, right_knee_valgus=2, distance_from_base=0.02,
                lower_body_confidence=0.9, is_reliable=True
            )
            features.append(feat)
            windowed.append({"avg_stance_ratio": 0.5, "avg_confidence": 0.9,
                           "avg_left_valgus": 2, "avg_right_valgus": 2})
        
        # Add 5 frames of NORMAL stance to trigger detection (transition out of narrow)
        for i in range(15, 20):
            feat = FrameFeatures(
                timestamp=i/30.0, frame_index=i,
                ankle_distance=0.2, shoulder_distance=0.2, stance_width_ratio=1.0,
                hip_center=(0.5, 0.5), shoulder_center=(0.5, 0.3), com_proxy=(0.5, 0.5),
                vertical_velocity=0, horizontal_velocity=0, com_velocity_magnitude=0,
                left_knee_angle=160, right_knee_angle=160, torso_lean_angle=10,
                left_knee_valgus=2, right_knee_valgus=2, distance_from_base=0.02,
                lower_body_confidence=0.9, is_reliable=True
            )
            features.append(feat)
            windowed.append({"avg_stance_ratio": 1.0, "avg_confidence": 0.9,
                           "avg_left_valgus": 2, "avg_right_valgus": 2})
        
        detector.detect_narrow_stance(features, windowed)
        mistakes = detector.get_mistakes()
        
        narrow = [m for m in mistakes if m.mistake_type == "narrow_stance"]
        assert len(narrow) >= 1, "Should detect narrow stance"


class TestStrokeAnalyzer:
    """Tests for overhead stroke analysis"""
    
    def test_overhead_confidence_calculation(self):
        """Overhead confidence should be high when wrist is above shoulder"""
        from core.stroke_analyzer import OverheadStrokeAnalyzer
        from core.feature_computer import LANDMARKS
        
        analyzer = OverheadStrokeAnalyzer()
        
        # Create mock landmarks with wrist above shoulder (good overhead position)
        landmarks = [{"x": 0, "y": 0, "visibility": 0.9} for _ in range(33)]
        
        # Right arm in overhead position
        landmarks[LANDMARKS["right_shoulder"]] = {"x": 0.6, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["right_elbow"]] = {"x": 0.65, "y": 0.25, "visibility": 0.9}
        landmarks[LANDMARKS["right_wrist"]] = {"x": 0.55, "y": 0.15, "visibility": 0.9}  # Above head
        
        # Left arm (non-dominant)
        landmarks[LANDMARKS["left_shoulder"]] = {"x": 0.4, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["left_elbow"]] = {"x": 0.35, "y": 0.45, "visibility": 0.9}
        landmarks[LANDMARKS["left_wrist"]] = {"x": 0.3, "y": 0.5, "visibility": 0.9}
        
        frames = [{"landmarks": landmarks, "timestamp": 0.5}]
        
        conf = analyzer._compute_overhead_confidence(
            frames, start_t=0.4, contact_t=0.5, end_t=0.6, side="right"
        )
        
        # Wrist is above shoulder, elbow elevated, arm extended -> high confidence
        assert conf >= 0.6, f"Should have high overhead confidence, got {conf}"
    
    def test_contact_too_low_detection(self):
        """Should detect contact point too low (wrist at/below shoulder)"""
        from core.stroke_analyzer import OverheadStrokeAnalyzer, StrokeWindow, StrokeAnalysis
        from core.feature_computer import LANDMARKS
        
        analyzer = OverheadStrokeAnalyzer()
        
        # Create mock stroke with wrist AT shoulder level (too low)
        landmarks = [{"x": 0, "y": 0, "visibility": 0.9} for _ in range(33)]
        
        # Right arm with wrist at shoulder level (LOW contact)
        landmarks[LANDMARKS["right_shoulder"]] = {"x": 0.6, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["right_elbow"]] = {"x": 0.65, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["right_wrist"]] = {"x": 0.7, "y": 0.35, "visibility": 0.9}  # Same level as shoulder!
        landmarks[LANDMARKS["left_shoulder"]] = {"x": 0.4, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["nose"]] = {"x": 0.5, "y": 0.2, "visibility": 0.9}
        
        frames = [{"landmarks": landmarks, "timestamp": 0.5}]
        
        stroke = StrokeWindow(
            stroke_id=1,
            start_timestamp=0.4,
            contact_proxy_timestamp=0.5,
            end_timestamp=0.6,
            duration=0.2,
            wrist_peak_speed=0.1,
            elbow_peak_speed=0.08,
            wrist_peak_timestamp=0.5,
            elbow_peak_timestamp=0.45,
            dominant_side="right",
            overhead_confidence=0.8,
            visibility_score=0.9,
            start_frame=0,
            contact_frame=0,
            end_frame=0
        )
        
        analysis = analyzer.analyze_stroke(stroke, frames, [])
        
        # Wrist at shoulder level = low contact
        assert analysis.contact_height_status == "low", f"Expected 'low', got {analysis.contact_height_status}"
        assert not analysis.wrist_above_shoulder, "Wrist should NOT be above shoulder"
    
    def test_elbow_leads_wrist_detection(self):
        """Should detect if elbow leads wrist in swing"""
        from core.stroke_analyzer import OverheadStrokeAnalyzer, StrokeWindow
        from core.feature_computer import LANDMARKS
        
        analyzer = OverheadStrokeAnalyzer()
        
        # Create mock landmarks
        landmarks = [{"x": 0, "y": 0, "visibility": 0.9} for _ in range(33)]
        landmarks[LANDMARKS["right_shoulder"]] = {"x": 0.6, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["right_elbow"]] = {"x": 0.65, "y": 0.25, "visibility": 0.9}
        landmarks[LANDMARKS["right_wrist"]] = {"x": 0.55, "y": 0.15, "visibility": 0.9}
        landmarks[LANDMARKS["left_shoulder"]] = {"x": 0.4, "y": 0.35, "visibility": 0.9}
        landmarks[LANDMARKS["nose"]] = {"x": 0.5, "y": 0.2, "visibility": 0.9}
        
        frames = [{"landmarks": landmarks, "timestamp": 0.5}]
        
        # Good elbow lead: elbow peak at 0.45, wrist peak at 0.5 (50ms lead)
        stroke_good = StrokeWindow(
            stroke_id=1, start_timestamp=0.4, contact_proxy_timestamp=0.5,
            end_timestamp=0.6, duration=0.2,
            wrist_peak_speed=0.1, elbow_peak_speed=0.08,
            wrist_peak_timestamp=0.5, elbow_peak_timestamp=0.45,  # 50ms lead
            dominant_side="right", overhead_confidence=0.8,
            visibility_score=0.9, start_frame=0, contact_frame=0, end_frame=0
        )
        
        analysis = analyzer.analyze_stroke(stroke_good, frames, [])
        assert analysis.elbow_leads_wrist, "Elbow should lead wrist with 50ms lead"
        assert 30 <= analysis.elbow_lead_time_ms <= 200
    
    def test_stroke_mistake_generation(self):
        """Should generate appropriate mistakes from stroke analysis"""
        from core.stroke_analyzer import OverheadStrokeAnalyzer, StrokeWindow, StrokeAnalysis
        
        analyzer = OverheadStrokeAnalyzer()
        
        # Create analysis with low contact
        stroke = StrokeWindow(
            stroke_id=1, start_timestamp=0.4, contact_proxy_timestamp=0.5,
            end_timestamp=0.6, duration=0.2,
            wrist_peak_speed=0.1, elbow_peak_speed=0.08,
            wrist_peak_timestamp=0.5, elbow_peak_timestamp=0.5,
            dominant_side="right", overhead_confidence=0.8,
            visibility_score=0.9, start_frame=0, contact_frame=0, end_frame=0
        )
        
        analysis = StrokeAnalysis(
            stroke=stroke,
            ready_position_good=True,
            ready_issues=[],
            contact_height_status="low",
            wrist_above_shoulder=False,
            wrist_above_head=False,
            contact_height_value=0.05,
            contact_in_front=True,
            contact_front_value=0.1,
            elbow_leads_wrist=False,
            elbow_lead_time_ms=0,
            is_valid_overhead=True,
            overhead_confidence=0.8
        )
        
        mistakes = analyzer.detect_mistakes([analysis])
        
        # Should have contact too low mistake
        low_contact = [m for m in mistakes if m.mistake_type == "overhead_contact_too_low"]
        assert len(low_contact) >= 1, "Should detect contact too low"
        
        # Should have elbow not leading
        elbow_issues = [m for m in mistakes if m.mistake_type == "elbow_not_leading"]
        assert len(elbow_issues) >= 1, "Should detect elbow not leading"


class TestDrillTypeDetection:
    """Test drill type detection helpers"""
    
    def test_stroke_drill_detection(self):
        """is_stroke_drill should recognize overhead drills"""
        from core.pipeline import is_stroke_drill, is_footwork_drill
        
        assert is_stroke_drill("overhead-shadow") == True
        assert is_stroke_drill("overhead-clear") == True
        assert is_stroke_drill("smash-shadow") == True
        assert is_stroke_drill("6-corner-shadow") == False
        
    def test_footwork_drill_detection(self):
        """is_footwork_drill should recognize footwork drills"""
        from core.pipeline import is_stroke_drill, is_footwork_drill
        
        assert is_footwork_drill("6-corner-shadow") == True
        assert is_footwork_drill("side-to-side") == True
        assert is_footwork_drill("footwork") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
