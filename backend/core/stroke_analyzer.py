"""
Overhead Stroke Analyzer for Shadow Practice
Detects and analyzes overhead shadow strokes (clear/smash style) using pose-only data.
NO racket detection, NO shuttle tracking, NO grip analysis.

V2 Accuracy Upgrades:
- Angle classifier for camera-aware "in front" checks
- Composite contact proxy (wrist speed + arm extension + height)
- Prep phase analysis (elbow up, non-racket arm)
- Confidence gating and episode clustering
- Calibration-aware baselines
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .feature_computer import FrameFeatures, LANDMARKS
from .angle_classifier import CameraAngleClassifier, AngleClassification, compute_frontness
from .contact_detection import EnhancedContactDetector, PrepPhaseAnalyzer, ContactProxyScore, PrepPhaseAnalysis
from .confidence_gating import VisibilityGate, EpisodeClustering, SmartFixFirstRanker, MistakeEpisode
from .calibration import BaseStanceCalibrator, CalibrationData
from config import get_thresholds

logger = logging.getLogger(__name__)


@dataclass
class StrokeWindow:
    """Detected stroke window with timing info"""
    stroke_id: int
    start_timestamp: float
    contact_proxy_timestamp: float
    end_timestamp: float
    duration: float
    
    # Peak velocities
    wrist_peak_speed: float
    elbow_peak_speed: float
    wrist_peak_timestamp: float
    elbow_peak_timestamp: float
    
    # Dominant arm (detected based on motion)
    dominant_side: str  # "left" or "right"
    
    # Confidence
    overhead_confidence: float
    visibility_score: float
    
    # Frame indices
    start_frame: int
    contact_frame: int
    end_frame: int


@dataclass
class StrokeAnalysis:
    """Analysis results for a single stroke"""
    stroke: StrokeWindow
    
    # Ready position (pre-stroke)
    ready_position_good: bool
    
    # Contact height
    contact_height_status: str  # "good", "medium", "low"
    wrist_above_shoulder: bool
    wrist_above_head: bool
    contact_height_value: float  # Relative position
    
    # Contact in front (camera-aware)
    contact_in_front: bool
    contact_front_value: float  # How far in front
    contact_front_confidence: float  # Confidence based on camera angle
    
    # Arm sequence
    elbow_leads_wrist: bool
    elbow_lead_time_ms: float
    
    # Prep phase (V2)
    prep_phase_good: bool
    elbow_prepared: bool
    non_racket_arm_up: bool
    
    # Overall
    is_valid_overhead: bool
    overhead_confidence: float
    
    # Camera angle info
    camera_angle: str  # "front", "side", "45deg"
    camera_confidence: float
    
    # Optional fields with defaults (must come last)
    ready_issues: List[str] = field(default_factory=list)
    prep_issues: List[str] = field(default_factory=list)


@dataclass
class StrokeMistake:
    """A detected stroke technique mistake"""
    mistake_type: str
    timestamp: float
    duration: float
    severity: float
    confidence: float
    evidence_timestamps: List[float] = field(default_factory=list)
    cue: str = ""
    description: str = ""
    metadata: Dict = field(default_factory=dict)


class OverheadStrokeAnalyzer:
    """
    Analyzes overhead shadow strokes from pose data.
    Uses rules-based detection, no ML.
    
    V2 Upgrades:
    - Camera angle classifier for better "in front" checks
    - Enhanced contact proxy with composite scoring
    - Prep phase analysis (elbow up, non-racket arm)
    - Visibility gating for anti-spam
    - Calibration integration
    """
    
    def __init__(self):
        self._thresholds = get_thresholds()
        self._strokes: List[StrokeWindow] = []
        self._analyses: List[StrokeAnalysis] = []
        self._mistakes: List[StrokeMistake] = []
        self._stroke_count = 0
        
        # V2 components
        self._angle_classifier = CameraAngleClassifier()
        self._contact_detector = EnhancedContactDetector()
        self._prep_analyzer = PrepPhaseAnalyzer()
        self._visibility_gate = VisibilityGate()
        self._calibrator = BaseStanceCalibrator()
        
        # Cached analysis data
        self._camera_angle: Optional[AngleClassification] = None
        self._calibration: Optional[CalibrationData] = None
    
    def reset(self):
        self._strokes = []
        self._analyses = []
        self._mistakes = []
        self._stroke_count = 0
        self._camera_angle = None
        self._calibration = None
    
    def _get_landmark_position(
        self, 
        landmarks: List[Dict], 
        name: str,
        use_smoothed: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """Get landmark (x, y, visibility)"""
        idx = LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        if use_smoothed and "x_smooth" in lm:
            return (lm["x_smooth"], lm["y_smooth"], lm.get("visibility", 0))
        return (lm["x"], lm["y"], lm.get("visibility", 0))
    
    def _compute_velocity(
        self,
        positions: List[Tuple[float, float]],
        timestamps: List[float]
    ) -> float:
        """Compute velocity magnitude from position history"""
        if len(positions) < 2:
            return 0.0
        
        total_vel = 0.0
        count = 0
        
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                vel = math.sqrt(dx*dx + dy*dy) / dt
                total_vel += vel
                count += 1
        
        return total_vel / count if count > 0 else 0.0
    
    def _compute_arm_velocities(
        self,
        frames: List[Dict],
        side: str
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute wrist and elbow velocities over frames.
        Returns (wrist_velocities, elbow_velocities, timestamps)
        """
        cfg = self._thresholds.stroke
        window = cfg.velocity_window_frames
        
        wrist_name = f"{side}_wrist"
        elbow_name = f"{side}_elbow"
        
        wrist_positions = []
        elbow_positions = []
        timestamps = []
        
        for frame in frames:
            landmarks = frame.get("landmarks")
            if not landmarks:
                continue
            
            wrist = self._get_landmark_position(landmarks, wrist_name)
            elbow = self._get_landmark_position(landmarks, elbow_name)
            ts = frame.get("timestamp", 0)
            
            if wrist and elbow and wrist[2] > cfg.stroke_visibility_min:
                wrist_positions.append((wrist[0], wrist[1]))
                elbow_positions.append((elbow[0], elbow[1]))
                timestamps.append(ts)
        
        # Compute rolling velocities
        wrist_vels = []
        elbow_vels = []
        vel_timestamps = []
        
        for i in range(window, len(wrist_positions)):
            w_vel = self._compute_velocity(
                wrist_positions[i-window:i+1],
                timestamps[i-window:i+1]
            )
            e_vel = self._compute_velocity(
                elbow_positions[i-window:i+1],
                timestamps[i-window:i+1]
            )
            wrist_vels.append(w_vel)
            elbow_vels.append(e_vel)
            vel_timestamps.append(timestamps[i])
        
        return wrist_vels, elbow_vels, vel_timestamps
    
    def _detect_dominant_arm(self, frames: List[Dict]) -> str:
        """Detect which arm is doing the stroke based on motion"""
        left_vels, _, _ = self._compute_arm_velocities(frames, "left")
        right_vels, _, _ = self._compute_arm_velocities(frames, "right")
        
        left_max = max(left_vels) if left_vels else 0
        right_max = max(right_vels) if right_vels else 0
        
        return "right" if right_max >= left_max else "left"
    
    def detect_strokes(self, frames: List[Dict]) -> List[StrokeWindow]:
        """
        Detect overhead stroke windows from frame sequence.
        A stroke is detected when wrist speed exceeds threshold.
        """
        cfg = self._thresholds.stroke
        
        # Detect dominant arm
        dominant = self._detect_dominant_arm(frames)
        
        # Compute velocities for dominant arm
        wrist_vels, elbow_vels, timestamps = self._compute_arm_velocities(frames, dominant)
        
        if not wrist_vels:
            return []
        
        strokes = []
        in_stroke = False
        stroke_start_idx = 0
        stroke_start_t = 0
        
        for i, (w_vel, e_vel, ts) in enumerate(zip(wrist_vels, elbow_vels, timestamps)):
            if not in_stroke:
                # Check for stroke start
                if w_vel >= cfg.wrist_speed_start_threshold:
                    in_stroke = True
                    stroke_start_idx = i
                    stroke_start_t = ts
            else:
                # Check for stroke end
                if w_vel < cfg.wrist_speed_end_threshold:
                    duration = ts - stroke_start_t
                    
                    if cfg.min_stroke_duration_sec <= duration <= cfg.max_stroke_duration_sec:
                        # Find peaks in this window
                        window_w_vels = wrist_vels[stroke_start_idx:i+1]
                        window_e_vels = elbow_vels[stroke_start_idx:i+1]
                        window_ts = timestamps[stroke_start_idx:i+1]
                        
                        if window_w_vels:
                            w_peak_idx = window_w_vels.index(max(window_w_vels))
                            e_peak_idx = window_e_vels.index(max(window_e_vels))
                            
                            w_peak_speed = window_w_vels[w_peak_idx]
                            e_peak_speed = window_e_vels[e_peak_idx]
                            w_peak_t = window_ts[w_peak_idx]
                            e_peak_t = window_ts[e_peak_idx]
                            
                            # Check if this is a valid stroke (peak speed high enough)
                            if w_peak_speed >= cfg.wrist_speed_peak_threshold:
                                # Compute overhead confidence
                                overhead_conf = self._compute_overhead_confidence(
                                    frames, stroke_start_t, w_peak_t, ts, dominant
                                )
                                
                                self._stroke_count += 1
                                stroke = StrokeWindow(
                                    stroke_id=self._stroke_count,
                                    start_timestamp=stroke_start_t,
                                    contact_proxy_timestamp=w_peak_t,
                                    end_timestamp=ts,
                                    duration=duration,
                                    wrist_peak_speed=w_peak_speed,
                                    elbow_peak_speed=e_peak_speed,
                                    wrist_peak_timestamp=w_peak_t,
                                    elbow_peak_timestamp=e_peak_t,
                                    dominant_side=dominant,
                                    overhead_confidence=overhead_conf,
                                    visibility_score=0.8,  # Placeholder
                                    start_frame=stroke_start_idx,
                                    contact_frame=stroke_start_idx + w_peak_idx,
                                    end_frame=i
                                )
                                strokes.append(stroke)
                    
                    in_stroke = False
        
        self._strokes = strokes
        return strokes
    
    def _compute_overhead_confidence(
        self,
        frames: List[Dict],
        start_t: float,
        contact_t: float,
        end_t: float,
        side: str
    ) -> float:
        """
        Compute confidence that this is an overhead stroke.
        Based on wrist/elbow height, arm extension, torso rotation.
        """
        cfg = self._thresholds.stroke
        
        # Find contact frame
        contact_frame = None
        for frame in frames:
            if abs(frame.get("timestamp", 0) - contact_t) < 0.05:
                contact_frame = frame
                break
        
        if not contact_frame or not contact_frame.get("landmarks"):
            return 0.0
        
        landmarks = contact_frame["landmarks"]
        wrist = self._get_landmark_position(landmarks, f"{side}_wrist")
        elbow = self._get_landmark_position(landmarks, f"{side}_elbow")
        shoulder = self._get_landmark_position(landmarks, f"{side}_shoulder")
        
        if not all([wrist, elbow, shoulder]):
            return 0.0
        
        confidence = 0.0
        
        # Check 1: Wrist above shoulder (in image coords, smaller y = higher)
        if wrist[1] < shoulder[1]:
            confidence += 0.4
        
        # Check 2: Elbow elevated
        if elbow[1] < shoulder[1] + cfg.elbow_elevation_threshold:
            confidence += 0.3
        
        # Check 3: Arm extension (wrist far from shoulder)
        arm_length = math.sqrt((wrist[0]-shoulder[0])**2 + (wrist[1]-shoulder[1])**2)
        if arm_length > 0.15:  # Reasonable extension
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def _find_frame_at_time(self, frames: List[Dict], timestamp: float) -> Optional[Dict]:
        """Find frame closest to timestamp"""
        best_frame = None
        best_diff = float('inf')
        
        for frame in frames:
            diff = abs(frame.get("timestamp", 0) - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        
        return best_frame if best_diff < 0.1 else None
    
    def analyze_stroke(
        self,
        stroke: StrokeWindow,
        frames: List[Dict],
        features: List[FrameFeatures]
    ) -> StrokeAnalysis:
        """Analyze a single detected stroke with V2 accuracy upgrades"""
        cfg = self._thresholds.stroke
        side = stroke.dominant_side
        
        # Get camera angle (cached)
        camera_angle = self._camera_angle or AngleClassification(
            angle_class="front", confidence=0.5, frontness_axis="x",
            shoulder_x_ratio=0.15, hip_shoulder_alignment=0.8, 
            occlusion_score=0.2, is_reliable=False
        )
        
        # Find contact frame
        contact_frame = self._find_frame_at_time(frames, stroke.contact_proxy_timestamp)
        
        # Default analysis for failures
        def default_analysis():
            return StrokeAnalysis(
                stroke=stroke,
                ready_position_good=True,
                contact_height_status="unknown",
                wrist_above_shoulder=False,
                wrist_above_head=False,
                contact_height_value=0,
                contact_in_front=True,
                contact_front_value=0,
                contact_front_confidence=0,
                elbow_leads_wrist=True,
                elbow_lead_time_ms=0,
                prep_phase_good=True,
                elbow_prepared=True,
                non_racket_arm_up=True,
                is_valid_overhead=False,
                overhead_confidence=stroke.overhead_confidence,
                camera_angle=camera_angle.angle_class,
                camera_confidence=camera_angle.confidence
            )
        
        if not contact_frame or not contact_frame.get("landmarks"):
            return default_analysis()
        
        landmarks = contact_frame["landmarks"]
        
        # Get key landmarks
        wrist = self._get_landmark_position(landmarks, f"{side}_wrist")
        elbow = self._get_landmark_position(landmarks, f"{side}_elbow")
        shoulder = self._get_landmark_position(landmarks, f"{side}_shoulder")
        nose = self._get_landmark_position(landmarks, "nose")
        
        # Get torso center (midpoint of shoulders)
        left_shoulder = self._get_landmark_position(landmarks, "left_shoulder")
        right_shoulder = self._get_landmark_position(landmarks, "right_shoulder")
        
        if not all([wrist, shoulder]):
            return default_analysis()
        
        # === V2: VISIBILITY GATING ===
        required_landmarks = [f"{side}_wrist", f"{side}_elbow", f"{side}_shoulder"]
        visibility = self._visibility_gate.check_stability(
            frames,
            stroke.start_timestamp,
            stroke.end_timestamp,
            required_landmarks
        )
        
        # === CONTACT HEIGHT ANALYSIS ===
        wrist_y = wrist[1]
        shoulder_y = shoulder[1]
        nose_y = nose[1] if nose else shoulder_y - 0.15
        
        height_diff = wrist_y - shoulder_y
        
        wrist_above_shoulder = height_diff < cfg.contact_low_threshold
        wrist_above_head = wrist_y < nose_y if nose else False
        
        if height_diff >= cfg.contact_low_threshold:
            contact_height_status = "low"
        elif height_diff >= cfg.contact_medium_threshold:
            contact_height_status = "medium"
        else:
            contact_height_status = "good"
        
        # === V2: CAMERA-AWARE CONTACT IN FRONT ===
        if left_shoulder and right_shoulder:
            torso_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                           (left_shoulder[1] + right_shoulder[1]) / 2)
        else:
            torso_center = (shoulder[0], shoulder[1])
        
        # Use camera-aware frontness computation
        frontness_value, frontness_confidence = compute_frontness(
            wrist_pos=(wrist[0], wrist[1]),
            shoulder_pos=(shoulder[0], shoulder[1]),
            torso_center=torso_center,
            angle_class=camera_angle
        )
        
        # Apply visibility-adjusted confidence
        frontness_confidence *= visibility.stability_score if visibility.is_stable else 0.5
        
        # Determine contact in front based on camera-aware threshold
        contact_in_front = frontness_value > -cfg.contact_front_tolerance
        contact_front_value = frontness_value
        contact_front_confidence = frontness_confidence
        
        # If camera angle is unreliable and frontness is marginal, reduce confidence
        if not camera_angle.is_reliable and abs(frontness_value) < cfg.contact_front_tolerance * 2:
            contact_front_confidence *= 0.5
        
        # === ELBOW LEADS WRIST ===
        elbow_lead_time_ms = (stroke.wrist_peak_timestamp - stroke.elbow_peak_timestamp) * 1000
        elbow_leads_wrist = cfg.elbow_lead_min_ms <= elbow_lead_time_ms <= cfg.elbow_lead_max_ms
        
        # === V2: PREP PHASE ANALYSIS ===
        prep_analysis = self._prep_analyzer.analyze_prep_phase(
            frames, stroke.start_timestamp, side
        )
        prep_phase_good = prep_analysis.prep_quality in ["good", "fair"]
        
        # === READY POSITION CHECK ===
        ready_issues = []
        ready_position_good = True
        
        ready_start_t = stroke.start_timestamp - cfg.ready_window_before_sec
        ready_features = [f for f in features 
                        if ready_start_t <= f.timestamp <= stroke.start_timestamp]
        
        if ready_features:
            avg_stance = sum(f.stance_width_ratio for f in ready_features) / len(ready_features)
            
            # Use calibration baseline if available
            if self._calibration and self._calibration.is_valid:
                stance_status, deviation = self._calibrator.check_stance_against_baseline(
                    avg_stance, self._calibration
                )
                if stance_status == "narrow":
                    ready_issues.append("narrow_stance")
                    ready_position_good = False
            else:
                # Fallback to fixed threshold
                if avg_stance < cfg.ready_stance_width_min:
                    ready_issues.append("narrow_stance")
                    ready_position_good = False
            
            # Check knee bend
            avg_knee = sum(min(f.left_knee_angle, f.right_knee_angle) for f in ready_features) / len(ready_features)
            if avg_knee > cfg.ready_knee_bend_min:
                ready_issues.append("knees_not_bent")
                ready_position_good = False
        
        # === OVERHEAD VALIDITY ===
        is_valid_overhead = stroke.overhead_confidence >= cfg.overhead_confidence_min
        
        return StrokeAnalysis(
            stroke=stroke,
            ready_position_good=ready_position_good,
            contact_height_status=contact_height_status,
            wrist_above_shoulder=wrist_above_shoulder,
            wrist_above_head=wrist_above_head,
            contact_height_value=height_diff,
            contact_in_front=contact_in_front,
            contact_front_value=contact_front_value,
            contact_front_confidence=contact_front_confidence,
            elbow_leads_wrist=elbow_leads_wrist,
            elbow_lead_time_ms=elbow_lead_time_ms,
            prep_phase_good=prep_phase_good,
            elbow_prepared=prep_analysis.elbow_above_shoulder,
            non_racket_arm_up=prep_analysis.non_racket_arm_up,
            is_valid_overhead=is_valid_overhead,
            overhead_confidence=stroke.overhead_confidence,
            camera_angle=camera_angle.angle_class,
            camera_confidence=camera_angle.confidence,
            ready_issues=ready_issues,
            prep_issues=prep_analysis.issues
        )
    
    def detect_mistakes(
        self,
        analyses: List[StrokeAnalysis]
    ) -> List[StrokeMistake]:
        """Detect mistakes from stroke analyses"""
        cfg = self._thresholds.stroke
        mistakes = []
        
        for analysis in analyses:
            stroke = analysis.stroke
            
            # Skip if not confident it's overhead
            if not analysis.is_valid_overhead:
                mistakes.append(StrokeMistake(
                    mistake_type="unclear_overhead_intent",
                    timestamp=stroke.contact_proxy_timestamp,
                    duration=stroke.duration,
                    severity=0.3,
                    confidence=1.0 - analysis.overhead_confidence,
                    evidence_timestamps=[stroke.start_timestamp, stroke.contact_proxy_timestamp],
                    cue="Check drill type",
                    description="This does not clearly look like an overhead shadow stroke. Select the correct drill or re-record.",
                    metadata={"overhead_confidence": analysis.overhead_confidence}
                ))
                continue
            
            # Contact height issues
            if analysis.contact_height_status == "low":
                mistakes.append(StrokeMistake(
                    mistake_type="overhead_contact_too_low",
                    timestamp=stroke.contact_proxy_timestamp,
                    duration=0.1,
                    severity=0.8,
                    confidence=analysis.overhead_confidence,
                    evidence_timestamps=[stroke.contact_proxy_timestamp],
                    cue="Higher contact",
                    description="Contact point is too low (at or below shoulder). Extend arm fully and contact above your head.",
                    metadata={"height_diff": analysis.contact_height_value}
                ))
            elif analysis.contact_height_status == "medium":
                mistakes.append(StrokeMistake(
                    mistake_type="overhead_contact_medium",
                    timestamp=stroke.contact_proxy_timestamp,
                    duration=0.1,
                    severity=0.4,
                    confidence=analysis.overhead_confidence,
                    evidence_timestamps=[stroke.contact_proxy_timestamp],
                    cue="Higher contact",
                    description="Contact point could be higher. Try to contact above your head for better power.",
                    metadata={"height_diff": analysis.contact_height_value}
                ))
            
            # Contact not in front (V2: camera-aware confidence)
            if not analysis.contact_in_front and analysis.contact_front_confidence > 0.4:
                mistakes.append(StrokeMistake(
                    mistake_type="contact_not_in_front",
                    timestamp=stroke.contact_proxy_timestamp,
                    duration=0.1,
                    severity=0.7,
                    confidence=analysis.contact_front_confidence,  # V2: camera-aware
                    evidence_timestamps=[stroke.contact_proxy_timestamp],
                    cue="Contact in front",
                    description="Contact point appears to be behind your body. Move contact point forward.",
                    metadata={"front_value": analysis.contact_front_value, 
                             "camera_angle": analysis.camera_angle}
                ))
            
            # Elbow not leading
            if not analysis.elbow_leads_wrist and analysis.elbow_lead_time_ms < cfg.elbow_lead_min_ms:
                mistakes.append(StrokeMistake(
                    mistake_type="elbow_not_leading",
                    timestamp=stroke.contact_proxy_timestamp,
                    duration=stroke.duration,
                    severity=0.5,
                    confidence=analysis.overhead_confidence * 0.7,
                    evidence_timestamps=[stroke.elbow_peak_timestamp, stroke.wrist_peak_timestamp],
                    cue="Elbow first",
                    description="Wrist is snapping too early. Lead with your elbow, then snap the wrist.",
                    metadata={"elbow_lead_ms": analysis.elbow_lead_time_ms}
                ))
            
            # Ready position issues
            if not analysis.ready_position_good:
                for issue in analysis.ready_issues:
                    if issue == "narrow_stance":
                        mistakes.append(StrokeMistake(
                            mistake_type="poor_ready_position",
                            timestamp=stroke.start_timestamp - 0.3,
                            duration=0.3,
                            severity=0.4,
                            confidence=analysis.overhead_confidence,
                            evidence_timestamps=[stroke.start_timestamp - 0.3],
                            cue="Wider base",
                            description="Stance too narrow before stroke. Widen your base for better balance.",
                            metadata={"issue": issue}
                        ))
                    elif issue == "knees_not_bent":
                        mistakes.append(StrokeMistake(
                            mistake_type="poor_ready_position",
                            timestamp=stroke.start_timestamp - 0.3,
                            duration=0.3,
                            severity=0.4,
                            confidence=analysis.overhead_confidence,
                            evidence_timestamps=[stroke.start_timestamp - 0.3],
                            cue="Lower stance",
                            description="Knees not bent enough before stroke. Lower your stance.",
                            metadata={"issue": issue}
                        ))
            
            # V2: Prep phase issues
            if not analysis.prep_phase_good:
                for issue in analysis.prep_issues:
                    if issue == "elbow_not_up":
                        mistakes.append(StrokeMistake(
                            mistake_type="elbow_not_prepared",
                            timestamp=stroke.start_timestamp - 0.3,
                            duration=0.3,
                            severity=0.6,
                            confidence=analysis.overhead_confidence,
                            evidence_timestamps=[stroke.start_timestamp - 0.3],
                            cue="Elbow up",
                            description="Racket arm elbow should be raised before swing. Prepare with elbow above shoulder.",
                            metadata={"issue": issue}
                        ))
                    elif issue == "non_racket_arm_down":
                        mistakes.append(StrokeMistake(
                            mistake_type="non_racket_arm_not_used",
                            timestamp=stroke.start_timestamp - 0.3,
                            duration=0.3,
                            severity=0.4,
                            confidence=analysis.overhead_confidence * 0.8,
                            evidence_timestamps=[stroke.start_timestamp - 0.3],
                            cue="Use non-racket arm",
                            description="Non-racket arm should be raised for balance and timing.",
                            metadata={"issue": issue}
                        ))
        
        self._mistakes = mistakes
        return mistakes
    
    def analyze_all(
        self,
        frames: List[Dict],
        features: List[FrameFeatures]
    ) -> Tuple[List[StrokeWindow], List[StrokeAnalysis], List[StrokeMistake]]:
        """
        Full analysis pipeline for overhead strokes.
        
        V2 Flow:
        1. Calibrate from initial frames
        2. Classify camera angle
        3. Detect strokes
        4. Analyze each stroke with V2 checks
        5. Detect mistakes with prep phase
        
        Args:
            frames: Raw frame data with landmarks
            features: Computed frame features
        
        Returns:
            (strokes, analyses, mistakes)
        """
        self.reset()
        
        if not frames:
            logger.info("No frames to analyze")
            return [], [], []
        
        # V2: Calibrate from initial frames
        self._calibration = self._calibrator.calibrate(frames)
        if self._calibration.is_valid:
            logger.info(f"Calibration complete: stance_ratio={self._calibration.baseline_stance_ratio:.2f}, conf={self._calibration.calibration_confidence:.2f}")
        else:
            logger.warning("Calibration incomplete or invalid")
        
        # V2: Classify camera angle
        self._camera_angle = self._angle_classifier.classify_video(frames)
        logger.info(f"Camera angle: {self._camera_angle.angle_class} (conf={self._camera_angle.confidence:.2f})")
        
        # Detect strokes
        strokes = self.detect_strokes(frames)
        
        if not strokes:
            logger.info("No overhead strokes detected")
            return [], [], []
        
        # Analyze each stroke
        analyses = []
        for stroke in strokes:
            analysis = self.analyze_stroke(stroke, frames, features)
            analyses.append(analysis)
        
        self._analyses = analyses
        
        # Detect mistakes (including prep phase issues)
        mistakes = self.detect_mistakes(analyses)
        
        logger.info(f"Detected {len(strokes)} strokes, {len(mistakes)} issues")
        
        return strokes, analyses, mistakes
    
    def get_camera_angle(self) -> Optional[AngleClassification]:
        """Get detected camera angle"""
        return self._camera_angle
    
    def get_calibration(self) -> Optional[CalibrationData]:
        """Get calibration data"""
        return self._calibration
    
    def get_strokes(self) -> List[StrokeWindow]:
        return self._strokes
    
    def get_analyses(self) -> List[StrokeAnalysis]:
        return self._analyses
    
    def get_mistakes(self) -> List[StrokeMistake]:
        return self._mistakes
