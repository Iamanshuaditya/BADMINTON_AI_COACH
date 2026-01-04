"""
Enhanced Contact Detection and Prep Phase Analysis
Improves contact proxy accuracy and adds prep phase checks.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .feature_computer import LANDMARKS
from config import get_thresholds

logger = logging.getLogger(__name__)


@dataclass
class ContactProxyScore:
    """Composite contact proxy with multiple signals"""
    timestamp: float
    frame_index: int
    
    # Individual scores (0-1)
    wrist_speed_score: float
    arm_extension_score: float
    wrist_height_score: float
    
    # Combined score
    composite_score: float
    
    # Raw values
    wrist_speed: float
    elbow_angle: float  # Degrees, higher = more extended
    wrist_height: float  # Relative to shoulder (negative = above)


@dataclass 
class PrepPhaseAnalysis:
    """Analysis of preparation phase before stroke"""
    # Timing
    prep_start_timestamp: float
    prep_end_timestamp: float  # When swing starts
    
    # Racket arm prep
    elbow_above_shoulder: bool
    elbow_elevation: float  # How much above shoulder (negative = above in y)
    arm_prepared: bool
    
    # Non-racket arm
    non_racket_arm_up: bool
    non_racket_wrist_height: float  # Relative to chest
    
    # Overall
    prep_quality: str  # "good", "fair", "poor"
    issues: List[str] = field(default_factory=list)


class EnhancedContactDetector:
    """
    Improved contact point detection using composite signals.
    """
    
    def __init__(self):
        self._thresholds = get_thresholds()
        
        # Weights for composite score
        self.wrist_speed_weight = 0.4
        self.arm_extension_weight = 0.35
        self.wrist_height_weight = 0.25
    
    def _get_landmark(self, landmarks: List[Dict], name: str) -> Optional[Tuple[float, float, float]]:
        """Get landmark (x, y, visibility)"""
        idx = LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        x = lm.get("x_smooth", lm.get("x", 0))
        y = lm.get("y_smooth", lm.get("y", 0))
        return (x, y, lm.get("visibility", 0))
    
    def _compute_elbow_angle(
        self, 
        shoulder: Tuple[float, float, float],
        elbow: Tuple[float, float, float],
        wrist: Tuple[float, float, float]
    ) -> float:
        """Compute elbow angle in degrees (180 = fully extended)"""
        # Vector from elbow to shoulder
        es_x = shoulder[0] - elbow[0]
        es_y = shoulder[1] - elbow[1]
        
        # Vector from elbow to wrist
        ew_x = wrist[0] - elbow[0]
        ew_y = wrist[1] - elbow[1]
        
        # Angle between vectors
        dot = es_x * ew_x + es_y * ew_y
        mag1 = math.sqrt(es_x**2 + es_y**2)
        mag2 = math.sqrt(ew_x**2 + ew_y**2)
        
        if mag1 * mag2 < 0.001:
            return 180.0
        
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.acos(cos_angle) * 180 / math.pi
        
        return angle
    
    def compute_contact_scores(
        self,
        frames: List[Dict],
        stroke_start_idx: int,
        stroke_end_idx: int,
        side: str,
        wrist_velocities: List[float],
        velocity_timestamps: List[float]
    ) -> List[ContactProxyScore]:
        """
        Compute composite contact scores for each frame in stroke window.
        
        Returns list of ContactProxyScore, one per frame in window.
        """
        scores = []
        
        window_frames = frames[stroke_start_idx:stroke_end_idx + 1]
        
        # Find max values for normalization
        max_wrist_speed = max(wrist_velocities) if wrist_velocities else 1.0
        
        for i, frame in enumerate(window_frames):
            landmarks = frame.get("landmarks")
            if not landmarks:
                continue
            
            timestamp = frame.get("timestamp", 0)
            frame_idx = stroke_start_idx + i
            
            # Get landmarks
            shoulder = self._get_landmark(landmarks, f"{side}_shoulder")
            elbow = self._get_landmark(landmarks, f"{side}_elbow")
            wrist = self._get_landmark(landmarks, f"{side}_wrist")
            
            if not all([shoulder, elbow, wrist]):
                continue
            
            # Check visibility
            if wrist[2] < 0.5:
                continue
            
            # === SCORE 1: Wrist Speed ===
            # Find closest velocity timestamp
            wrist_speed = 0.0
            for j, vt in enumerate(velocity_timestamps):
                if abs(vt - timestamp) < 0.05:
                    wrist_speed = wrist_velocities[j]
                    break
            wrist_speed_score = wrist_speed / max_wrist_speed if max_wrist_speed > 0 else 0
            
            # === SCORE 2: Arm Extension ===
            elbow_angle = self._compute_elbow_angle(shoulder, elbow, wrist)
            # Normalize: 90 degrees = 0, 180 degrees = 1
            arm_extension_score = max(0, (elbow_angle - 90) / 90)
            
            # === SCORE 3: Wrist Height ===
            # Relative to shoulder (negative = above)
            wrist_height = wrist[1] - shoulder[1]
            # Normalize: at shoulder = 0, well above = 1
            wrist_height_score = max(0, min(1, -wrist_height / 0.2))
            
            # === COMPOSITE SCORE ===
            composite = (
                wrist_speed_score * self.wrist_speed_weight +
                arm_extension_score * self.arm_extension_weight +
                wrist_height_score * self.wrist_height_weight
            )
            
            scores.append(ContactProxyScore(
                timestamp=timestamp,
                frame_index=frame_idx,
                wrist_speed_score=wrist_speed_score,
                arm_extension_score=arm_extension_score,
                wrist_height_score=wrist_height_score,
                composite_score=composite,
                wrist_speed=wrist_speed,
                elbow_angle=elbow_angle,
                wrist_height=wrist_height
            ))
        
        return scores
    
    def find_best_contact_proxy(
        self,
        frames: List[Dict],
        stroke_start_idx: int,
        stroke_end_idx: int,
        side: str,
        wrist_velocities: List[float],
        velocity_timestamps: List[float]
    ) -> Optional[ContactProxyScore]:
        """
        Find the best contact proxy using composite score.
        
        Returns the ContactProxyScore with highest composite score.
        """
        scores = self.compute_contact_scores(
            frames, stroke_start_idx, stroke_end_idx, side,
            wrist_velocities, velocity_timestamps
        )
        
        if not scores:
            return None
        
        return max(scores, key=lambda s: s.composite_score)


class PrepPhaseAnalyzer:
    """
    Analyzes preparation phase before overhead stroke.
    Checks elbow preparation and non-racket arm usage.
    """
    
    def __init__(self):
        self._thresholds = get_thresholds()
        
        # Thresholds for prep phase
        self.elbow_above_shoulder_threshold = 0.02  # How much elbow should be above shoulder (y)
        self.non_racket_arm_threshold = 0.1  # How high non-racket wrist should be
        self.prep_window_sec = 0.4  # How long before swing to check
    
    def _get_landmark(self, landmarks: List[Dict], name: str) -> Optional[Tuple[float, float, float]]:
        """Get landmark (x, y, visibility)"""
        idx = LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        x = lm.get("x_smooth", lm.get("x", 0))
        y = lm.get("y_smooth", lm.get("y", 0))
        return (x, y, lm.get("visibility", 0))
    
    def analyze_prep_phase(
        self,
        frames: List[Dict],
        stroke_start_timestamp: float,
        dominant_side: str
    ) -> PrepPhaseAnalysis:
        """
        Analyze preparation phase before stroke.
        
        Args:
            frames: All frames
            stroke_start_timestamp: When swing starts
            dominant_side: "left" or "right"
        
        Returns:
            PrepPhaseAnalysis with findings
        """
        non_dominant = "left" if dominant_side == "right" else "right"
        
        # Find prep frames (before swing)
        prep_start = stroke_start_timestamp - self.prep_window_sec
        prep_frames = [f for f in frames 
                      if prep_start <= f.get("timestamp", 0) <= stroke_start_timestamp]
        
        if not prep_frames:
            return PrepPhaseAnalysis(
                prep_start_timestamp=prep_start,
                prep_end_timestamp=stroke_start_timestamp,
                elbow_above_shoulder=True,
                elbow_elevation=0,
                arm_prepared=True,
                non_racket_arm_up=True,
                non_racket_wrist_height=0,
                prep_quality="unknown"
            )
        
        # Analyze each prep frame
        elbow_elevations = []
        non_racket_heights = []
        
        for frame in prep_frames:
            landmarks = frame.get("landmarks")
            if not landmarks:
                continue
            
            # Dominant arm (racket arm)
            dom_shoulder = self._get_landmark(landmarks, f"{dominant_side}_shoulder")
            dom_elbow = self._get_landmark(landmarks, f"{dominant_side}_elbow")
            
            if dom_shoulder and dom_elbow and dom_elbow[2] > 0.5:
                # Negative = elbow above shoulder (in image coords)
                elbow_elev = dom_elbow[1] - dom_shoulder[1]
                elbow_elevations.append(elbow_elev)
            
            # Non-dominant arm
            non_shoulder = self._get_landmark(landmarks, f"{non_dominant}_shoulder")
            non_wrist = self._get_landmark(landmarks, f"{non_dominant}_wrist")
            
            if non_shoulder and non_wrist and non_wrist[2] > 0.5:
                # How high is non-racket wrist relative to shoulder
                wrist_height = non_shoulder[1] - non_wrist[1]  # Positive = above
                non_racket_heights.append(wrist_height)
        
        # Compute averages
        avg_elbow_elev = sum(elbow_elevations) / len(elbow_elevations) if elbow_elevations else 0
        avg_non_racket = sum(non_racket_heights) / len(non_racket_heights) if non_racket_heights else 0
        
        # Checks
        elbow_above = avg_elbow_elev < -self.elbow_above_shoulder_threshold
        arm_prepared = elbow_above
        non_racket_up = avg_non_racket > self.non_racket_arm_threshold
        
        # Determine issues
        issues = []
        if not elbow_above:
            issues.append("elbow_not_up")
        if not non_racket_up:
            issues.append("non_racket_arm_down")
        
        # Quality
        if not issues:
            prep_quality = "good"
        elif len(issues) == 1:
            prep_quality = "fair"
        else:
            prep_quality = "poor"
        
        return PrepPhaseAnalysis(
            prep_start_timestamp=prep_start,
            prep_end_timestamp=stroke_start_timestamp,
            elbow_above_shoulder=elbow_above,
            elbow_elevation=avg_elbow_elev,
            arm_prepared=arm_prepared,
            non_racket_arm_up=non_racket_up,
            non_racket_wrist_height=avg_non_racket,
            prep_quality=prep_quality,
            issues=issues
        )
