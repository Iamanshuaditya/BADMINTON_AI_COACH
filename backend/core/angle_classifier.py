"""
Camera Angle Classifier
Auto-classifies camera angle from pose landmarks to route rules appropriately.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from config import get_thresholds

logger = logging.getLogger(__name__)


@dataclass
class AngleClassification:
    """Camera angle classification result"""
    angle_class: str  # "front", "side", "45deg"
    confidence: float
    
    # Derived axis for "in front" checks
    frontness_axis: str  # "x", "z_proxy", "combined"
    
    # Raw measurements
    shoulder_x_ratio: float  # shoulder width / image width proxy
    hip_shoulder_alignment: float  # how aligned are hip/shoulders
    occlusion_score: float  # how much one side occludes the other
    
    # Flags
    is_reliable: bool


# Landmark indices
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "nose": 0,
    "left_ankle": 27,
    "right_ankle": 28,
}


class CameraAngleClassifier:
    """
    Classifies camera angle from pose landmarks.
    
    Front-ish: Shoulders wide in x, both hips/shoulders visible
    Side-ish: Shoulders narrow in x, one side occluding
    45°: In between
    """
    
    def __init__(self):
        # Thresholds for classification
        self.front_shoulder_ratio_min = 0.15  # shoulders > 15% of frame width = front-ish
        self.side_shoulder_ratio_max = 0.08   # shoulders < 8% = side-ish
        self.visibility_threshold = 0.5
        self.angle_confidence_min = 0.6
    
    def _get_landmark(self, landmarks: List[Dict], name: str) -> Optional[Tuple[float, float, float]]:
        """Get landmark (x, y, visibility)"""
        idx = LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        return (lm.get("x", 0), lm.get("y", 0), lm.get("visibility", 0))
    
    def classify_single_frame(self, landmarks: List[Dict]) -> AngleClassification:
        """Classify camera angle from a single frame's landmarks"""
        
        # Get key landmarks
        left_shoulder = self._get_landmark(landmarks, "left_shoulder")
        right_shoulder = self._get_landmark(landmarks, "right_shoulder")
        left_hip = self._get_landmark(landmarks, "left_hip")
        right_hip = self._get_landmark(landmarks, "right_hip")
        
        # Default if can't classify
        default = AngleClassification(
            angle_class="unknown",
            confidence=0.0,
            frontness_axis="x",
            shoulder_x_ratio=0,
            hip_shoulder_alignment=0,
            occlusion_score=0,
            is_reliable=False
        )
        
        if not all([left_shoulder, right_shoulder]):
            return default
        
        # Check visibility
        ls_vis = left_shoulder[2]
        rs_vis = right_shoulder[2]
        
        if ls_vis < self.visibility_threshold and rs_vis < self.visibility_threshold:
            return default
        
        # === COMPUTE METRICS ===
        
        # 1. Shoulder width ratio (x distance between shoulders)
        shoulder_x_dist = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_x_ratio = shoulder_x_dist
        
        # 2. Hip-shoulder alignment (are they in same plane?)
        hip_shoulder_alignment = 0.0
        if left_hip and right_hip:
            hip_x_dist = abs(right_hip[0] - left_hip[0])
            hip_shoulder_alignment = 1.0 - abs(shoulder_x_dist - hip_x_dist) / max(shoulder_x_dist, 0.01)
        
        # 3. Occlusion score (visibility difference between sides)
        lh_vis = left_hip[2] if left_hip else 0
        rh_vis = right_hip[2] if right_hip else 0
        
        left_side_vis = (ls_vis + lh_vis) / 2
        right_side_vis = (rs_vis + rh_vis) / 2
        occlusion_score = abs(left_side_vis - right_side_vis)
        
        # === CLASSIFY ===
        
        # Front-ish: wide shoulders, both sides visible, good alignment
        if shoulder_x_ratio >= self.front_shoulder_ratio_min and occlusion_score < 0.3:
            angle_class = "front"
            confidence = min(1.0, shoulder_x_ratio / 0.2) * (1 - occlusion_score)
            frontness_axis = "x"
        
        # Side-ish: narrow shoulders, one side occluding
        elif shoulder_x_ratio <= self.side_shoulder_ratio_max or occlusion_score > 0.5:
            angle_class = "side"
            confidence = min(1.0, (1 - shoulder_x_ratio / 0.15)) * occlusion_score
            frontness_axis = "z_proxy"  # Use depth proxy (shoulder-to-wrist direction)
        
        # 45 degrees: in between
        else:
            angle_class = "45deg"
            confidence = 0.7  # Medium confidence for in-between
            frontness_axis = "combined"
        
        # Ensure confidence is reasonable
        confidence = max(0.3, min(1.0, confidence))
        
        return AngleClassification(
            angle_class=angle_class,
            confidence=confidence,
            frontness_axis=frontness_axis,
            shoulder_x_ratio=shoulder_x_ratio,
            hip_shoulder_alignment=hip_shoulder_alignment,
            occlusion_score=occlusion_score,
            is_reliable=confidence >= self.angle_confidence_min
        )
    
    def classify_video(self, frames: List[Dict]) -> AngleClassification:
        """
        Classify camera angle from multiple frames (voting).
        Uses majority voting with confidence weighting.
        """
        if not frames:
            return AngleClassification(
                angle_class="unknown", confidence=0, frontness_axis="x",
                shoulder_x_ratio=0, hip_shoulder_alignment=0, occlusion_score=0,
                is_reliable=False
            )
        
        # Classify each frame
        classifications = []
        for frame in frames:
            landmarks = frame.get("landmarks")
            if landmarks:
                clf = self.classify_single_frame(landmarks)
                if clf.confidence > 0.3:
                    classifications.append(clf)
        
        if not classifications:
            # Default to front if can't determine
            return AngleClassification(
                angle_class="front", confidence=0.5, frontness_axis="x",
                shoulder_x_ratio=0.15, hip_shoulder_alignment=0.8, occlusion_score=0.2,
                is_reliable=False
            )
        
        # Vote
        votes = {"front": 0, "side": 0, "45deg": 0}
        weighted_metrics = {"shoulder_x_ratio": 0, "hip_shoulder_alignment": 0, "occlusion_score": 0}
        total_weight = 0
        
        for clf in classifications:
            weight = clf.confidence
            if clf.angle_class in votes:
                votes[clf.angle_class] += weight
            weighted_metrics["shoulder_x_ratio"] += clf.shoulder_x_ratio * weight
            weighted_metrics["hip_shoulder_alignment"] += clf.hip_shoulder_alignment * weight
            weighted_metrics["occlusion_score"] += clf.occlusion_score * weight
            total_weight += weight
        
        # Find winner
        winner = max(votes.keys(), key=lambda k: votes[k])
        winner_votes = votes[winner]
        total_votes = sum(votes.values())
        
        confidence = winner_votes / total_votes if total_votes > 0 else 0.5
        
        # Determine frontness axis based on winner
        if winner == "front":
            frontness_axis = "x"
        elif winner == "side":
            frontness_axis = "z_proxy"
        else:
            frontness_axis = "combined"
        
        # Average metrics
        if total_weight > 0:
            for k in weighted_metrics:
                weighted_metrics[k] /= total_weight
        
        return AngleClassification(
            angle_class=winner,
            confidence=confidence,
            frontness_axis=frontness_axis,
            shoulder_x_ratio=weighted_metrics["shoulder_x_ratio"],
            hip_shoulder_alignment=weighted_metrics["hip_shoulder_alignment"],
            occlusion_score=weighted_metrics["occlusion_score"],
            is_reliable=confidence >= self.angle_confidence_min
        )


def compute_frontness(
    wrist_pos: Tuple[float, float],
    shoulder_pos: Tuple[float, float],
    torso_center: Tuple[float, float],
    angle_class: AngleClassification
) -> Tuple[float, float]:
    """
    Compute how far "in front" the wrist is based on camera angle.
    
    Returns: (frontness_value, confidence)
    - Positive = wrist in front
    - Negative = wrist behind
    """
    
    if angle_class.frontness_axis == "x":
        # Front view: use x-axis displacement from torso center
        # In front = wrist further from torso center in direction of swing
        frontness = abs(wrist_pos[0] - torso_center[0])
        # For front view, wrist being above/forward is good
        # Use y-axis: lower y = higher = more in front for overhead
        y_component = (shoulder_pos[1] - wrist_pos[1]) * 0.3
        frontness = frontness + y_component
        confidence = angle_class.confidence * 0.9
        
    elif angle_class.frontness_axis == "z_proxy":
        # Side view: use shoulder-to-wrist direction as depth proxy
        # Wrist should be "leading" (x > shoulder x for right side, etc.)
        dx = wrist_pos[0] - shoulder_pos[0]
        dy = wrist_pos[1] - shoulder_pos[1]
        
        # For side view, "in front" means wrist is in direction of swing
        # Approximate using arm extension direction
        frontness = dy * -0.5 + abs(dx) * 0.5  # Negative dy = wrist above = good
        confidence = angle_class.confidence * 0.6  # Lower confidence for side view
        
    else:  # "combined" for 45 degree
        # Blend x-axis and depth proxy
        x_frontness = abs(wrist_pos[0] - torso_center[0])
        y_frontness = (shoulder_pos[1] - wrist_pos[1]) * 0.5
        frontness = (x_frontness + y_frontness) / 2
        confidence = angle_class.confidence * 0.75
    
    return (frontness, confidence)
