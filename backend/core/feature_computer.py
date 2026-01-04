"""
Feature Computation Module
Computes biomechanical features from pose landmarks for event detection.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from config import get_thresholds


# MediaPipe landmark indices
LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32
}


@dataclass
class FrameFeatures:
    """Computed features for a single frame"""
    timestamp: float
    frame_index: int
    
    # Distances (normalized 0-1)
    ankle_distance: float
    shoulder_distance: float
    stance_width_ratio: float
    
    # Centers
    hip_center: Tuple[float, float]
    shoulder_center: Tuple[float, float]
    com_proxy: Tuple[float, float]  # Center of mass approximation (midpoint of hips)
    
    # Velocities (computed from deltas)
    vertical_velocity: float  # Positive = moving up
    horizontal_velocity: float
    com_velocity_magnitude: float
    
    # Angles (degrees)
    left_knee_angle: float
    right_knee_angle: float
    torso_lean_angle: float  # From vertical
    
    # Knee valgus (collapse) - deviation from hip->ankle line
    left_knee_valgus: float
    right_knee_valgus: float
    
    # Base position tracking
    distance_from_base: float
    
    # Confidence/reliability
    lower_body_confidence: float
    is_reliable: bool
    
    # Raw landmark positions for reference
    landmarks: Optional[Dict] = None


class FeatureComputer:
    """
    Computes per-frame features from pose landmarks.
    Maintains state for velocity computation and base position calibration.
    """
    
    def __init__(self):
        self._thresholds = get_thresholds()
        self._prev_frame: Optional[Dict] = None
        self._prev_timestamp: float = 0
        self._base_position: Optional[Tuple[float, float]] = None
        self._calibration_frames: List[Tuple[float, float]] = []
        self._frame_count: int = 0
    
    def reset(self):
        """Reset state for new video"""
        self._prev_frame = None
        self._prev_timestamp = 0
        self._base_position = None
        self._calibration_frames = []
        self._frame_count = 0
    
    def _get_landmark(
        self, 
        landmarks: List[Dict], 
        name: str,
        use_smoothed: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """Get landmark coordinates by name"""
        idx = LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        
        lm = landmarks[idx]
        if use_smoothed and "x_smooth" in lm:
            return (lm["x_smooth"], lm["y_smooth"], lm.get("visibility", 0))
        return (lm["x"], lm["y"], lm.get("visibility", 0))
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _midpoint(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """Midpoint between two points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def _angle_between_points(
        self, 
        p1: Tuple[float, float], 
        p2: Tuple[float, float], 
        p3: Tuple[float, float]
    ) -> float:
        """
        Compute angle at p2 between p1-p2-p3 in degrees.
        Returns angle in range [0, 180]
        """
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 * mag2 == 0:
            return 0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp for numerical stability
        
        return math.degrees(math.acos(cos_angle))
    
    def _compute_knee_valgus(
        self,
        hip: Tuple[float, float],
        knee: Tuple[float, float],
        ankle: Tuple[float, float]
    ) -> float:
        """
        Compute knee valgus (collapse) in degrees.
        Measures deviation of knee from the line between hip and ankle.
        Positive = knee collapsing inward (valgus)
        """
        # Vector from hip to ankle
        ha = (ankle[0] - hip[0], ankle[1] - hip[1])
        ha_len = math.sqrt(ha[0]**2 + ha[1]**2)
        
        if ha_len == 0:
            return 0
        
        # Unit vector
        ha_unit = (ha[0] / ha_len, ha[1] / ha_len)
        
        # Vector from hip to knee
        hk = (knee[0] - hip[0], knee[1] - hip[1])
        
        # Project knee onto hip-ankle line
        proj_len = hk[0] * ha_unit[0] + hk[1] * ha_unit[1]
        proj = (hip[0] + ha_unit[0] * proj_len, hip[1] + ha_unit[1] * proj_len)
        
        # Perpendicular distance from knee to line
        perp_dist = self._distance(knee, proj)
        
        # Sign: positive if knee is medial (inward), negative if lateral
        # For frontal view, this depends on which leg
        # Simplified: just return absolute deviation as proxy
        
        # Convert to angle approximation using small angle: tan(theta) ≈ theta for small angles
        # angle ≈ atan(perp_dist / proj_len) in degrees
        if proj_len > 0:
            angle = math.degrees(math.atan(perp_dist / proj_len))
        else:
            angle = 0
        
        return abs(angle)
    
    def _compute_torso_lean(
        self,
        shoulder_center: Tuple[float, float],
        hip_center: Tuple[float, float]
    ) -> float:
        """
        Compute torso lean angle from vertical in degrees.
        0 = perfectly upright, positive = leaning forward
        """
        # Vector from hip to shoulder
        dx = shoulder_center[0] - hip_center[0]
        dy = hip_center[1] - shoulder_center[1]  # Flip Y (image coords)
        
        if dy == 0:
            return 90  # Horizontal
        
        # Angle from vertical
        angle = math.degrees(math.atan(abs(dx) / dy))
        return angle
    
    def _calibrate_base_position(self, com: Tuple[float, float]):
        """Collect calibration frames for base position"""
        cfg = self._thresholds.recovery
        
        if len(self._calibration_frames) < cfg.calibration_frames:
            self._calibration_frames.append(com)
        
        if len(self._calibration_frames) == cfg.calibration_frames:
            # Average position as base
            avg_x = sum(p[0] for p in self._calibration_frames) / len(self._calibration_frames)
            avg_y = sum(p[1] for p in self._calibration_frames) / len(self._calibration_frames)
            self._base_position = (avg_x, avg_y)
    
    def compute(self, frame_data: Dict) -> Optional[FrameFeatures]:
        """
        Compute features for a single frame.
        
        Args:
            frame_data: Dict with landmarks from pose extractor
        
        Returns:
            FrameFeatures or None if landmarks unavailable
        """
        landmarks = frame_data.get("landmarks")
        if not landmarks:
            return None
        
        timestamp = frame_data.get("timestamp", 0)
        frame_index = frame_data.get("frame_index", self._frame_count)
        self._frame_count += 1
        
        # Get key landmarks
        left_shoulder = self._get_landmark(landmarks, "left_shoulder")
        right_shoulder = self._get_landmark(landmarks, "right_shoulder")
        left_hip = self._get_landmark(landmarks, "left_hip")
        right_hip = self._get_landmark(landmarks, "right_hip")
        left_knee = self._get_landmark(landmarks, "left_knee")
        right_knee = self._get_landmark(landmarks, "right_knee")
        left_ankle = self._get_landmark(landmarks, "left_ankle")
        right_ankle = self._get_landmark(landmarks, "right_ankle")
        
        # Check if we have minimum required landmarks
        required = [left_shoulder, right_shoulder, left_hip, right_hip,
                   left_knee, right_knee, left_ankle, right_ankle]
        
        if not all(required):
            return None
        
        # Compute lower body confidence
        lower_body_vis = [
            left_hip[2], right_hip[2], 
            left_knee[2], right_knee[2],
            left_ankle[2], right_ankle[2]
        ]
        lower_body_confidence = sum(lower_body_vis) / len(lower_body_vis)
        is_reliable = lower_body_confidence >= self._thresholds.visibility.high_confidence
        
        # Distances
        ankle_distance = self._distance(left_ankle[:2], right_ankle[:2])
        shoulder_distance = self._distance(left_shoulder[:2], right_shoulder[:2])
        
        stance_width_ratio = ankle_distance / shoulder_distance if shoulder_distance > 0 else 1.0
        
        # Centers
        hip_center = self._midpoint(left_hip[:2], right_hip[:2])
        shoulder_center = self._midpoint(left_shoulder[:2], right_shoulder[:2])
        com_proxy = hip_center  # Simple COM approximation
        
        # Calibrate base position from first N frames
        self._calibrate_base_position(com_proxy)
        
        # Distance from base
        if self._base_position:
            distance_from_base = self._distance(com_proxy, self._base_position)
        else:
            distance_from_base = 0
        
        # Velocities
        dt = timestamp - self._prev_timestamp if self._prev_frame else 0
        
        if self._prev_frame and dt > 0:
            prev_com = self._prev_frame.get("com_proxy", com_proxy)
            vertical_velocity = (prev_com[1] - com_proxy[1]) / dt  # Positive = up (y decreases)
            horizontal_velocity = (com_proxy[0] - prev_com[0]) / dt
            com_velocity_magnitude = math.sqrt(vertical_velocity**2 + horizontal_velocity**2)
        else:
            vertical_velocity = 0
            horizontal_velocity = 0
            com_velocity_magnitude = 0
        
        # Knee angles (hip-knee-ankle)
        left_knee_angle = self._angle_between_points(
            left_hip[:2], left_knee[:2], left_ankle[:2]
        )
        right_knee_angle = self._angle_between_points(
            right_hip[:2], right_knee[:2], right_ankle[:2]
        )
        
        # Torso lean
        torso_lean_angle = self._compute_torso_lean(shoulder_center, hip_center)
        
        # Knee valgus
        left_knee_valgus = self._compute_knee_valgus(
            left_hip[:2], left_knee[:2], left_ankle[:2]
        )
        right_knee_valgus = self._compute_knee_valgus(
            right_hip[:2], right_knee[:2], right_ankle[:2]
        )
        
        # Store for next frame
        self._prev_frame = {
            "com_proxy": com_proxy,
            "hip_center": hip_center
        }
        self._prev_timestamp = timestamp
        
        return FrameFeatures(
            timestamp=timestamp,
            frame_index=frame_index,
            ankle_distance=ankle_distance,
            shoulder_distance=shoulder_distance,
            stance_width_ratio=stance_width_ratio,
            hip_center=hip_center,
            shoulder_center=shoulder_center,
            com_proxy=com_proxy,
            vertical_velocity=vertical_velocity,
            horizontal_velocity=horizontal_velocity,
            com_velocity_magnitude=com_velocity_magnitude,
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            torso_lean_angle=torso_lean_angle,
            left_knee_valgus=left_knee_valgus,
            right_knee_valgus=right_knee_valgus,
            distance_from_base=distance_from_base,
            lower_body_confidence=lower_body_confidence,
            is_reliable=is_reliable
        )
    
    def compute_all(self, frames_data: List[Dict]) -> List[FrameFeatures]:
        """Compute features for all frames"""
        self.reset()
        features = []
        for frame in frames_data:
            feat = self.compute(frame)
            if feat:
                features.append(feat)
        return features


def compute_windowed_features(
    features: List[FrameFeatures],
    window_size: int = 7
) -> List[Dict]:
    """
    Compute windowed statistics for smoothed detection.
    
    Args:
        features: List of per-frame features
        window_size: Window size for averaging
    
    Returns:
        List of dicts with windowed stats
    """
    windowed = []
    half_win = window_size // 2
    
    for i, feat in enumerate(features):
        start = max(0, i - half_win)
        end = min(len(features), i + half_win + 1)
        window = features[start:end]
        
        # Compute windowed averages
        avg_stance = sum(f.stance_width_ratio for f in window) / len(window)
        avg_v_vel = sum(f.vertical_velocity for f in window) / len(window)
        avg_h_vel = sum(f.horizontal_velocity for f in window) / len(window)
        avg_left_knee = sum(f.left_knee_angle for f in window) / len(window)
        avg_right_knee = sum(f.right_knee_angle for f in window) / len(window)
        avg_left_valgus = sum(f.left_knee_valgus for f in window) / len(window)
        avg_right_valgus = sum(f.right_knee_valgus for f in window) / len(window)
        avg_confidence = sum(f.lower_body_confidence for f in window) / len(window)
        
        # Compute velocity peaks for event detection
        v_vel_std = np.std([f.vertical_velocity for f in window])
        h_vel_std = np.std([f.horizontal_velocity for f in window])
        
        windowed.append({
            "timestamp": feat.timestamp,
            "frame_index": feat.frame_index,
            "raw": feat,
            "avg_stance_ratio": avg_stance,
            "avg_vertical_velocity": avg_v_vel,
            "avg_horizontal_velocity": avg_h_vel,
            "avg_left_knee_angle": avg_left_knee,
            "avg_right_knee_angle": avg_right_knee,
            "avg_left_valgus": avg_left_valgus,
            "avg_right_valgus": avg_right_valgus,
            "avg_confidence": avg_confidence,
            "vertical_velocity_std": v_vel_std,
            "horizontal_velocity_std": h_vel_std
        })
    
    return windowed
