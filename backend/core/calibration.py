"""
Base Stance Calibration
2-second calibration phase to establish baselines for more accurate detection.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .feature_computer import LANDMARKS

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Calibration baselines from initial stance"""
    # Timing
    calibration_start: float
    calibration_end: float
    frames_used: int
    
    # Stance baselines
    baseline_stance_width: float  # Average ankle distance
    baseline_stance_ratio: float  # Average stance width ratio
    shoulder_width_reference: float  # For normalizing
    
    # Position baselines
    base_position: Tuple[float, float]  # Hip center at rest
    base_zone_radius: float  # How much movement is normal at rest
    
    # Posture baselines
    baseline_knee_angle: float  # Average knee angle (less = more bent)
    baseline_torso_lean: float  # Average torso lean angle
    baseline_hip_height: float  # Y position of hip center
    
    # Confidence
    calibration_confidence: float  # 0-1, based on stability
    is_valid: bool
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


@dataclass
class CalibrationConfig:
    """Configuration for calibration phase"""
    duration_sec: float = 2.0
    min_frames: int = 30  # Minimum frames for valid calibration
    stability_threshold: float = 0.02  # Max variance for stable calibration
    visibility_min: float = 0.5


class BaseStanceCalibrator:
    """
    Establishes baseline measurements from initial stance.
    
    Usage:
    1. User stands in ready position for 2 seconds
    2. System calibrates stance width, position, posture
    3. All subsequent checks use these baselines
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self._calibration: Optional[CalibrationData] = None
    
    def _get_landmark(self, landmarks: List[Dict], name: str) -> Optional[Tuple[float, float, float]]:
        """Get landmark (x, y, visibility)"""
        idx = LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        x = lm.get("x_smooth", lm.get("x", 0))
        y = lm.get("y_smooth", lm.get("y", 0))
        return (x, y, lm.get("visibility", 0))
    
    def calibrate(self, frames: List[Dict]) -> CalibrationData:
        """
        Perform calibration from initial frames.
        
        Args:
            frames: All frames from video. Uses first N seconds.
        
        Returns:
            CalibrationData with baselines
        """
        if not frames:
            return self._empty_calibration()
        
        # Find calibration window
        start_time = frames[0].get("timestamp", 0)
        end_time = start_time + self.config.duration_sec
        
        cal_frames = [f for f in frames 
                     if start_time <= f.get("timestamp", 0) <= end_time]
        
        if len(cal_frames) < self.config.min_frames:
            warnings = [f"Only {len(cal_frames)} frames for calibration"]
            # Use what we have
            if not cal_frames:
                return self._empty_calibration()
        else:
            warnings = []
        
        # Collect measurements
        stance_widths = []
        stance_ratios = []
        shoulder_widths = []
        hip_positions = []
        knee_angles = []
        torso_leans = []
        hip_heights = []
        
        for frame in cal_frames:
            landmarks = frame.get("landmarks")
            if not landmarks:
                continue
            
            # Get landmarks
            left_ankle = self._get_landmark(landmarks, "left_ankle")
            right_ankle = self._get_landmark(landmarks, "right_ankle")
            left_shoulder = self._get_landmark(landmarks, "left_shoulder")
            right_shoulder = self._get_landmark(landmarks, "right_shoulder")
            left_hip = self._get_landmark(landmarks, "left_hip")
            right_hip = self._get_landmark(landmarks, "right_hip")
            left_knee = self._get_landmark(landmarks, "left_knee")
            right_knee = self._get_landmark(landmarks, "right_knee")
            
            # Skip if key landmarks not visible
            if not all([left_ankle, right_ankle, left_shoulder, right_shoulder]):
                continue
            
            if any(lm[2] < self.config.visibility_min for lm in [left_ankle, right_ankle, left_shoulder, right_shoulder]):
                continue
            
            # Stance width
            stance_width = math.sqrt(
                (right_ankle[0] - left_ankle[0])**2 + 
                (right_ankle[1] - left_ankle[1])**2
            )
            stance_widths.append(stance_width)
            
            # Shoulder width
            shoulder_width = math.sqrt(
                (right_shoulder[0] - left_shoulder[0])**2 + 
                (right_shoulder[1] - left_shoulder[1])**2
            )
            shoulder_widths.append(shoulder_width)
            
            # Stance ratio
            if shoulder_width > 0.01:
                stance_ratios.append(stance_width / shoulder_width)
            
            # Hip position
            if left_hip and right_hip:
                hip_center = (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                )
                hip_positions.append(hip_center)
                hip_heights.append(hip_center[1])
            
            # Knee angles (simplified - angle at knee joint)
            if left_knee and left_hip:
                # Vector from knee to hip
                kh_y = left_hip[1] - left_knee[1]
                # Approximate knee angle from vertical
                if left_ankle:
                    ka_y = left_ankle[1] - left_knee[1]
                    if ka_y > 0.001:
                        angle = 180 - abs(math.degrees(math.atan2(kh_y, 0.1)))
                        knee_angles.append(angle)
            
            # Torso lean (angle from vertical)
            if left_hip and right_hip:
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                
                dx = shoulder_center[0] - hip_center[0]
                dy = hip_center[1] - shoulder_center[1]  # Positive = shoulders above hips
                
                if dy > 0.001:
                    lean_angle = abs(math.degrees(math.atan2(dx, dy)))
                    torso_leans.append(lean_angle)
        
        # Compute averages
        if not stance_widths:
            return self._empty_calibration()
        
        avg_stance_width = sum(stance_widths) / len(stance_widths)
        avg_stance_ratio = sum(stance_ratios) / len(stance_ratios) if stance_ratios else 1.0
        avg_shoulder_width = sum(shoulder_widths) / len(shoulder_widths) if shoulder_widths else 0.2
        avg_knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else 160
        avg_torso_lean = sum(torso_leans) / len(torso_leans) if torso_leans else 10
        avg_hip_height = sum(hip_heights) / len(hip_heights) if hip_heights else 0.5
        
        # Base position
        if hip_positions:
            avg_hip_x = sum(p[0] for p in hip_positions) / len(hip_positions)
            avg_hip_y = sum(p[1] for p in hip_positions) / len(hip_positions)
            base_position = (avg_hip_x, avg_hip_y)
            
            # Compute movement variance for base zone radius
            variances = [(p[0] - avg_hip_x)**2 + (p[1] - avg_hip_y)**2 for p in hip_positions]
            base_zone_radius = math.sqrt(sum(variances) / len(variances)) * 2  # 2 sigma
        else:
            base_position = (0.5, 0.5)
            base_zone_radius = 0.05
        
        # Stability check
        stance_variance = sum((w - avg_stance_width)**2 for w in stance_widths) / len(stance_widths)
        is_stable = stance_variance < self.config.stability_threshold
        
        if not is_stable:
            warnings.append("Stance was not stable during calibration")
        
        # Confidence based on frames used and stability
        frame_ratio = len(stance_widths) / self.config.min_frames
        confidence = min(1.0, frame_ratio) * (0.5 + 0.5 * (1 if is_stable else 0.5))
        
        return CalibrationData(
            calibration_start=start_time,
            calibration_end=end_time,
            frames_used=len(stance_widths),
            baseline_stance_width=avg_stance_width,
            baseline_stance_ratio=avg_stance_ratio,
            shoulder_width_reference=avg_shoulder_width,
            base_position=base_position,
            base_zone_radius=base_zone_radius,
            baseline_knee_angle=avg_knee_angle,
            baseline_torso_lean=avg_torso_lean,
            baseline_hip_height=avg_hip_height,
            calibration_confidence=confidence,
            is_valid=confidence >= 0.5,
            warnings=warnings
        )
    
    def _empty_calibration(self) -> CalibrationData:
        """Return empty/default calibration"""
        return CalibrationData(
            calibration_start=0,
            calibration_end=0,
            frames_used=0,
            baseline_stance_width=0.2,
            baseline_stance_ratio=1.0,
            shoulder_width_reference=0.2,
            base_position=(0.5, 0.5),
            base_zone_radius=0.08,
            baseline_knee_angle=160,
            baseline_torso_lean=10,
            baseline_hip_height=0.5,
            calibration_confidence=0.3,
            is_valid=False,
            warnings=["No calibration data available"]
        )
    
    def get_calibration(self) -> Optional[CalibrationData]:
        """Get current calibration data"""
        return self._calibration
    
    def check_stance_against_baseline(
        self,
        current_stance_ratio: float,
        calibration: CalibrationData
    ) -> Tuple[str, float]:
        """
        Check current stance against calibrated baseline.
        
        Returns:
            (status, deviation)
            status: "good", "narrow", "wide"
            deviation: How far from baseline (0 = perfect match)
        """
        baseline = calibration.baseline_stance_ratio
        deviation = (current_stance_ratio - baseline) / baseline
        
        if deviation < -0.2:  # 20% narrower than baseline
            return ("narrow", abs(deviation))
        elif deviation > 0.3:  # 30% wider (usually fine)
            return ("wide", deviation)
        else:
            return ("good", abs(deviation))
    
    def check_position_against_base(
        self,
        current_position: Tuple[float, float],
        calibration: CalibrationData
    ) -> Tuple[bool, float]:
        """
        Check if current position is within base zone.
        
        Returns:
            (is_at_base, distance_from_base)
        """
        base = calibration.base_position
        distance = math.sqrt(
            (current_position[0] - base[0])**2 + 
            (current_position[1] - base[1])**2
        )
        
        is_at_base = distance <= calibration.base_zone_radius
        return (is_at_base, distance)
