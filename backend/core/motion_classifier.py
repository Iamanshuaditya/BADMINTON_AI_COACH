"""
Motion Classifier Module
Auto-detects drill type from pose frame motion patterns.

Classifies drills into:
- overhead-shadow: Repeated overhead arm movements (wrist above shoulder, vertical arm velocity spikes)
- footwork: Lateral hip movement with direction changes, arms mostly below shoulder
- 6-corner-shadow: Combined footwork patterns with direction changes
- unknown: Cannot confidently classify

Robust to camera angles (side/front/back/45°) by using relative comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import get_thresholds

logger = logging.getLogger(__name__)


# MediaPipe landmark indices
LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


@dataclass
class ClassificationResult:
    """Result of drill type classification"""
    detected_drill_type: str  # "overhead-shadow", "footwork", "6-corner-shadow", "unknown"
    confidence: float  # 0.0 - 1.0
    debug_features: Dict = field(default_factory=dict)
    reason: str = ""  # Human-readable reason for classification


@dataclass
class MotionFeatures:
    """Extracted motion features for classification"""
    # Overhead indicators
    wrist_above_shoulder_fraction: float = 0.0
    avg_wrist_vertical_velocity: float = 0.0
    wrist_velocity_spikes: int = 0
    max_wrist_height_relative: float = 0.0  # max(shoulder_y - wrist_y) / shoulder_distance
    
    # Footwork indicators
    hip_lateral_range: float = 0.0  # max - min of hip x position
    direction_change_count: int = 0
    avg_hip_lateral_velocity: float = 0.0
    arms_below_shoulder_fraction: float = 0.0
    
    # Quality metrics
    valid_frame_count: int = 0
    total_frame_count: int = 0
    avg_pose_confidence: float = 0.0


def get_landmark(landmarks: List[Dict], name: str) -> Optional[Tuple[float, float, float]]:
    """Get landmark by name from pose frame landmarks list."""
    idx = LANDMARKS.get(name)
    if idx is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    return (lm.get("x", 0), lm.get("y", 0), lm.get("visibility", 0))


def compute_frame_confidence(landmarks: List[Dict]) -> float:
    """Compute average visibility confidence for key landmarks."""
    key_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                 "left_wrist", "right_wrist"]
    confidences = []
    for name in key_names:
        lm = get_landmark(landmarks, name)
        if lm:
            confidences.append(lm[2])
    return sum(confidences) / len(confidences) if confidences else 0.0


def extract_motion_features(frames: List[Dict], config) -> MotionFeatures:
    """
    Extract motion features from a list of pose frames.
    
    Args:
        frames: List of pose frame dicts with format:
                {
                    "landmarks": [{"x":.., "y":.., "z":.., "visibility":..}, ...],
                    "timestamp": float,
                    "confidence": float (optional)
                }
        config: DrillClassifierConfig with thresholds
    
    Returns:
        MotionFeatures for classification
    """
    features = MotionFeatures()
    features.total_frame_count = len(frames)
    
    if len(frames) < config.min_frames:
        return features
    
    # Filter frames by confidence
    valid_frames = []
    for frame in frames:
        landmarks = frame.get("landmarks")
        if not landmarks:
            continue
        
        conf = frame.get("confidence", compute_frame_confidence(landmarks))
        if conf >= config.min_pose_confidence:
            valid_frames.append((frame, conf))
    
    features.valid_frame_count = len(valid_frames)
    
    if len(valid_frames) < config.min_frames:
        return features
    
    features.avg_pose_confidence = sum(c for _, c in valid_frames) / len(valid_frames)
    
    # Collect per-frame measurements
    wrist_above_shoulder_count = 0
    arms_below_shoulder_count = 0
    hip_x_positions = []
    wrist_heights = []  # relative to shoulder
    prev_wrist_y = None
    prev_hip_x = None
    wrist_velocity_sum = 0
    hip_velocity_sum = 0
    direction_changes = 0
    prev_hip_direction = 0  # -1, 0, 1
    velocity_spike_count = 0
    
    for frame, conf in valid_frames:
        landmarks = frame.get("landmarks", [])
        
        # Get key landmarks
        left_shoulder = get_landmark(landmarks, "left_shoulder")
        right_shoulder = get_landmark(landmarks, "right_shoulder")
        left_wrist = get_landmark(landmarks, "left_wrist")
        right_wrist = get_landmark(landmarks, "right_wrist")
        left_hip = get_landmark(landmarks, "left_hip")
        right_hip = get_landmark(landmarks, "right_hip")
        
        if not all([left_shoulder, right_shoulder, left_wrist, right_wrist, left_hip, right_hip]):
            continue
        
        # Compute centers and distances for normalization
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_width < 0.01:
            shoulder_width = 0.1  # fallback
        
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        
        # Use the higher wrist (dominant arm indicator)
        # In image coords, lower y = higher position
        if left_wrist[1] < right_wrist[1]:
            wrist_y = left_wrist[1]
            wrist_x = left_wrist[0]
        else:
            wrist_y = right_wrist[1]
            wrist_x = right_wrist[0]
        
        avg_shoulder_y = shoulder_center_y
        
        # Check wrist above shoulder (lower y = higher in image)
        # Relative: wrist above shoulder by some margin
        wrist_relative_height = avg_shoulder_y - wrist_y  # positive = above shoulder
        wrist_heights.append(wrist_relative_height)
        
        if wrist_relative_height > 0.02:  # wrist noticeably above shoulder
            wrist_above_shoulder_count += 1
        
        # Check arms below shoulder for footwork signature
        both_wrists_below = left_wrist[1] > avg_shoulder_y and right_wrist[1] > avg_shoulder_y
        if both_wrists_below:
            arms_below_shoulder_count += 1
        
        # Track hip lateral position (normalized by shoulder width)
        hip_x_normalized = hip_center_x / shoulder_width if shoulder_width > 0 else 0
        hip_x_positions.append(hip_x_normalized)
        
        # Compute velocities
        if prev_wrist_y is not None:
            wrist_v = prev_wrist_y - wrist_y  # positive = moving up
            wrist_velocity_sum += abs(wrist_v)
            
            # Detect velocity spikes (rapid arm movements upward)
            if wrist_v > 0.03:  # significant upward motion
                velocity_spike_count += 1
        
        if prev_hip_x is not None:
            hip_v = hip_center_x - prev_hip_x
            hip_velocity_sum += abs(hip_v)
            
            # Detect direction changes
            current_dir = 1 if hip_v > 0.005 else (-1 if hip_v < -0.005 else 0)
            if current_dir != 0 and prev_hip_direction != 0 and current_dir != prev_hip_direction:
                direction_changes += 1
            if current_dir != 0:
                prev_hip_direction = current_dir
        
        prev_wrist_y = wrist_y
        prev_hip_x = hip_center_x
    
    # Compute final features
    n = len(valid_frames)
    if n > 0:
        features.wrist_above_shoulder_fraction = wrist_above_shoulder_count / n
        features.arms_below_shoulder_fraction = arms_below_shoulder_count / n
        features.avg_wrist_vertical_velocity = wrist_velocity_sum / n
        features.avg_hip_lateral_velocity = hip_velocity_sum / n
        features.wrist_velocity_spikes = velocity_spike_count
    
    if wrist_heights:
        features.max_wrist_height_relative = max(wrist_heights)
    
    if hip_x_positions:
        features.hip_lateral_range = max(hip_x_positions) - min(hip_x_positions)
    
    features.direction_change_count = direction_changes
    
    return features


def classify_drill(frames: List[Dict]) -> ClassificationResult:
    """
    Classify drill type from pose frames using rules-based analysis.
    
    Args:
        frames: List of pose frame dicts with format:
                {
                    "landmarks": [{"x":.., "y":.., "z":.., "visibility":..}, ...],
                    "timestamp": float,
                    "confidence": float (optional)
                }
    
    Returns:
        ClassificationResult with drill type, confidence, and debug info
    """
    config = get_thresholds().drill_classifier
    
    # Extract motion features
    features = extract_motion_features(frames, config)
    
    # Check minimum requirements
    if features.valid_frame_count < config.min_frames:
        logger.debug(
            "Drill classification failed: insufficient frames "
            f"({features.valid_frame_count}/{config.min_frames})"
        )
        return ClassificationResult(
            detected_drill_type="unknown",
            confidence=0.0,
            debug_features={
                "valid_frames": features.valid_frame_count,
                "required_frames": config.min_frames
            },
            reason=f"Insufficient valid frames ({features.valid_frame_count}/{config.min_frames})"
        )
    
    # Score each drill type
    overhead_score = 0.0
    footwork_score = 0.0
    corner_score = 0.0
    
    # Overhead scoring
    # - High wrist above shoulder fraction
    # - Velocity spikes from arm movements
    # - Max wrist height above head level
    if features.wrist_above_shoulder_fraction >= config.overhead_wrist_above_shoulder_fraction:
        overhead_score += 0.4
    if features.wrist_above_shoulder_fraction >= 0.5:
        overhead_score += 0.2
    if features.wrist_velocity_spikes >= 3:
        overhead_score += 0.2
    if features.max_wrist_height_relative > 0.1:
        overhead_score += 0.2
    
    # Footwork scoring
    # - High hip lateral movement range
    # - Direction changes
    # - Arms mostly below shoulder
    if features.hip_lateral_range >= config.footwork_hip_lateral_threshold:
        footwork_score += 0.3
    if features.hip_lateral_range >= 0.1:
        footwork_score += 0.2
    if features.direction_change_count >= 3:
        footwork_score += 0.2
    if features.arms_below_shoulder_fraction >= 0.6:
        footwork_score += 0.3
    
    # 6-corner scoring (combination of footwork + some overhead elements)
    if features.direction_change_count >= 5:
        corner_score += 0.4
    if features.hip_lateral_range >= 0.08:
        corner_score += 0.3
    if features.wrist_above_shoulder_fraction >= 0.15:
        corner_score += 0.3
    
    # Determine winner
    scores = {
        "overhead-shadow": overhead_score,
        "footwork": footwork_score,
        "6-corner-shadow": corner_score
    }
    
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    
    # Compute confidence based on margin
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        margin = sorted_scores[0] - sorted_scores[1]
        # Confidence = score * (1 + margin) capped at 1.0
        confidence = min(1.0, best_score * (1 + margin))
    else:
        confidence = best_score
    
    # If confidence is below threshold, return unknown
    if confidence < config.confidence_threshold or best_score < 0.3:
        logger.debug(
            f"Drill classification low confidence: {confidence:.2f} "
            f"(threshold {config.confidence_threshold}), scores={scores}"
        )
        return ClassificationResult(
            detected_drill_type="unknown",
            confidence=confidence,
            debug_features={
                "scores": scores,
                "features": {
                    "wrist_above_shoulder_fraction": round(features.wrist_above_shoulder_fraction, 3),
                    "hip_lateral_range": round(features.hip_lateral_range, 4),
                    "direction_changes": features.direction_change_count,
                    "arms_below_shoulder_fraction": round(features.arms_below_shoulder_fraction, 3),
                    "wrist_velocity_spikes": features.wrist_velocity_spikes,
                    "valid_frames": features.valid_frame_count
                }
            },
            reason=f"Low confidence ({confidence:.2f} < {config.confidence_threshold})"
        )
    
    # Build debug info
    debug_features = {
        "scores": scores,
        "features": {
            "wrist_above_shoulder_fraction": round(features.wrist_above_shoulder_fraction, 3),
            "hip_lateral_range": round(features.hip_lateral_range, 4),
            "direction_changes": features.direction_change_count,
            "arms_below_shoulder_fraction": round(features.arms_below_shoulder_fraction, 3),
            "wrist_velocity_spikes": features.wrist_velocity_spikes,
            "max_wrist_height_relative": round(features.max_wrist_height_relative, 4),
            "avg_pose_confidence": round(features.avg_pose_confidence, 3),
            "valid_frames": features.valid_frame_count
        }
    }
    
    reasons = {
        "overhead-shadow": f"High wrist elevation ({features.wrist_above_shoulder_fraction:.0%} above shoulder)",
        "footwork": f"Strong lateral movement (range={features.hip_lateral_range:.3f}, {features.direction_change_count} direction changes)",
        "6-corner-shadow": f"Multi-directional pattern ({features.direction_change_count} changes)"
    }
    
    logger.debug(f"Drill classified as '{best_type}' with confidence {confidence:.2f}. Scores: {scores}")
    
    return ClassificationResult(
        detected_drill_type=best_type,
        confidence=confidence,
        debug_features=debug_features,
        reason=reasons.get(best_type, "")
    )
