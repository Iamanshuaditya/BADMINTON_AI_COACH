"""
ShuttleSense V1 - Configurable Thresholds
All thresholds can be tuned without code changes by modifying this file.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SplitStepConfig:
    """Split step detection thresholds"""
    # Minimum vertical velocity (normalized) to detect hop
    min_vertical_velocity: float = 0.02
    # Window duration in milliseconds
    window_ms: int = 400  # 200-500ms typical
    # Minimum height change ratio for valid split step
    min_height_change_ratio: float = 0.08
    # Minimum duration in seconds
    min_duration_sec: float = 0.15
    # Maximum duration in seconds
    max_duration_sec: float = 0.6


@dataclass
class StanceConfig:
    """Stance width thresholds"""
    # Stance width ratio = ankle_distance / shoulder_distance
    min_ratio: float = 0.7
    max_ratio: float = 1.5
    # Optimal range
    optimal_min: float = 0.9
    optimal_max: float = 1.3
    # Sustained window for narrow stance detection (frames)
    sustained_window_frames: int = 10


@dataclass
class KneeConfig:
    """Knee angle and valgus thresholds"""
    # Knee flexion angle for lunge detection (degrees)
    lunge_flexion_min: float = 45.0
    lunge_flexion_max: float = 120.0
    # Knee valgus (collapse) threshold in degrees
    valgus_threshold: float = 8.0  # >8° deviation flags issue
    # Confidence gate - ignore if landmark confidence < this
    confidence_gate: float = 0.6


@dataclass
class RecoveryConfig:
    """Recovery to base thresholds"""
    # Maximum time to return to base (seconds)
    max_recovery_time_sec: float = 2.0
    # Base zone tolerance (normalized distance from calibrated base)
    base_zone_tolerance: float = 0.08
    # Slow recovery threshold
    slow_recovery_sec: float = 1.5
    # Calibration frames (first N frames to establish base position)
    calibration_frames: int = 30


@dataclass
class BalanceConfig:
    """Balance and stability thresholds"""
    # Maximum acceleration for stable landing (m/s^2 proxy)
    max_landing_acceleration: float = 0.05
    # Post-landing window for stability check (frames)
    stability_window_frames: int = 10
    # Oscillation threshold (normalized movement)
    oscillation_threshold: float = 0.015


@dataclass
class PostureConfig:
    """Posture thresholds"""
    # Torso lean angle thresholds (degrees from vertical)
    min_lean: float = 5.0
    max_lean: float = 35.0
    # Upright (too tall) threshold
    upright_threshold: float = 8.0


@dataclass
class FSMConfig:
    """Finite State Machine configuration"""
    # Hysteresis - minimum time in a state before transition (seconds)
    min_state_duration_sec: float = 0.1
    # Cooldown between same cue emission (seconds)
    cue_cooldown_sec: float = 1.5
    # Window size for windowed computations (frames)
    window_size_frames: int = 7
    # Minimum frames for event detection
    min_event_frames: int = 3


@dataclass
class VisibilityConfig:
    """Landmark visibility thresholds"""
    # Minimum visibility to trust a landmark
    min_visibility: float = 0.5
    # High confidence threshold
    high_confidence: float = 0.7
    # Critical landmarks that must be visible for analysis
    critical_landmarks: tuple = ("left_ankle", "right_ankle", "left_hip", 
                                  "right_hip", "left_knee", "right_knee")


@dataclass
class SmoothingConfig:
    """OneEuro filter parameters"""
    # Minimum cutoff frequency (lower = more smoothing)
    min_cutoff: float = 1.0
    # Speed coefficient (higher = less lag for fast movements)
    beta: float = 0.007
    # Derivative cutoff
    d_cutoff: float = 1.0
    # Feet-specific parameters (more smoothing needed)
    feet_min_cutoff: float = 0.8
    feet_beta: float = 0.01


@dataclass
class ThresholdConfig:
    """Master threshold configuration"""
    split_step: SplitStepConfig = field(default_factory=SplitStepConfig)
    stance: StanceConfig = field(default_factory=StanceConfig)
    knee: KneeConfig = field(default_factory=KneeConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


# Global config instance - modify this to tune thresholds
THRESHOLDS = ThresholdConfig()


def get_thresholds() -> ThresholdConfig:
    """Get the current threshold configuration"""
    return THRESHOLDS
