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
class EmbeddingsConfig:
    """Embeddings and semantic search configuration"""
    # Similarity threshold - below this, answer is "not grounded"
    similarity_threshold: float = 0.35
    # Top-k chunks to retrieve
    top_k: int = 5
    # Minimum similarity to include in response
    min_include_similarity: float = 0.25
    # Model name for sentence-transformers
    model_name: str = "all-MiniLM-L6-v2"
    # Cache embeddings to disk
    cache_embeddings: bool = True


@dataclass
class DrillClassifierConfig:
    """Drill type auto-detection configuration"""
    # Minimum confidence to accept auto-detection
    confidence_threshold: float = 0.55
    # Minimum frames required for classification
    min_frames: int = 30
    # Minimum pose confidence to include frame
    min_pose_confidence: float = 0.7
    # Overhead signature: minimum fraction of frames with wrist above shoulder
    overhead_wrist_above_shoulder_fraction: float = 0.3
    # Footwork signature: minimum hip lateral movement (normalized)
    footwork_hip_lateral_threshold: float = 0.05
    # Default drill type when detection fails
    default_drill_type: str = "footwork"


@dataclass
class ChatGroundingConfig:
    """Chat grounding and response configuration"""
    # Maximum chunks to include in evidence
    max_evidence_chunks: int = 5
    # Minimum relevant chunks required for grounded response
    min_evidence_chunks: int = 2
    # Require timestamp citations in response
    require_citations: bool = True
    # Enable debug info in response
    include_debug: bool = False
    # Suggested questions when grounded=false
    suggested_questions: tuple = (
        "What mistakes did you detect in this session?",
        "How was my split step timing?",
        "What should I focus on first?",
    )


@dataclass
class StrokeConfig:
    """Overhead stroke detection thresholds (for shadow practice)"""
    
    # Wrist speed thresholds (normalized velocity)
    wrist_speed_start_threshold: float = 0.04  # Swing initiation
    wrist_speed_peak_threshold: float = 0.08   # Minimum for valid stroke
    wrist_speed_end_threshold: float = 0.02    # Swing end
    
    # Stroke window timing
    min_stroke_duration_sec: float = 0.15
    max_stroke_duration_sec: float = 0.8
    ready_window_before_sec: float = 0.5  # Check ready position this long before swing
    
    # Contact height rules (relative positions)
    # Wrist should be above shoulder for overhead
    contact_above_shoulder_required: bool = True
    # Ideal: wrist above nose/eyes for full extension
    contact_above_head_ideal: bool = True
    # Height bands (wrist_y relative to shoulder_y, normalized)
    contact_low_threshold: float = 0.0   # At or below shoulder = too low
    contact_medium_threshold: float = -0.1  # Between shoulder and head
    contact_high_threshold: float = -0.2   # Above head level (negative = higher in image coords)
    
    # Contact in front of body
    # Wrist should lead torso center in swing direction
    contact_front_tolerance: float = 0.03  # How much wrist should be in front
    
    # Elbow leads wrist timing
    elbow_lead_min_ms: int = 30   # Elbow should peak this many ms before wrist
    elbow_lead_max_ms: int = 200  # But not more than this
    
    # Overhead intent validation
    overhead_confidence_min: float = 0.6  # Min confidence to give overhead corrections
    # Wrist must be above shoulder for overhead intent
    wrist_above_shoulder_required: bool = True
    # Elbow elevation threshold (elbow_y < shoulder_y by this much)
    elbow_elevation_threshold: float = 0.05
    
    # Ready position checks
    ready_stance_width_min: float = 0.6  # Minimum stance width ratio
    ready_knee_bend_min: float = 155.0   # Max knee angle (less = more bent)
    ready_posture_check: bool = True
    
    # Visibility requirements for stroke analysis
    stroke_visibility_min: float = 0.5
    # Critical landmarks for overhead stroke
    stroke_critical_landmarks: tuple = (
        "left_wrist", "right_wrist",
        "left_elbow", "right_elbow", 
        "left_shoulder", "right_shoulder",
        "nose"
    )
    
    # Cooldown between stroke detections (seconds)
    stroke_cooldown_sec: float = 0.5
    
    # Rolling window for velocity computation (frames)
    velocity_window_frames: int = 5


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
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    drill_classifier: DrillClassifierConfig = field(default_factory=DrillClassifierConfig)
    chat_grounding: ChatGroundingConfig = field(default_factory=ChatGroundingConfig)


# Global config instance - modify this to tune thresholds
THRESHOLDS = ThresholdConfig()


def get_thresholds() -> ThresholdConfig:
    """Get the current threshold configuration"""
    return THRESHOLDS


# Commonly referenced thresholds (aliases)
SIMILARITY_THRESHOLD = THRESHOLDS.embeddings.similarity_threshold
DRILL_CLASSIFIER_CONFIDENCE_THRESHOLD = THRESHOLDS.drill_classifier.confidence_threshold
MIN_POSE_FRAMES_FOR_CLASSIFICATION = THRESHOLDS.drill_classifier.min_frames
