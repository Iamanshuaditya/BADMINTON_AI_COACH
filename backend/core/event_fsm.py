"""
Finite State Machine (FSM) for Badminton Footwork Event Detection
Rules-first approach with hysteresis for stable detection.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .feature_computer import FrameFeatures
from config import get_thresholds

logger = logging.getLogger(__name__)


class FootworkState(Enum):
    """Player footwork states"""
    IDLE = auto()           # Standing at base position
    SPLIT_STEP = auto()     # Performing split step (hop)
    PUSH_OFF = auto()       # Pushing off to move
    MOVING = auto()         # In motion to target
    LUNGE = auto()          # Lunging to reach shuttle
    RECOVERY = auto()       # Returning to base
    DIRECTION_CHANGE = auto()  # Changing movement direction


@dataclass
class DetectedEvent:
    """A detected footwork event"""
    event_type: str
    start_timestamp: float
    end_timestamp: float
    duration: float
    confidence: float
    severity: float = 0.0  # For mistakes: 0-1 scale
    evidence_timestamps: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FSMContext:
    """Context for FSM state tracking"""
    current_state: FootworkState = FootworkState.IDLE
    state_start_time: float = 0
    state_start_frame: int = 0
    state_duration: float = 0
    
    # Split step detection
    split_step_peak_velocity: float = 0
    split_step_start_time: float = 0
    
    # Recovery tracking
    recovery_start_time: float = 0
    recovery_start_distance: float = 0
    
    # Direction change
    last_horizontal_direction: int = 0  # -1=left, 0=stationary, 1=right
    direction_change_time: float = 0
    
    # Lunge tracking
    lunge_start_time: float = 0
    lunge_min_knee_angle: float = 180


class FootworkFSM:
    """
    Finite State Machine for detecting badminton footwork events.
    
    Uses hysteresis and cooldowns for stable detection.
    Follows "rules before ML" principle.
    """
    
    def __init__(self):
        self._thresholds = get_thresholds()
        self._ctx = FSMContext()
        self._events: List[DetectedEvent] = []
        self._last_cue_times: Dict[str, float] = {}
        self._frame_buffer: List[Dict] = []
    
    def reset(self):
        """Reset FSM for new video"""
        self._ctx = FSMContext()
        self._events = []
        self._last_cue_times = {}
        self._frame_buffer = []
    
    def _can_emit_cue(self, cue_type: str, timestamp: float) -> bool:
        """Check if cooldown has elapsed for this cue type"""
        last_time = self._last_cue_times.get(cue_type, -999)
        cooldown = self._thresholds.fsm.cue_cooldown_sec
        return (timestamp - last_time) >= cooldown
    
    def _emit_event(
        self,
        event_type: str,
        start_time: float,
        end_time: float,
        confidence: float,
        severity: float = 0.0,
        evidence: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ):
        """Record a detected event"""
        event = DetectedEvent(
            event_type=event_type,
            start_timestamp=start_time,
            end_timestamp=end_time,
            duration=end_time - start_time,
            confidence=confidence,
            severity=severity,
            evidence_timestamps=evidence or [start_time, end_time],
            metadata=metadata or {}
        )
        self._events.append(event)
        self._last_cue_times[event_type] = end_time
        logger.debug(f"Event detected: {event_type} at {start_time:.2f}s (conf: {confidence:.2f})")
    
    def _transition_to(
        self,
        new_state: FootworkState,
        timestamp: float,
        frame_index: int
    ):
        """Transition to a new state with hysteresis check"""
        old_state = self._ctx.current_state
        
        # Check minimum state duration (hysteresis)
        min_duration = self._thresholds.fsm.min_state_duration_sec
        if self._ctx.state_duration < min_duration:
            return  # Don't transition yet
        
        logger.debug(f"State: {old_state.name} -> {new_state.name} at {timestamp:.2f}s")
        
        self._ctx.current_state = new_state
        self._ctx.state_start_time = timestamp
        self._ctx.state_start_frame = frame_index
        self._ctx.state_duration = 0
    
    def _detect_split_step(self, feat: FrameFeatures, windowed: Dict) -> bool:
        """Detect split step (hop before movement)"""
        cfg = self._thresholds.split_step
        
        # Split step is characterized by:
        # 1. Quick upward velocity (hop)
        # 2. Followed by landing with wider stance
        # 3. Short duration (200-500ms)
        
        v_vel = feat.vertical_velocity
        
        # Detect upward peak
        if v_vel > cfg.min_vertical_velocity:
            if self._ctx.split_step_start_time == 0:
                self._ctx.split_step_start_time = feat.timestamp
                self._ctx.split_step_peak_velocity = v_vel
                return False
            
            # Track peak
            if v_vel > self._ctx.split_step_peak_velocity:
                self._ctx.split_step_peak_velocity = v_vel
        
        # Check for landing (velocity reversal)
        elif self._ctx.split_step_start_time > 0 and v_vel < -cfg.min_vertical_velocity * 0.5:
            duration = feat.timestamp - self._ctx.split_step_start_time
            
            if cfg.min_duration_sec <= duration <= cfg.max_duration_sec:
                # Valid split step
                confidence = min(1.0, self._ctx.split_step_peak_velocity / (cfg.min_vertical_velocity * 3)) * feat.lower_body_confidence
                
                self._emit_event(
                    event_type="split_step",
                    start_time=self._ctx.split_step_start_time,
                    end_time=feat.timestamp,
                    confidence=confidence,
                    metadata={"peak_velocity": self._ctx.split_step_peak_velocity}
                )
                
                # Reset
                self._ctx.split_step_start_time = 0
                self._ctx.split_step_peak_velocity = 0
                return True
        
        # Timeout - reset if too long
        elif self._ctx.split_step_start_time > 0:
            if feat.timestamp - self._ctx.split_step_start_time > cfg.max_duration_sec * 1.5:
                self._ctx.split_step_start_time = 0
                self._ctx.split_step_peak_velocity = 0
        
        return False
    
    def _detect_push_off(self, feat: FrameFeatures, windowed: Dict) -> bool:
        """Detect push-off initiation (start of movement)"""
        # Push-off detected when horizontal velocity exceeds threshold
        # after being relatively stationary
        
        h_vel = abs(feat.horizontal_velocity)
        v_std = windowed.get("horizontal_velocity_std", 0)
        
        # Check if accelerating from low movement
        if h_vel > 0.03 and v_std < 0.02:  # Threshold for push-off
            if self._can_emit_cue("push_off", feat.timestamp):
                self._emit_event(
                    event_type="push_off",
                    start_time=feat.timestamp - 0.1,
                    end_time=feat.timestamp,
                    confidence=feat.lower_body_confidence,
                    metadata={"horizontal_velocity": h_vel}
                )
                return True
        
        return False
    
    def _detect_direction_change(self, feat: FrameFeatures, windowed: Dict) -> bool:
        """Detect direction change during movement"""
        h_vel = feat.horizontal_velocity
        
        # Determine direction
        if h_vel > 0.02:
            direction = 1
        elif h_vel < -0.02:
            direction = -1
        else:
            direction = 0
        
        # Check for direction reversal
        if direction != 0 and direction != self._ctx.last_horizontal_direction and self._ctx.last_horizontal_direction != 0:
            if self._can_emit_cue("direction_change", feat.timestamp):
                self._emit_event(
                    event_type="direction_change",
                    start_time=feat.timestamp - 0.05,
                    end_time=feat.timestamp,
                    confidence=feat.lower_body_confidence,
                    metadata={
                        "from_direction": "right" if self._ctx.last_horizontal_direction > 0 else "left",
                        "to_direction": "right" if direction > 0 else "left"
                    }
                )
                self._ctx.direction_change_time = feat.timestamp
                self._ctx.last_horizontal_direction = direction
                return True
        
        if direction != 0:
            self._ctx.last_horizontal_direction = direction
        
        return False
    
    def _detect_lunge(self, feat: FrameFeatures, windowed: Dict) -> bool:
        """Detect lunge phase (deep knee bend + forward reach)"""
        cfg = self._thresholds.knee
        
        # Use minimum knee angle (more flexed = smaller angle)
        min_knee = min(feat.left_knee_angle, feat.right_knee_angle)
        
        # Lunge detected when knee angle is low enough
        if cfg.lunge_flexion_min <= min_knee <= cfg.lunge_flexion_max:
            if self._ctx.lunge_start_time == 0:
                self._ctx.lunge_start_time = feat.timestamp
                self._ctx.lunge_min_knee_angle = min_knee
            else:
                # Track minimum angle during lunge
                if min_knee < self._ctx.lunge_min_knee_angle:
                    self._ctx.lunge_min_knee_angle = min_knee
        
        # Lunge ends when knee extends
        elif self._ctx.lunge_start_time > 0 and min_knee > cfg.lunge_flexion_max:
            duration = feat.timestamp - self._ctx.lunge_start_time
            
            if duration >= 0.1:  # Minimum lunge duration
                confidence = feat.lower_body_confidence
                
                self._emit_event(
                    event_type="lunge",
                    start_time=self._ctx.lunge_start_time,
                    end_time=feat.timestamp,
                    confidence=confidence,
                    metadata={
                        "min_knee_angle": self._ctx.lunge_min_knee_angle,
                        "duration": duration
                    }
                )
            
            self._ctx.lunge_start_time = 0
            self._ctx.lunge_min_knee_angle = 180
            return True
        
        return False
    
    def _detect_recovery(self, feat: FrameFeatures, windowed: Dict) -> bool:
        """Detect recovery to base position"""
        cfg = self._thresholds.recovery
        
        distance = feat.distance_from_base
        
        # Check if returning to base
        if distance > cfg.base_zone_tolerance:
            # Still away from base
            if self._ctx.recovery_start_time == 0:
                self._ctx.recovery_start_time = feat.timestamp
                self._ctx.recovery_start_distance = distance
        else:
            # At base - check if this is end of recovery
            if self._ctx.recovery_start_time > 0:
                recovery_time = feat.timestamp - self._ctx.recovery_start_time
                
                self._emit_event(
                    event_type="recovery",
                    start_time=self._ctx.recovery_start_time,
                    end_time=feat.timestamp,
                    confidence=feat.lower_body_confidence,
                    metadata={
                        "recovery_time": recovery_time,
                        "start_distance": self._ctx.recovery_start_distance
                    }
                )
                
                self._ctx.recovery_start_time = 0
                self._ctx.recovery_start_distance = 0
                return True
        
        return False
    
    def process_frame(self, feat: FrameFeatures, windowed: Dict) -> List[str]:
        """
        Process a single frame and detect events.
        
        Args:
            feat: Frame features
            windowed: Windowed statistics
        
        Returns:
            List of event types detected this frame
        """
        detected = []
        
        # Update state duration
        self._ctx.state_duration = feat.timestamp - self._ctx.state_start_time
        
        # Detect all event types
        if self._detect_split_step(feat, windowed):
            detected.append("split_step")
        
        if self._detect_push_off(feat, windowed):
            detected.append("push_off")
        
        if self._detect_direction_change(feat, windowed):
            detected.append("direction_change")
        
        if self._detect_lunge(feat, windowed):
            detected.append("lunge")
        
        if self._detect_recovery(feat, windowed):
            detected.append("recovery")
        
        return detected
    
    def process_all(
        self,
        features: List[FrameFeatures],
        windowed: List[Dict]
    ) -> List[DetectedEvent]:
        """
        Process all frames and return detected events.
        
        Args:
            features: List of frame features
            windowed: List of windowed statistics (same length as features)
        
        Returns:
            List of detected events
        """
        self.reset()
        
        for feat, wind in zip(features, windowed):
            self.process_frame(feat, wind)
        
        return self._events
    
    def get_events(self) -> List[DetectedEvent]:
        """Get all detected events"""
        return self._events
