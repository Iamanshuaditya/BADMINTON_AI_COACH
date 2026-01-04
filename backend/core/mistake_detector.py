"""
Mistake Detection Module
Detects footwork mistakes based on computed features and events.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

from .feature_computer import FrameFeatures
from .event_fsm import DetectedEvent
from config import get_thresholds

logger = logging.getLogger(__name__)


@dataclass
class DetectedMistake:
    """A detected footwork mistake"""
    mistake_type: str
    timestamp: float
    duration: float
    severity: float
    confidence: float
    evidence_timestamps: List[float] = field(default_factory=list)
    cue: str = ""
    description: str = ""
    metadata: Dict = field(default_factory=dict)


class MistakeDetector:
    """Detects footwork mistakes based on features and events."""
    
    def __init__(self):
        self._thresholds = get_thresholds()
        self._mistakes: List[DetectedMistake] = []
    
    def reset(self):
        self._mistakes = []
    
    def _emit_mistake(self, mistake_type: str, timestamp: float, duration: float,
                      severity: float, confidence: float, evidence: List[float],
                      cue: str, description: str, metadata: Optional[Dict] = None):
        mistake = DetectedMistake(
            mistake_type=mistake_type, timestamp=timestamp, duration=duration,
            severity=severity, confidence=confidence, evidence_timestamps=evidence,
            cue=cue, description=description, metadata=metadata or {}
        )
        self._mistakes.append(mistake)
    
    def detect_narrow_stance(self, features: List[FrameFeatures], windowed: List[Dict]):
        cfg = self._thresholds.stance
        narrow_count, narrow_start, evidence = 0, None, []
        
        for feat, wind in zip(features, windowed):
            ratio = wind["avg_stance_ratio"]
            if ratio < cfg.min_ratio:
                if narrow_start is None:
                    narrow_start = feat.timestamp
                narrow_count += 1
                evidence.append(feat.timestamp)
            else:
                if narrow_count >= cfg.sustained_window_frames and narrow_start is not None:
                    self._emit_mistake("narrow_stance", narrow_start, 
                                       feat.timestamp - narrow_start, 
                                       min(1.0, (cfg.min_ratio - ratio) / 0.3),
                                       wind["avg_confidence"], evidence[-5:],
                                       "Wider base", f"Stance too narrow ({ratio:.2f})")
                narrow_count, narrow_start, evidence = 0, None, []
    
    def detect_knee_collapse(self, features: List[FrameFeatures], 
                             windowed: List[Dict], events: List[DetectedEvent]):
        cfg = self._thresholds.knee
        lunges = [e for e in events if e.event_type == "lunge"]
        
        for lunge in lunges:
            frames = [(f, w) for f, w in zip(features, windowed)
                      if lunge.start_timestamp <= f.timestamp <= lunge.end_timestamp]
            max_valgus, max_time = 0, lunge.start_timestamp
            
            for feat, wind in frames:
                valgus = max(wind["avg_left_valgus"], wind["avg_right_valgus"])
                if valgus > max_valgus and feat.lower_body_confidence >= cfg.confidence_gate:
                    max_valgus, max_time = valgus, feat.timestamp
            
            if max_valgus > cfg.valgus_threshold:
                self._emit_mistake("knee_collapse", max_time, lunge.duration,
                                   min(1.0, (max_valgus - cfg.valgus_threshold) / 10),
                                   lunge.confidence, [lunge.start_timestamp, max_time],
                                   "Knee out", f"Knee collapsing inward ({max_valgus:.1f}°)")
    
    def detect_missing_split_step(self, events: List[DetectedEvent]):
        splits = [e for e in events if e.event_type == "split_step"]
        dir_changes = [e for e in events if e.event_type == "direction_change"]
        
        for dc in dir_changes:
            has_split = any(dc.start_timestamp - 0.5 <= s.end_timestamp <= dc.start_timestamp 
                           for s in splits)
            if not has_split:
                self._emit_mistake("missing_split_step", dc.start_timestamp, 0.3, 0.7,
                                   dc.confidence, [dc.start_timestamp], "Split step",
                                   "No split step before direction change")
    
    def detect_slow_recovery(self, events: List[DetectedEvent]):
        cfg = self._thresholds.recovery
        for rec in [e for e in events if e.event_type == "recovery"]:
            time = rec.metadata.get("recovery_time", 0)
            if time > cfg.max_recovery_time_sec:
                self._emit_mistake("slow_recovery", rec.start_timestamp, time,
                                   min(1.0, (time - cfg.max_recovery_time_sec) / 2.0),
                                   rec.confidence, [rec.start_timestamp, rec.end_timestamp],
                                   "Recover faster", f"Recovery took {time:.1f}s")
    
    def detect_all(self, features: List[FrameFeatures], windowed: List[Dict],
                   events: List[DetectedEvent]) -> List[DetectedMistake]:
        self.reset()
        self.detect_narrow_stance(features, windowed)
        self.detect_knee_collapse(features, windowed, events)
        self.detect_missing_split_step(events)
        self.detect_slow_recovery(events)
        return self._mistakes
    
    def get_mistakes(self) -> List[DetectedMistake]:
        return self._mistakes


def rank_mistakes(mistakes: List[DetectedMistake]) -> List[DetectedMistake]:
    if not mistakes:
        return []
    type_counts = {}
    for m in mistakes:
        type_counts[m.mistake_type] = type_counts.get(m.mistake_type, 0) + 1
    
    def score(m):
        return m.severity * m.confidence * (0.5 + 0.5 * type_counts[m.mistake_type] / len(mistakes))
    return sorted(mistakes, key=score, reverse=True)


def get_top_mistakes(mistakes: List[DetectedMistake], n: int = 3) -> List[Dict]:
    ranked = rank_mistakes(mistakes)
    agg = {}
    for m in ranked:
        if m.mistake_type not in agg:
            agg[m.mistake_type] = {"type": m.mistake_type, "count": 0, "max_severity": 0,
                                   "avg_confidence": 0, "cue": m.cue, "timestamps": []}
        a = agg[m.mistake_type]
        a["count"] += 1
        a["max_severity"] = max(a["max_severity"], m.severity)
        a["avg_confidence"] += m.confidence
        a["timestamps"].append(m.timestamp)
    
    for a in agg.values():
        a["avg_confidence"] /= a["count"]
    
    return sorted(agg.values(), key=lambda x: x["count"] * x["max_severity"], reverse=True)[:n]


def get_fix_first_plan(mistakes: List[DetectedMistake]) -> Optional[Dict]:
    top = get_top_mistakes(mistakes, 1)
    if not top:
        return None
    p = top[0]
    drills = {"narrow_stance": "Shadow footwork with floor markers",
              "knee_collapse": "Single-leg squats - knee over toes",
              "missing_split_step": "Split step timing drill",
              "slow_recovery": "Recovery shuttle run drill"}
    return {"primary_issue": p["type"], "occurrences": p["count"],
            "cue": p["cue"], "focus_drill": drills.get(p["type"], "General footwork"),
            "key_timestamps": p["timestamps"][:3]}
