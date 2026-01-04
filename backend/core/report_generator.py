"""
Coach Report Generator
Generates structured JSON coach reports from analysis results.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .feature_computer import FrameFeatures
from .event_fsm import DetectedEvent
from .mistake_detector import DetectedMistake, get_top_mistakes, get_fix_first_plan


@dataclass
class CoachReport:
    """Structured coach report"""
    session_id: str
    created_at: str
    
    # Video metadata
    video_duration: float
    fps: float
    resolution: Dict[str, int]
    
    # Drill info
    drill_type: str
    
    # Processing stats
    processing_time_sec: float
    frames_analyzed: int
    frames_with_pose: int
    model_version: str
    
    # Events timeline
    events: List[Dict]
    
    # Mistakes
    mistakes: List[Dict]
    top_mistakes: List[Dict]
    
    # Fix first plan
    fix_first_plan: Optional[Dict]
    
    # Metrics summary
    metrics_summary: Dict
    
    # Confidence notes
    confidence_notes: List[str]


def generate_report(
    session_id: str,
    video_metadata: Dict,
    processing_stats: Dict,
    features: List[FrameFeatures],
    events: List[DetectedEvent],
    mistakes: List[DetectedMistake],
    drill_type: str = "unknown"
) -> CoachReport:
    """Generate a coach report from analysis results."""
    
    # Compute metrics summary
    if features:
        avg_stance = sum(f.stance_width_ratio for f in features) / len(features)
        avg_confidence = sum(f.lower_body_confidence for f in features) / len(features)
        low_conf_frames = sum(1 for f in features if f.lower_body_confidence < 0.5)
    else:
        avg_stance, avg_confidence, low_conf_frames = 0, 0, 0
    
    # Events to dict
    events_list = []
    for e in events:
        events_list.append({
            "type": e.event_type,
            "timestamp": round(e.start_timestamp, 2),
            "end_timestamp": round(e.end_timestamp, 2),
            "duration": round(e.duration, 2),
            "confidence": round(e.confidence, 2),
            "metadata": e.metadata
        })
    
    # Mistakes to dict
    mistakes_list = []
    for m in mistakes:
        mistakes_list.append({
            "type": m.mistake_type,
            "timestamp": round(m.timestamp, 2),
            "duration": round(m.duration, 2),
            "severity": round(m.severity, 2),
            "confidence": round(m.confidence, 2),
            "cue": m.cue,
            "description": m.description,
            "evidence": [round(t, 2) for t in m.evidence_timestamps]
        })
    
    # Confidence notes
    conf_notes = []
    if low_conf_frames > len(features) * 0.1:
        conf_notes.append(f"Low visibility in {low_conf_frames} frames ({low_conf_frames/len(features)*100:.0f}%)")
    if avg_confidence < 0.6:
        conf_notes.append("Overall pose confidence is low - consider better camera angle")
    
    return CoachReport(
        session_id=session_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        video_duration=video_metadata.get("duration", 0),
        fps=video_metadata.get("fps", 30),
        resolution={"width": video_metadata.get("width", 0), 
                    "height": video_metadata.get("height", 0)},
        drill_type=drill_type,
        processing_time_sec=processing_stats.get("processing_time_sec", 0),
        frames_analyzed=processing_stats.get("frames_processed", 0),
        frames_with_pose=processing_stats.get("frames_with_pose", 0),
        model_version="mediapipe-0.10.9",
        events=events_list,
        mistakes=mistakes_list,
        top_mistakes=get_top_mistakes(mistakes, 3),
        fix_first_plan=get_fix_first_plan(mistakes),
        metrics_summary={
            "avg_stance_width_ratio": round(avg_stance, 2),
            "avg_pose_confidence": round(avg_confidence, 2),
            "total_events": len(events),
            "total_mistakes": len(mistakes),
            "split_steps_detected": sum(1 for e in events if e.event_type == "split_step"),
            "lunges_detected": sum(1 for e in events if e.event_type == "lunge"),
            "direction_changes": sum(1 for e in events if e.event_type == "direction_change")
        },
        confidence_notes=conf_notes
    )


def report_to_dict(report: CoachReport) -> Dict:
    """Convert report to JSON-serializable dict."""
    return asdict(report)


def create_evidence_chunks(report: CoachReport) -> List[str]:
    """Create evidence chunks for RAG/chat grounding."""
    chunks = []
    
    # Event chunks
    for e in report.events:
        chunk = (f"{e['timestamp']}s-{e['end_timestamp']}s: {e['type'].upper()} detected. "
                f"Duration: {e['duration']}s. Confidence: {e['confidence']}")
        chunks.append(chunk)
    
    # Mistake chunks
    for m in report.mistakes:
        chunk = (f"{m['timestamp']}s: MISTAKE - {m['type'].replace('_', ' ')}. "
                f"Severity: {m['severity']}/1.0. Confidence: {m['confidence']}. "
                f"Cue: '{m['cue']}'. Evidence at: {m['evidence']}")
        chunks.append(chunk)
    
    # Summary chunk
    if report.fix_first_plan:
        plan = report.fix_first_plan
        chunks.append(
            f"PRIORITY FIX: {plan['primary_issue'].replace('_', ' ')} "
            f"({plan['occurrences']} occurrences). Focus: {plan['cue']}. "
            f"Recommended drill: {plan['focus_drill']}"
        )
    
    # Metrics chunk
    m = report.metrics_summary
    chunks.append(
        f"SESSION SUMMARY: {report.video_duration:.1f}s video. "
        f"{m['total_events']} events detected. {m['total_mistakes']} issues found. "
        f"Avg stance ratio: {m['avg_stance_width_ratio']}. "
        f"Pose confidence: {m['avg_pose_confidence']}"
    )
    
    return chunks
