"""
Video Analysis Pipeline
Orchestrates the full analysis workflow for both footwork and stroke analysis.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .pose_extractor import PoseExtractor, save_poses_to_json
from .feature_computer import FeatureComputer, compute_windowed_features
from .event_fsm import FootworkFSM
from .mistake_detector import MistakeDetector, DetectedMistake
from .stroke_analyzer import OverheadStrokeAnalyzer, StrokeMistake
from .report_generator import generate_report, report_to_dict, create_evidence_chunks

logger = logging.getLogger(__name__)


# Drill types that trigger stroke analysis
STROKE_DRILL_TYPES = [
    "overhead-shadow",
    "overhead-clear",
    "overhead-smash",
    "clear-shadow",
    "smash-shadow"
]

# Drill types that trigger footwork analysis
FOOTWORK_DRILL_TYPES = [
    "6-corner-shadow",
    "side-to-side",
    "front-back",
    "footwork",
    "shadow-footwork"
]


def is_stroke_drill(drill_type: str) -> bool:
    """Check if drill type requires stroke analysis"""
    return any(s in drill_type.lower() for s in ["overhead", "clear", "smash", "stroke"])


def is_footwork_drill(drill_type: str) -> bool:
    """Check if drill type requires footwork analysis"""
    return any(s in drill_type.lower() for s in ["footwork", "shadow", "corner", "side", "front", "back", "lunge"])


class AnalysisPipeline:
    """
    Full video analysis pipeline:
    1. Extract poses
    2. Smooth keypoints
    3. Compute features
    4. Detect events (footwork)
    5. Analyze strokes (if overhead drill)
    6. Detect mistakes
    7. Generate report
    """
    
    def __init__(self, output_dir: str = "./data/sessions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pose_extractor = PoseExtractor(enable_smoothing=True)
        self.feature_computer = FeatureComputer()
        self.event_fsm = FootworkFSM()
        self.mistake_detector = MistakeDetector()
        self.stroke_analyzer = OverheadStrokeAnalyzer()
    
    def analyze(
        self,
        video_path: str,
        drill_type: str = "unknown",
        session_id: Optional[str] = None,
        save_poses: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Run full analysis on a video.
        
        Args:
            video_path: Path to video file
            drill_type: Type of drill being performed
            session_id: Optional session ID (auto-generated if not provided)
            save_poses: Whether to save raw pose data to JSON
            progress_callback: Optional callback(stage, progress)
        
        Returns:
            Analysis results including report
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting analysis for session {session_id}, drill: {drill_type}")
        
        # Determine analysis mode
        do_footwork = is_footwork_drill(drill_type) or drill_type == "unknown"
        do_stroke = is_stroke_drill(drill_type)
        
        # Stage 1: Extract poses
        if progress_callback:
            progress_callback("pose_extraction", 0)
        
        pose_data = self.pose_extractor.extract_from_video(
            video_path,
            progress_callback=lambda c, t: progress_callback("pose_extraction", c/t) if progress_callback else None
        )
        
        if save_poses:
            poses_path = session_dir / "poses.json"
            save_poses_to_json(pose_data, str(poses_path))
        
        # Stage 2: Compute features
        if progress_callback:
            progress_callback("feature_computation", 0)
        
        frames = pose_data.get("frames", [])
        features = self.feature_computer.compute_all(frames)
        
        if not features:
            logger.warning("No features computed - no poses detected?")
            return {"error": "No poses detected in video", "session_id": session_id}
        
        # Stage 3: Compute windowed features
        windowed = compute_windowed_features(features)
        
        if progress_callback:
            progress_callback("feature_computation", 1.0)
        
        # Initialize results
        events = []
        all_mistakes = []
        stroke_data = None
        
        # Stage 4: Footwork analysis (if applicable)
        if do_footwork:
            if progress_callback:
                progress_callback("footwork_analysis", 0)
            
            events = self.event_fsm.process_all(features, windowed)
            footwork_mistakes = self.mistake_detector.detect_all(features, windowed, events)
            
            # Convert to list for combining
            all_mistakes.extend(footwork_mistakes)
            
            if progress_callback:
                progress_callback("footwork_analysis", 1.0)
        
        # Stage 5: Stroke analysis (if applicable)
        if do_stroke:
            if progress_callback:
                progress_callback("stroke_analysis", 0)
            
            strokes, stroke_analyses, stroke_mistakes = self.stroke_analyzer.analyze_all(
                frames, features
            )
            
            # Convert stroke mistakes to common format
            for sm in stroke_mistakes:
                all_mistakes.append(DetectedMistake(
                    mistake_type=sm.mistake_type,
                    timestamp=sm.timestamp,
                    duration=sm.duration,
                    severity=sm.severity,
                    confidence=sm.confidence,
                    evidence_timestamps=sm.evidence_timestamps,
                    cue=sm.cue,
                    description=sm.description,
                    metadata=sm.metadata
                ))
            
            # Store stroke-specific data
            stroke_data = {
                "strokes_detected": len(strokes),
                "strokes": [
                    {
                        "stroke_id": s.stroke_id,
                        "start_timestamp": round(s.start_timestamp, 2),
                        "contact_timestamp": round(s.contact_proxy_timestamp, 2),
                        "end_timestamp": round(s.end_timestamp, 2),
                        "duration": round(s.duration, 2),
                        "overhead_confidence": round(s.overhead_confidence, 2),
                        "dominant_side": s.dominant_side
                    }
                    for s in strokes
                ],
                "analyses": [
                    {
                        "stroke_id": a.stroke.stroke_id,
                        "is_valid_overhead": a.is_valid_overhead,
                        "contact_height_status": a.contact_height_status,
                        "wrist_above_shoulder": a.wrist_above_shoulder,
                        "contact_in_front": a.contact_in_front,
                        "elbow_leads_wrist": a.elbow_leads_wrist,
                        "ready_position_good": a.ready_position_good
                    }
                    for a in stroke_analyses
                ]
            }
            
            if progress_callback:
                progress_callback("stroke_analysis", 1.0)
        
        # Stage 6: Generate report
        if progress_callback:
            progress_callback("report_generation", 0)
        
        report = generate_report(
            session_id=session_id,
            video_metadata=pose_data.get("video_metadata", {}),
            processing_stats=pose_data.get("processing_stats", {}),
            features=features,
            events=events,
            mistakes=all_mistakes,
            drill_type=drill_type,
            stroke_data=stroke_data
        )
        
        report_dict = report_to_dict(report)
        
        # Save report
        report_path = session_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Generate evidence chunks for chat
        evidence_chunks = create_evidence_chunks(report)
        chunks_path = session_dir / "evidence_chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump(evidence_chunks, f, indent=2)
        
        if progress_callback:
            progress_callback("report_generation", 1.0)
        
        logger.info(f"Analysis complete for {session_id}: {len(events)} events, {len(all_mistakes)} mistakes")
        if stroke_data:
            logger.info(f"  Strokes detected: {stroke_data['strokes_detected']}")
        
        return {
            "session_id": session_id,
            "report": report_dict,
            "evidence_chunks": evidence_chunks,
            "output_dir": str(session_dir)
        }
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Load a previous session's data."""
        session_dir = self.output_dir / session_id
        
        if not session_dir.exists():
            return None
        
        result = {"session_id": session_id}
        
        report_path = session_dir / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                result["report"] = json.load(f)
        
        chunks_path = session_dir / "evidence_chunks.json"
        if chunks_path.exists():
            with open(chunks_path) as f:
                result["evidence_chunks"] = json.load(f)
        
        return result
    
    def list_sessions(self) -> list:
        """List all session IDs."""
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]
    
    def cleanup(self):
        """Release resources."""
        self.pose_extractor.close()
