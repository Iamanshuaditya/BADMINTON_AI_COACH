"""
Video Analysis Pipeline
Orchestrates the full analysis workflow for both footwork and stroke analysis.
Now with comprehensive error handling and recovery.
"""

import os
import json
import uuid
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from .pose_extractor import PoseExtractor, save_poses_to_json
from .feature_computer import FeatureComputer, compute_windowed_features
from .event_fsm import FootworkFSM
from .mistake_detector import MistakeDetector, DetectedMistake
from .stroke_analyzer import OverheadStrokeAnalyzer, StrokeMistake
from .report_generator import generate_report, report_to_dict, create_evidence_chunks

# Import exceptions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from exceptions import (
    ShuttleSenseException,
    VideoProcessingError,
    PoseExtractionError,
    FeatureComputationError,
    ReportGenerationError,
    InsufficientPoseData,
    VideoNotFound
)

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

# Minimum requirements for valid analysis
MIN_FRAMES_WITH_POSE = 30  # At least 1 second at 30fps
MIN_FEATURES_REQUIRED = 10


def is_stroke_drill(drill_type: str) -> bool:
    """Check if drill type requires stroke analysis"""
    return any(s in drill_type.lower() for s in ["overhead", "clear", "smash", "stroke"])


def is_footwork_drill(drill_type: str) -> bool:
    """Check if drill type requires footwork analysis"""
    return any(s in drill_type.lower() for s in ["footwork", "shadow", "corner", "side", "front", "back", "lunge"])


class AnalysisResult:
    """Structured analysis result with status tracking"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.success = False
        self.stages_completed = []
        self.stages_failed = []
        self.warnings = []
        self.processing_time_sec = 0
        self.data = {}
    
    def add_warning(self, warning: str):
        """Add a non-fatal warning"""
        self.warnings.append(warning)
        logger.warning(f"[{self.session_id}] {warning}")
    
    def complete_stage(self, stage: str):
        """Mark a stage as completed"""
        self.stages_completed.append(stage)
        logger.debug(f"[{self.session_id}] Stage completed: {stage}")
    
    def fail_stage(self, stage: str, error: str):
        """Mark a stage as failed"""
        self.stages_failed.append({"stage": stage, "error": error})
        logger.error(f"[{self.session_id}] Stage failed: {stage} - {error}")
    
    def to_dict(self) -> Dict:
        """Convert to API response format"""
        result = {
            "session_id": self.session_id,
            "success": self.success,
            "processing_time_sec": round(self.processing_time_sec, 2),
        }
        result.update(self.data)
        
        if self.warnings:
            result["warnings"] = self.warnings
        
        if self.stages_failed:
            result["failed_stages"] = self.stages_failed
        
        return result


class AnalysisPipeline:
    """
    Full video analysis pipeline with error handling:
    1. Validate input
    2. Extract poses
    3. Smooth keypoints
    4. Compute features
    5. Detect events (footwork)
    6. Analyze strokes (if overhead drill)
    7. Detect mistakes
    8. Generate report
    
    Each stage has error handling and the pipeline can produce
    partial results if later stages fail.
    """
    
    def __init__(self, output_dir: str = "./data/sessions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pose_extractor = PoseExtractor(enable_smoothing=True)
        self.feature_computer = FeatureComputer()
        self.event_fsm = FootworkFSM()
        self.mistake_detector = MistakeDetector()
        self.stroke_analyzer = OverheadStrokeAnalyzer()
    
    def _validate_video_path(self, video_path: str) -> Path:
        """Validate video file exists and is accessible"""
        path = Path(video_path)
        
        if not path.exists():
            raise VideoNotFound(str(path))
        
        if not path.is_file():
            raise VideoProcessingError(f"Not a file: {path}", stage="validation")
        
        # Check file size (must be > 0)
        if path.stat().st_size == 0:
            raise VideoProcessingError("Video file is empty", stage="validation")
        
        return path
    
    def _extract_poses_safe(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Extract poses with error handling"""
        try:
            pose_data = self.pose_extractor.extract_from_video(
                video_path,
                progress_callback=lambda c, t: progress_callback("pose_extraction", c/t) if progress_callback else None
            )
            
            # Validate extraction results
            frames = pose_data.get("frames", [])
            frames_with_pose = sum(1 for f in frames if f.get("landmarks"))
            
            if frames_with_pose == 0:
                raise InsufficientPoseData("No poses detected in video. Ensure full body is visible.")
            
            if frames_with_pose < MIN_FRAMES_WITH_POSE:
                # Warning but continue
                logger.warning(
                    f"Low pose count: {frames_with_pose} frames. "
                    f"Analysis may be limited."
                )
            
            return pose_data
            
        except ShuttleSenseException:
            raise
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}", exc_info=True)
            raise PoseExtractionError(f"Failed to extract poses: {str(e)}")
    
    def _compute_features_safe(
        self,
        frames: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> tuple:
        """Compute features with error handling"""
        try:
            if progress_callback:
                progress_callback("feature_computation", 0)
            
            features = self.feature_computer.compute_all(frames)
            
            if not features:
                raise FeatureComputationError("No features could be computed from poses")
            
            if len(features) < MIN_FEATURES_REQUIRED:
                logger.warning(f"Low feature count: {len(features)}. Analysis may be limited.")
            
            # Compute windowed features
            windowed = compute_windowed_features(features)
            
            if progress_callback:
                progress_callback("feature_computation", 1.0)
            
            return features, windowed
            
        except ShuttleSenseException:
            raise
        except Exception as e:
            logger.error(f"Feature computation failed: {e}", exc_info=True)
            raise FeatureComputationError(f"Failed to compute features: {str(e)}")
    
    def _analyze_footwork_safe(
        self,
        features: List[Dict],
        windowed: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> tuple:
        """Analyze footwork with error handling"""
        try:
            if progress_callback:
                progress_callback("footwork_analysis", 0)
            
            events = self.event_fsm.process_all(features, windowed)
            mistakes = self.mistake_detector.detect_all(features, windowed, events)
            
            if progress_callback:
                progress_callback("footwork_analysis", 1.0)
            
            return events, mistakes
            
        except Exception as e:
            logger.error(f"Footwork analysis failed: {e}", exc_info=True)
            # Return empty results instead of failing completely
            return [], []
    
    def _analyze_strokes_safe(
        self,
        frames: List[Dict],
        features: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> tuple:
        """Analyze strokes with error handling"""
        try:
            if progress_callback:
                progress_callback("stroke_analysis", 0)
            
            strokes, analyses, mistakes = self.stroke_analyzer.analyze_all(frames, features)
            
            if progress_callback:
                progress_callback("stroke_analysis", 1.0)
            
            return strokes, analyses, mistakes
            
        except Exception as e:
            logger.error(f"Stroke analysis failed: {e}", exc_info=True)
            # Return empty results instead of failing completely
            return [], [], []
    
    def _generate_report_safe(
        self,
        session_id: str,
        video_metadata: Dict,
        processing_stats: Dict,
        features: List[Dict],
        events: List,
        mistakes: List,
        drill_type: str,
        stroke_data: Optional[Dict],
        progress_callback: Optional[Callable] = None
    ) -> tuple:
        """Generate report with error handling"""
        try:
            if progress_callback:
                progress_callback("report_generation", 0)
            
            report = generate_report(
                session_id=session_id,
                video_metadata=video_metadata,
                processing_stats=processing_stats,
                features=features,
                events=events,
                mistakes=mistakes,
                drill_type=drill_type,
                stroke_data=stroke_data
            )
            
            report_dict = report_to_dict(report)
            evidence_chunks = create_evidence_chunks(report)
            
            if progress_callback:
                progress_callback("report_generation", 1.0)
            
            return report_dict, evidence_chunks
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            raise ReportGenerationError(f"Failed to generate report: {str(e)}")
    
    def analyze(
        self,
        video_path: str,
        drill_type: str = "unknown",
        session_id: Optional[str] = None,
        save_poses: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run full analysis on a video with comprehensive error handling.
        
        Args:
            video_path: Path to video file
            drill_type: Type of drill being performed
            session_id: Optional session ID (auto-generated if not provided)
            save_poses: Whether to save raw pose data to JSON
            progress_callback: Optional callback(stage, progress)
        
        Returns:
            Analysis results including report
        
        Raises:
            VideoNotFound: If video file doesn't exist
            VideoProcessingError: If video cannot be processed
            PoseExtractionError: If pose extraction fails
            InsufficientPoseData: If not enough poses detected
            FeatureComputationError: If feature computation fails
            ReportGenerationError: If report generation fails
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())[:8]
        result = AnalysisResult(session_id)
        
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting analysis for session {session_id}, drill: {drill_type}")
        
        try:
            # Stage 1: Validate input
            video_path = str(self._validate_video_path(video_path))
            result.complete_stage("validation")
            
            # Determine analysis mode
            do_footwork = is_footwork_drill(drill_type) or drill_type == "unknown"
            do_stroke = is_stroke_drill(drill_type)
            
            # Stage 2: Extract poses
            pose_data = self._extract_poses_safe(video_path, progress_callback)
            result.complete_stage("pose_extraction")
            
            if save_poses:
                poses_path = session_dir / "poses.json"
                save_poses_to_json(pose_data, str(poses_path))
            
            # Stage 3: Compute features
            frames = pose_data.get("frames", [])
            features, windowed = self._compute_features_safe(frames, progress_callback)
            result.complete_stage("feature_computation")
            
            # Track stats
            frames_with_pose = sum(1 for f in frames if f.get("landmarks"))
            if frames_with_pose < len(frames) * 0.5:
                result.add_warning(
                    f"Only {frames_with_pose}/{len(frames)} frames had detected poses. "
                    "Consider better lighting or camera angle."
                )
            
            # Initialize results
            events = []
            all_mistakes = []
            stroke_data = None
            
            # Stage 4: Footwork analysis (if applicable)
            if do_footwork:
                events, footwork_mistakes = self._analyze_footwork_safe(
                    features, windowed, progress_callback
                )
                all_mistakes.extend(footwork_mistakes)
                result.complete_stage("footwork_analysis")
            
            # Stage 5: Stroke analysis (if applicable)
            if do_stroke:
                strokes, analyses, stroke_mistakes = self._analyze_strokes_safe(
                    frames, features, progress_callback
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
                if strokes:
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
                            for a in analyses
                        ]
                    }
                
                result.complete_stage("stroke_analysis")
            
            # Stage 6: Generate report
            report_dict, evidence_chunks = self._generate_report_safe(
                session_id=session_id,
                video_metadata=pose_data.get("video_metadata", {}),
                processing_stats=pose_data.get("processing_stats", {}),
                features=features,
                events=events,
                mistakes=all_mistakes,
                drill_type=drill_type,
                stroke_data=stroke_data,
                progress_callback=progress_callback
            )
            result.complete_stage("report_generation")
            
            # Save outputs
            report_path = session_dir / "report.json"
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            chunks_path = session_dir / "evidence_chunks.json"
            with open(chunks_path, 'w') as f:
                json.dump(evidence_chunks, f, indent=2)
            
            # Mark success
            result.success = True
            result.processing_time_sec = time.time() - start_time
            result.data = {
                "report": report_dict,
                "evidence_chunks": evidence_chunks,
                "output_dir": str(session_dir)
            }
            
            logger.info(
                f"Analysis complete for {session_id}: "
                f"{len(events)} events, {len(all_mistakes)} mistakes "
                f"in {result.processing_time_sec:.1f}s"
            )
            
            if stroke_data:
                logger.info(f"  Strokes detected: {stroke_data['strokes_detected']}")
            
            return result.to_dict()
            
        except ShuttleSenseException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            logger.error(f"Unexpected error in analysis: {e}", exc_info=True)
            raise VideoProcessingError(f"Analysis failed unexpectedly: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Load a previous session's data."""
        session_dir = self.output_dir / session_id
        
        if not session_dir.exists():
            return None
        
        result = {"session_id": session_id}
        
        report_path = session_dir / "report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    result["report"] = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load report for {session_id}: {e}")
                result["report_error"] = str(e)
        
        chunks_path = session_dir / "evidence_chunks.json"
        if chunks_path.exists():
            try:
                with open(chunks_path) as f:
                    result["evidence_chunks"] = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load evidence chunks for {session_id}: {e}")
                result["chunks_error"] = str(e)
        
        return result
    
    def list_sessions(self) -> list:
        """List all session IDs."""
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and return success status."""
        session_dir = self.output_dir / session_id
        
        if not session_dir.exists():
            return False
        
        try:
            import shutil
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def cleanup(self):
        """Release resources."""
        try:
            self.pose_extractor.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
