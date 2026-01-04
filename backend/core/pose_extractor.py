"""
Pose Extraction using MediaPipe BlazePose
Extracts 33 landmarks per frame from video.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import json
import logging
import time

from .one_euro_filter import KeypointSmoother
from config import get_thresholds

logger = logging.getLogger(__name__)


# MediaPipe landmark names (33 landmarks)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Key landmark indices for badminton analysis
LANDMARK_INDICES = {name: i for i, name in enumerate(LANDMARK_NAMES)}


class PoseExtractor:
    """
    Extracts pose landmarks from video using MediaPipe.
    Supports both batch processing and frame-by-frame streaming.
    """
    
    def __init__(
        self,
        model_complexity: int = 1,  # 0=lite, 1=full, 2=heavy
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_smoothing: bool = True
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_smoothing = enable_smoothing
        
        self.pose: Optional[mp.solutions.pose.Pose] = None
        self.smoother: Optional[KeypointSmoother] = None
        
        self._thresholds = get_thresholds()
    
    def _init_pose(self):
        """Initialize MediaPipe Pose"""
        if self.pose is None:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        
        if self.enable_smoothing and self.smoother is None:
            cfg = self._thresholds.smoothing
            self.smoother = KeypointSmoother(
                num_landmarks=33,
                min_cutoff=cfg.min_cutoff,
                beta=cfg.beta,
                d_cutoff=cfg.d_cutoff,
                feet_min_cutoff=cfg.feet_min_cutoff,
                feet_beta=cfg.feet_beta
            )
    
    def extract_from_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Extract pose from a single frame.
        
        Args:
            frame: BGR image (OpenCV format)
            timestamp: Optional timestamp in seconds
        
        Returns:
            Dict with landmarks and metadata, or None if no pose detected
        """
        self._init_pose()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        h, w = frame.shape[:2]
        landmarks_raw = []
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_raw.append({
                "name": LANDMARK_NAMES[i],
                "index": i,
                "x": lm.x,  # Normalized 0-1
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "x_px": int(lm.x * w),  # Pixel coordinates
                "y_px": int(lm.y * h)
            })
        
        # Apply smoothing if enabled
        if self.enable_smoothing and self.smoother:
            landmarks_smoothed = self.smoother.smooth(
                landmarks_raw,
                timestamp=timestamp,
                visibility_threshold=self._thresholds.visibility.min_visibility
            )
            # Merge smoothed coordinates back
            for i, (raw, smooth) in enumerate(zip(landmarks_raw, landmarks_smoothed)):
                raw["x_smooth"] = smooth["x"]
                raw["y_smooth"] = smooth["y"]
                raw["reliable"] = smooth["reliable"]
        
        return {
            "timestamp": timestamp,
            "landmarks": landmarks_raw,
            "frame_width": w,
            "frame_height": h
        }
    
    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Extract poses from entire video.
        
        Args:
            video_path: Path to video file
            max_frames: Optional limit on frames to process
            progress_callback: Optional callback(current_frame, total_frames)
        
        Returns:
            Dict with all frame poses and video metadata
        """
        self._init_pose()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {total_frames} frames, {fps} FPS, {duration:.2f}s")
        
        frames_data = []
        frame_idx = 0
        start_time = time.time()
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            timestamp = frame_idx / fps
            pose_data = self.extract_from_frame(frame, timestamp)
            
            if pose_data:
                pose_data["frame_index"] = frame_idx
                frames_data.append(pose_data)
            else:
                # No pose detected - record empty frame
                frames_data.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "landmarks": None,
                    "no_pose_detected": True
                })
            
            frame_idx += 1
            
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        
        processing_time = time.time() - start_time
        
        return {
            "video_metadata": {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
                "path": str(video_path)
            },
            "processing_stats": {
                "processing_time_sec": processing_time,
                "frames_processed": len(frames_data),
                "frames_with_pose": sum(1 for f in frames_data if f.get("landmarks")),
                "avg_fps": len(frames_data) / processing_time if processing_time > 0 else 0
            },
            "frames": frames_data
        }
    
    def stream_from_video(
        self,
        video_path: str
    ) -> Generator[Dict, None, None]:
        """
        Stream pose extraction frame by frame (generator).
        Memory efficient for large videos.
        """
        self._init_pose()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            pose_data = self.extract_from_frame(frame, timestamp)
            
            if pose_data:
                pose_data["frame_index"] = frame_idx
                yield pose_data
            
            frame_idx += 1
        
        cap.release()
    
    def close(self):
        """Release resources"""
        if self.pose:
            self.pose.close()
            self.pose = None
        if self.smoother:
            self.smoother.reset()
            self.smoother = None


def save_poses_to_json(pose_data: Dict, output_path: str):
    """Save extracted poses to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)
    logger.info(f"Saved pose data to {output_path}")


def load_poses_from_json(json_path: str) -> Dict:
    """Load poses from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)
