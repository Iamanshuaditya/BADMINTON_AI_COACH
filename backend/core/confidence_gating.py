"""
Confidence Gating and Episode Clustering
Anti-spam shields for more coach-like reports.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisibilityWindow:
    """Visibility analysis for a decision window"""
    start_timestamp: float
    end_timestamp: float
    required_landmarks: List[str]
    
    # Per-landmark stats
    visibility_counts: Dict[str, int]  # Frames where visible
    total_frames: int
    
    # Overall
    is_stable: bool
    stability_score: float  # 0-1


@dataclass
class MistakeEpisode:
    """Clustered mistake episode"""
    mistake_type: str
    episode_id: int
    
    # Timing
    start_timestamp: float
    end_timestamp: float
    duration: float
    
    # Aggregated data
    occurrence_count: int
    avg_severity: float
    avg_confidence: float
    peak_severity: float
    
    # Representative instance
    representative_timestamp: float
    representative_cue: str
    representative_description: str
    
    # Evidence
    all_timestamps: List[float] = field(default_factory=list)


class VisibilityGate:
    """
    Ensures landmarks are stably visible before triggering rules.
    """
    
    def __init__(self, visibility_min: float = 0.5, stable_frame_ratio: float = 0.6):
        self.visibility_min = visibility_min
        self.stable_frame_ratio = stable_frame_ratio
    
    def check_stability(
        self,
        frames: List[Dict],
        start_timestamp: float,
        end_timestamp: float,
        required_landmarks: List[str]
    ) -> VisibilityWindow:
        """
        Check if required landmarks are stably visible in window.
        
        Args:
            frames: All frames
            start_timestamp: Window start
            end_timestamp: Window end
            required_landmarks: List of landmark names that must be visible
        
        Returns:
            VisibilityWindow with analysis
        """
        # Landmark indices
        LANDMARK_INDICES = {
            "left_shoulder": 11, "right_shoulder": 12,
            "left_hip": 23, "right_hip": 24,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
            "nose": 0,
        }
        
        # Find frames in window
        window_frames = [f for f in frames 
                        if start_timestamp <= f.get("timestamp", 0) <= end_timestamp]
        
        if not window_frames:
            return VisibilityWindow(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                required_landmarks=required_landmarks,
                visibility_counts={lm: 0 for lm in required_landmarks},
                total_frames=0,
                is_stable=False,
                stability_score=0
            )
        
        total_frames = len(window_frames)
        visibility_counts = {lm: 0 for lm in required_landmarks}
        
        for frame in window_frames:
            landmarks = frame.get("landmarks", [])
            if not landmarks:
                continue
            
            for lm_name in required_landmarks:
                idx = LANDMARK_INDICES.get(lm_name)
                if idx is not None and idx < len(landmarks):
                    vis = landmarks[idx].get("visibility", 0)
                    if vis >= self.visibility_min:
                        visibility_counts[lm_name] += 1
        
        # Check if all landmarks meet stability threshold
        all_stable = True
        total_visibility = 0
        
        for lm_name in required_landmarks:
            ratio = visibility_counts[lm_name] / total_frames if total_frames > 0 else 0
            total_visibility += ratio
            if ratio < self.stable_frame_ratio:
                all_stable = False
        
        stability_score = total_visibility / len(required_landmarks) if required_landmarks else 0
        
        return VisibilityWindow(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            required_landmarks=required_landmarks,
            visibility_counts=visibility_counts,
            total_frames=total_frames,
            is_stable=all_stable,
            stability_score=stability_score
        )


class EpisodeClustering:
    """
    Clusters repeated mistakes into episodes.
    Reports top episodes instead of every instance.
    """
    
    def __init__(self, merge_window_sec: float = 2.0, max_episodes: int = 3):
        self.merge_window_sec = merge_window_sec
        self.max_episodes = max_episodes
    
    def cluster_mistakes(
        self,
        mistakes: List[Dict]
    ) -> List[MistakeEpisode]:
        """
        Cluster mistakes into episodes.
        
        Args:
            mistakes: List of mistake dicts (from detector)
        
        Returns:
            List of MistakeEpisode, max self.max_episodes per type
        """
        if not mistakes:
            return []
        
        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        for m in mistakes:
            mt = m.get("type", "unknown")
            if mt not in by_type:
                by_type[mt] = []
            by_type[mt].append(m)
        
        all_episodes = []
        episode_id = 0
        
        for mistake_type, type_mistakes in by_type.items():
            # Sort by timestamp
            sorted_mistakes = sorted(type_mistakes, key=lambda x: x.get("timestamp", 0))
            
            # Cluster
            clusters = []
            current_cluster = [sorted_mistakes[0]]
            
            for i in range(1, len(sorted_mistakes)):
                prev_ts = current_cluster[-1].get("timestamp", 0)
                curr_ts = sorted_mistakes[i].get("timestamp", 0)
                
                if curr_ts - prev_ts <= self.merge_window_sec:
                    current_cluster.append(sorted_mistakes[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [sorted_mistakes[i]]
            
            clusters.append(current_cluster)
            
            # Create episodes from clusters
            for cluster in clusters:
                if not cluster:
                    continue
                
                episode_id += 1
                timestamps = [m.get("timestamp", 0) for m in cluster]
                severities = [m.get("severity", 0) for m in cluster]
                confidences = [m.get("confidence", 0) for m in cluster]
                
                # Find representative (highest severity)
                best_idx = severities.index(max(severities))
                rep = cluster[best_idx]
                
                episode = MistakeEpisode(
                    mistake_type=mistake_type,
                    episode_id=episode_id,
                    start_timestamp=min(timestamps),
                    end_timestamp=max(timestamps),
                    duration=max(timestamps) - min(timestamps),
                    occurrence_count=len(cluster),
                    avg_severity=sum(severities) / len(severities),
                    avg_confidence=sum(confidences) / len(confidences),
                    peak_severity=max(severities),
                    representative_timestamp=rep.get("timestamp", 0),
                    representative_cue=rep.get("cue", ""),
                    representative_description=rep.get("description", ""),
                    all_timestamps=timestamps
                )
                all_episodes.append(episode)
        
        # Sort by score and return top
        def episode_score(ep: MistakeEpisode) -> float:
            return ep.peak_severity * math.log(1 + ep.occurrence_count) * ep.avg_confidence
        
        all_episodes.sort(key=episode_score, reverse=True)
        
        return all_episodes[:self.max_episodes * len(by_type)]  # Top 3 per type
    
    def get_top_episodes(
        self,
        episodes: List[MistakeEpisode],
        n: int = 3
    ) -> List[MistakeEpisode]:
        """Get top N episodes across all types."""
        def score(ep: MistakeEpisode) -> float:
            return ep.peak_severity * math.log(1 + ep.occurrence_count) * ep.avg_confidence
        
        sorted_eps = sorted(episodes, key=score, reverse=True)
        return sorted_eps[:n]


class SmartFixFirstRanker:
    """
    Smarter Fix-First ranking for stroke mode.
    Accounts for drill relevance and foundational issues.
    """
    
    # Priority multipliers (higher = more important to fix first)
    PRIORITY_MULTIPLIERS = {
        # Foundation issues (fix these first)
        "poor_ready_position": 1.5,
        "narrow_stance": 1.4,
        
        # Core technique
        "overhead_contact_too_low": 1.3,
        "contact_not_in_front": 1.2,
        "knee_collapse": 1.2,
        
        # Timing/sequence
        "elbow_not_leading": 1.0,
        "slow_recovery": 1.0,
        "missing_split_step": 1.0,
        
        # Minor
        "overhead_contact_medium": 0.7,
        "unclear_overhead_intent": 0.5,
    }
    
    # Drill relevance mapping
    DRILL_RELEVANCE = {
        "overhead": {
            "overhead_contact_too_low": 1.5,
            "contact_not_in_front": 1.3,
            "elbow_not_leading": 1.2,
            "poor_ready_position": 1.3,
        },
        "footwork": {
            "narrow_stance": 1.3,
            "knee_collapse": 1.3,
            "slow_recovery": 1.2,
            "missing_split_step": 1.2,
        }
    }
    
    def compute_fix_score(
        self,
        episode: MistakeEpisode,
        drill_type: str
    ) -> float:
        """
        Compute fix-first score for an episode.
        
        Formula: severity * log(1+frequency) * confidence * priority * drill_relevance
        """
        # Base score
        severity = episode.peak_severity
        frequency = episode.occurrence_count
        confidence = episode.avg_confidence
        
        base_score = severity * math.log(1 + frequency) * confidence
        
        # Priority multiplier
        priority = self.PRIORITY_MULTIPLIERS.get(episode.mistake_type, 1.0)
        
        # Drill relevance
        drill_mode = "overhead" if "overhead" in drill_type.lower() or "smash" in drill_type.lower() or "clear" in drill_type.lower() else "footwork"
        relevance_map = self.DRILL_RELEVANCE.get(drill_mode, {})
        relevance = relevance_map.get(episode.mistake_type, 1.0)
        
        return base_score * priority * relevance
    
    def get_fix_first(
        self,
        episodes: List[MistakeEpisode],
        drill_type: str
    ) -> Optional[MistakeEpisode]:
        """Get the single most important issue to fix first."""
        if not episodes:
            return None
        
        scored = [(self.compute_fix_score(ep, drill_type), ep) for ep in episodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return scored[0][1] if scored else None
