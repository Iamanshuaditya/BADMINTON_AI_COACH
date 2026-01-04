"""
OneEuro Filter Implementation for Keypoint Smoothing
Based on: https://cristal.univ-lille.fr/~casiez/1euro/

The 1€ filter is a simple speed-based low-pass filter that reduces jitter
while minimizing lag. Perfect for real-time pose estimation smoothing.
"""

import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class OneEuroFilterConfig:
    """Configuration for OneEuro filter"""
    min_cutoff: float = 1.0  # Minimum cutoff frequency (Hz)
    beta: float = 0.007  # Speed coefficient
    d_cutoff: float = 1.0  # Derivative cutoff frequency


class LowPassFilter:
    """Simple exponential low-pass filter"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.y: Optional[float] = None
        self.s: Optional[float] = None
    
    def set_alpha(self, alpha: float):
        self.alpha = max(0.0, min(1.0, alpha))
    
    def filter(self, value: float) -> float:
        if self.y is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s
    
    def has_last_raw_value(self) -> bool:
        return self.y is not None
    
    def last_raw_value(self) -> float:
        return self.y if self.y is not None else 0.0
    
    def reset(self):
        self.y = None
        self.s = None


class OneEuroFilter:
    """
    One Euro Filter for smooth, low-latency signal filtering.
    
    Automatically adapts the cutoff frequency based on signal speed:
    - Slow movements get more smoothing (lower cutoff)
    - Fast movements get less smoothing (higher cutoff) to reduce lag
    """
    
    def __init__(
        self,
        freq: float = 30.0,  # Sampling frequency (FPS)
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None
    
    def _alpha(self, cutoff: float) -> float:
        """Compute alpha for exponential smoothing"""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter a value.
        
        Args:
            x: Input value
            timestamp: Optional timestamp (seconds). If provided, will compute
                      dynamic frequency from time delta.
        
        Returns:
            Filtered value
        """
        # Update frequency from timestamp if available
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 0:
                self.freq = 1.0 / dt
        self.last_time = timestamp
        
        # Estimate derivative
        if self.x_filter.has_last_raw_value():
            dx = (x - self.x_filter.last_raw_value()) * self.freq
        else:
            dx = 0.0
        
        # Filter derivative
        self.dx_filter.set_alpha(self._alpha(self.d_cutoff))
        edx = self.dx_filter.filter(dx)
        
        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Filter the signal
        self.x_filter.set_alpha(self._alpha(cutoff))
        return self.x_filter.filter(x)
    
    def reset(self):
        """Reset filter state"""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class KeypointSmoother:
    """
    Applies OneEuro filtering to pose estimation keypoints.
    Maintains separate filters for each landmark's x, y coordinates.
    """
    
    def __init__(
        self,
        num_landmarks: int = 33,  # MediaPipe has 33 landmarks
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        feet_indices: Optional[Tuple[int, ...]] = None,
        feet_min_cutoff: float = 0.8,
        feet_beta: float = 0.01
    ):
        self.num_landmarks = num_landmarks
        self.freq = freq
        
        # Default feet indices (MediaPipe: ankles 27,28 and heels 29,30 and toes 31,32)
        self.feet_indices = feet_indices or (27, 28, 29, 30, 31, 32)
        
        # Create filters for each landmark (x, y)
        self.filters_x: Dict[int, OneEuroFilter] = {}
        self.filters_y: Dict[int, OneEuroFilter] = {}
        
        for i in range(num_landmarks):
            # Feet get different parameters (more smoothing)
            if i in self.feet_indices:
                mc, b = feet_min_cutoff, feet_beta
            else:
                mc, b = min_cutoff, beta
            
            self.filters_x[i] = OneEuroFilter(freq, mc, b, d_cutoff)
            self.filters_y[i] = OneEuroFilter(freq, mc, b, d_cutoff)
    
    def smooth(
        self,
        landmarks: list,
        timestamp: Optional[float] = None,
        visibility_threshold: float = 0.5
    ) -> list:
        """
        Smooth a frame of landmarks.
        
        Args:
            landmarks: List of landmark dicts with x, y, visibility keys
            timestamp: Optional timestamp for adaptive frequency
            visibility_threshold: Don't aggressively filter low-visibility landmarks
        
        Returns:
            List of smoothed landmarks
        """
        smoothed = []
        
        for i, lm in enumerate(landmarks):
            if i >= self.num_landmarks:
                break
            
            x = lm.get("x", 0)
            y = lm.get("y", 0)
            visibility = lm.get("visibility", 1.0)
            
            # Gate: if visibility is low, don't update filter aggressively
            # Just return raw value and mark as unreliable
            if visibility < visibility_threshold:
                smoothed.append({
                    "x": x,
                    "y": y,
                    "z": lm.get("z", 0),
                    "visibility": visibility,
                    "smoothed": False,
                    "reliable": False
                })
                continue
            
            # Apply filtering
            sx = self.filters_x[i].filter(x, timestamp)
            sy = self.filters_y[i].filter(y, timestamp)
            
            smoothed.append({
                "x": sx,
                "y": sy,
                "z": lm.get("z", 0),
                "visibility": visibility,
                "smoothed": True,
                "reliable": visibility >= visibility_threshold
            })
        
        return smoothed
    
    def reset(self):
        """Reset all filters"""
        for f in self.filters_x.values():
            f.reset()
        for f in self.filters_y.values():
            f.reset()
