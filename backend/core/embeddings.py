"""
Embeddings Module
Provides embedding-based semantic search for grounded chat.

Features:
- Compute embeddings for evidence chunks using sentence-transformers
- Cache embeddings per session to disk
- Semantic similarity search for chat grounding
- Threshold-based grounding validation
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from config import get_thresholds

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Falling back to keyword-based retrieval.")


@dataclass
class ChunkMeta:
    """Metadata for an evidence chunk"""
    chunk_id: int
    chunk_type: str  # "mistake", "event", "priority", "summary", "stroke"
    timestamps: List[float] = field(default_factory=list)
    severity: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class EmbeddingsCache:
    """Cached embeddings for a session"""
    session_id: str
    chunks: List[str]
    embeddings: List[List[float]]
    meta: List[Dict]
    model_name: str
    version: str = "1.0"


@dataclass
class RetrievalResult:
    """Result of semantic retrieval"""
    chunk: str
    chunk_id: int
    similarity: float
    meta: ChunkMeta


class EmbeddingsManager:
    """
    Manages embeddings for session evidence chunks.
    Handles caching, computation, and retrieval.
    """
    
    def __init__(self, data_dir: str = "./data/sessions"):
        self.config = get_thresholds().embeddings
        self.data_dir = Path(data_dir)
        self._model: Optional["SentenceTransformer"] = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy-load the embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            return
        
        if self._model_loaded:
            return
        
        try:
            self._model = SentenceTransformer(self.config.model_name)
            self._model_loaded = True
            logger.info(f"Loaded embedding model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None
    
    def _get_cache_path(self, session_id: str) -> Path:
        """Get path for embeddings cache file"""
        return self.data_dir / session_id / "embeddings.json"
    
    def _parse_chunk_meta(self, chunk: str, chunk_id: int) -> ChunkMeta:
        """Parse chunk text to extract metadata"""
        import re
        
        # Extract timestamps from chunk
        timestamp_pattern = r'(\d+\.?\d*)s'
        timestamps = [float(m) for m in re.findall(timestamp_pattern, chunk)]
        
        # Determine chunk type
        chunk_upper = chunk.upper()
        if "MISTAKE" in chunk_upper:
            chunk_type = "mistake"
        elif "PRIORITY" in chunk_upper:
            chunk_type = "priority"
        elif "CONFIDENCE NOTES" in chunk_upper:
            chunk_type = "confidence"
        elif "SUMMARY" in chunk_upper or "SESSION" in chunk_upper:
            chunk_type = "summary"
        elif "STROKE" in chunk_upper or "OVERHEAD" in chunk_upper:
            chunk_type = "stroke"
        else:
            chunk_type = "event"
        
        # Extract severity if present
        severity_match = re.search(r'severity[:\s]+(\d+\.?\d*)', chunk.lower())
        severity = float(severity_match.group(1)) if severity_match else None
        
        # Extract confidence if present
        conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', chunk.lower())
        confidence = float(conf_match.group(1)) if conf_match else None
        
        return ChunkMeta(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            timestamps=timestamps[:3],  # Limit to first 3
            severity=severity,
            confidence=confidence
        )
    
    def compute_embeddings(
        self, 
        session_id: str, 
        chunks: List[str],
        force_recompute: bool = False
    ) -> Optional[EmbeddingsCache]:
        """
        Compute embeddings for chunks and cache to disk.
        
        Args:
            session_id: Session ID for caching
            chunks: List of evidence chunk strings
            force_recompute: If True, recompute even if cache exists
        
        Returns:
            EmbeddingsCache or None if embeddings not available
        """
        cache_path = self._get_cache_path(session_id)
        
        # Check cache first
        if not force_recompute and self.config.cache_embeddings and cache_path.exists():
            try:
                cached = self.load_cached_embeddings(session_id)
                if cached and cached.chunks == chunks:
                    logger.debug(f"Using cached embeddings for session {session_id}")
                    return cached
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Embeddings not available - sentence-transformers not installed")
            return None
        
        self._load_model()
        if not self._model:
            return None
        
        # Compute embeddings
        try:
            embeddings = self._model.encode(chunks, convert_to_numpy=True)
            embeddings_list = embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return None
        
        # Parse metadata
        meta_list = []
        for i, chunk in enumerate(chunks):
            meta = self._parse_chunk_meta(chunk, i)
            meta_list.append(asdict(meta))
        
        # Create cache object
        cache = EmbeddingsCache(
            session_id=session_id,
            chunks=chunks,
            embeddings=embeddings_list,
            meta=meta_list,
            model_name=self.config.model_name
        )
        
        # Save to disk
        if self.config.cache_embeddings:
            self._save_cache(cache)
        
        logger.info(f"Computed embeddings for {len(chunks)} chunks, session {session_id}")
        return cache
    
    def _save_cache(self, cache: EmbeddingsCache):
        """Save embeddings cache to disk"""
        cache_path = self._get_cache_path(cache.session_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(asdict(cache), f)
            logger.debug(f"Saved embeddings cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
    
    def load_cached_embeddings(self, session_id: str) -> Optional[EmbeddingsCache]:
        """Load embeddings from cache"""
        cache_path = self._get_cache_path(session_id)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return EmbeddingsCache(**data)
        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {e}")
            return None
    
    def semantic_search(
        self,
        query: str,
        cache: EmbeddingsCache,
        top_k: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Perform semantic search over cached chunks.
        
        Args:
            query: User query string
            cache: Cached embeddings for session
            top_k: Number of results to return (default from config)
        
        Returns:
            Tuple of (results list, max similarity score)
        """
        if not EMBEDDINGS_AVAILABLE:
            # Fallback to keyword-based
            return self._keyword_search(query, cache.chunks, cache.meta, top_k)
        
        self._load_model()
        if not self._model:
            return self._keyword_search(query, cache.chunks, cache.meta, top_k)
        
        top_k = top_k or self.config.top_k
        
        try:
            # Embed query
            query_embedding = self._model.encode([query], convert_to_numpy=True)[0]
            
            # Compute cosine similarities
            import numpy as np
            chunk_embeddings = np.array(cache.embeddings)
            
            # Normalize
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            chunk_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-10)
            
            # Cosine similarity
            similarities = np.dot(chunk_norms, query_norm)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            max_sim = 0.0
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim >= self.config.min_include_similarity:
                    meta = ChunkMeta(**cache.meta[idx])
                    results.append(RetrievalResult(
                        chunk=cache.chunks[idx],
                        chunk_id=idx,
                        similarity=sim,
                        meta=meta
                    ))
                max_sim = max(max_sim, sim)
            
            logger.debug(f"Semantic search: query='{query[:50]}...', max_sim={max_sim:.3f}, results={len(results)}")
            return results, max_sim
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._keyword_search(query, cache.chunks, cache.meta, top_k)
    
    def _keyword_search(
        self,
        query: str,
        chunks: List[str],
        meta_list: List[Dict],
        top_k: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], float]:
        """Fallback keyword-based search"""
        top_k = top_k or self.config.top_k
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Keyword categories for badminton
        keywords = {
            "split": ["split", "hop", "jump", "step"],
            "lunge": ["lunge", "reach", "stretch"],
            "knee": ["knee", "valgus", "collapse", "inward"],
            "stance": ["stance", "width", "narrow", "wide", "feet"],
            "recovery": ["recovery", "base", "center", "back"],
            "mistake": ["mistake", "error", "wrong", "problem", "issue"],
            "timing": ["timing", "late", "early", "slow", "fast"],
            "overhead": ["overhead", "stroke", "contact", "wrist", "elbow"],
            "summary": ["overall", "summary", "total", "session"]
        }
        
        scored = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = 0.0
            
            # Direct word matches
            for word in query_words:
                if word in chunk_lower:
                    score += 0.2
            
            # Keyword category matches
            for category, kws in keywords.items():
                query_has_category = any(kw in query_lower for kw in kws)
                chunk_has_category = any(kw in chunk_lower for kw in kws)
                if query_has_category and chunk_has_category:
                    score += 0.3
            
            # Boost priority/mistake chunks
            if "PRIORITY" in chunk or "MISTAKE" in chunk:
                score += 0.1
            
            # Cap at 1.0
            score = min(1.0, score)
            
            if score > 0:
                meta = ChunkMeta(**meta_list[i]) if meta_list else ChunkMeta(chunk_id=i, chunk_type="unknown")
                scored.append((score, i, chunk, meta))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        max_sim = scored[0][0] if scored else 0.0
        results = [
            RetrievalResult(chunk=s[2], chunk_id=s[1], similarity=s[0], meta=s[3])
            for s in scored[:top_k]
            if s[0] >= self.config.min_include_similarity
        ]
        
        return results, max_sim
    
    def is_grounded(self, max_similarity: float) -> bool:
        """Check if max similarity meets grounding threshold"""
        return max_similarity >= self.config.similarity_threshold


def create_structured_chunks(report_dict: Dict) -> List[str]:
    """
    Create structured evidence chunks from a report dict.
    Each chunk includes timestamps for citation.
    
    Args:
        report_dict: Report dict from report_generator.report_to_dict()
    
    Returns:
        List of evidence chunk strings
    """
    chunks = []

    def format_ts(ts: float) -> str:
        value = f"{ts:.2f}".rstrip("0").rstrip(".")
        return f"{value}s"

    def format_ts_list(timestamps: List[float]) -> str:
        if not timestamps:
            return "[]"
        return "[" + ", ".join(format_ts(t) for t in timestamps) + "]"
    
    # Mistakes grouped by type
    mistakes = report_dict.get("mistakes", [])
    for m in mistakes:
        chunk = (
            f"{format_ts(m['timestamp'])}: MISTAKE - {m['type'].replace('_', ' ')}. "
            f"Severity: {m['severity']}/1.0. Confidence: {m['confidence']}. "
            f"Cue: '{m.get('cue', '')}'. Evidence at: {format_ts_list(m.get('evidence', []))}"
        )
        chunks.append(chunk)

    # Mistake summaries by type
    if mistakes:
        grouped = {}
        for m in mistakes:
            key = m.get("type", "unknown")
            grouped.setdefault(key, []).append(m.get("timestamp", 0))
        for key, ts_list in grouped.items():
            ts_list = sorted(t for t in ts_list if t is not None)
            chunk = (
                f"MISTAKE SUMMARY: {key.replace('_', ' ')} "
                f"({len(ts_list)} occurrences). Timestamps: {format_ts_list(ts_list[:5])}"
            )
            chunks.append(chunk)
    
    # Events
    events = report_dict.get("events", [])
    for e in events:
        chunk = (
            f"{format_ts(e['timestamp'])}-{format_ts(e['end_timestamp'])}: {e['type'].upper()} detected. "
            f"Duration: {format_ts(e['duration'])}. Confidence: {e['confidence']}"
        )
        chunks.append(chunk)
    
    # Stroke analysis
    stroke_analysis = report_dict.get("stroke_analysis")
    if stroke_analysis:
        strokes = stroke_analysis.get("strokes", [])
        analyses = stroke_analysis.get("analyses", [])
        
        for stroke in strokes:
            chunk = (
                f"{format_ts(stroke['contact_timestamp'])}: OVERHEAD STROKE detected. "
                f"Duration: {format_ts(stroke['duration'])}. "
                f"Overhead confidence: {stroke['overhead_confidence']}. "
                f"Dominant arm: {stroke['dominant_side']}"
            )
            chunks.append(chunk)
        
        for analysis in analyses:
            status_parts = []
            if not analysis.get("wrist_above_shoulder"):
                status_parts.append("contact below shoulder")
            if not analysis.get("contact_in_front"):
                status_parts.append("contact behind body")
            if not analysis.get("elbow_leads_wrist"):
                status_parts.append("elbow not leading")
            if not analysis.get("ready_position_good"):
                status_parts.append("ready position issues")
            
            if status_parts:
                chunk = (
                    f"Stroke #{analysis['stroke_id']}: Issues detected - {', '.join(status_parts)}. "
                    f"Contact height: {analysis.get('contact_height_status', 'unknown')}."
                )
            else:
                chunk = f"Stroke #{analysis['stroke_id']}: Good form. Contact height: {analysis.get('contact_height_status', 'ok')}."
            chunks.append(chunk)
    
    # Fix first plan (priority)
    fix_first = report_dict.get("fix_first_plan")
    if fix_first:
        chunk = (
            f"PRIORITY FIX: {fix_first['primary_issue'].replace('_', ' ')} "
            f"({fix_first['occurrences']} occurrences). Focus: {fix_first['cue']}. "
            f"Recommended drill: {fix_first['focus_drill']}. "
            f"Key timestamps: {format_ts_list(fix_first.get('key_timestamps', []))}"
        )
        chunks.append(chunk)
    
    # Top mistakes summary
    top_mistakes = report_dict.get("top_mistakes", [])
    if top_mistakes:
        for tm in top_mistakes:
            chunk = (
                f"TOP ISSUE: {tm['type'].replace('_', ' ')} - {tm['count']} occurrences. "
                f"Max severity: {tm['max_severity']}. Cue: '{tm['cue']}'. "
                f"Timestamps: {format_ts_list(tm.get('timestamps', [])[:3])}"
            )
            chunks.append(chunk)
    
    # Metrics summary
    metrics = report_dict.get("metrics_summary", {})
    if metrics:
        duration = report_dict.get("video_duration", 0)
        chunk = (
            f"SESSION SUMMARY: {duration:.1f}s video. "
            f"{metrics.get('total_events', 0)} footwork events. "
            f"{metrics.get('total_mistakes', 0)} issues found. "
            f"Split steps: {metrics.get('split_steps_detected', 0)}. "
            f"Lunges: {metrics.get('lunges_detected', 0)}. "
            f"Pose confidence: {metrics.get('avg_pose_confidence', 0)}"
        )
        if metrics.get("strokes_detected"):
            chunk += f". Strokes detected: {metrics['strokes_detected']}"
        chunks.append(chunk)
    
    # Confidence notes
    conf_notes = report_dict.get("confidence_notes", [])
    if conf_notes:
        chunk = f"CONFIDENCE NOTES: {'; '.join(conf_notes)}"
        chunks.append(chunk)
    
    # Drill type info
    drill_type = report_dict.get("drill_type", "unknown")
    drill_source = report_dict.get("drill_type_source", "user")
    drill_conf = report_dict.get("drill_type_confidence", 1.0)
    chunk = f"DRILL INFO: Type='{drill_type}', Source='{drill_source}', Confidence={drill_conf:.2f}"
    chunks.append(chunk)
    
    return chunks
