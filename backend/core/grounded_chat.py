"""
Grounded Chat Module
Implements RAG-based chat that answers ONLY from evidence with embedding-based retrieval.
Supports both direct Google Gemini API and Anthropic-compatible proxies.

Features:
- Embedding-based semantic retrieval (sentence-transformers)
- Cached embeddings per session
- Similarity threshold for grounding validation
- Structured response with citations and grounded status
- One correction principle enforcement
"""

import re
from typing import Dict, List, Optional
import logging

# Import settings and thresholds
from config.settings import get_settings
from config import get_thresholds

# Import embeddings manager
from core.embeddings import EmbeddingsManager, RetrievalResult

logger = logging.getLogger(__name__)

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import Anthropic
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


STRICT_SYSTEM_PROMPT = """You are ShuttleSense AI, an elite badminton technical coach analyzing a player's video session.

YOUR COACHING STYLE:
- Be encouraging but technically precise
- Give ACTIONABLE advice the player can immediately use
- Focus on ONE key correction at a time (the most impactful issue first)
- Provide a specific 30-60 second micro-drill for each correction
- Use timestamps [Xs] to reference specific moments from their session

WHEN ANALYZING ISSUES:
- If the evidence shows MISTAKES (like "missing split step"), acknowledge the issue and provide the FIX
- Reference specific timestamps where the issue occurred
- Explain WHY this matters for their game
- Give them a concrete drill to practice

RESPONSE FORMAT:
1. Acknowledge what you see in their session
2. Explain the technical issue briefly
3. Provide the correction cue (short, memorable phrase)
4. Give a specific micro-drill (30-60 seconds)

Example good response:
"I can see at [12.3s] and [18.5s] you're initiating your split step late - about 0.2 seconds after your opponent contacts the shuttle. This delays your reaction.

**The Fix:** Time your split step to land AS your opponent's racket contacts the shuttle.

**Quick Drill (60s):** Shadow footwork - have a partner call 'now!' randomly. Practice landing your split step on their call. Do 20 reps, focusing on soft, balanced landings."

Be the coach every player wishes they had - supportive, specific, and practical."""


def format_evidence_for_prompt(results: List[RetrievalResult], confidence_notes: Optional[str] = None) -> str:
    """Format retrieved evidence for the LLM prompt."""
    if not results:
        return "No evidence available for this session."
    
    lines = ["EVIDENCE FROM VIDEO SESSION (sorted by relevance):"]
    for i, result in enumerate(results, 1):
        sim_pct = int(result.similarity * 100)
        lines.append(f"{i}. [{result.meta.chunk_type}] (relevance: {sim_pct}%) {result.chunk}")
    if confidence_notes:
        lines.append(f"CONFIDENCE NOTES: {confidence_notes}")
    return "\n".join(lines)


def build_chat_prompt(
    question: str,
    retrieved_results: List[RetrievalResult],
    session_summary: Optional[Dict] = None,
    confidence_notes: Optional[str] = None
) -> str:
    """Build the full prompt for the LLM."""
    evidence = format_evidence_for_prompt(retrieved_results, confidence_notes)
    
    prompt = f"""{STRICT_SYSTEM_PROMPT}

{evidence}

PLAYER'S QUESTION: {question}

Now coach them! Reference the timestamps from their session, explain what you see, and give them a specific drill to fix it. Be their supportive coach."""
    
    return prompt


def extract_citations(text: str) -> List[Dict]:
    """
    Extract timestamp citations from response text.
    Returns list of citation dicts with timestamp and type.
    """
    pattern = r'\[(\d+\.?\d*)s?\]'
    matches = re.findall(pattern, text)
    
    citations = []
    seen = set()
    for m in matches:
        ts = float(m)
        if ts not in seen:
            citations.append({
                "timestamp": ts,
                "type": "evidence_reference"
            })
            seen.add(ts)
    
    return citations


class GroundedChat:
    """
    Chat interface that grounds all responses in evidence.
    Uses embedding-based semantic retrieval with caching.
    """
    
    def __init__(self, api_key: Optional[str] = None, data_dir: str = "./data/sessions"):
        self.settings = get_settings()
        self.thresholds = get_thresholds()
        self.api_key = api_key or self.settings.GOOGLE_API_KEY or "dummy"
        self.provider = self.settings.LLM_PROVIDER
        self.proxy_url = self.settings.LLM_PROXY_URL
        
        self.genai_model = None
        self.anthropic_client = None
        
        # Initialize embeddings manager
        self.embeddings_manager = EmbeddingsManager(data_dir)
        
        # Cache for session embeddings
        self._session_cache: Dict[str, any] = {}
        
        # Initialize provider
        if self.provider == "anthropic" or "proxy" in self.provider:
            if ANTHROPIC_AVAILABLE:
                try:
                    self.anthropic_client = Anthropic(
                        api_key=self.api_key,
                        base_url=self.proxy_url
                    )
                    logger.info(f"Initialized Anthropic client with proxy: {self.proxy_url}")
                except Exception as e:
                    logger.error(f"Failed to init Anthropic client: {e}")
            else:
                logger.error("Anthropic library not installed. Cannot use proxy.")
        
        elif GEMINI_AVAILABLE and self.settings.GOOGLE_API_KEY:
            try:
                genai.configure(api_key=self.settings.GOOGLE_API_KEY)
                self.genai_model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini model initialized (Direct API)")
            except Exception as e:
                logger.warning(f"Failed to init Gemini: {e}")

    def _format_ts_value(self, ts: float) -> str:
        """Format a timestamp value for citations."""
        value = f"{ts:.2f}".rstrip("0").rstrip(".")
        return value or "0"

    def _normalize_issue(self, text: str) -> str:
        return " ".join(text.lower().replace("_", " ").split())

    def _is_generic_improvement_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        improvement_markers = [
            "what should i improve",
            "how can i improve",
            "what should i work on",
            "what should i focus on",
            "what should i fix",
            "what do i need to do",
            "what do i need to improve",
            "fix first",
            "top mistake",
            "primary issue",
            "main issue",
            "biggest issue",
            "key issue",
            "recommendation",
            "suggestions",
            "next step",
            "next steps",
            "improve",
            "improvement"
        ]
        specific_terms = [
            "split",
            "lunge",
            "knee",
            "stance",
            "recovery",
            "overhead",
            "stroke",
            "wrist",
            "elbow",
            "timing",
            "footwork",
            "balance",
            "posture",
            "base",
            "serve",
            "hip",
            "shoulder",
            "ankle"
        ]
        if not any(marker in q for marker in improvement_markers):
            return False
        return not any(term in q for term in specific_terms)

    def _build_fix_first_results(
        self,
        evidence_chunks: List[str],
        min_results: int
    ) -> List[RetrievalResult]:
        max_results = self.thresholds.chat_grounding.max_evidence_chunks
        candidate_indices: List[int] = []

        priority_idx = None
        for i, chunk in enumerate(evidence_chunks):
            if "PRIORITY FIX" in chunk:
                priority_idx = i
                candidate_indices.append(i)
                break

        if priority_idx is None:
            for marker in ("TOP ISSUE", "MISTAKE SUMMARY", "MISTAKE -"):
                for i, chunk in enumerate(evidence_chunks):
                    if marker in chunk:
                        priority_idx = i
                        candidate_indices.append(i)
                        break
                if candidate_indices:
                    break

        if not candidate_indices:
            return []

        issue = self._extract_issue(evidence_chunks[candidate_indices[0]])
        issue_norm = self._normalize_issue(issue) if issue else None

        def add_matching_chunks(predicate) -> None:
            for i, chunk in enumerate(evidence_chunks):
                if i in candidate_indices:
                    continue
                if predicate(chunk):
                    candidate_indices.append(i)
                    if len(candidate_indices) >= max_results:
                        return
                if len(candidate_indices) >= min_results:
                    return

        if issue_norm:
            def is_issue_chunk(chunk: str) -> bool:
                chunk_norm = self._normalize_issue(chunk)
                return (
                    issue_norm in chunk_norm and
                    ("MISTAKE" in chunk or "TOP ISSUE" in chunk or "MISTAKE SUMMARY" in chunk)
                )
            add_matching_chunks(is_issue_chunk)

        if len(candidate_indices) < min_results:
            add_matching_chunks(lambda chunk: "MISTAKE" in chunk)

        if len(candidate_indices) < min_results:
            add_matching_chunks(lambda chunk: "SESSION SUMMARY" in chunk)

        results = []
        for rank, idx in enumerate(candidate_indices[:max_results]):
            chunk = evidence_chunks[idx]
            meta = self.embeddings_manager._parse_chunk_meta(chunk, idx)
            similarity = max(0.5, 1.0 - (rank * 0.05))
            results.append(RetrievalResult(
                chunk=chunk,
                chunk_id=idx,
                similarity=similarity,
                meta=meta
            ))

        return results

    def _extract_issue(self, chunk: str) -> Optional[str]:
        match = re.search(r"PRIORITY FIX:\s*([^\(\.]+)", chunk, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"MISTAKE\s*-\s*([^.]+)", chunk, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"TOP ISSUE:\s*([^\-]+)", chunk, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_cue(self, chunk: str) -> Optional[str]:
        match = re.search(r"Cue:\s*'([^']+)'", chunk, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"Focus:\s*([^\.]+)", chunk, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_recommended_drills(self, chunks: List[str]) -> List[str]:
        drills = []
        for chunk in chunks:
            match = re.search(r"Recommended drill:\s*([^\.]+)", chunk, re.IGNORECASE)
            if match:
                drills.append(match.group(1).strip())
        return drills

    def _find_confidence_notes(self, chunks: List[str]) -> Optional[str]:
        for chunk in chunks:
            if chunk.upper().startswith("CONFIDENCE NOTES"):
                return chunk.split(":", 1)[-1].strip()
        return None

    def _select_primary_result(self, results: List[RetrievalResult]) -> Optional[RetrievalResult]:
        if not results:
            return None
        for chunk_type in ["priority", "mistake", "event", "summary"]:
            for result in results:
                if result.meta.chunk_type == chunk_type and result.meta.timestamps:
                    return result
        for result in results:
            if result.meta.timestamps:
                return result
        return results[0]

    def _collect_evidence_timestamps(self, results: List[RetrievalResult]) -> set:
        timestamps = set()
        for result in results:
            for ts in result.meta.timestamps:
                timestamps.add(self._format_ts_value(ts))
        return timestamps

    def _map_citations_to_types(
        self,
        citations: List[Dict],
        results: List[RetrievalResult]
    ) -> List[Dict]:
        if not citations:
            return citations
        ts_to_type = {}
        for result in results:
            issue = self._extract_issue(result.chunk)
            issue_type = self._normalize_issue(issue).replace(" ", "_") if issue else result.meta.chunk_type
            for ts in result.meta.timestamps:
                key = self._format_ts_value(ts)
                ts_to_type.setdefault(key, issue_type)
        mapped = []
        for citation in citations:
            ts_key = self._format_ts_value(citation.get("timestamp", 0))
            mapped.append({
                "timestamp": citation.get("timestamp", 0),
                "type": ts_to_type.get(ts_key, "evidence_reference")
            })
        return mapped

    def _get_recommended_drill_for_issue(
        self,
        results: List[RetrievalResult],
        issue: Optional[str]
    ) -> Optional[str]:
        issue_norm = self._normalize_issue(issue) if issue else None
        for result in results:
            if "Recommended drill:" not in result.chunk:
                continue
            chunk_issue = self._extract_issue(result.chunk)
            chunk_issue_norm = self._normalize_issue(chunk_issue) if chunk_issue else None
            if not issue_norm or not chunk_issue_norm or issue_norm == chunk_issue_norm:
                match = re.search(r"Recommended drill:\s*([^\.]+)", result.chunk, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return None

    def _find_recommended_drill_in_chunks(
        self,
        chunks: List[str],
        issue: Optional[str]
    ) -> Optional[str]:
        issue_norm = self._normalize_issue(issue) if issue else None
        fallback = None
        for chunk in chunks:
            if "Recommended drill:" not in chunk:
                continue
            chunk_issue = self._extract_issue(chunk)
            chunk_issue_norm = self._normalize_issue(chunk_issue) if chunk_issue else None
            match = re.search(r"Recommended drill:\s*([^\.]+)", chunk, re.IGNORECASE)
            if not match:
                continue
            drill = match.group(1).strip()
            if fallback is None:
                fallback = drill
            if not issue_norm or not chunk_issue_norm or issue_norm == chunk_issue_norm:
                return drill
        return fallback

    def _build_evidence_answer(
        self,
        results: List[RetrievalResult],
        confidence_notes: Optional[str],
        evidence_chunks: Optional[List[str]] = None
    ) -> Optional[Dict]:
        primary = self._select_primary_result(results)
        if not primary or not primary.meta.timestamps:
            return None

        issue = self._extract_issue(primary.chunk) or "technique issue"
        cue = self._extract_cue(primary.chunk)
        drill = self._get_recommended_drill_for_issue(results, issue)
        if not drill and evidence_chunks:
            drill = self._find_recommended_drill_in_chunks(evidence_chunks, issue)

        if not drill:
            return None

        ts_value = self._format_ts_value(primary.meta.timestamps[0])
        ts_tag = f"[{ts_value}s]"

        parts = [f"Focus on {issue} at {ts_tag}."]
        if cue:
            parts.append(f"Cue: {cue} {ts_tag}.")
        parts.append(f"30-60s micro-drill: {drill} {ts_tag}.")
        if confidence_notes:
            parts.append(f"Confidence note: {confidence_notes} {ts_tag}.")

        answer = " ".join(parts)
        citations = extract_citations(answer)

        return {
            "answer": answer,
            "citations": citations
        }
    
    def _get_or_create_embeddings(
        self, 
        session_id: str, 
        evidence_chunks: List[str]
    ):
        """Get cached embeddings or compute new ones."""
        # Check in-memory cache
        if session_id in self._session_cache:
            cached = self._session_cache[session_id]
            if cached.chunks == evidence_chunks:
                return cached
        
        # Compute embeddings (will use disk cache if available)
        cache = self.embeddings_manager.compute_embeddings(session_id, evidence_chunks)
        
        if cache:
            self._session_cache[session_id] = cache
        
        return cache
    
    def chat(
        self,
        question: str,
        evidence_chunks: List[str],
        session_id: str = "unknown",
        session_summary: Optional[Dict] = None,
        include_debug: bool = False
    ) -> Dict:
        """
        Answer a question grounded in evidence using semantic retrieval.
        
        Args:
            question: User's question
            evidence_chunks: List of evidence chunk strings
            session_id: Session ID for caching
            session_summary: Optional session metrics summary
            include_debug: Include debug info in response
        
        Returns:
            Dict with structured response:
            {
                "answer": str,
                "grounded": bool,
                "citations": [{"timestamp": float, "type": str}],
                "debug": {...}  # if include_debug
                "missing_evidence": {...}  # if grounded=false
            }
        """
        config = self.thresholds.chat_grounding
        include_debug = include_debug or config.include_debug

        if not evidence_chunks:
            return self._build_ungrounded_response(
                question=question,
                max_similarity=0.0,
                threshold=self.thresholds.embeddings.similarity_threshold,
                include_debug=include_debug,
                reason="No evidence chunks"
            )

        if self._is_generic_improvement_question(question):
            fix_results = self._build_fix_first_results(evidence_chunks, config.min_evidence_chunks)
            if len(fix_results) >= config.min_evidence_chunks:
                confidence_notes = self._find_confidence_notes(evidence_chunks)
                fallback = self._build_evidence_answer(fix_results, confidence_notes, evidence_chunks)
                if fallback:
                    response = {
                        "answer": fallback["answer"],
                        "grounded": True,
                        "citations": self._map_citations_to_types(fallback["citations"], fix_results),
                        "evidence_used": [r.chunk for r in fix_results]
                    }
                    if include_debug:
                        response["debug"] = {
                            "top_similarity": 1.0,
                            "selected_chunk_types": [r.meta.chunk_type for r in fix_results],
                            "retrieval_count": len(fix_results),
                            "llm_model": "fix-first"
                        }
                    return response
        
        # Get or create embeddings
        embeddings_cache = self._get_or_create_embeddings(session_id, evidence_chunks)
        
        if not embeddings_cache:
            # Fallback to keyword-based if embeddings not available
            return self._keyword_fallback_chat(question, evidence_chunks, session_summary)
        
        # Perform semantic search
        results, max_similarity = self.embeddings_manager.semantic_search(
            query=question,
            cache=embeddings_cache,
            top_k=config.max_evidence_chunks
        )
        logger.debug(
            f"[{session_id}] Chat similarity max={max_similarity:.3f}, "
            f"results={len(results)}"
        )
        
        if not results:
            return self._build_ungrounded_response(
                question=question,
                max_similarity=max_similarity,
                threshold=self.thresholds.embeddings.similarity_threshold,
                include_debug=include_debug,
                reason="No relevant evidence chunks"
            )
        if len(results) < config.min_evidence_chunks:
            return self._build_ungrounded_response(
                question=question,
                max_similarity=max_similarity,
                threshold=self.thresholds.embeddings.similarity_threshold,
                include_debug=include_debug,
                reason="Too few relevant chunks"
            )

        # Get confidence notes for context
        confidence_notes = self._find_confidence_notes(evidence_chunks)
        
        # Build prompt with ALL retrieved evidence (no strict filtering)
        prompt = build_chat_prompt(question, results, session_summary, confidence_notes)

        # Try LLM first, fall back to evidence-based response
        answer = None
        citations = []
        
        if self.anthropic_client or self.genai_model:
            try:
                candidate = self._get_llm_response(prompt, results)
                if candidate and len(candidate) > 20:
                    candidate_citations = extract_citations(candidate)
                    if not (config.require_citations and not candidate_citations):
                        answer = candidate
                        citations = self._map_citations_to_types(candidate_citations, results)
            except Exception as e:
                logger.warning(f"LLM failed, using evidence fallback: {e}")

        # Fallback to evidence-only response
        if not answer:
            fallback = self._build_evidence_answer(results, confidence_notes, evidence_chunks)
            if fallback:
                answer = fallback["answer"]
                citations = self._map_citations_to_types(fallback["citations"], results)
            else:
                # Last resort: just summarize the top evidence
                answer = self._simple_evidence_summary(results, evidence_chunks)
                citations = [
                    {"timestamp": ts, "type": "evidence"}
                    for r in results[:3]
                    for ts in r.meta.timestamps[:1]
                ]
        
        # Build response
        response = {
            "answer": answer,
            "grounded": max_similarity >= self.thresholds.embeddings.similarity_threshold,
            "citations": citations,
            "evidence_used": [r.chunk for r in results]
        }
        
        if include_debug:
            response["debug"] = {
                "top_similarity": round(max_similarity, 3),
                "selected_chunk_types": [r.meta.chunk_type for r in results],
                "retrieval_count": len(results),
                "llm_model": self.provider if (self.anthropic_client or self.genai_model) else "evidence-only"
            }
        
        return response
    
    def _simple_evidence_summary(self, results: List[RetrievalResult], chunks: List[str]) -> str:
        """Create a simple summary from evidence when everything else fails."""
        if not results:
            return "No relevant evidence found in this session."
        
        # Find the priority fix first
        priority_chunks = [c for c in chunks if "PRIORITY FIX:" in c]
        if priority_chunks:
            # Extract key info from priority fix
            pc = priority_chunks[0]
            return f"Based on your session: {pc}"
        
        # Otherwise summarize top result
        top = results[0]
        return f"From your session at [{top.meta.timestamps[0] if top.meta.timestamps else 0}s]: {top.chunk}"
    
    def _get_llm_response(self, prompt: str, results: List[RetrievalResult]) -> str:
        """Get response from LLM provider."""
        fallback = self._fallback_response_from_results(results)
        
        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.genai_model:
                response = self.genai_model.generate_content(prompt)
                return response.text
            
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
        
        return fallback
    
    def _stream_llm_response(self, prompt: str, results: List[RetrievalResult]):
        """Stream response from LLM provider, yielding text chunks."""
        try:
            if self.anthropic_client:
                with self.anthropic_client.messages.stream(
                    model="claude-sonnet-4-5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for text in stream.text_stream:
                        yield text
            else:
                # Fallback for non-streaming
                full_response = self._get_llm_response(prompt, results)
                # Simulate streaming by yielding words
                for word in full_response.split(' '):
                    yield word + ' '
                    
        except Exception as e:
            logger.error(f"LLM streaming failed ({self.provider}): {e}")
            # Yield fallback on error
            fallback = self._fallback_response_from_results(results)
            yield fallback
    
    def chat_stream(
        self,
        question: str,
        evidence_chunks: List[str],
        session_id: str = "unknown",
        session_summary: Optional[Dict] = None
    ):
        """
        Stream chat response - yields text chunks as they are generated.
        
        Yields:
            str: Text chunks as they are generated
        """
        config = self.thresholds.chat_grounding
        
        if not evidence_chunks:
            yield "No evidence available for this session."
            return

        if self._is_generic_improvement_question(question):
            fix_results = self._build_fix_first_results(evidence_chunks, config.min_evidence_chunks)
            if len(fix_results) >= config.min_evidence_chunks:
                confidence_notes = self._find_confidence_notes(evidence_chunks)
                fallback = self._build_evidence_answer(fix_results, confidence_notes, evidence_chunks)
                if fallback:
                    yield fallback["answer"]
                    return
        
        # Get or create embeddings
        embeddings_cache = self._get_or_create_embeddings(session_id, evidence_chunks)
        
        if not embeddings_cache:
            yield "Unable to process session data."
            return
        
        # Perform semantic search
        results, max_similarity = self.embeddings_manager.semantic_search(
            query=question,
            cache=embeddings_cache,
            top_k=config.max_evidence_chunks
        )
        
        if not results:
            yield "No relevant evidence found for your question. Try asking about specific movements or mistakes."
            return
        if len(results) < config.min_evidence_chunks:
            yield (
                "I don't have enough evidence from this session to answer that question. "
                "Try asking about specific mistakes or events detected in the video."
            )
            return
        
        # Get confidence notes for context
        confidence_notes = self._find_confidence_notes(evidence_chunks)
        
        # Build prompt
        prompt = build_chat_prompt(question, results, session_summary, confidence_notes)
        
        # Stream response
        for chunk in self._stream_llm_response(prompt, results):
            yield chunk
    
    def _fallback_response_from_results(self, results: List[RetrievalResult]) -> str:
        """Generate fallback response from retrieved results."""
        if not results:
            return "Unable to provide analysis. Please check if the video was processed correctly."
        
        # Find best chunk to respond with
        for result in results:
            if result.meta.chunk_type == "priority":
                ts = result.meta.timestamps[0] if result.meta.timestamps else 0
                return f"Based on the analysis at [{ts}s]: {result.chunk}"
            if result.meta.chunk_type == "mistake":
                ts = result.meta.timestamps[0] if result.meta.timestamps else 0
                return f"Issue detected at [{ts}s]: {result.chunk}"
        
        # Default to first result
        result = results[0]
        ts = result.meta.timestamps[0] if result.meta.timestamps else 0
        return f"From the session data at [{ts}s]: {result.chunk}"
    
    def _build_ungrounded_response(
        self,
        question: str,
        max_similarity: float,
        threshold: float,
        include_debug: bool,
        reason: str
    ) -> Dict:
        """Build response when evidence is insufficient."""
        config = self.thresholds.chat_grounding
        
        response = {
            "answer": (
                "I don't have enough evidence from this session to answer that question. "
                "The available data doesn't closely match what you're asking about. "
                "Try asking about specific mistakes or events detected in the video."
            ),
            "grounded": False,
            "citations": [],
            "missing_evidence": {
                "reason": reason,
                "max_similarity": round(max_similarity, 3),
                "threshold": threshold,
                "suggested_questions": list(config.suggested_questions)
            }
        }
        
        if include_debug:
            response["debug"] = {
                "top_similarity": round(max_similarity, 3),
                "threshold": threshold,
                "grounding_failed": True
            }
        
        return response
    
    def _keyword_fallback_chat(
        self,
        question: str,
        evidence_chunks: List[str],
        session_summary: Optional[Dict] = None
    ) -> Dict:
        """Fallback to keyword-based retrieval when embeddings unavailable."""
        config = self.thresholds.embeddings
        
        # Simple keyword matching
        query_lower = question.lower()
        scored = []
        
        for i, chunk in enumerate(evidence_chunks):
            chunk_lower = chunk.lower()
            score = 0.0
            
            for word in query_lower.split():
                if len(word) > 2 and word in chunk_lower:
                    score += 0.15
            
            if "PRIORITY" in chunk or "MISTAKE" in chunk:
                score += 0.1
            
            scored.append((min(1.0, score), chunk))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        max_sim = scored[0][0] if scored else 0.0
        
        if max_sim < config.similarity_threshold:
            return self._build_ungrounded_response(
                question, max_sim, config.similarity_threshold, False, "Similarity below threshold"
            )
        
        # Use top chunks
        top_chunks = [c for s, c in scored[:5] if s > 0]
        if len(top_chunks) < self.thresholds.chat_grounding.min_evidence_chunks:
            return self._build_ungrounded_response(
                question,
                max_sim,
                config.similarity_threshold,
                False,
                "Too few relevant chunks"
            )
        
        if not top_chunks:
            return self._build_ungrounded_response(
                question, 0.0, config.similarity_threshold, False, "No relevant evidence chunks"
            )
        
        # Build a deterministic answer from top chunks
        results = []
        for i, (score, chunk) in enumerate(scored[:5]):
            if score <= 0:
                continue
            meta = self.embeddings_manager._parse_chunk_meta(chunk, i)
            results.append(RetrievalResult(chunk=chunk, chunk_id=i, similarity=score, meta=meta))

        confidence_notes = self._find_confidence_notes(evidence_chunks)
        fallback = self._build_evidence_answer(results, confidence_notes, evidence_chunks)
        if not fallback:
            return self._build_ungrounded_response(
                question, max_sim, config.similarity_threshold, False, "No drill recommendation in evidence"
            )

        return {
            "answer": fallback["answer"],
            "grounded": True,
            "citations": self._map_citations_to_types(fallback["citations"], results),
            "evidence_used": top_chunks
        }


class StubChat:
    """Stub chat for testing - shows how prompts would work."""
    
    def __init__(self, data_dir: str = "./data/sessions"):
        self.embeddings_manager = EmbeddingsManager(data_dir)
        self.thresholds = get_thresholds()
    
    def chat(
        self,
        question: str,
        evidence_chunks: List[str],
        session_id: str = "test",
        session_summary: Optional[Dict] = None,
        include_debug: bool = True
    ) -> Dict:
        """Stub chat that returns what the prompt would look like."""
        # Try to use embeddings
        cache = self.embeddings_manager.compute_embeddings(session_id, evidence_chunks)
        
        if cache:
            results, max_sim = self.embeddings_manager.semantic_search(question, cache)
            is_grounded = self.embeddings_manager.is_grounded(max_sim)
        else:
            # Fallback
            results = []
            max_sim = 0.5
            is_grounded = True
        
        if not is_grounded:
            return {
                "answer": "Not enough evidence to answer.",
                "grounded": False,
                "citations": [],
                "missing_evidence": {
                    "reason": "Low similarity",
                    "max_similarity": max_sim
                }
            }
        
        # Build prompt
        if results:
            prompt = build_chat_prompt(question, results, session_summary)
            timestamps = []
            for r in results:
                timestamps.extend(r.meta.timestamps)
            ts_str = ", ".join(f"[{t}s]" for t in timestamps[:3])
            answer = f"Based on the evidence at {ts_str}: {results[0].chunk[:200]}..."
        else:
            # Keyword fallback
            timestamps = []
            for chunk in evidence_chunks[:3]:
                matches = re.findall(r'(\d+\.?\d*)s', chunk)
                timestamps.extend(matches[:2])
            ts_str = ", ".join(f"[{t}s]" for t in timestamps[:3])
            answer = f"Based on the evidence at {ts_str}: {evidence_chunks[0][:200]}..."
            prompt = "N/A"
        
        return {
            "answer": answer,
            "citations": [{"timestamp": float(t), "type": "evidence"} for t in timestamps[:3]],
            "grounded": True,
            "evidence_used": [r.chunk for r in results] if results else evidence_chunks[:3],
            "debug": {
                "top_similarity": round(max_sim, 3),
                "retrieval_method": "embeddings" if results else "keyword"
            },
            "debug_prompt": prompt
        }
