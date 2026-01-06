"""
Grounded Chat Module
Implements RAG-based chat that answers ONLY from evidence.
Supports both direct Google Gemini API and Anthropic-compatible proxies.
"""

import os
import re
from typing import Dict, List, Optional
import logging

# Import settings
from config.settings import get_settings

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


STRICT_SYSTEM_PROMPT = """You are a badminton footwork coach assistant. You analyze video session data.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. You may ONLY use the provided evidence to answer questions.
2. Every claim you make MUST cite at least one timestamp in [brackets] like [12.3s].
3. If evidence is insufficient, say: "I can't confirm this from the video. Consider refilming with full-body view."
4. Do NOT make up information or speculate beyond the evidence.
5. Keep responses concise and actionable.
6. Focus on the most important issue first (one correction at a time principle).

When asked about technique, always ground your answer in specific timestamps from the evidence."""


def retrieve_relevant_chunks(
    query: str,
    chunks: List[str],
    top_k: int = 5
) -> List[str]:
    """
    Simple keyword-based retrieval.
    Retrieves chunks most relevant to the query.
    """
    query_lower = query.lower()
    
    # Keywords to look for
    keywords = {
        "split": ["split", "hop", "jump", "step"],
        "lunge": ["lunge", "reach", "stretch"],
        "knee": ["knee", "valgus", "collapse", "inward"],
        "stance": ["stance", "width", "narrow", "wide", "feet"],
        "recovery": ["recovery", "base", "center", "back"],
        "mistake": ["mistake", "error", "wrong", "problem", "issue"],
        "timing": ["timing", "late", "early", "slow", "fast"],
        "summary": ["overall", "summary", "total", "session"]
    }
    
    # Score each chunk
    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        
        # Direct query word matches
        for word in query_lower.split():
            if word in chunk_lower:
                score += 2
        
        # Keyword category matches
        for category, kws in keywords.items():
            if any(kw in query_lower for kw in kws):
                if any(kw in chunk_lower for kw in kws):
                    score += 3
        
        # Boost mistake/priority chunks for general questions
        if "MISTAKE" in chunk or "PRIORITY" in chunk:
            score += 1
        
        scored.append((score, chunk))
    
    # Sort by score and return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def format_evidence_for_prompt(chunks: List[str]) -> str:
    """Format evidence chunks for the LLM prompt."""
    if not chunks:
        return "No evidence available for this session."
    
    lines = ["EVIDENCE FROM VIDEO SESSION:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"{i}. {chunk}")
    return "\n".join(lines)


def build_chat_prompt(
    question: str,
    evidence_chunks: List[str],
    session_summary: Optional[Dict] = None
) -> str:
    """Build the full prompt for the LLM."""
    evidence = format_evidence_for_prompt(evidence_chunks)
    
    prompt = f"""{STRICT_SYSTEM_PROMPT}

{evidence}

USER QUESTION: {question}

Remember: Only use the evidence above. Cite timestamps. If unsure, say so."""
    
    return prompt


class GroundedChat:
    """Chat interface that grounds all responses in evidence."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GOOGLE_API_KEY or "dummy"
        self.provider = self.settings.LLM_PROVIDER
        self.proxy_url = self.settings.LLM_PROXY_URL
        
        self.genai_model = None
        self.anthropic_client = None
        
        # Initialize provider
        if self.provider == "anthropic" or "proxy" in self.provider:
            if ANTHROPIC_AVAILABLE:
                try:
                    # Point to proxy
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
    
    def chat(
        self,
        question: str,
        evidence_chunks: List[str],
        session_summary: Optional[Dict] = None
    ) -> Dict:
        """
        Answer a question grounded in evidence.
        
        Returns:
            Dict with 'answer', 'citations', 'grounded' fields
        """
        # Retrieve relevant chunks
        relevant = retrieve_relevant_chunks(question, evidence_chunks, top_k=5)
        
        if not relevant:
            return {
                "answer": "I don't have enough evidence from this session to answer that question.",
                "citations": [],
                "grounded": False,
                "evidence_used": []
            }
        
        prompt = build_chat_prompt(question, relevant, session_summary)
        answer = self._fallback_response(question, relevant)
        
        # Try LLM
        try:
            if self.anthropic_client:
                # Use proxy
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-5", # Model name usually ignored by proxy or maps to Gemini
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.content[0].text
                
            elif self.genai_model:
                # Use Gemini Direct
                response = self.genai_model.generate_content(prompt)
                answer = response.text
                
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
            # Fallback already set
        
        # Extract citations
        citations = self._extract_citations(answer)
        
        return {
            "answer": answer,
            "citations": citations,
            "grounded": len(citations) > 0,
            "evidence_used": relevant
        }
    
    def _fallback_response(self, question: str, chunks: List[str]) -> str:
        """Fallback when LLM is unavailable - structured response from evidence."""
        q_lower = question.lower()
        
        # Find most relevant chunk
        for chunk in chunks:
            if "PRIORITY" in chunk:
                return f"Based on the analysis: {chunk}"
            if "MISTAKE" in chunk and ("issue" in q_lower or "problem" in q_lower or "wrong" in q_lower):
                return f"Issue detected: {chunk}"
        
        # Default to first chunk
        if chunks:
            return f"From the session data: {chunks[0]}"
        
        return "Unable to provide analysis. Please check if the video was processed correctly."
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract timestamp citations from response."""
        pattern = r'\[(\d+\.?\d*)s?\]'
        matches = re.findall(pattern, text)
        return [f"{m}s" for m in matches]


# Stub for testing without API
class StubChat:
    """Stub chat for testing - shows how prompts would work."""
    
    def chat(self, question: str, evidence_chunks: List[str], 
             session_summary: Optional[Dict] = None) -> Dict:
        relevant = retrieve_relevant_chunks(question, evidence_chunks, top_k=5)
        prompt = build_chat_prompt(question, relevant, session_summary)
        
        # Generate stub response based on evidence
        if not relevant:
            answer = "No evidence available for this question."
        else:
            # Extract timestamps from evidence
            timestamps = []
            for chunk in relevant:
                matches = re.findall(r'(\d+\.?\d*)s', chunk)
                timestamps.extend(matches[:2])
            
            ts_str = ", ".join(f"[{t}s]" for t in timestamps[:3]) if timestamps else "[N/A]"
            answer = f"Based on the evidence at {ts_str}: {relevant[0][:200]}..."
        
        return {
            "answer": answer,
            "citations": re.findall(r'\[(\d+\.?\d*s?)\]', answer),
            "grounded": True,
            "evidence_used": relevant,
            "debug_prompt": prompt
        }
