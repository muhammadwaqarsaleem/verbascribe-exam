"""
Data contracts connecting Pillar I (ASR), Pillar II (SLU), and Pillar III (State & Audio).
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class WordConfidence(BaseModel):
    word: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ASRTranscriptionResult(BaseModel):
    """Output contract from Pillar I (Ali) to Pillar II & III"""
    raw_transcript: str
    word_confidences: List[WordConfidence]
    utterance_confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float


class IntentSlotResult(BaseModel):
    """Output contract from Pillar II (Waqar) to Pillar III"""
    intent: str  # "DICTATION", "EDIT_ACTION", "NAVIGATION"
    action_type: Optional[str] = None  # "INSERT", "DELETE", "SUBSTITUTE", "READBACK"
    target_text: Optional[str] = None
    replacement_text: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    is_collision_risk: bool = False


class SystemResponse(BaseModel):
    """Output contract from Pillar III (Saad) to User Interface"""
    buffer_state: str
    spoken_feedback: str
    tts_rate: int  # 200 for normal, 140 for uncertain readback
    flagged_low_confidence_tokens: List[str]