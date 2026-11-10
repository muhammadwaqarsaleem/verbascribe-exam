"""
Data contracts connecting Pillar I (ASR), Pillar II (SLU), and Pillar III (State & Audio).
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# Unambiguous action primitives for hands-free state control
ActionType = Literal[
    "DELETE_LAST_WORD",
    "DELETE_LAST_SENTENCE",
    "DELETE_SPAN",
    "SUBSTITUTE_SPAN",
    "CLEAR_BUFFER",
    "READBACK_ALL",
    "READBACK_LAST_SENTENCE"
]

IntentType = Literal["DICTATION", "EDIT_ACTION", "NAVIGATION"]


class WordConfidence(BaseModel):
    """Word-level acoustic confidence extracted from Distil-Whisper decoder."""
    word: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ASRTranscriptionResult(BaseModel):
    """Output contract from Pillar I (Ali) to Pillar II & III."""
    raw_transcript: str
    word_confidences: List[WordConfidence]
    utterance_confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float


class IntentSlotResult(BaseModel):
    """Output contract from Pillar II (Waqar) to Pillar III."""
    intent: IntentType
    action_type: Optional[ActionType] = None
    target_text: Optional[str] = None
    replacement_text: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    is_collision_risk: bool = False  # True when dictation contains command words like 'delete'


class SystemResponse(BaseModel):
    """Output contract from Pillar III (Saad) to Gradio/UI layer."""
    buffer_state: str
    spoken_feedback: str
    tts_rate: int = Field(default=200, description="Playback speed in WPM (200=normal, 140=slowed)")
    flagged_low_confidence_tokens: List[str] = Field(default_factory=list)