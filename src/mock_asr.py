from src.schemas import ASRTranscriptionResult, WordConfidence


class MockASR:
    """Use this until the real model is ready. Same output contract as ASREngine."""

    def transcribe(self, audio=None, sr=16000):
        words = [("kinetic", 0.98), ("energy", 0.97), ("equals", 0.62), ("half", 0.95)]
        return ASRTranscriptionResult(
            raw_transcript=" ".join(w for w, _ in words),
            word_confidences=[WordConfidence(word=w, confidence=c) for w, c in words],
            utterance_confidence=0.88,
            latency_ms=120.0,
        )