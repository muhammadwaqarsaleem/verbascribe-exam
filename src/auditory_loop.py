"""
Pillar III: auditory feedback loop (dynamic TTS rate control).

Turns a SystemResponse from the state machine into speech. The idea from
Nowrin & Vertanen (2025) is to read at normal speed (200 WPM) and slow down
(140 WPM) only where the ASR was unsure, instead of slowing everything.

* The state machine decides *what* to say and *which words* were uncertain.
* This module decides *how* to say it: it splits the feedback into segments,
  each with its own rate, and hands them to a TTS backend.
* The backend is swappable, so everything except the actual audio output can be
  tested without a speaker (see tests/test_auditory_loop.py).

Rate check on your own machine (day-1 task):   python -m src.auditory_loop
"""
from __future__ import annotations

import shutil
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from src.schemas import SystemResponse

_PUNCT = string.punctuation + " "


def _key(token: str) -> str:
    return token.strip(_PUNCT).lower()


@dataclass(frozen=True)
class Segment:
    text: str
    rate: int  # words per minute

    @property
    def n_words(self) -> int:
        return len(self.text.split())


class TTSBackend(Protocol):
    def speak(self, segments: Sequence[Segment]) -> None: ...


class Pyttsx3Backend:
    """Offline TTS via pyttsx3 (Windows/macOS, or Linux if its audio path works).
    Imported lazily so the rest of the project works without it."""

    def __init__(self):
        import pyttsx3

        self._engine = pyttsx3.init()

    def speak(self, segments: Sequence[Segment]) -> None:
        # pyttsx3 queues setProperty and say in order, so the rate changes
        # between segments take effect at the right place.
        for seg in segments:
            self._engine.setProperty("rate", seg.rate)
            self._engine.say(seg.text)
        self._engine.runAndWait()


class EspeakNgBackend:
    """Calls the espeak-ng binary directly (Linux).

    Avoids pyttsx3 and its audio plumbing, so it works wherever
    `espeak-ng "hello"` already works (PipeWire, PulseAudio, ...). espeak-ng's
    -s option is approximate words per minute, which matches how we use rates.
    Segments are spoken one after another, so expect a very short gap between
    them.
    """

    def __init__(self, voice: str = "en", binary: Optional[str] = None):
        self._bin = binary or shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._bin:
            raise RuntimeError("espeak-ng not found on PATH")
        self.voice = voice

    def speak(self, segments: Sequence[Segment]) -> None:
        for seg in segments:
            subprocess.run(
                [self._bin, "-v", self.voice, "-s", str(seg.rate), "--", seg.text],
                check=True,
            )


def default_backend() -> TTSBackend:
    """espeak-ng on Linux when it is installed, otherwise pyttsx3."""
    if sys.platform.startswith("linux") and (shutil.which("espeak-ng") or shutil.which("espeak")):
        return EspeakNgBackend()
    return Pyttsx3Backend()


class AuditoryLoop:
    def __init__(
        self,
        backend: Optional[TTSBackend] = None,
        normal_wpm: int = 200,
        slow_wpm: int = 140,
    ):
        self._backend = backend
        self.normal_wpm = normal_wpm
        self.slow_wpm = slow_wpm

    # ---- planning (pure, no audio) --------------------------------------- #
    def plan(self, response: SystemResponse) -> List[Segment]:
        """Split the spoken feedback into segments, slowing only flagged words."""
        text = response.spoken_feedback.strip()
        if not text:
            return []

        flagged = {_key(t) for t in response.flagged_low_confidence_tokens if _key(t)}
        tokens = text.split()
        slow = [_key(t) in flagged for t in tokens]

        # Nothing flagged, or flagged words are not in the sentence we are about
        # to read (so we cannot place the slowdown): fall back to the single rate
        # the state machine chose for the whole response.
        if not any(slow):
            return [Segment(text, response.tts_rate)]

        segments: List[Segment] = []
        cur_words: List[str] = []
        cur_slow = slow[0]
        for tok, is_slow in zip(tokens, slow):
            if is_slow != cur_slow:
                segments.append(self._segment(cur_words, cur_slow))
                cur_words, cur_slow = [], is_slow
            cur_words.append(tok)
        segments.append(self._segment(cur_words, cur_slow))
        return segments

    def _segment(self, words: List[str], slow: bool) -> Segment:
        return Segment(" ".join(words), self.slow_wpm if slow else self.normal_wpm)

    # ---- speaking -------------------------------------------------------- #
    def speak(self, response: SystemResponse) -> List[Segment]:
        """Speak the response and return the segments used (handy for logging)."""
        segments = self.plan(response)
        if segments:
            self._get_backend().speak(segments)
        return segments

    def _get_backend(self) -> TTSBackend:
        if self._backend is None:
            self._backend = default_backend()
        return self._backend

    # ---- timing (for the report: the "+7% listening time" comparison) ----- #
    @staticmethod
    def estimated_seconds(segments: Sequence[Segment]) -> float:
        return sum(s.n_words / s.rate * 60 for s in segments)

    def overhead_vs_normal(self, segments: Sequence[Segment]) -> float:
        """Extra listening time relative to reading everything at normal speed."""
        words = sum(s.n_words for s in segments)
        if words == 0:
            return 0.0
        baseline = words / self.normal_wpm * 60
        return self.estimated_seconds(segments) / baseline - 1


# --------------------------------------------------------------------------- #
# Rate check: does 200 vs 140 actually sound 200 vs 140 on this machine?
# --------------------------------------------------------------------------- #
SAMPLE = (
    "Kinetic energy equals half mass times velocity squared. "
    "The second law of thermodynamics states that entropy never decreases "
    "in a closed system."
)


def measure_rates(backend: Optional[TTSBackend] = None, rates=(200, 140), text: str = SAMPLE):
    """Speak the same text at each rate and report the effective words per minute.

    pyttsx3's 'rate' is only nominally WPM and differs between engines (SAPI5 on
    Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux), so measure it before
    quoting 200/140 WPM in the report. Timing includes engine start-up, so treat
    the numbers as approximate.
    """
    backend = backend or default_backend()
    n_words = len(text.split())
    results = []
    for rate in rates:
        t0 = time.perf_counter()
        backend.speak([Segment(text, rate)])
        dt = time.perf_counter() - t0
        results.append((rate, dt, n_words / dt * 60))
    return results


if __name__ == "__main__":
    print("Speaking a test passage at each rate...")
    silent = False
    for rate, dt, wpm in measure_rates():
        print(f"requested {rate:>3} WPM -> took {dt:5.1f}s -> effective {wpm:5.0f} WPM")
        silent = silent or wpm > 400  # nobody speaks this fast: audio never played
    if silent:
        print(
            "\nWARNING: speech finished almost instantly, so nothing was actually played "
            "and these numbers are meaningless. Check that espeak-ng works on its own "
            "(espeak-ng 'hello') and that `aplay -l` lists a sound device."
        )
