"""
Pillar III: deterministic answer-buffer state machine.

Takes the Pillar I output (ASRTranscriptionResult) and the Pillar II output
(IntentSlotResult), updates the active exam answer, and returns a SystemResponse
for the auditory loop / UI.

Design notes
------------
* The buffer is a list of sentences, each a list of Word(text, conf). Confidence
  is kept per word so that readbacks can re-flag words that are still uncertain.
* Nothing here is "smart": every operation is a plain string manipulation, so
  CESR (command execution success rate) can be tested exactly.
* Destructive actions are never silent. Edits that Pillar II is unsure about are
  held until the student confirms by voice, and every mutation can be undone.
"""
from __future__ import annotations

import copy
import re
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.schemas import ASRTranscriptionResult, IntentSlotResult, SystemResponse

SENTENCE_END = (".", "?", "!")
_PUNCT = string.punctuation + " "

_YES = {"yes", "yeah", "confirm", "do it"}
_NO = {"no", "cancel", "nope"}


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
def _key(text: str) -> str:
    """Lower-cased token with surrounding punctuation removed (matching key)."""
    return text.strip(_PUNCT).lower()


def _normalize(text: str) -> str:
    return " ".join(k for k in (_key(t) for t in text.split()) if k)


def _split_trailing(text: str) -> Tuple[str, str]:
    """'energy.' -> ('energy', '.'); 'n)' -> ('n', ')')."""
    m = re.search(r"[^\w]*$", text)
    return text[: m.start()], m.group()


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass
class Word:
    text: str  # as shown to the user, including trailing punctuation
    conf: float = 1.0

    @property
    def key(self) -> str:
        return _key(self.text)

    @property
    def ends_sentence(self) -> bool:
        return self.text.endswith(SENTENCE_END)

    @property
    def bare(self) -> str:
        return self.text.strip(_PUNCT)


def _say(words: List[Word]) -> str:
    return " ".join(w.text for w in words).strip(_PUNCT)


def _align(tokens: List[str], confs, fallback: float) -> List[Word]:
    """
    Attach ASR confidences to the raw transcript tokens.

    ASREngine strips punctuation, so word_confidences can be shorter than
    raw_transcript.split(). We match by key, looking a few positions ahead so a
    single dropped/merged token does not shift everything after it. Tokens that
    cannot be matched get `fallback` (the utterance confidence).
    """
    words: List[Word] = []
    j = 0
    for tok in tokens:
        k = _key(tok)
        if not k:  # stray punctuation token: glue it to the previous word
            if words:
                words[-1].text += tok
            continue
        conf = fallback
        for m in range(j, min(j + 3, len(confs))):
            if _key(confs[m].word) == k:
                conf = confs[m].confidence
                j = m + 1
                break
        words.append(Word(tok, conf))
    return words


def _to_sentences(words: List[Word]) -> List[List[Word]]:
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if w.ends_sentence:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
class AnswerStateMachine:
    def __init__(
        self,
        tau: float = 0.93,
        normal_wpm: int = 200,
        slow_wpm: int = 140,
        min_intent_conf: float = 0.60,
        max_history: int = 50,
    ):
        self.tau = tau
        self.normal_wpm = normal_wpm
        self.slow_wpm = slow_wpm
        self.min_intent_conf = min_intent_conf
        self.max_history = max_history

        self.sentences: List[List[Word]] = []
        self._history: List[List[List[Word]]] = []
        self._pending: Optional[Tuple[ASRTranscriptionResult, IntentSlotResult]] = None

    # ---- public API ------------------------------------------------------- #
    def render(self) -> str:
        return " ".join(w.text for s in self.sentences for w in s)

    def low_confidence_words(self) -> List[str]:
        return [w.bare for s in self.sentences for w in s if w.conf < self.tau]

    def process(
        self, asr: ASRTranscriptionResult, intent: IntentSlotResult
    ) -> SystemResponse:
        if self._pending is not None:
            reply = self._resolve_pending(asr)
            if reply is not None:
                return reply

        if self._is_clear(intent):
            if not self.sentences:
                return self._respond("Your answer is already empty.")
            self._pending = (asr, intent)
            return self._respond(
                "Clear your whole answer? Say yes to clear it, or no to keep it."
            )
        if self._needs_confirmation(intent):
            self._pending = (asr, intent)
            return self._respond(
                "I wasn't sure that was a command. "
                "Say yes to run it, or no to add it to your answer as text."
            )
        return self._dispatch(asr, intent)

    def undo(self) -> SystemResponse:
        if not self._history:
            return self._respond("Nothing to undo.")
        self.sentences = self._history.pop()
        return self._respond(f"Undone. {self._active_phrase(len(self.sentences) - 1)}")

    # ---- routing ---------------------------------------------------------- #
    @staticmethod
    def _is_clear(intent: IntentSlotResult) -> bool:
        return intent.intent == "EDIT_ACTION" and intent.action_type == "CLEAR_BUFFER"

    def _needs_confirmation(self, intent: IntentSlotResult) -> bool:
        return intent.intent == "EDIT_ACTION" and (
            intent.is_collision_risk or intent.confidence < self.min_intent_conf
        )

    def _resolve_pending(self, asr: ASRTranscriptionResult) -> Optional[SystemResponse]:
        heard = _normalize(asr.raw_transcript)
        p_asr, p_intent = self._pending
        if heard in _YES:
            self._pending = None
            return self._dispatch(p_asr, p_intent)
        if heard in _NO:
            self._pending = None
            if self._is_clear(p_intent):  # never turn "clear" into answer text
                return self._respond("Okay, your answer is unchanged.")
            return self._dictate(p_asr)
        self._pending = None  # anything else: drop it and treat as a new utterance
        return None

    def _dispatch(
        self, asr: ASRTranscriptionResult, intent: IntentSlotResult
    ) -> SystemResponse:
        kind = (intent.intent or "").upper()
        action = (intent.action_type or "").upper()

        if kind == "DICTATION":
            return self._dictate(asr)
        if kind == "NAVIGATION":
            return self._navigate(asr, intent)
        if kind == "EDIT_ACTION":
            if action in ("DELETE_SPAN", "DELETE_LAST_WORD", "DELETE_LAST_SENTENCE", "CLEAR_BUFFER"):
                return self._delete(asr, intent)
            if action == "SUBSTITUTE_SPAN":
                return self._substitute(asr, intent)
        return self._respond("Sorry, I didn't understand that. Your answer is unchanged.")

    # ---- operations ------------------------------------------------------- #
    def _dictate(
        self, asr: ASRTranscriptionResult, text: Optional[str] = None
    ) -> SystemResponse:
        raw = asr.raw_transcript if text is None else text
        tokens = raw.split()
        if not tokens:
            return self._respond("I didn't catch that.")

        words = _align(tokens, asr.word_confidences, asr.utterance_confidence)
        self._snapshot()
        self.sentences.extend(_to_sentences(words))
        flagged = [w.bare for w in words if w.conf < self.tau]
        return self._respond(_say(words), flagged)

    def _delete(
        self, asr: ASRTranscriptionResult, intent: IntentSlotResult
    ) -> SystemResponse:
        if not self.sentences:
            return self._respond("Your answer is empty. Nothing to delete.")

        action = (intent.action_type or "").upper()
        if action == "CLEAR_BUFFER":  # reached only after the student said yes
            self._snapshot()
            self.sentences = []
            return self._respond("Cleared. Your answer is now empty.")

        si = len(self.sentences) - 1
        label = None
        if action == "DELETE_LAST_SENTENCE":
            wi, n = 0, len(self.sentences[si])
            label = "the last sentence"
        elif action == "DELETE_LAST_WORD":
            wi, n = len(self.sentences[si]) - 1, 1
        else:  # DELETE_SPAN
            if not intent.target_text:
                return self._respond("I didn't hear which words to delete. Nothing deleted.")
            loc = self._find(intent.target_text)
            if loc is None:
                return self._respond(
                    f"I couldn't find {intent.target_text} in your answer. Nothing deleted."
                )
            si, wi, n = loc

        self._snapshot()
        removed, active = self._remove_span(si, wi, n)
        if label:
            said = f"Deleted {label}, starting with {' '.join(_say(removed).split()[:3])}."
        else:
            said = f"Deleted {_say(removed)}."
        return self._respond(f"{said} {self._active_phrase(active)}")

    def _substitute(
        self, asr: ASRTranscriptionResult, intent: IntentSlotResult
    ) -> SystemResponse:
        if not intent.replacement_text:
            return self._respond("I didn't hear what to replace it with. Nothing changed.")

        if intent.target_text:
            loc = self._find(intent.target_text)
            if loc is None:
                return self._respond(
                    f"I couldn't find {intent.target_text} in your answer. Nothing changed."
                )
        else:
            # Target-only repair: "I meant force" fixes the latest uncertain word.
            loc = self._last_uncertain()
            if loc is None:
                return self._respond(
                    "There is no uncertain word to replace. Say which word to change."
                )
        si, wi, n = loc

        conf_lookup = {_key(c.word): c.confidence for c in asr.word_confidences}
        sent = self.sentences[si]
        old = sent[wi : wi + n]
        _, trail = _split_trailing(old[-1].text)

        new = [Word(t, conf_lookup.get(_key(t), 1.0)) for t in intent.replacement_text.split()]
        new[-1].text += trail  # keep the full stop / comma of what we replaced
        if wi == 0 and old[0].text[:1].isupper():
            new[0].text = _cap(new[0].text)

        self._snapshot()
        sent[wi : wi + n] = new
        flagged = [w.bare for w in new if w.conf < self.tau]
        return self._respond(
            f"Replaced {_say(old)} with {_say(new)}. {self._active_phrase(si)}", flagged
        )

    def _navigate(
        self, asr: ASRTranscriptionResult, intent: IntentSlotResult
    ) -> SystemResponse:
        heard = _normalize(asr.raw_transcript)
        if "question" in heard:
            return self._respond("Moving between questions is not supported yet.")
        if not self.sentences:
            return self._respond("Your answer is empty.")

        action = intent.action_type
        if action == "READBACK_LAST_SENTENCE":
            span = self.sentences[-1]
        elif action == "READBACK_ALL":
            span = [w for s in self.sentences for w in s]
        elif re.search(r"\b(last|previous) sentence\b", heard):  # no action_type given
            span = self.sentences[-1]
        else:
            span = [w for s in self.sentences for w in s]
        flagged = [w.bare for w in span if w.conf < self.tau]
        return self._respond(_say(span), flagged)

    # ---- helpers ---------------------------------------------------------- #
    def _respond(self, feedback: str, flagged: Optional[List[str]] = None) -> SystemResponse:
        flagged = flagged or []
        return SystemResponse(
            buffer_state=self.render(),
            spoken_feedback=feedback,
            tts_rate=self.slow_wpm if flagged else self.normal_wpm,
            flagged_low_confidence_tokens=flagged,
        )

    def _snapshot(self) -> None:
        self._history.append(copy.deepcopy(self.sentences))
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def _active_phrase(self, si: Optional[int]) -> str:
        if si is None or not self.sentences:
            return "Your answer is now empty."
        si = max(0, min(si, len(self.sentences) - 1))
        return f"Active sentence: {_say(self.sentences[si])}."

    def _find(self, target: str) -> Optional[Tuple[int, int, int]]:
        """Most recent occurrence of a (multi-word) phrase: (sentence, word, length)."""
        keys = [k for k in (_key(t) for t in target.split()) if k]
        n = len(keys)
        if n == 0:
            return None
        for si in range(len(self.sentences) - 1, -1, -1):
            sent_keys = [w.key for w in self.sentences[si]]
            for wi in range(len(sent_keys) - n, -1, -1):
                if sent_keys[wi : wi + n] == keys:
                    return si, wi, n
        return None

    def _last_uncertain(self) -> Optional[Tuple[int, int, int]]:
        for si in range(len(self.sentences) - 1, -1, -1):
            for wi in range(len(self.sentences[si]) - 1, -1, -1):
                if self.sentences[si][wi].conf < self.tau:
                    return si, wi, 1
        return None

    def _remove_span(self, si: int, wi: int, n: int) -> Tuple[List[Word], Optional[int]]:
        """Remove words, repairing the full stop and capital letter. Returns
        (removed words, index of the sentence to announce as active)."""
        sent = self.sentences[si]
        removed = sent[wi : wi + n]
        terminal = ""
        if wi + n == len(sent):
            tail = _split_trailing(sent[-1].text)[1]
            if tail.endswith(SENTENCE_END):
                terminal = tail
        was_capitalised = wi == 0 and removed[0].text[:1].isupper()

        del sent[wi : wi + n]
        if not sent:
            del self.sentences[si]
            active = si - 1 if si > 0 else (0 if self.sentences else None)
            return removed, active

        if terminal:
            sent[-1].text = _split_trailing(sent[-1].text)[0] + terminal
        if was_capitalised:
            sent[0].text = _cap(sent[0].text)
        return removed, si
