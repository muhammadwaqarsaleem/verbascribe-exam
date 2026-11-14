"""
VerbaScribe-Exam: live demo interface (Pillar III, Saad).

Pipeline
--------
    speech / typed text -> ASR -> IntentParser -> AnswerStateMachine -> AuditoryLoop (TTS)
          (Pillar I)     (Pillar II)             (Pillar III)          (Pillar III)

Run from the repository root:

    python app.py                      # auto: real IntentParser if weights exist
    python app.py --parser rules       # force the built-in rule-based stand-in
    python app.py --parser real        # fail loudly if the weights cannot be loaded
    python app.py --adapter models/whisper_lora   # Ali's LoRA adapter for the mic demo
    python app.py --no-tts             # never speak aloud (silent demo / no speakers)

Speech sources (pick one in the UI)
-----------------------------------
* Mock ASR           : MockASR's fixed sample. Proves the wiring end to end.
* Typed text         : you type what the student "says". Words listed as uncertain get a low
                       confidence, so you can demo the 140 WPM slowdown without a microphone.
* Microphone         : live audio through ASREngine (Distil-Whisper, optional LoRA adapter).
                       The model is loaded on first use, so the app starts instantly.

The IntentParser needs Waqar's weights in models/intent_slot_model/ (pytorch_model.bin). Until
they are there, `--parser auto` falls back to RuleBasedParser below so the UI still works.
"""
from __future__ import annotations

import argparse
import html
import re
import string
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditory_loop import AuditoryLoop, Segment  # noqa: E402
from src.mock_asr import MockASR  # noqa: E402
from src.schemas import (  # noqa: E402
    ASRTranscriptionResult,
    IntentSlotResult,
    SystemResponse,
    WordConfidence,
)
from src.state_machine import AnswerStateMachine  # noqa: E402

DEFAULT_MODEL_DIR = ROOT / "models" / "intent_slot_model"
TAU_DEFAULT = 0.93  # confidence threshold from the Pillar III execution contract
NORMAL_WPM, SLOW_WPM = 200, 140
SLU_BUDGET_MS = 50.0  # latency target for IntentParser.parse() (final report, Section 4)

SRC_MOCK = "Mock ASR (fixed sample)"
SRC_TYPED = "Typed text (simulated speech)"
SRC_MIC = "Microphone (Distil-Whisper)"

_PUNCT = string.punctuation + " "


def _key(token: str) -> str:
    return token.strip(_PUNCT).lower()


# --------------------------------------------------------------------------- #
# Pillar II stand-in (only used while the real weights are unavailable)
# --------------------------------------------------------------------------- #
class RuleBasedParser:
    """Regex stand-in with the same `parse(text) -> IntentSlotResult` interface as IntentParser.

    NOT the project's SLU model: it only exists so the UI and state machine can be demonstrated
    before the weights are downloaded. Commands must *start* with the command word, which is
    how it stays correct on collision sentences such as "the delete operation in red-black trees".
    """

    name = "rule-based stand-in"

    _CLEAR = re.compile(r"^(?:clear|wipe|erase|delete)\s+(?:the\s+|my\s+)?(?:whole\s+|entire\s+)?"
                        r"(?:buffer|answer|everything|all)\b|^start over\b")
    _SUBST = re.compile(r"^(?:replace|change)\s+(?P<t>.+?)\s+(?:with|to)\s+(?P<r>.+)$")
    _MEANT = re.compile(r"^i meant\s+(?P<r>.+)$")
    _DEL_SENT = re.compile(r"^(?:delete|erase|remove|scratch)\b.*\b(?:last|previous)\s+sentence\b")
    _DEL_WORD = re.compile(r"^(?:delete|erase|remove|scratch)\b.*\b(?:last|previous)\s+word\b")
    _DEL_SPAN = re.compile(r"^(?:delete|erase|remove)\s+(?:the\s+(?:phrase|words?)\s+)?(?P<t>.+)$")
    _NAV = re.compile(r"^(?:read|what is|what's|go to|repeat)\b")
    _TRIGGERS = ("delete", "replace", "clear", "scratch")

    def parse(self, text: str) -> IntentSlotResult:
        t = " ".join(_key(w) for w in text.split() if _key(w))

        def edit(action, target=None, replacement=None):
            return IntentSlotResult(
                intent="EDIT_ACTION", action_type=action, target_text=target,
                replacement_text=replacement, confidence=0.95,
            )

        if self._CLEAR.search(t):
            return edit("CLEAR_BUFFER")
        if m := self._SUBST.match(t):
            return edit("SUBSTITUTE_SPAN", m["t"], m["r"])
        if m := self._MEANT.match(t):
            return edit("SUBSTITUTE_SPAN", None, m["r"])
        if self._DEL_SENT.match(t):
            return edit("DELETE_LAST_SENTENCE")
        if self._DEL_WORD.match(t):
            return edit("DELETE_LAST_WORD")
        if m := self._DEL_SPAN.match(t):
            return edit("DELETE_SPAN", m["t"])
        if self._NAV.match(t):
            whole = any(k in t for k in ("paragraph", "buffer", "beginning", "everything", "all"))
            return IntentSlotResult(
                intent="NAVIGATION",
                action_type="READBACK_ALL" if whole else "READBACK_LAST_SENTENCE",
                confidence=0.95,
            )
        return IntentSlotResult(
            intent="DICTATION",
            confidence=0.90,
            is_collision_risk=any(k in t for k in self._TRIGGERS),
        )


def load_parser(mode: str, model_dir: Path) -> Tuple[Any, str, Optional[str]]:
    """Return (parser, label, warning). `mode` is auto | real | rules."""
    if mode == "rules":
        return RuleBasedParser(), RuleBasedParser.name, None
    try:
        weights = Path(model_dir) / "pytorch_model.bin"
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found. Extract Waqar's weights into models/intent_slot_model/."
            )
        from src.intent_parser import IntentParser  # imports torch + transformers

        return IntentParser(model_dir=str(model_dir)), "IntentParser (DistilBERT joint intent-slot)", None
    except Exception as exc:  # missing weights, no torch, no internet for the base model, ...
        if mode == "real":
            raise
        return RuleBasedParser(), RuleBasedParser.name, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Pillar I: ASR sources
# --------------------------------------------------------------------------- #
def simulated_asr(text: str, uncertain: set, high: float = 0.97, low: float = 0.62) -> ASRTranscriptionResult:
    """Build an ASRTranscriptionResult from typed text, mirroring ASREngine's output shape:
    raw_transcript keeps punctuation, word_confidences are punctuation-stripped."""
    t0 = time.perf_counter()
    confs = []
    for tok in text.split():
        bare = tok.strip(_PUNCT)
        if bare:
            confs.append(WordConfidence(word=bare, confidence=low if _key(tok) in uncertain else high))
    utt = float(np.exp(np.mean(np.log([c.confidence for c in confs])))) if confs else 0.0
    return ASRTranscriptionResult(
        raw_transcript=text.strip(),
        word_confidences=confs,
        utterance_confidence=min(utt, 1.0),
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def prepare_audio(audio: Tuple[int, np.ndarray]) -> Tuple[np.ndarray, int]:
    """Gradio mic output (sr, int16 array) -> mono float32 at 16 kHz for Whisper."""
    sr, data = audio
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):  # scale first: mean() would turn ints into floats
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data, 16000


# --------------------------------------------------------------------------- #
# Pipeline (UI independent, so it can be tested without Gradio)
# --------------------------------------------------------------------------- #
@dataclass
class TurnResult:
    source: str
    transcript: str
    response: SystemResponse
    plan: List[Segment]
    asr: Optional[ASRTranscriptionResult] = None
    intent: Optional[IntentSlotResult] = None
    asr_ms: float = 0.0
    slu_ms: float = 0.0
    state_ms: float = 0.0


@dataclass
class Session:
    """Per-browser-tab state: one exam answer and its turn log."""
    sm: AnswerStateMachine = field(default_factory=lambda: AnswerStateMachine(
        tau=TAU_DEFAULT, normal_wpm=NORMAL_WPM, slow_wpm=SLOW_WPM))
    log: List[List[Any]] = field(default_factory=list)


class Pipeline:
    """Models shared by every session: parser, (lazy) Whisper engine, TTS loop."""

    def __init__(self, parser, adapter_path: Optional[str] = None, backend=None):
        self.parser = parser
        self.adapter_path = adapter_path
        self.mock = MockASR()
        self.loop = AuditoryLoop(backend=backend, normal_wpm=NORMAL_WPM, slow_wpm=SLOW_WPM)
        self._engine = None
        self._engine_lock = threading.Lock()
        self._tts_lock = threading.Lock()

    # ---- ASR ---------------------------------------------------------------- #
    def _asr_engine(self):
        with self._engine_lock:
            if self._engine is None:
                try:
                    from src.asr_engine import ASREngine

                    self._engine = ASREngine(adapter_path=self.adapter_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load the ASR model ({type(exc).__name__}: {exc}). "
                        "It needs torch/transformers, an internet connection for distil-small.en "
                        "on first run, and a valid --adapter path if one is given."
                    ) from exc
            return self._engine

    def transcribe(self, source: str, typed: str = "", uncertain: str = "", audio=None) -> ASRTranscriptionResult:
        if source == SRC_MOCK:
            return self.mock.transcribe()
        if source == SRC_MIC:
            if audio is None:
                raise ValueError("Record something with the microphone first.")
            samples, sr = prepare_audio(audio)
            return self._asr_engine().transcribe(samples, sr)
        if not (typed or "").strip():
            raise ValueError("Type what the student says first.")
        words = {_key(w) for w in re.split(r"[,\s]+", uncertain or "") if _key(w)}
        return simulated_asr(typed, words)

    # ---- SLU + state machine ----------------------------------------------- #
    def run_turn(self, sm: AnswerStateMachine, asr: ASRTranscriptionResult, source: str) -> TurnResult:
        text = asr.raw_transcript.strip()
        t0 = time.perf_counter()
        if text:
            intent = self.parser.parse(text)
        else:  # silence: skip the model, the state machine answers "I didn't catch that."
            intent = IntentSlotResult(intent="DICTATION", confidence=0.0)
        t1 = time.perf_counter()
        response = sm.process(asr, intent)
        t2 = time.perf_counter()
        return TurnResult(
            source=source, transcript=text, response=response, plan=self.loop.plan(response),
            asr=asr, intent=intent, asr_ms=asr.latency_ms,
            slu_ms=(t1 - t0) * 1000, state_ms=(t2 - t1) * 1000,
        )

    def run_undo(self, sm: AnswerStateMachine) -> TurnResult:
        response = sm.undo()
        return TurnResult(source="Undo button", transcript="(undo)", response=response,
                          plan=self.loop.plan(response))

    # ---- TTS ----------------------------------------------------------------- #
    def speak(self, response: SystemResponse) -> Optional[str]:
        """Speak on the machine running the app. Returns an error message, or None on success."""
        with self._tts_lock:
            try:
                self.loop.speak(response)
                return None
            except Exception as exc:
                return (f"{type(exc).__name__}: {exc}. On Linux install espeak-ng; "
                        "on Windows/macOS check that pyttsx3 is installed.")


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def highlight(sm: AnswerStateMachine) -> str:
    """Answer buffer as HTML. Words below tau get a fixed amber highlight; hover shows the
    confidence. (gr.HighlightedText ignored our color_map and picked random colours.)"""
    spans = []
    for s in sm.sentences:
        for w in s:
            text = html.escape(w.text)
            if w.conf < sm.tau:
                spans.append(
                    f'<mark style="background:#f59e0b;color:#111;padding:1px 5px;border-radius:4px" '
                    f'title="confidence {w.conf:.2f}">{text}</mark>'
                )
            else:
                spans.append(text)
    if not spans:
        return '<div style="opacity:.6;padding:8px">(answer is empty)</div>'
    return f'<div style="line-height:2;font-size:1.05em;padding:8px">{" ".join(spans)}</div>'


def plan_rows(plan: List[Segment]) -> List[List[Any]]:
    return [[s.text, s.rate, "slowed" if s.rate == SLOW_WPM else "normal"] for s in plan]


def metrics_md(pipe: Pipeline, tr: TurnResult) -> str:
    if tr.asr is None:
        return ""
    overhead = pipe.loop.overhead_vs_normal(tr.plan) if tr.plan else 0.0
    ok = "✅" if tr.slu_ms <= SLU_BUDGET_MS else "⚠️"
    return (
        f"**ASR** {tr.asr_ms:.0f} ms &nbsp;·&nbsp; **SLU** {tr.slu_ms:.1f} ms {ok} "
        f"(target ≤ {SLU_BUDGET_MS:.0f} ms) &nbsp;·&nbsp; **State machine** {tr.state_ms:.2f} ms "
        f"&nbsp;·&nbsp; **Extra listening time vs. all-normal speed** {overhead:+.1%}"
    )


def status_md(sm: AnswerStateMachine, note: str = "") -> str:
    parts = []
    if getattr(sm, "_pending", None) is not None:
        parts.append("⚠️ **Waiting for confirmation.** Say or type *yes* to run it, *no* to cancel.")
    if note:
        parts.append(note)
    return "  \n".join(parts)


def trace_dict(tr: TurnResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": tr.source}
    if tr.asr is not None:
        out["asr"] = tr.asr.model_dump()
    if tr.intent is not None:
        out["intent_slot"] = tr.intent.model_dump()
    out["system_response"] = tr.response.model_dump()
    return out


def log_row(n: int, tr: TurnResult) -> List[Any]:
    i = tr.intent
    return [
        n, tr.source, tr.transcript,
        i.intent if i else "-", (i.action_type or "-") if i else "-",
        f"{i.confidence:.2f}" if i else "-",
        tr.response.spoken_feedback, tr.response.tts_rate,
    ]


LOG_HEADERS = ["#", "Source", "Transcript", "Intent", "Action", "Intent conf.", "Spoken feedback", "WPM"]


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
def build_ui(pipe: Pipeline, parser_label: str, parser_warning: Optional[str], tts_default: bool):
    import gradio as gr

    def pack(sess: Session, tr: TurnResult, note: str = "", clear_inputs: bool = True):
        return (
            sess,
            highlight(sess.sm),
            tr.response.spoken_feedback,
            plan_rows(tr.plan),
            metrics_md(pipe, tr),
            status_md(sess.sm, note),
            trace_dict(tr),
            sess.log,
            "" if clear_inputs else gr.update(),
            None if clear_inputs else gr.update(),
        )

    def empty_pack(sess: Session, note: str = ""):
        return (sess, highlight(sess.sm), "", [], "", status_md(sess.sm, note), {}, sess.log, "", None)

    def turn(sess, source, typed, uncertain, audio, tau, speak):
        sess = sess or Session()
        sess.sm.tau = float(tau)
        try:
            asr = pipe.transcribe(source, typed, uncertain, audio)
        except ValueError as exc:  # user mistake: keep what they typed, just tell them
            gr.Warning(str(exc))
            yield pack(sess, TurnResult("-", "", SystemResponse(
                buffer_state=sess.sm.render(), spoken_feedback="", tts_rate=NORMAL_WPM), []),
                note=f"ℹ️ {exc}", clear_inputs=False)
            return
        except RuntimeError as exc:
            raise gr.Error(str(exc))

        tr = pipe.run_turn(sess.sm, asr, source)
        sess.log.append(log_row(len(sess.log) + 1, tr))
        will_speak = bool(speak and tr.plan)
        yield pack(sess, tr, note="🔊 Speaking…" if will_speak else "")
        if will_speak:
            err = pipe.speak(tr.response)
            yield (sess, *[gr.update()] * 4,
                   status_md(sess.sm, f"❌ TTS failed: {err}" if err else "✅ Spoken."),
                   *[gr.update()] * 4)

    def quick(word):
        def handler(sess, tau, speak):
            yield from turn(sess, SRC_TYPED, word, "", None, tau, speak)
        return handler

    def undo(sess, tau, speak):
        sess = sess or Session()
        sess.sm.tau = float(tau)
        tr = pipe.run_undo(sess.sm)
        sess.log.append(log_row(len(sess.log) + 1, tr))
        will_speak = bool(speak and tr.plan)
        yield pack(sess, tr, note="🔊 Speaking…" if will_speak else "")
        if will_speak:
            err = pipe.speak(tr.response)
            yield (sess, *[gr.update()] * 4,
                   status_md(sess.sm, f"❌ TTS failed: {err}" if err else "✅ Spoken."),
                   *[gr.update()] * 4)

    def reset(_sess):
        return empty_pack(Session(), "Started a new answer.")

    def toggle_source(source):
        typed = source == SRC_TYPED
        return (gr.update(visible=typed), gr.update(visible=typed),
                gr.update(visible=source == SRC_MIC))

    with gr.Blocks(title="VerbaScribe-Exam") as demo:
        gr.Markdown(
            "# 🎙️ VerbaScribe-Exam\n"
            "Hands-free exam answering for students with motor impairments. Dictate your answer, "
            "edit it by voice, and have it read back. Words the speech recogniser was unsure "
            "about are read **more slowly** (140 WPM instead of 200 WPM) so you can catch errors."
        )
        banner = f"**Intent parser:** {parser_label}"
        if parser_warning:
            banner += (f"  \n⚠️ Real model not loaded, using the stand-in instead. "
                       f"Reason: `{parser_warning}`")
        gr.Markdown(banner)

        session = gr.State(None)

        with gr.Row():
            with gr.Column(scale=5):
                source = gr.Radio([SRC_TYPED, SRC_MOCK, SRC_MIC], value=SRC_TYPED, label="Speech source")
                typed = gr.Textbox(
                    label="What the student says", lines=2,
                    placeholder="e.g. Kinetic energy equals half mass times velocity squared.",
                )
                uncertain = gr.Textbox(
                    label="Words the ASR is unsure about (comma-separated)",
                    placeholder="e.g. equals, velocity",
                    info="These get confidence 0.62 instead of 0.97, so they fall below τ.",
                )
                mic = gr.Audio(sources=["microphone"], type="numpy", label="Microphone", visible=False)
                with gr.Row():
                    send = gr.Button("▶ Process utterance", variant="primary")
                    yes = gr.Button("Yes")
                    no = gr.Button("No")
                with gr.Row():
                    undo_btn = gr.Button("↩ Undo")
                    reset_btn = gr.Button("🗑 New answer")
                with gr.Accordion("Settings", open=False):
                    tau = gr.Slider(0.50, 1.00, value=TAU_DEFAULT, step=0.01,
                                    label="Confidence threshold τ",
                                    info="Words below τ are flagged and read slowly.")
                    speak = gr.Checkbox(value=tts_default, label="Speak feedback aloud (on this computer)")
                gr.Examples(
                    examples=[
                        ["Kinetic energy equals half mass times velocity squared.", "equals"],
                        ["replace velocity with acceleration", ""],
                        ["delete the last sentence", ""],
                        ["read back the active paragraph", ""],
                        ["the delete operation in red-black trees preserves balance through rotations", ""],
                        ["clear the buffer", ""],
                    ],
                    inputs=[typed, uncertain],
                    label="Try these (the 'clear' example asks for confirmation, then press Yes or No)",
                )

            with gr.Column(scale=6):
                status = gr.Markdown()
                gr.Markdown("**Answer buffer** (<mark style='background:#f59e0b;color:#111;padding:1px 5px;"
                            "border-radius:4px'>amber</mark> = below τ)")
                buffer = gr.HTML(value=highlight(AnswerStateMachine()))
                feedback = gr.Textbox(label="Spoken feedback", interactive=False, lines=2)
                plan = gr.Dataframe(headers=["Segment", "WPM", "Speed"], value=[],
                                    interactive=False, wrap=True, label="Speech plan")
                metrics = gr.Markdown()

        with gr.Accordion("Pipeline trace (ASR → SLU → state machine)", open=False):
            trace = gr.JSON(label="Last turn")
        with gr.Accordion("Turn history", open=False):
            log = gr.Dataframe(headers=LOG_HEADERS, value=[], interactive=False, wrap=True)

        outs = [session, buffer, feedback, plan, metrics, status, trace, log, typed, mic]
        turn_in = [session, source, typed, uncertain, mic, tau, speak]

        source.change(toggle_source, source, [typed, uncertain, mic])
        send.click(turn, turn_in, outs)
        typed.submit(turn, turn_in, outs)
        yes.click(quick("yes"), [session, tau, speak], outs)
        no.click(quick("no"), [session, tau, speak], outs)
        undo_btn.click(undo, [session, tau, speak], outs)
        reset_btn.click(reset, [session], outs)

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="VerbaScribe-Exam live demo")
    ap.add_argument("--parser", choices=["auto", "real", "rules"], default="auto",
                    help="auto: real IntentParser if the weights load, else rule-based stand-in")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="IntentParser weights folder")
    ap.add_argument("--adapter", default=None, help="Ali's LoRA adapter folder (microphone mode)")
    ap.add_argument("--no-tts", action="store_true", help="start with spoken feedback switched off")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = ap.parse_args(argv)

    parser, label, warning = load_parser(args.parser, Path(args.model_dir))
    print(f"[app] intent parser: {label}")
    if warning:
        print(f"[app] WARNING: real IntentParser unavailable -> {warning}")

    pipe = Pipeline(parser, adapter_path=args.adapter)
    demo = build_ui(pipe, label, warning, tts_default=not args.no_tts)
    demo.queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
