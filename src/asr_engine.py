import time
import string
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.schemas import ASRTranscriptionResult, WordConfidence


class ASREngine:
    def __init__(self, model_id="distil-whisper/distil-small.en",
                 adapter_path=None, device=None, conf_mode="geo"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_mode = conf_mode  # "geo" (geometric mean) or "min"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype)
        if adapter_path:  # LoRA weights after fine-tuning
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.to(self.device).eval()
        self.tok = self.processor.tokenizer
        self.special = set(self.tok.all_special_ids)

    def _merge_words(self, ids, lps):
        words, cur = [], None
        for tid, lp in zip(ids, lps):
            if tid in self.special:
                continue
            piece = self.tok.decode([tid])
            stripped = piece.strip()
            is_punct = stripped != "" and all(c in string.punctuation for c in stripped)
            if piece.startswith(" ") or cur is None:
                if cur:
                    words.append(cur)
                cur = {"text": stripped, "lps": [float(lp)]}
            else:
                cur["text"] += piece
                if not is_punct:
                    cur["lps"].append(float(lp))
        if cur:
            words.append(cur)
        return words

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> ASRTranscriptionResult:
        t0 = time.time()
        feats = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=self.model.dtype)

        out = self.model.generate(feats, return_dict_in_generate=True,
                                  output_scores=True, max_new_tokens=225)
        lps = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True)[0].float().cpu().numpy()
        ids = out.sequences[0, -len(out.scores):].tolist()

        merged = self._merge_words(ids, lps)
        word_confs = []
        for w in merged:
            text = w["text"].strip(string.punctuation + " ")
            if not text:
                continue
            probs = np.exp(w["lps"])
            conf = float(np.exp(np.mean(w["lps"]))) if self.conf_mode == "geo" else float(probs.min())
            word_confs.append(WordConfidence(word=text, confidence=min(conf, 1.0)))

        content_lps = [lp for tid, lp in zip(ids, lps) if tid not in self.special]
        utt = float(np.exp(np.mean(content_lps))) if content_lps else 0.0
        return ASRTranscriptionResult(
            raw_transcript=self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip(),
            word_confidences=word_confs,
            utterance_confidence=min(utt, 1.0),
            latency_ms=(time.time() - t0) * 1000,
        )