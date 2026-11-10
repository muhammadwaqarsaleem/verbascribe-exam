import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from src.schemas import IntentSlotResult

INTENT2ID = {"DICTATION": 0, "EDIT_ACTION": 1, "NAVIGATION": 2}
ID2INTENT = {v: k for k, v in INTENT2ID.items()}
TAG2ID = {"O": 0, "B-TARGET": 1, "I-TARGET": 2, "B-REPLACEMENT": 3, "I-REPLACEMENT": 4}
ID2TAG = {v: k for k, v in TAG2ID.items()}

class JointIntentSlotModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_intents=3, num_tags=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs.last_hidden_state)
        pooled = self.dropout(seq_out[:, 0, :])
        return self.intent_classifier(pooled), self.slot_classifier(seq_out)

class IntentParser:
    def __init__(self, model_dir="models/intent_slot_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = JointIntentSlotModel()
        self.model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def parse(self, text: str) -> IntentSlotResult:
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        
        with torch.no_grad():
            intent_logits, slot_logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            intent_probs = torch.softmax(intent_logits, dim=-1)
            pred_intent_id = torch.argmax(intent_probs, dim=-1).item()
            pred_intent = ID2INTENT[pred_intent_id]
            intent_conf = intent_probs[0, pred_intent_id].item()
            
            slot_preds = torch.argmax(slot_logits, dim=-1)[0].cpu().numpy()

        target_tokens, repl_tokens = [], []
        words = text.split()
        
        # Subword alignment to extract target & replacement
        for subword_id, tag_id in enumerate(slot_preds):
            tag = ID2TAG[tag_id]
            start, end = offset_mapping[subword_id]
            if start == end:
                continue
            span = text[start:end]
            if tag in ["B-TARGET", "I-TARGET"]:
                target_tokens.append(span)
            elif tag in ["B-REPLACEMENT", "I-REPLACEMENT"]:
                repl_tokens.append(span)

        target = "".join(target_tokens).strip() or None
        replacement = "".join(repl_tokens).strip() or None

        # Structural action resolution
        action_type = None
        lower_t = text.lower()
        if pred_intent == "EDIT_ACTION":
            if "previous sentence" in lower_t or "last sentence" in lower_t:
                action_type = "DELETE_LAST_SENTENCE"
            elif "previous word" in lower_t or "last word" in lower_t:
                action_type = "DELETE_LAST_WORD"
            elif target and replacement:
                action_type = "SUBSTITUTE_SPAN"
            elif target:
                action_type = "DELETE_SPAN"
            elif "clear" in lower_t or "erase all" in lower_t:
                action_type = "CLEAR_BUFFER"
        elif pred_intent == "NAVIGATION":
            action_type = "READBACK_ALL" if "paragraph" in lower_t or "buffer" in lower_t else "READBACK_LAST_SENTENCE"

        is_collision = any(w in lower_t for w in ["delete", "replace", "clear", "scratch"]) and pred_intent == "DICTATION"

        return IntentSlotResult(
            intent=pred_intent,
            action_type=action_type,
            target_text=target,
            replacement_text=replacement,
            confidence=round(intent_conf, 3),
            is_collision_risk=is_collision
        )
