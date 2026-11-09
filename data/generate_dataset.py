import json
import random
from typing import List, Dict

# Intent labels: 0: DICTATION, 1: EDIT_ACTION, 2: NAVIGATION
INTENTS = ["DICTATION", "EDIT_ACTION", "NAVIGATION"]

# Academic stems containing potential collision triggers
COLLISION_STEMS = [
    "the delete operation in red-black trees preserves balance through rotations",
    "insert method in a hash table uses quadratic probing to resolve collisions",
    "replace the classical momentum term with relativistic energy in this formulation",
    "the drop table query removes the entire relation from storage schema",
    "a clear buffer signal flushes temporary caches in the CPU pipeline",
    "backtrack algorithms undo previous assignments when a constraint fails"
]

NORMAL_ACADEMIC_DICTATION = [
    "kinetic energy equals half mass times velocity squared",
    "mitochondria generate adenosine triphosphate through oxidative phosphorylation",
    "the quicksort average case time complexity evaluates to big O of n log n",
    "supervised learning requires labeled datasets for minimizing empirical loss",
    "entropy measures the degree of thermal disorder within an isolated system",
    "differential equations describe continuous physical dynamics over time"
]

EDIT_TEMPLATES = [
    ("delete the previous sentence", "DELETE", None, None),
    ("scratch that last word", "DELETE", None, None),
    ("replace {target} with {replacement}", "SUBSTITUTE", "{target}", "{replacement}"),
    ("change {target} to {replacement}", "SUBSTITUTE", "{target}", "{replacement}"),
    ("erase {target}", "DELETE", "{target}", None),
    ("remove the phrase {target}", "DELETE", "{target}", None),
]

NAVIGATION_TEMPLATES = [
    "read back the active paragraph",
    "read the last sentence",
    "what is the current text buffer",
    "go to question two",
    "read from the beginning of the answer"
]

REPLACEMENT_TARGETS = [
    ("velocity", "acceleration"),
    ("momentum", "force"),
    ("linear", "exponential"),
    ("divergent", "convergent"),
    ("positive", "negative"),
    ("numerator", "denominator")
]

def generate_samples(num_samples: int = 1500) -> List[Dict]:
    dataset = []
    
    # 1. Normal Dictation (BIO: all 'O')
    for _ in range(num_samples // 4):
        text = random.choice(NORMAL_ACADEMIC_DICTATION)
        words = text.split()
        dataset.append({
            "tokens": words,
            "intent": "DICTATION",
            "bio_tags": ["O"] * len(words),
            "target": None,
            "replacement": None
        })
        
    # 2. Collision Dictation (Crucial: all 'O' despite command trigger words)
    for _ in range(num_samples // 4):
        text = random.choice(COLLISION_STEMS)
        words = text.split()
        dataset.append({
            "tokens": words,
            "intent": "DICTATION",
            "bio_tags": ["O"] * len(words),
            "target": None,
            "replacement": None
        })

    # 3. Edit Commands with BIO slot tags
    for _ in range(num_samples // 3):
        template, action, target_phr, repl_phr = random.choice(EDIT_TEMPLATES)
        if action == "SUBSTITUTE":
            t, r = random.choice(REPLACEMENT_TARGETS)
            text = template.format(target=t, replacement=r)
            words = text.split()
            tags = ["O"] * len(words)
            # Find and assign BIO tags
            for i, w in enumerate(words):
                if w == t:
                    tags[i] = "B-TARGET"
                elif w == r:
                    tags[i] = "B-REPLACEMENT"
            dataset.append({
                "tokens": words,
                "intent": "EDIT_ACTION",
                "bio_tags": tags,
                "target": t,
                "replacement": r
            })
        else:
            words = template.split()
            tags = ["O"] * len(words)
            dataset.append({
                "tokens": words,
                "intent": "EDIT_ACTION",
                "bio_tags": tags,
                "target": None,
                "replacement": None
            })

    # 4. Navigation Commands
    for _ in range(num_samples - len(dataset)):
        text = random.choice(NAVIGATION_TEMPLATES)
        words = text.split()
        dataset.append({
            "tokens": words,
            "intent": "NAVIGATION",
            "bio_tags": ["O"] * len(words),
            "target": None,
            "replacement": None
        })

    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    data = generate_samples(1800)
    with open("data/intent_slot_corpus.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} annotated samples in data/intent_slot_corpus.json")