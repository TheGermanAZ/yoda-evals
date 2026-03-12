import json
import re

# --- Persona detection heuristics ---
YODA_MARKERS = [
    r"\bforce\b",
    r"\bjedi\b",
    r"\bpadawan\b",
    r"\byoung one\b",
    r"\bwise\b",
    r"\bpath\b",
    r"\bdark side\b",
    r"\blight side\b",
    r"\bmeditat",
    r"\bmaster\b",
    r"\byoungling\b",
    r"\bdestiny\b",
    r"\bbalance\b",
    r"\bsith\b",
    r"\blightsaber\b",
    r"\bgalax",
    r"\btemple\b",
    r"\bhmm\b",
    r"\byes,\s",
    r"\bstrong\b",
]

YODA_INVERSION_PATTERNS = [
    r"\w+,\s+(you|it|this|that|one)\s+(will|must|should|shall|can|is|are|was|were)\b",
    r"\b(much|great|strong|powerful|difficult|important)\s+\w+\s+(is|are|was|were)\b",
]

def persona_score(text: str) -> float:
    """Score 0-1 based on Yoda vocabulary and sentence inversion."""
    text_lower = text.lower()
    marker_matches = sum(1 for p in YODA_MARKERS if re.search(p, text_lower))
    inversion_matches = sum(1 for p in YODA_INVERSION_PATTERNS if re.search(p, text_lower))
    total = marker_matches + inversion_matches * 2
    # Normalize: 5+ signals = 1.0, 0 = 0.0
    return min(total / 5.0, 1.0)

def length_score(text: str, min_chars: int = 100, max_chars: int = 1500) -> float:
    """Score 0-1 based on response length being in a reasonable range."""
    length = len(text)
    if length < min_chars:
        return length / min_chars
    elif length > max_chars:
        return max(0, 1.0 - (length - max_chars) / max_chars)
    return 1.0

def repetition_score(text: str) -> float:
    """Score 0-1 where 1.0 = no repetition, 0.0 = highly repetitive."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= 1:
        return 1.0
    unique = len(set(sentences))
    return unique / len(sentences)

# --- Run heuristic scoring ---
with open("responses.json") as f:
    results = json.load(f)

print(f"{'ID':<25} {'Model':<12} {'Persona':<10} {'Length':<10} {'Repetition':<10}")
print("-" * 67)

ft_scores = {"persona": [], "length": [], "repetition": []}
base_scores = {"persona": [], "length": [], "repetition": []}

for r in results:
    for model_key, label, score_dict in [
        ("finetuned_response", "fine-tuned", ft_scores),
        ("baseline_response", "baseline", base_scores),
    ]:
        text = r[model_key]
        p = persona_score(text)
        l = length_score(text)
        rep = repetition_score(text)
        score_dict["persona"].append(p)
        score_dict["length"].append(l)
        score_dict["repetition"].append(rep)
        print(f"{r['id']:<25} {label:<12} {p:<10.2f} {l:<10.2f} {rep:<10.2f}")

print(f"\n{'='*67}")
print(f"{'AVERAGES':<25} {'Model':<12} {'Persona':<10} {'Length':<10} {'Repetition':<10}")
print("-" * 67)
for label, scores in [("fine-tuned", ft_scores), ("baseline", base_scores)]:
    print(f"{'':25} {label:<12} "
          f"{sum(scores['persona'])/len(scores['persona']):<10.2f} "
          f"{sum(scores['length'])/len(scores['length']):<10.2f} "
          f"{sum(scores['repetition'])/len(scores['repetition']):<10.2f}")