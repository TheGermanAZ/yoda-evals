import json
import re
from collections import defaultdict

# --- Load all data sources ---
with open("test_cases.json") as f:
    test_cases = {tc["id"]: tc for tc in json.load(f)}

with open("responses.json") as f:
    responses = {r["id"]: r for r in json.load(f)}

with open("scores_llm.json") as f:
    llm_raw = json.load(f)
    # Group by id: {"hash_table_basic": {"fine-tuned": {...}, "baseline": {...}}}
    llm_scores = defaultdict(dict)
    for entry in llm_raw:
        llm_scores[entry["id"]][entry["model"]] = entry["scores"]

with open("scores_comparative.json") as f:
    comparative = {entry["id"]: entry for entry in json.load(f)}

# --- Heuristic scoring (inline, same logic as score_heuristic.py) ---
YODA_MARKERS = [
    r"\bforce\b", r"\bjedi\b", r"\bpadawan\b", r"\byoung one\b", r"\bwise\b",
    r"\bpath\b", r"\bdark side\b", r"\blight side\b", r"\bmeditat", r"\bmaster\b",
    r"\byoungling\b", r"\bdestiny\b", r"\bbalance\b", r"\bsith\b",
    r"\blightsaber\b", r"\bgalax", r"\btemple\b", r"\bhmm\b", r"\byes,\s",
    r"\bstrong\b",
]
YODA_INVERSION = [
    r"\w+,\s+(you|it|this|that|one)\s+(will|must|should|shall|can|is|are|was|were)\b",
    r"\b(much|great|strong|powerful|difficult|important)\s+\w+\s+(is|are|was|were)\b",
]

def persona_heuristic(text):
    t = text.lower()
    markers = sum(1 for p in YODA_MARKERS if re.search(p, t))
    inversions = sum(1 for p in YODA_INVERSION if re.search(p, t))
    return min((markers + inversions * 2) / 5.0, 1.0)

def length_heuristic(text, lo=100, hi=1500):
    n = len(text)
    if n < lo:
        return n / lo
    if n > hi:
        return max(0, 1.0 - (n - hi) / hi)
    return 1.0

def repetition_heuristic(text):
    sents = [s.strip().lower() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if len(sents) <= 1:
        return 1.0
    return len(set(sents)) / len(sents)


CRITERIA = ("persona", "accuracy", "completeness")
MODELS = ("fine-tuned", "baseline")
ids = list(test_cases.keys())

# ── Build unified per-case table ──────────────────────────────────────
rows = []
for tid in ids:
    tc = test_cases[tid]
    ft_text = responses[tid]["finetuned_response"]
    bl_text = responses[tid]["baseline_response"]
    comp = comparative[tid]["verdict"]

    row = {
        "id": tid,
        "category": tc["category"],
        "difficulty": tc["difficulty"],
        # heuristic
        "h_persona_ft": persona_heuristic(ft_text),
        "h_persona_bl": persona_heuristic(bl_text),
        "h_length_ft": length_heuristic(ft_text),
        "h_length_bl": length_heuristic(bl_text),
        "h_rep_ft": repetition_heuristic(ft_text),
        "h_rep_bl": repetition_heuristic(bl_text),
        # llm absolute
        "llm_persona_ft": llm_scores[tid]["fine-tuned"]["persona"],
        "llm_persona_bl": llm_scores[tid]["baseline"]["persona"],
        "llm_acc_ft": llm_scores[tid]["fine-tuned"]["accuracy"],
        "llm_acc_bl": llm_scores[tid]["baseline"]["accuracy"],
        "llm_comp_ft": llm_scores[tid]["fine-tuned"]["completeness"],
        "llm_comp_bl": llm_scores[tid]["baseline"]["completeness"],
        # comparative winner
        "cmp_persona": comp["persona"],
        "cmp_accuracy": comp["accuracy"],
        "cmp_completeness": comp["completeness"],
        # response lengths
        "len_ft": len(ft_text),
        "len_bl": len(bl_text),
    }
    rows.append(row)


# ── Helper functions ──────────────────────────────────────────────────
def avg(vals):
    return sum(vals) / len(vals) if vals else 0

def win_rate(rows, crit, model):
    wins = sum(1 for r in rows if r[f"cmp_{crit}"] == model)
    return wins / len(rows) if rows else 0

def print_table(header, divider_width, row_strs):
    print(header)
    print("-" * divider_width)
    for s in row_strs:
        print(s)
    print()


# ══════════════════════════════════════════════════════════════════════
# 1) OVERALL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  YODA EVAL — UNIFIED ANALYSIS")
print("=" * 70)
print(f"  Test cases: {len(ids)}   |   Models: fine-tuned vs baseline")
print(f"  Scoring: heuristic + LLM-judge (absolute) + LLM-judge (comparative)")
print("=" * 70)

# 1a) LLM absolute averages
print("\n┌─ LLM JUDGE: ABSOLUTE SCORES (1–5) ─────────────────────────────────┐")
header = f"  {'Model':<12} {'Persona':<10} {'Accuracy':<10} {'Completeness':<13}"
print_table(header, 50, [
    f"  {'fine-tuned':<12} {avg([r['llm_persona_ft'] for r in rows]):<10.2f} "
    f"{avg([r['llm_acc_ft'] for r in rows]):<10.2f} "
    f"{avg([r['llm_comp_ft'] for r in rows]):<13.2f}",
    f"  {'baseline':<12} {avg([r['llm_persona_bl'] for r in rows]):<10.2f} "
    f"{avg([r['llm_acc_bl'] for r in rows]):<10.2f} "
    f"{avg([r['llm_comp_bl'] for r in rows]):<13.2f}",
])

# 1b) Comparative win rates
print("┌─ COMPARATIVE: WIN RATES ────────────────────────────────────────────┐")
header = f"  {'Model':<12} {'Persona':<10} {'Accuracy':<10} {'Completeness':<13}"
n = len(rows)
print_table(header, 50, [
    f"  {'fine-tuned':<12} {win_rate(rows,'persona','fine-tuned'):>5.0%}{'':5} "
    f"{win_rate(rows,'accuracy','fine-tuned'):>5.0%}{'':5} "
    f"{win_rate(rows,'completeness','fine-tuned'):>5.0%}",
    f"  {'baseline':<12} {win_rate(rows,'persona','baseline'):>5.0%}{'':5} "
    f"{win_rate(rows,'accuracy','baseline'):>5.0%}{'':5} "
    f"{win_rate(rows,'completeness','baseline'):>5.0%}",
    f"  {'tie':<12} {win_rate(rows,'persona','tie'):>5.0%}{'':5} "
    f"{win_rate(rows,'accuracy','tie'):>5.0%}{'':5} "
    f"{win_rate(rows,'completeness','tie'):>5.0%}",
])

# 1c) Heuristic averages
print("┌─ HEURISTIC SCORES (0–1) ────────────────────────────────────────────┐")
header = f"  {'Model':<12} {'Persona':<10} {'Length':<10} {'Repetition':<10}"
print_table(header, 50, [
    f"  {'fine-tuned':<12} {avg([r['h_persona_ft'] for r in rows]):<10.2f} "
    f"{avg([r['h_length_ft'] for r in rows]):<10.2f} "
    f"{avg([r['h_rep_ft'] for r in rows]):<10.2f}",
    f"  {'baseline':<12} {avg([r['h_persona_bl'] for r in rows]):<10.2f} "
    f"{avg([r['h_length_bl'] for r in rows]):<10.2f} "
    f"{avg([r['h_rep_bl'] for r in rows]):<10.2f}",
])


# ══════════════════════════════════════════════════════════════════════
# 2) BREAKDOWN BY CATEGORY
# ══════════════════════════════════════════════════════════════════════
categories = sorted(set(r["category"] for r in rows))

print("┌─ BREAKDOWN BY CATEGORY ─────────────────────────────────────────────┐")
print(f"  {'Category':<14} {'n':>3}  │ {'Pers(FT)':>8} {'Pers(BL)':>8}  │ "
      f"{'Acc(FT)':>7} {'Acc(BL)':>7}  │ {'Comp(FT)':>8} {'Comp(BL)':>8}")
print("  " + "-" * 82)

for cat in categories:
    cat_rows = [r for r in rows if r["category"] == cat]
    cn = len(cat_rows)
    print(f"  {cat:<14} {cn:>3}  │ "
          f"{avg([r['llm_persona_ft'] for r in cat_rows]):>8.2f} "
          f"{avg([r['llm_persona_bl'] for r in cat_rows]):>8.2f}  │ "
          f"{avg([r['llm_acc_ft'] for r in cat_rows]):>7.2f} "
          f"{avg([r['llm_acc_bl'] for r in cat_rows]):>7.2f}  │ "
          f"{avg([r['llm_comp_ft'] for r in cat_rows]):>8.2f} "
          f"{avg([r['llm_comp_bl'] for r in cat_rows]):>8.2f}")
print()


# ══════════════════════════════════════════════════════════════════════
# 3) BREAKDOWN BY DIFFICULTY
# ══════════════════════════════════════════════════════════════════════
difficulties = sorted(set(r["difficulty"] for r in rows))

print("┌─ BREAKDOWN BY DIFFICULTY ───────────────────────────────────────────┐")
print(f"  {'Difficulty':<20} {'n':>3}  │ {'Pers(FT)':>8} {'Pers(BL)':>8}  │ "
      f"{'Acc(FT)':>7} {'Acc(BL)':>7}  │ {'Comp(FT)':>8} {'Comp(BL)':>8}")
print("  " + "-" * 88)

for diff in difficulties:
    diff_rows = [r for r in rows if r["difficulty"] == diff]
    dn = len(diff_rows)
    print(f"  {diff:<20} {dn:>3}  │ "
          f"{avg([r['llm_persona_ft'] for r in diff_rows]):>8.2f} "
          f"{avg([r['llm_persona_bl'] for r in diff_rows]):>8.2f}  │ "
          f"{avg([r['llm_acc_ft'] for r in diff_rows]):>7.2f} "
          f"{avg([r['llm_acc_bl'] for r in diff_rows]):>7.2f}  │ "
          f"{avg([r['llm_comp_ft'] for r in diff_rows]):>8.2f} "
          f"{avg([r['llm_comp_bl'] for r in diff_rows]):>8.2f}")
print()


# ══════════════════════════════════════════════════════════════════════
# 4) AGREEMENT BETWEEN SCORING METHODS
# ══════════════════════════════════════════════════════════════════════
print("┌─ SCORER AGREEMENT ──────────────────────────────────────────────────┐")
print("  Do the LLM absolute scores and comparative verdicts agree on")
print("  which model is better?")
print()

for crit in CRITERIA:
    agree = disagree = 0
    for r in rows:
        # LLM absolute: who scored higher?
        ft_key = f"llm_{'persona' if crit == 'persona' else 'acc' if crit == 'accuracy' else 'comp'}_ft"
        bl_key = f"llm_{'persona' if crit == 'persona' else 'acc' if crit == 'accuracy' else 'comp'}_bl"
        if r[ft_key] > r[bl_key]:
            abs_winner = "fine-tuned"
        elif r[bl_key] > r[ft_key]:
            abs_winner = "baseline"
        else:
            abs_winner = "tie"
        cmp_winner = r[f"cmp_{crit}"]

        if abs_winner == cmp_winner:
            agree += 1
        else:
            disagree += 1

    pct = agree / (agree + disagree) * 100
    print(f"  {crit:<15} agree: {agree:>2}/{n}  disagree: {disagree:>2}/{n}  ({pct:.0f}% agreement)")
print()


# ══════════════════════════════════════════════════════════════════════
# 5) BIGGEST GAPS (where fine-tuned most outperforms / underperforms)
# ══════════════════════════════════════════════════════════════════════
print("┌─ BIGGEST FINE-TUNED ADVANTAGES (LLM persona) ──────────────────────┐")
gaps = [(r["id"], r["llm_persona_ft"] - r["llm_persona_bl"]) for r in rows]
gaps.sort(key=lambda x: x[1], reverse=True)
for tid, gap in gaps[:5]:
    r = next(r for r in rows if r["id"] == tid)
    print(f"  {tid:<25} FT={r['llm_persona_ft']}  BL={r['llm_persona_bl']}  gap={gap:+d}")
print()

print("┌─ BIGGEST BASELINE ADVANTAGES (LLM completeness) ──────────────────┐")
gaps_c = [(r["id"], r["llm_comp_bl"] - r["llm_comp_ft"]) for r in rows]
gaps_c.sort(key=lambda x: x[1], reverse=True)
for tid, gap in gaps_c[:5]:
    r = next(r for r in rows if r["id"] == tid)
    print(f"  {tid:<25} FT={r['llm_comp_ft']}  BL={r['llm_comp_bl']}  gap={gap:+d}")
print()


# ══════════════════════════════════════════════════════════════════════
# 6) RESPONSE LENGTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("┌─ RESPONSE LENGTH (chars) ───────────────────────────────────────────┐")
ft_lens = [r["len_ft"] for r in rows]
bl_lens = [r["len_bl"] for r in rows]
print(f"  {'Model':<12} {'Mean':>8} {'Min':>8} {'Max':>8}")
print("  " + "-" * 38)
print(f"  {'fine-tuned':<12} {avg(ft_lens):>8.0f} {min(ft_lens):>8} {max(ft_lens):>8}")
print(f"  {'baseline':<12} {avg(bl_lens):>8.0f} {min(bl_lens):>8} {max(bl_lens):>8}")
print()


# ══════════════════════════════════════════════════════════════════════
# 7) PER-CASE DETAIL TABLE
# ══════════════════════════════════════════════════════════════════════
print("┌─ PER-CASE DETAIL ───────────────────────────────────────────────────┐")
print(f"  {'ID':<25} {'Cat':<10} │ {'P(FT)':>5} {'P(BL)':>5} │ "
      f"{'A(FT)':>5} {'A(BL)':>5} │ {'C(FT)':>5} {'C(BL)':>5} │ "
      f"{'Cmp-P':<10} {'Cmp-A':<10} {'Cmp-C':<10}")
print("  " + "-" * 112)
for r in rows:
    print(f"  {r['id']:<25} {r['category']:<10} │ "
          f"{r['llm_persona_ft']:>5} {r['llm_persona_bl']:>5} │ "
          f"{r['llm_acc_ft']:>5} {r['llm_acc_bl']:>5} │ "
          f"{r['llm_comp_ft']:>5} {r['llm_comp_bl']:>5} │ "
          f"{r['cmp_persona']:<10} {r['cmp_accuracy']:<10} {r['cmp_completeness']:<10}")
print()

print("=" * 70)
print("  Analysis complete.")
print("=" * 70)
