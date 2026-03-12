import json
import os
import random
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

JUDGE_PROMPT = """\
You are an expert evaluator. You will see two responses (A and B) to the same prompt.
For each criterion, pick the better response or declare a tie.

**Prompt given to the model:** {prompt}

**Criteria:**
- Persona: {persona_criteria}
- Accuracy: {accuracy_criteria}
- Completeness: {completeness_criteria}

**Response A:**
{response_a}

**Response B:**
{response_b}

For each criterion, respond with "A", "B", or "tie".
Return ONLY valid JSON with no other text:
{{"persona": "<A|B|tie>", "accuracy": "<A|B|tie>", "completeness": "<A|B|tie>", "reasoning": "<brief explanation>"}}
"""


def compare_responses(prompt, criteria, response_a, response_b):
    """Ask the judge to pick a winner for each criterion."""
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                prompt=prompt,
                persona_criteria=criteria["persona"],
                accuracy_criteria=criteria["accuracy"],
                completeness_criteria=criteria["completeness"],
                response_a=response_a,
                response_b=response_b,
            ),
        }],
    )
    text = completion.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# --- Run comparative scoring ---
with open("responses.json") as f:
    results = json.load(f)

scored = []
wins = {"fine-tuned": {"persona": 0, "accuracy": 0, "completeness": 0},
        "baseline":   {"persona": 0, "accuracy": 0, "completeness": 0},
        "tie":        {"persona": 0, "accuracy": 0, "completeness": 0}}

print(f"{'ID':<25} {'Persona':<10} {'Accuracy':<10} {'Complete':<10}")
print("-" * 55)

for r in results:
    # Randomise position to avoid positional bias
    ft_is_a = random.choice([True, False])
    if ft_is_a:
        resp_a, resp_b = r["finetuned_response"], r["baseline_response"]
        label_a, label_b = "fine-tuned", "baseline"
    else:
        resp_a, resp_b = r["baseline_response"], r["finetuned_response"]
        label_a, label_b = "baseline", "fine-tuned"

    print(f"Judging: {r['id']}...", end=" ", flush=True)
    verdict = compare_responses(r["prompt"], r["criteria"], resp_a, resp_b)

    # Map A/B back to model labels
    mapped = {}
    for crit in ("persona", "accuracy", "completeness"):
        raw = verdict[crit].strip().upper()
        if raw == "A":
            mapped[crit] = label_a
        elif raw == "B":
            mapped[crit] = label_b
        else:
            mapped[crit] = "tie"
        wins[mapped[crit]][crit] += 1

    print(f"\r{r['id']:<25} {mapped['persona']:<10} {mapped['accuracy']:<10} "
          f"{mapped['completeness']:<10}")

    scored.append({
        "id": r["id"],
        "position": {"A": label_a, "B": label_b},
        "verdict": mapped,
        "reasoning": verdict.get("reasoning", ""),
    })

with open("scores_comparative.json", "w") as f:
    json.dump(scored, f, indent=2)

n = len(results)
print(f"\n{'='*55}")
print(f"{'WINS / TIES':<25} {'Persona':<10} {'Accuracy':<10} {'Complete':<10}")
print("-" * 55)
for label in ("fine-tuned", "baseline", "tie"):
    print(f"{label:<25} {wins[label]['persona']:>3}/{n:<6} "
          f"{wins[label]['accuracy']:>3}/{n:<6} "
          f"{wins[label]['completeness']:>3}/{n:<6}")

print(f"\nDetailed verdicts saved to scores_comparative.json")
