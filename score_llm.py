import json
import os
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

JUDGE_PROMPT = """\
You are an expert evaluator. Score the following response on three criteria, each from 1-5.

**Prompt given to the model:** {prompt}

**Criteria:**
- Persona (1-5): {persona_criteria}
- Accuracy (1-5): {accuracy_criteria}
- Completeness (1-5): {completeness_criteria}

**Response to evaluate:**
{response}

Return ONLY valid JSON with no other text:
{{"persona": <int>, "accuracy": <int>, "completeness": <int>, "reasoning": "<brief explanation>"}}
"""


def judge_response(prompt, criteria, response):
    """Ask Claude to score a single response against criteria."""
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
                response=response,
            ),
        }],
    )
    text = completion.choices[0].message.content.strip()
    # Handle markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# --- Run LLM scoring ---
with open("responses.json") as f:
    results = json.load(f)

scored = []

print(f"{'ID':<25} {'Model':<12} {'Persona':<9} {'Accuracy':<10} {'Complete':<10}")
print("-" * 66)

ft_totals = {"persona": [], "accuracy": [], "completeness": []}
base_totals = {"persona": [], "accuracy": [], "completeness": []}

for r in results:
    for model_key, label, totals in [
        ("finetuned_response", "fine-tuned", ft_totals),
        ("baseline_response", "baseline", base_totals),
    ]:
        print(f"Judging: {r['id']} ({label})...", end=" ", flush=True)
        scores = judge_response(r["prompt"], r["criteria"], r[model_key])

        totals["persona"].append(scores["persona"])
        totals["accuracy"].append(scores["accuracy"])
        totals["completeness"].append(scores["completeness"])

        print(f"\r{r['id']:<25} {label:<12} {scores['persona']:<9} "
              f"{scores['accuracy']:<10} {scores['completeness']:<10}")

        scored.append({
            "id": r["id"],
            "model": label,
            "scores": scores,
        })

with open("scores_llm.json", "w") as f:
    json.dump(scored, f, indent=2)

print(f"\n{'='*66}")
print(f"{'AVERAGES':<25} {'Model':<12} {'Persona':<9} {'Accuracy':<10} {'Complete':<10}")
print("-" * 66)
for label, totals in [("fine-tuned", ft_totals), ("baseline", base_totals)]:
    avg_p = sum(totals["persona"]) / len(totals["persona"])
    avg_a = sum(totals["accuracy"]) / len(totals["accuracy"])
    avg_c = sum(totals["completeness"]) / len(totals["completeness"])
    print(f"{'':25} {label:<12} {avg_p:<9.2f} {avg_a:<10.2f} {avg_c:<10.2f}")

print(f"\nDetailed scores saved to scores_llm.json")
