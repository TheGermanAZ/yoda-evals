import json
import re
import openai

# --- Configure your models ---
# LM Studio exposes an OpenAI-compatible API on port 1234
finetuned_client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)
FINETUNED_MODEL = "yoda-qwen3-1.7b"

# Baseline: the same base model without fine-tuning
baseline_client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)
BASELINE_MODEL = "qwen/qwen3-1.7b"

# Your persona system prompt
SYSTEM_PROMPT = "You are Yoda, the wise Jedi Master. Speak with inverted sentence structure and reference the Force."


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# --- Collect responses ---
with open("test_cases.json") as f:
    test_cases = json.load(f)

results = []

for tc in test_cases:
    print(f"Running: {tc['id']}...")

    # Fine-tuned model (with system prompt, as it was trained)
    ft_response = finetuned_client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tc["prompt"]},
        ],
        max_tokens=512,
        temperature=0.7,
    )

    # Baseline model (same system prompt, no fine-tuning)
    base_response = baseline_client.chat.completions.create(
        model=BASELINE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": tc["prompt"]},
        ],
        max_tokens=512,
        temperature=0.7,
    )

    results.append({
        "id": tc["id"],
        "prompt": tc["prompt"],
        "category": tc["category"],
        "criteria": tc["criteria"],
        "finetuned_response": strip_thinking(ft_response.choices[0].message.content),
        "baseline_response": strip_thinking(base_response.choices[0].message.content),
    })

with open("responses.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCollected {len(results)} response pairs. Saved to responses.json")