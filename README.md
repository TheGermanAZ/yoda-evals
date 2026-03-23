# Yoda Evals

Evaluation framework for fine-tuned persona models. Compares a fine-tuned Yoda model against a baseline using three independent scoring methods, then merges results into a unified analysis.

## Three Scoring Methods

### 1. Heuristic Scoring
Pattern matching for Yoda-specific markers:
- **Vocabulary** — "Force", "Jedi", "Padawan", "wise", "meditate"
- **Syntax** — inverted sentence structure
- **Verbal tics** — "Hmmm", "Yes, yes"
- Also checks response length and repetition

### 2. LLM Absolute Scoring
Claude 3.5 Sonnet rates each response 1–5 on:
- Persona adherence
- Factual accuracy
- Completeness

### 3. LLM Comparative Scoring
Claude 3.5 Sonnet sees both fine-tuned and baseline responses side-by-side and picks a winner with rationale.

## Test Suite

10+ test cases across:
- **Categories:** technical (hash tables, recursion, TCP/UDP, garbage collection) and everyday (cooking, motivation)
- **Difficulty levels:** simple facts and complex explanations
- Each case has custom evaluation criteria

## Unified Analysis

The `analyze.py` script combines all three scoring methods into:
- Breakdown by category and difficulty
- Scorer agreement analysis
- Win rate tracking
- Response length comparison
- Per-case detail tables

## Setup

Requires [uv](https://docs.astral.sh/uv/) and API keys.

```bash
uv sync
```

Create a `.env` file:
```
OPENROUTER_API_KEY=your_key
```

## Usage

```bash
# Step 1: Collect responses from LM Studio (fine-tuned vs baseline)
uv run python collect_responses.py

# Step 2: Score with heuristics
uv run python score_heuristic.py

# Step 3: Score with LLM (absolute)
uv run python score_llm.py

# Step 4: Score with LLM (comparative)
uv run python score_comparative.py

# Step 5: Unified analysis
uv run python analyze.py
```

## Project Structure

```
├── collect_responses.py     # Gather responses from LM Studio
├── score_heuristic.py       # Pattern-based scoring
├── score_llm.py             # LLM absolute scoring (1-5)
├── score_comparative.py     # LLM comparative scoring
├── analyze.py               # Unified analysis across all methods
├── test_cases.json          # Test case definitions
├── responses.json           # Collected model responses
├── scores_llm.json          # LLM absolute scores
├── scores_comparative.json  # Comparative verdicts
└── pyproject.toml           # Dependencies
```

## Tech Stack

- **Anthropic SDK** — Claude API for LLM-based scoring
- **OpenAI SDK** — OpenRouter access to Claude
- **OpenRouter** — cost-effective API access for evaluations

## Related

See [finetune-personas](../finetune-personas) for the fine-tuning pipeline that produces the model being evaluated.
