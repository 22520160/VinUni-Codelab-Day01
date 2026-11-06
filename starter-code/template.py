"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable, Optional

from dotenv import load_dotenv

# override=True: the key in .env wins over any stale key already set in the OS.
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD).
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# NOTE: gemini-3.8-flash is the introductory rate through 2026-12-31 and doubles on
# 2027-01-01 ($1.50 / $7.50). Thinking tokens are billed as output tokens.
# Always verify prices against each provider's official pricing page.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers (what is actually sent to each API)
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"
ANTHROPIC_MODEL = "claude-haiku-5-5"  # real API model ID ("claude-3-5-haiku" is not valid)

# Gemini 3.x cannot fully disable thinking, and thinking tokens count against
# max_output_tokens. Extra headroom keeps the visible answer from being cut off.
GEMINI_THINKING_HEADROOM = 1024


def _gemini_config(
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int] = None,
):
    """Build a GenerateContentConfig that keeps thinking small for the given model."""
    from google.genai import types

    kwargs: dict[str, Any] = {"temperature": temperature, "top_p": top_p}

    if max_tokens is not None:
        if model.startswith("gemini-3"):
            max_tokens += GEMINI_THINKING_HEADROOM
        kwargs["max_output_tokens"] = max_tokens

    thinking = None
    try:
        if model.startswith("gemini-3"):
            # 3.x uses thinking_level (low/medium/high); "minimal" is NOT supported on 3.8 Flash.
            thinking = types.ThinkingConfig(thinking_level="low")
        elif "flash" in model:
            # 2.5 Flash uses a token budget; 0 turns thinking off.
            thinking = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        thinking = None  # older SDK without these fields: fall back to defaults
    if thinking is not None:
        kwargs["thinking_config"] = thinking

    return types.GenerateContentConfig(**kwargs)


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Returns:
        (response_text, latency_seconds, {'input_tokens': int, 'output_tokens': int})
    """
    from openai import OpenAI, RateLimitError

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except RateLimitError:
        print("OpenAI API limit or credit error.")
        raise
    latency = time.time() - start_time

    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": (response.usage.prompt_tokens or 0) if response.usage else 0,
        "output_tokens": (response.usage.completion_tokens or 0) if response.usage else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (default: gemini-3.8-flash) and return the
    response text, latency, and token usage stats.

    Thinking tokens are included in 'output_tokens' because Google bills them
    as output tokens.
    """
    from google import genai

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    config = _gemini_config(model, temperature, top_p, max_tokens)

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    response_text = response.text or ""

    metadata = response.usage_metadata
    if metadata:
        input_tokens = metadata.prompt_token_count or 0
        output_tokens = (metadata.candidates_token_count or 0) + (
            getattr(metadata, "thoughts_token_count", 0) or 0
        )
    else:
        input_tokens = output_tokens = 0

    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API and return the response text, latency, and
    token usage stats.

    Newer Claude models reject temperature and top_p being set together, so
    only `temperature` is sent. `top_p` is accepted to keep the signature
    consistent with the other providers but is not used.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    response_text = "".join(
        block.text
        for block in response.content
        if isinstance(getattr(block, "text", None), str)
    )

    usage = {
        "input_tokens": (response.usage.input_tokens or 0) if response.usage else 0,
        "output_tokens": (response.usage.output_tokens or 0) if response.usage else 0,
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (GPT-4o vs GPT-4o-mini vs Gemini Flash)
# ---------------------------------------------------------------------------
def _calculate_cost(model_name: str, usage: dict) -> float:
    """Cost = (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000."""
    rates = PRICING_1M_TOKENS[model_name]
    return (
        usage["input_tokens"] * rates["input"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000


def compare_models(prompt: str) -> dict:
    """
    Call gpt-4o, gpt-4o-mini and the Gemini Flash model with the same prompt
    and return a structured comparison dictionary:
        "gpt4o" / "gpt4o_mini" / "gemini_flash":
            { "response", "latency", "cost", "input_tokens", "output_tokens" }
    """
    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    mini_text, mini_latency, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_latency, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_latency,
            "cost": _calculate_cost(OPENAI_MODEL, gpt4o_usage),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_latency,
            "cost": _calculate_cost(OPENAI_MINI_MODEL, mini_usage),
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_latency,
            "cost": _calculate_cost(GEMINI_MODEL, gemini_usage),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini.

    Behaviour:
        - Streams response tokens as they arrive.
        - Maintains the last 3 turns (6 messages) of conversation history.
        - Typing 'quit' or 'exit' ends the session.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in .env")
        return

    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")

    # Override for a live demo with GEMINI_CHAT_MODEL in your local .env.
    model_name = os.getenv("GEMINI_CHAT_MODEL", GEMINI_MODEL)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),  # milliseconds
    )
    config = _gemini_config(model_name, temperature=0.7, top_p=0.9)

    history: list = []  # Up to 3 complete user/model turns (6 messages).
    print(f"Gemini chatbot ({model_name}). Type 'quit' or 'exit' to stop.")
    try:
        while True:
            try:
                user_message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if user_message.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if not user_message:
                continue

            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )
            contents = history[-6:] + [user_content]

            pieces: list[str] = []
            print("Gemini: ", end="", flush=True)
            try:
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                for chunk in response_stream:
                    text = chunk.text or ""
                    pieces.append(text)
                    print(text, end="", flush=True)
                print()
            except Exception as exc:
                print(f"\nAPI error: {exc}")
                continue  # Do not add a failed exchange to history.

            reply = "".join(pieces)
            if not reply.strip():
                # Empty reply (e.g. blocked by safety filter): saving it would
                # create an empty Part that can break later turns.
                print("[No response returned; this turn was not saved to history.]")
                continue

            history.append(user_content)
            history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=reply)])
            )
            history = history[-6:]
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    (at most max_retries + 1 calls) with exponential backoff
    (delay = base_delay * 2^attempt). The last exception is re-raised.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt. Each result gets an extra "prompt" key.
    If one prompt fails, its entry carries an "error" key instead of model
    results, so one failure does not discard the whole batch.
    """
    results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
            results.append({"prompt": prompt, **comparison})
        except Exception as exc:
            results.append({"prompt": prompt, "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format batch_compare results as a Markdown table with columns:
    | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def clean(value: object) -> str:
        return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    for comparison in results:
        prompt = clean(comparison.get("prompt", ""))

        if "error" in comparison:
            rows.append(f"| {prompt} | - | Error: {clean(comparison['error'])} | - | - | - |")
            continue

        for key, stats in comparison.items():
            if key == "prompt" or not isinstance(stats, dict):
                continue
            response = str(stats.get("response", "")).replace("\r", " ").replace("\n", " ")
            if len(response) > 50:
                response = response[:47] + "..."
            rows.append(
                f"| {prompt} | {clean(model_names.get(key, key))} | {clean(response)} | "
                f"{stats.get('latency', 0.0):.2f}s | "
                f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)} | "
                f"${stats.get('cost', 0.0):.6f} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")