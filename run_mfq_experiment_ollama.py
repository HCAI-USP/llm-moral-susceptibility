#!/usr/bin/env python3
"""MFQ experiment runner that targets local Ollama models."""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import ollama
    from ollama import Client
    from ollama import ResponseError as OllamaResponseError
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Missing dependency: install the Ollama Python client with `pip install ollama`."
    ) from exc

from mfq_questions import iter_questions

# Map model names to custom CSV filenames (without path). Populate this mapping
# if you want specific Ollama models to write to a fixed filename instead of the
# auto-sanitized default.
CUSTOM_MODEL_FILENAMES: Dict[str, str] = {}

DEFAULT_BASE_URL = "http://localhost:11434"


def create_persona_prompt(persona: str, question: str) -> str:
    """Create a prompt that includes the persona and the MFQ question."""
    return f"""You are roleplaying as the following persona: {persona}

Please answer the following question from the perspective of this persona.

{question}"""

def extract_rating(response: str) -> int:
    """Extract numeric rating from LLM response"""
    try:
        number = re.search(r'\b[0-5]\b', response.strip())
        if number:
            return int(number.group())
        print(f"No valid rating found in response: {response}")
        return -1
    except Exception:
        print(f"Could not parse response: {response}")
        return -1


def resolve_model_filename(model_name: str, suffix: str = "", directory: Optional[Path] = None) -> Path:
    """Return the CSV path for the given model, applying any custom mapping."""

    base_dir = directory or Path("data")
    custom = CUSTOM_MODEL_FILENAMES.get(model_name)
    if custom:
        stem = custom[:-4] if custom.lower().endswith(".csv") else custom
    else:
        stem = model_name.replace(":", "_").replace("/", "_")

    filename = f"ollama__{stem}{suffix}.csv"
    return base_dir / filename


def list_models(client: Client) -> List[str]:
    """Return sorted Ollama model names known to the daemon."""
    # try:
    #     payload = client.list()
    # except Exception as exc:
    #     print(f"Could not list Ollama models: {exc}")
    #     return []

    # models = payload.get("models", []) if isinstance(payload, dict) else []
    # names: List[str] = []
    # for item in models:
    #     if not isinstance(item, dict):
    #         continue
    #     name = item.get("model") or item.get("name")
    #     if isinstance(name, str):
    #         names.append(name)
    # # Preserve order while deduplicating
    # deduped: List[str] = []
    # for name in names:
    #     if name not in deduped:
    #         deduped.append(name)
    # return deduped
    try:
        models = [m.model for m in ollama.list().models]
        return sorted(models)
    except Exception as exc:
        print(f"Could not list Ollama models: {exc}")
        return []


def prompt_for_model_selection(
    client: Client, base_url: str, cached_models: Optional[List[str]] = None
) -> str:
    """Prompt the user to pick an Ollama model from the running instance."""

    models = cached_models or list_models(client)
    if not models:
        raise SystemExit(
            dedent(
                f"""
                No Ollama models are available at {base_url}.
                Pull one with `ollama pull <model>` and retry.
                """
            ).strip()
        )

    print("Select the Ollama model to run:")
    for idx, model_name in enumerate(models, start=1):
        print(f"  {idx}. {model_name}")

    while True:
        choice = input("Enter the number (or exact name) of the model to use: ").strip()
        if not choice:
            continue

        # Allow entering an exact model name.
        if choice in models:
            return choice

        try:
            index = int(choice)
        except ValueError:
            print("Invalid selection. Please enter a valid number or model name.")
            continue

        if 1 <= index <= len(models):
            return models[index - 1]

        print("Invalid selection. Please enter a valid number or model name.")


def generate_ollama_response(
    client: Client,
    model_name: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a response from Ollama using a single-turn chat call."""
    try:
        result = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options=options or None,
        )
    except OllamaResponseError as exc:
        print(f"Ollama chat error: {exc}")
        return "ERROR"
    except Exception as exc:
        print(f"Ollama chat error: {exc}")
        return "ERROR"

    if isinstance(result, dict):
        message = result.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        for key in ("response", "output", "state"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return str(result).strip()


def run_mfq_experiment(
    client: Client,
    personas: List[str],
    model_name: str,
    *,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    existing_valid_slots: Optional[Set[Tuple[int, int, int]]] = None,
    collect_new_rows: bool = False,
    slot_failures: Optional[Dict[Tuple[int, int, int], int]] = None,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Run the MFQ experiment.

    - If ``csv_writer`` is supplied, rows are streamed directly to the CSV.
    - ``existing_valid_slots`` marks (persona_id, question_id, run_index) entries
      that already have valid (>=0) ratings and should be skipped.
    - When ``collect_new_rows`` is True, the function returns any newly
      generated rows so the caller can handle persistence (e.g., rewrite files).
    - ``slot_failures`` tracks how many invalid attempts (rating -1) have been
      recorded for each (persona, question, run_index) slot.
    """

    if csv_writer is None and not collect_new_rows and row_callback is None:
        raise ValueError(
            "run_mfq_experiment requires a csv_writer unless collect_new_rows or row_callback is provided"
        )

    questions = list(iter_questions())

    personas_processed = 0
    responses_written = 0
    existing_valid_slots = existing_valid_slots or set()
    slot_failures = slot_failures or {}
    new_rows: List[Dict[str, Any]] = []

    print(f"Running MFQ experiment with {len(personas)} personas using ollama:{model_name}")

    for persona_id, persona in enumerate(personas):
        persona_text = str(persona)
        print(f"\nProgress: {persona_id + 1}/{len(personas)} - {persona_text[:50]}...")
        personas_processed += 1

        for question in questions:
            prompt = create_persona_prompt(persona_text, question.prompt)

            for run_index in range(1, n + 1):
                slot_key = (persona_id, question.id, run_index)
                if slot_key in existing_valid_slots:
                    continue

                response = generate_ollama_response(client, model_name, prompt, options)
                rating = extract_rating(response)
                response_text = response.strip() if isinstance(response, str) else str(response)

                prior_failures = slot_failures.get(slot_key, 0)
                failures = prior_failures + (1 if rating < 0 else 0)

                row = {
                    "persona_id": persona_id,
                    "question_id": question.id,
                    "run_index": run_index,
                    "rating": rating,
                    "failures": failures,
                    "response": response_text,
                    "collected_at": datetime.now().isoformat(),
                }

                if csv_writer is not None:
                    csv_writer.writerow(row)
                    responses_written += 1
                    if csv_file is not None:
                        csv_file.flush()
                else:
                    responses_written += 1

                slot_failures[slot_key] = failures

                if row_callback is not None:
                    row_callback(dict(row))

                if collect_new_rows:
                    # Store a copy to avoid accidental mutation
                    new_rows.append(dict(row))

                if rating >= 0:
                    existing_valid_slots.add(slot_key)

    return personas_processed, responses_written, new_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MFQ experiments with personas using local Ollama models."
    )
    parser.add_argument(
        "--personas-file",
        type=Path,
        default=Path("personas.json"),
        help="Path to the personas JSON file (default: personas.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit the number of personas to test (default: 100).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of times each persona answers each question (default: 10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name to use (skips interactive selection).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Ollama server base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature passed to Ollama (default: 0.1).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Maximum tokens to sample per response (default: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for Ollama requests (default: 60).",
    )

    args = parser.parse_args()

    personas_path = args.personas_file
    try:
        personas = json.loads(personas_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: Could not find personas file at {personas_path}")
        return
    except json.JSONDecodeError as exc:
        print(f"Error: Failed to parse personas file {personas_path}: {exc}")
        return

    if not isinstance(personas, list):
        print("Error: personas file must contain a JSON array of persona descriptions.")
        return

    if args.limit:
        personas = personas[: args.limit]

    if not personas:
        print("No personas to run after applying the limit.")
        return

    print(f"Loaded {len(personas)} personas")

    base_url = args.base_url.rstrip("/")
    try:
        client = Client(host=base_url, timeout=args.timeout)
    except TypeError:
        # Older versions of the client do not accept timeout; fall back gracefully.
        client = Client(host=base_url)
    except Exception as exc:
        print(f"Failed to initialise Ollama client at {base_url}: {exc}")
        return

    available_models = list_models(client)

    if args.model:
        model_name = args.model.strip()
        if model_name not in available_models:
            if available_models:
                print("The requested model is not available. Models currently pulled:")
                for name in available_models:
                    print(f"  - {name}")
            else:
                print(
                    f"The requested model '{model_name}' is not available and no models were found on {base_url}."
                )
            return
    else:
        model_name = prompt_for_model_selection(client, base_url, cached_models=available_models)

    print(f"Selected model: ollama:{model_name} @ {base_url}")

    ollama_options = {
        "temperature": args.temperature,
        "num_predict": args.max_tokens,
    }

    output_path = resolve_model_filename(model_name)
    file_exists = output_path.exists()

    fieldnames = [
        "persona_id",
        "question_id",
        "run_index",
        "rating",
        "failures",
        "response",
        "collected_at",
    ]

    existing_rows: List[Dict[str, Any]] = []
    existing_valid_slots: Set[Tuple[int, int, int]] = set()
    slot_failures: Dict[Tuple[int, int, int], int] = {}
    rows_by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    had_missing_failures = False

    if file_exists:
        try:
            with open(output_path, "r", newline="", encoding="utf-8") as existing_file:
                reader = csv.DictReader(existing_file)
                for row in reader:
                    try:
                        persona_id = int(row["persona_id"])
                        question_id = int(row["question_id"])
                        run_index = int(row["run_index"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    rating_value = row.get("rating", -1)
                    try:
                        rating = int(rating_value)
                    except (TypeError, ValueError):
                        rating = -1

                    raw_failures = row.get("failures")
                    if raw_failures in (None, ""):
                        failures = 0
                        had_missing_failures = True
                    else:
                        try:
                            failures = int(raw_failures)
                        except (TypeError, ValueError):
                            failures = 0
                            had_missing_failures = True

                    if rating < 0 and failures <= 0:
                        failures = 1

                    row_dict = {
                        "persona_id": persona_id,
                        "question_id": question_id,
                        "run_index": run_index,
                        "rating": rating,
                        "failures": failures,
                        "response": row.get("response", ""),
                        "collected_at": row.get("collected_at", ""),
                    }

                    existing_rows.append(row_dict)
                    rows_by_key[(persona_id, question_id, run_index)] = row_dict

                    if rating >= 0:
                        existing_valid_slots.add((persona_id, question_id, run_index))

                    slot_failures[(persona_id, question_id, run_index)] = failures

            if existing_valid_slots:
                print(
                    f"Found {len(existing_valid_slots)} previously completed slots. Only missing or invalid entries will be re-run."
                )
        except FileNotFoundError:
            file_exists = False

    if file_exists:
        def write_rows_to_disk() -> None:
            if not rows_by_key:
                return
            tmp_path = output_path.parent / f"{output_path.name}.tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as tmp_file:
                writer = csv.DictWriter(tmp_file, fieldnames=fieldnames)
                writer.writeheader()
                for key in sorted(rows_by_key.keys()):
                    writer.writerow(rows_by_key[key])
            os.replace(tmp_path, output_path)

        def handle_new_row(row: Dict[str, Any]) -> None:
            key = (row["persona_id"], row["question_id"], row["run_index"])
            rows_by_key[key] = row
            slot_failures[key] = row.get("failures", 0)
            write_rows_to_disk()

        personas_processed, responses_written, _ = run_mfq_experiment(
            client,
            personas,
            model_name,
            n=args.n,
            csv_writer=None,
            csv_file=None,
            existing_valid_slots=set(existing_valid_slots),
            collect_new_rows=False,
            slot_failures=slot_failures,
            row_callback=handle_new_row,
            options=ollama_options,
        )

        if responses_written == 0 and had_missing_failures and rows_by_key:
            write_rows_to_disk()

    else:
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            personas_processed, responses_written, _ = run_mfq_experiment(
                client,
                personas,
                model_name,
                n=args.n,
                csv_writer=writer,
                csv_file=csv_file,
                existing_valid_slots=None,
                collect_new_rows=False,
                slot_failures=slot_failures,
                options=ollama_options,
            )

    if file_exists and responses_written == 0:
        print("\nNo new runs were required; all slots were already filled with valid ratings.")

    print(
        f"\nExperiment completed! Processed {personas_processed} personas and logged {responses_written} responses to {output_path}."
    )

if __name__ == "__main__":
    main()
