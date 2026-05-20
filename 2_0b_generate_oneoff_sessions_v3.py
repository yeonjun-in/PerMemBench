"""
2_0b_generate_oneoff_sessions_v2.py

Stage 2b: Generate one-off session events for memory_required=False domains.

For each persona's memory_required=False domains, generates a set of independent,
self-contained events — each representing a single AI agent interaction with no
continuity requirements across sessions.

Unlike memory_required=True domains (which have longitudinal project structures),
these events are:
  - Independent of each other (no cross-session continuity)
  - Self-contained within a single conversation
  - Realistic one-time requests grounded in the persona's life

Number of events per domain is derived from the actual timeline duration estimated
from the memory_required=True domain skeletons, using empirically grounded
interaction frequencies:

  high   -> once every 4 weeks
  medium -> once every 8 weeks
  low    -> once every 12 weeks

Timeline duration is estimated from the skeleton's project structure (same logic
as 2_1_integrate_timeline_v2.py) so that 2_0b can run before 2_1 without
needing the final integrated timeline.

Pipeline position:
  2_0  -> life_skeletons/{uuid}.json       (memory_required=True domains)
  2_0b -> injects oneoff_sessions[] into life_skeletons/{uuid}.json  <- this script
  2_1  -> integrates both into unified timeline

Iteration target: skeleton files in --skeleton_dir (only processes UUIDs that 2_0 completed).
Persona metadata is looked up from --persona_dir for domain info (memory_required, frequency).

Usage:
  # single skeleton file
  python 2_0b_generate_oneoff_sessions_v2.py \\
      --skeleton_file ./life_skeletons/0a0dcec0.json \\
      --persona_dir ./final_persona_metadata_v3

  # full skeleton directory
  python 2_0b_generate_oneoff_sessions_v2.py \\
      --skeleton_dir ./life_skeletons \\
      --persona_dir ./final_persona_metadata_v3 \\
      --provider openai --model gpt-4o-mini
"""

import json
import os
import argparse
import random
from pathlib import Path
from glob import glob
from tqdm import tqdm
import sys, math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM import UnifiedLLM


# ============================================================
# CONSTANTS
# ============================================================

# Interaction frequency: how often the user turns to AI in this domain.
# Expressed as weeks between sessions, grounded in realistic incidental usage.
#   high   -> once every 4 weeks  (e.g., active hobbyist, frequent traveler)
#   medium -> once every 8 weeks  (e.g., occasional cook, periodic DIY project)
#   low    -> once every 12 weeks (e.g., rare one-off need)
WEEKS_PER_SESSION = {
    "high":   4,
    "medium": 8,
    "low":    12,
}

WEEKS_PER_MONTH = 4  # average weeks per calendar month

# Fallback if frequency label is missing or unrecognized
DEFAULT_WEEKS_PER_SESSION = 8

# Max domains to process per persona (keeps timeline manageable)
MAX_ONEOFF_DOMAINS = 4


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are designing one-off AI agent interaction events for a personalized memory benchmark.

## Context
These events represent self-contained, single-session interactions where a user turns to an
AI agent for a specific need — and then never needs to follow up on it again. There is no
longitudinal engagement, no project structure, and no memory required across sessions.

## What makes a good one-off event
- Grounded specifically in THIS user's life situation, personality, and circumstances
- A realistic, concrete moment when the user would open an AI agent
- Self-contained: the user gets what they need and moves on
- Diverse: each event in the same domain should cover a DIFFERENT situation or need
- Not artificially padded: if the user would genuinely ask this once, that is enough

## What to avoid
- Generic events that could apply to anyone (e.g., "looking up a recipe")
- Events that imply follow-up or continuity ("starting a new project", "beginning a journey")
- Repetitive events covering the same scenario within the same domain
- Events that contradict the persona's actual interests or circumstances
"""

ONEOFF_PROMPT = """\
## User Persona
{persona}

## Domain
- Name      : {domain_name}
- Frequency : {frequency}
- Why used  : {reason}

## Task
Generate exactly {n_events} one-off interaction events for this user in the "{domain_name}" domain.

These events will be spread across a ~{total_months}-month period, so they should reflect
genuinely different moments and needs across that timeframe — not the same situation repeated.
The user interacts with an AI agent in this domain roughly once every {interval_weeks} weeks,
so make sure the events feel naturally spaced and varied in their triggers.

### Requirements per event
- event_title       : Short (4-8 words), specific to the situation
- event_description : 2-3 sentences. What is happening in the user's life right now that
                      triggers this specific request? What does the user need from the agent?
                      Be concrete and grounded in the persona.

### Output Format (strict JSON, no trailing commas)
{{
  "domain_name": "{domain_name}",
  "events": [
    {{
      "event_id": 1,
      "event_title": "Short specific title",
      "event_description": "2-3 sentences grounding the request in this user's life."
    }},
    ...
  ]
}}

Return ONLY the JSON object. No explanation before or after.
"""


# ============================================================
# HELPERS
# ============================================================

def format_persona(persona_dict: dict) -> str:
    skip_keys = {"uuid"}
    lines = []
    for key, value in persona_dict.items():
        if key in skip_keys:
            continue
        lines.append(f"[{key}]\n{value}")
    return "\n\n".join(lines)


def estimate_total_months(skeleton_data: dict) -> int:
    """
    Estimate total timeline duration in months from the memory_required=True
    domain skeletons. Uses the same heuristic as 2_1_integrate_timeline_v2.py:
    sum of (project count × 3 months per project), clamped to [12, 24].
    """
    domain_skeletons = skeleton_data.get("domain_skeletons", [])
    total = 0
    for ds in domain_skeletons:
        total += len(ds.get("skeleton", {}).get("projects", [])) * 3
    return min(max(total, 12), 24)


def get_session_count(frequency: str, total_months: int) -> int:
    """
    Derive number of one-off sessions from timeline duration and interaction frequency.

    Formula: n = total_weeks / weeks_per_session
    where total_weeks = total_months × 4.33
    """
    total_weeks = total_months * WEEKS_PER_MONTH
    interval_weeks = WEEKS_PER_SESSION.get(frequency, DEFAULT_WEEKS_PER_SESSION)
    return max(1, math.floor(total_weeks / interval_weeks))


def generate_oneoff_events(
    llm: "UnifiedLLM",
    persona: dict,
    domain: dict,
    total_months: int,
) -> dict:
    """
    Generate one-off events for a single memory_required=False domain.
    Returns the parsed JSON with domain_name and events list.
    """
    persona_text = format_persona(persona)
    frequency = domain.get("frequency", "medium")
    interval_weeks = WEEKS_PER_SESSION.get(frequency, DEFAULT_WEEKS_PER_SESSION)
    n_events = get_session_count(frequency, total_months)

    prompt = ONEOFF_PROMPT.format(
        persona=persona_text,
        domain_name=domain["domain_name"],
        frequency=frequency,
        reason=domain.get("reason", ""),
        n_events=n_events,
        total_months=total_months,
        interval_weeks=interval_weeks,
    )

    raw = llm.chat(prompt, system=SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(cleaned)

    # Inject metadata fields
    result["memory_required"] = False
    result["frequency"] = frequency
    result["interval_weeks"] = interval_weeks
    result["n_events_expected"] = n_events

    return result


def process_skeleton(
    skeleton_path: str,
    persona_dir: str,
    llm: "UnifiedLLM",
    overwrite: bool = False,
) -> dict:
    """
    For one skeleton file:
      1. Load skeleton -> estimate total_months from project structure
      2. Look up persona metadata -> extract memory_required=False domains
      3. Generate one-off events per domain (count derived from total_months)
      4. Inject into the skeleton file as oneoff_sessions[]
    """
    # Load existing skeleton
    with open(skeleton_path, encoding="utf-8") as f:
        skeleton_data = json.load(f)

    uuid = skeleton_data["uuid"]

    # Skip if already processed and overwrite not set
    if not overwrite and skeleton_data.get("oneoff_sessions") is not None:
        return {"uuid": uuid, "skipped": True, "reason": "already has oneoff_sessions"}

    # Estimate timeline duration from memory_required=True domain structure
    total_months = estimate_total_months(skeleton_data)

    # Look up persona metadata for domain info
    persona_path = Path(persona_dir) / f"{uuid}.json"
    if not persona_path.exists():
        return {"uuid": uuid, "skipped": True, "reason": "no persona metadata file found"}

    with open(persona_path, encoding="utf-8") as f:
        persona_data = json.load(f)

    persona = persona_data["persona"]

    # Filter: use=True, memory_required=False, frequency not null
    oneoff_domains = [
        d for d in persona_data.get("domains", [])
        if d.get("use") is True
        and d.get("memory_required") is False
        and d.get("frequency") is not None
    ]
    random.shuffle(oneoff_domains)
    oneoff_domains = oneoff_domains[:MAX_ONEOFF_DOMAINS]

    if not oneoff_domains:
        skeleton_data["oneoff_sessions"] = []
        with open(skeleton_path, "w", encoding="utf-8") as f:
            json.dump(skeleton_data, f, ensure_ascii=False, indent=2)
        return {"uuid": uuid, "total_months": total_months, "n_domains": 0, "n_events": 0}

    results = []
    errors = []
    total_events = 0

    for domain in tqdm(oneoff_domains, desc=f"  {uuid[:8]} oneoff", leave=False):
        try:
            domain_result = generate_oneoff_events(llm, persona, domain, total_months)
            n = len(domain_result.get("events", []))
            total_events += n
            results.append(domain_result)
        except Exception as e:
            errors.append({
                "domain_name": domain["domain_name"],
                "error": str(e),
            })
            print(f"    ✗ {domain['domain_name']}: {e}")

    # Inject into skeleton file
    skeleton_data["oneoff_sessions"] = results
    if errors:
        skeleton_data["oneoff_errors"] = errors

    with open(skeleton_path, "w", encoding="utf-8") as f:
        json.dump(skeleton_data, f, ensure_ascii=False, indent=2)

    return {
        "uuid": uuid,
        "total_months": total_months,
        "n_domains": len(results),
        "n_events": total_events,
        "n_errors": len(errors),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2b: Generate one-off events for memory_required=False domains"
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--skeleton_file", type=str, default=None,
                             help="Path to a single life skeleton JSON file (2_0 output)")
    input_group.add_argument("--skeleton_dir", type=str, default='life_skeletons_v5_clean',
                             help="Directory containing life skeleton JSON files (2_0 output)")

    parser.add_argument("--persona_dir", type=str, default='final_persona_metadata_v3',
                        help="Directory containing persona metadata JSON files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of skeleton files to process")
    parser.add_argument("--provider", type=str, default="openai",
                        help="LLM provider: openai | claude | together | gemini")
    parser.add_argument("--model", type=str, default="gpt-5.4",
                        help="Model name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing oneoff_sessions in skeleton files")

    args = parser.parse_args()

    llm = UnifiedLLM(args.provider, args.model)

    if args.skeleton_file:
        files = [args.skeleton_file]
    else:
        files = sorted(glob(os.path.join(args.skeleton_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]

    if args.limit:
        files = files[:args.limit]

    skeleton_label = args.skeleton_dir or args.skeleton_file
    print(f"Provider      : {args.provider} / {args.model}")
    print(f"Skeleton      : {skeleton_label}")
    print(f"Persona dir   : {args.persona_dir}")
    print(f"Files         : {len(files)}")
    print(f"Max domains   : {MAX_ONEOFF_DOMAINS}")
    print(f"Frequency     : high={WEEKS_PER_SESSION['high']}w | "
          f"medium={WEEKS_PER_SESSION['medium']}w | "
          f"low={WEEKS_PER_SESSION['low']}w\n")

    success = skipped = errors = 0
    total_events = 0

    for filepath in tqdm(files, desc="Skeletons"):
        try:
            result = process_skeleton(
                skeleton_path=filepath,
                persona_dir=args.persona_dir,
                llm=llm,
                overwrite=args.overwrite,
            )
            if result.get("skipped"):
                skipped += 1
                print(f"  skip {result['uuid'][:8]}: {result.get('reason', 'skipped')}")
            else:
                success += 1
                total_events += result.get("n_events", 0)
                n_err = result.get("n_errors", 0)
                months = result.get("total_months", "?")
                print(
                    f"  ok   {result['uuid'][:8]}: "
                    f"~{months}mo timeline → "
                    f"{result['n_domains']} domains, "
                    f"{result['n_events']} events"
                    + (f"  [{n_err} errors]" if n_err else "")
                )
        except Exception as e:
            errors += 1
            print(f"  err  {filepath}: {e}")

    print(f"\nDone -- success: {success}, skipped: {skipped}, errors: {errors}")
    print(f"Total one-off events generated: {total_events}")


if __name__ == "__main__":
    main()
