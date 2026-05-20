"""
2_1_integrate_timeline_v3.py

Stage 2: Cross-domain timeline integration.

Takes the per-domain Life Skeletons (output of 2_0_generate_life_skeleton_v4.py) and
one-off session events (injected by 2_0b_generate_oneoff_sessions_v3.py), then produces
a single unified session timeline.

Design:
  - LLM is responsible ONLY for placing memory_required=True events on the timeline.
    It handles: anchor life event detection, cross-domain linking, project sequentiality,
    and realistic interleaving across longitudinal domains.
  - memory_required=False (one-off) sessions are placed programmatically after the LLM
    call, using frequency-based sequential placement derived from interval_weeks stored
    in each oneoff domain block (set by 2_0b).

One-off session placement:
  For each domain block, events are placed starting at month=interval_months,
  incrementing by interval_months each time. Events whose computed month exceeds
  the actual total_months from the LLM timeline are silently dropped. This means
  the final session count naturally adapts to the real timeline duration without
  requiring any re-generation.

Pipeline position:
  2_0  -> life_skeletons/{uuid}.json
  2_0b -> injects oneoff_sessions[] into life_skeletons/{uuid}.json
  2_1  -> life_timelines/{uuid}.json   <- this script

Usage:
    # single file
    python 2_1_integrate_timeline_v3.py \\
        --input_file ./life_skeletons/0a0dcec0.json \\
        --output_dir ./life_timelines

    # full directory
    python 2_1_integrate_timeline_v3.py \\
        --input_dir ./life_skeletons \\
        --output_dir ./life_timelines

    # with specific model
    python 2_1_integrate_timeline_v3.py \\
        --input_dir ./life_skeletons \\
        --output_dir ./life_timelines \\
        --provider openai --model gpt-4o
"""

import json
import os
import argparse
from glob import glob
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM import UnifiedLLM


# ============================================================
# CONSTANTS
# ============================================================

WEEKS_PER_MONTH = 4.33          # average weeks per calendar month
DEFAULT_INTERVAL_WEEKS = 8      # fallback if interval_weeks missing (medium frequency)


# ============================================================
# PROMPTS  (memory_required=True sessions only)
# ============================================================

SYSTEM_PROMPT = """\
You are designing an integrated session timeline for a personalized AI agent memory benchmark.

You will be given a user persona and a collection of per-domain Life Skeletons.
Each skeleton contains a sequence of Projects and Events that describe how this user
engages with an AI agent in that domain over 1-2 years.

Your job is to arrange all events into a single, realistic session
timeline — a chronologically ordered list of agent sessions that this person would
actually have, given their real life circumstances.

## Key Principles

1. **Ground in the persona's life.**
   Events across different domains often stem from the same real-world situation.
   For example, deciding to pursue a GED would simultaneously trigger sessions in
   Career Development, Academic Study, and Personal Finance.
   Identify these "anchor life events" and cluster related domain sessions around them.

2. **Respect project sequentiality within each domain.**
   Within a single domain, Project 2 can only begin after Project 1 ends.
   Do not reorder events within a project.

3. **Interleave domains realistically.**
   The user does not finish all sessions in Domain A before starting Domain B.
   They use different domains at different times based on what is happening in their life.

4. **Assign concrete month numbers.**
   Use integer months (1, 2, 3, ..., up to ~24).
   Multiple sessions can share the same month — that is realistic.
   Ensure the overall spread feels natural given the persona's pace and circumstances.

5. **Flag cross-domain links.**
   When a session is causally or contextually connected to a session in another domain
   (both triggered by the same life event), record that link explicitly.

## Output Format (strict JSON, no trailing commas)

{
  "total_months": <integer, total span of the timeline>,
  "anchor_life_events": [
    {
      "event_name": "Short name for the real-world trigger",
      "description": "1 sentence: what happened and why it triggers multiple domains",
      "month": <integer>,
      "triggered_sessions": [
        {"domain": "...", "project_id": <int>, "event_id": <int>}
      ]
    }
  ],
  "sessions": [
    {
      "session_id": <integer, 1-indexed, sequential>,
      "month": <integer>,
      "domain": "<domain name>",
      "memory_required": true,
      "project_id": <integer>,
      "event_id": <integer>,
      "event_title": "<copy from skeleton>",
      "event_description": null,
      "anchor_life_event": "<name of anchor life event this belongs to, or null>",
      "cross_domain_links": [
        {"domain": "...", "project_id": <int>, "event_id": <int>}
      ]
    }
  ]
}

Return ONLY the JSON object. No explanation before or after.
"""

INTEGRATION_PROMPT = """\
## User Persona
{persona}

## Domain Life Skeletons (memory_required=True)
{skeletons}

## Task
Produce a unified, chronologically ordered session timeline for this user across all domains.

### Events (every one must appear exactly once in the sessions array):
{event_inventory}

### Requirements
- Every event listed above must appear exactly once in the sessions array.
- Set memory_required=true, include project_id and event_id for each session.
- Set event_description to null (downstream code will look it up from the skeleton).
- Assign a realistic month (1-{max_month}) to each session.
- Identify anchor life events that trigger sessions across multiple domains simultaneously.
- Respect project ordering within each domain (project N must precede project N+1).
- session_id must be sequential (1, 2, 3, ...) in chronological order.
- cross_domain_links should reference sessions triggered by the same anchor life event.

Return ONLY the JSON object described above.
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


def build_skeleton_summary(domain_skeletons: list) -> str:
    """Compact summary of all memory_required=True domain skeletons for the prompt."""
    lines = []
    for ds in domain_skeletons:
        domain = ds["domain_name"]
        freq = ds.get("frequency", "medium")
        lines.append(f"\n### Domain: {domain} (frequency: {freq})")
        for proj in ds["skeleton"].get("projects", []):
            lines.append(
                f"  Project {proj['project_id']}: {proj['title']} "
                f"[{proj.get('approximate_duration', '?')}]"
            )
            lines.append(f"    {proj.get('description', '')}")
            for evt in proj.get("events", []):
                lines.append(f"    Event {evt['event_id']}: {evt['title']}")
                lines.append(f"      {evt.get('description', '')}")
    return "\n".join(lines)


def build_event_inventory(domain_skeletons: list) -> str:
    """Flat list of all (domain, project_id, event_id, title) for memory_required=True events."""
    lines = ["domain | project_id | event_id | event_title"]
    lines.append("-" * 70)
    for ds in domain_skeletons:
        domain = ds["domain_name"]
        for proj in ds["skeleton"].get("projects", []):
            for evt in proj.get("events", []):
                lines.append(
                    f"{domain} | {proj['project_id']} | {evt['event_id']} | {evt['title']}"
                )
    return "\n".join(lines)


def estimate_max_month(domain_skeletons: list) -> int:
    """Rough upper bound: sum of all project durations, capped at 24."""
    total = 0
    for ds in domain_skeletons:
        total += len(ds["skeleton"].get("projects", [])) * 3
    return min(max(total, 12), 24)


def count_events(domain_skeletons: list) -> int:
    total = 0
    for ds in domain_skeletons:
        for proj in ds["skeleton"].get("projects", []):
            total += len(proj.get("events", []))
    return total


def count_oneoff_events(oneoff_sessions: list) -> int:
    total = 0
    for domain_block in oneoff_sessions:
        total += len(domain_block.get("events", []))
    return total


def merge_oneoff_sessions(timeline: dict, oneoff_sessions: list) -> dict:
    """
    Programmatically place memory_required=False one-off sessions into the timeline
    using frequency-based sequential placement, then re-sort and reassign session_ids.

    Placement logic per domain:
      - interval_months = interval_weeks / WEEKS_PER_MONTH  (from domain_block)
      - Place first event at month = round(interval_months)
      - Each subsequent event: month += round(interval_months)  [cumulative]
      - If month > total_months: TRUNCATE — drop this and all remaining events

    total_months comes from the actual LLM-generated timeline, NOT the pre-estimated
    value from 2_0b. Events generated in 2_0b that fall outside the real timeline
    are silently dropped, so the final session count naturally adapts to the actual
    timeline duration without requiring re-generation.

    Fallback: if interval_weeks is missing, defaults to DEFAULT_INTERVAL_WEEKS (8w).
    """
    if not oneoff_sessions:
        return timeline

    total_months = timeline.get("total_months", 24)
    existing_sessions = timeline.get("sessions", [])

    oneoff_flat = []
    for domain_block in oneoff_sessions:
        domain_name = domain_block["domain_name"]
        events = domain_block.get("events", [])
        if not events:
            continue

        interval_weeks = domain_block.get("interval_weeks", DEFAULT_INTERVAL_WEEKS)
        interval_months = interval_weeks / WEEKS_PER_MONTH

        placed = 0
        current_month = interval_months   # first event lands at 1 × interval

        for evt in events:
            month = max(1, round(current_month))
            if month > total_months:
                # Timeline exhausted — truncate remaining events for this domain
                dropped = len(events) - placed
                print(f"    [{domain_name}] placed {placed}, "
                      f"dropped {dropped} (month {month} > {total_months}mo timeline)")
                break

            oneoff_flat.append({
                "session_id": -1,       # placeholder; reassigned below
                "month": month,
                "domain": domain_name,
                "memory_required": False,
                "project_id": None,
                "event_id": evt["event_id"],
                "event_title": evt["event_title"],
                "event_description": evt["event_description"],
                "anchor_life_event": None,
                "cross_domain_links": [],
            })
            placed += 1
            current_month += interval_months

    # Merge with memory_required=True sessions
    # Within the same month: memory_required=True sessions come first
    merged = existing_sessions + oneoff_flat
    merged.sort(key=lambda s: (
      s["month"],
        0 if s["memory_required"] else 1,
        s["session_id"] if s["memory_required"] else 0,  # LLM order tiebreaker
    ))

    # Reassign sequential session_ids
    for idx, session in enumerate(merged, start=1):
        session["session_id"] = idx

    timeline["sessions"] = merged
    return timeline


def validate_timeline(
    timeline: dict,
    domain_skeletons: list,
) -> list:
    """
    Validates memory_required=True sessions only:
    - All expected events are present (no missing, no duplicates)
    - Project order is respected within each domain
    """
    warnings = []

    expected = set()
    for ds in domain_skeletons:
        for proj in ds["skeleton"].get("projects", []):
            for evt in proj.get("events", []):
                expected.add((ds["domain_name"], proj["project_id"], evt["event_id"]))

    sessions = timeline.get("sessions", [])
    seen = set()

    for s in sessions:
        if not s.get("memory_required", True):
            continue
        key = (s["domain"], s["project_id"], s["event_id"])
        if key in seen:
            warnings.append(f"Duplicate memory_required=True session: {key}")
        seen.add(key)

    for m in expected - seen:
        warnings.append(f"Missing memory_required=True session: {m}")
    for e in seen - expected:
        warnings.append(f"Unknown memory_required=True session (not in skeleton): {e}")

    # Check project order per domain
    domain_proj_months: dict = {}
    for s in sessions:
        if not s.get("memory_required", True):
            continue
        d = s["domain"]
        p = s["project_id"]
        m = s["month"]
        domain_proj_months.setdefault(d, {}).setdefault(p, []).append(m)

    for domain, proj_map in domain_proj_months.items():
        proj_ids = sorted(proj_map.keys())
        for i in range(len(proj_ids) - 1):
            p_cur = proj_ids[i]
            p_next = proj_ids[i + 1]
            max_cur = max(proj_map[p_cur])
            min_next = min(proj_map[p_next])
            if max_cur > min_next:
                warnings.append(
                    f"Project order violated in {domain}: "
                    f"Project {p_cur} (max month {max_cur}) overlaps "
                    f"Project {p_next} (min month {min_next})"
                )

    return warnings


def generate_timeline(
    llm: UnifiedLLM,
    persona: dict,
    domain_skeletons: list,
) -> dict:
    """Call LLM to place memory_required=True events on a timeline."""
    persona_text = format_persona(persona)
    skeleton_summary = build_skeleton_summary(domain_skeletons)
    event_inventory = build_event_inventory(domain_skeletons)
    max_month = estimate_max_month(domain_skeletons)

    prompt = INTEGRATION_PROMPT.format(
        persona=persona_text,
        skeletons=skeleton_summary,
        event_inventory=event_inventory,
        max_month=max_month,
    )

    raw = llm.chat(prompt, system=SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def process_skeleton_file(
    filepath: str,
    llm: UnifiedLLM,
    output_dir: str,
    overwrite: bool = False,
) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    uuid = data["uuid"]
    output_path = os.path.join(output_dir, f"{uuid}.json")

    if not overwrite and os.path.exists(output_path):
        return {"uuid": uuid, "skipped": True}

    persona = data["persona"]
    domain_skeletons = data.get("domain_skeletons", [])
    oneoff_sessions = data.get("oneoff_sessions", [])

    valid_skeletons = [
        ds for ds in domain_skeletons
        if ds.get("skeleton") and ds["skeleton"].get("projects")
    ]
    if not valid_skeletons:
        return {"uuid": uuid, "skipped": True, "reason": "no valid skeletons"}

    n_true = count_events(valid_skeletons)
    n_false_generated = count_oneoff_events(oneoff_sessions)
    print(f"  {uuid[:8]}: {len(valid_skeletons)} domains ({n_true} mem-required) | "
          f"{n_false_generated} one-off events generated")

    # Step 1: LLM places memory_required=True events
    try:
        timeline = generate_timeline(llm, persona, valid_skeletons)
    except Exception as e:
        return {"uuid": uuid, "error": str(e)}

    actual_months = timeline.get("total_months", "?")
    print(f"  {uuid[:8]}: LLM produced {actual_months}-month timeline")

    # Step 2: validate memory_required=True placement
    warnings = validate_timeline(timeline, valid_skeletons)
    if warnings:
        print(f"  Validation warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")

    # Step 3: programmatically merge one-off sessions
    # Uses actual total_months from LLM output; truncates events beyond timeline
    timeline = merge_oneoff_sessions(timeline, oneoff_sessions)

    n_false_placed = sum(
        1 for s in timeline.get("sessions", []) if not s.get("memory_required", True)
    )
    total_sessions = len(timeline.get("sessions", []))
    print(f"  {uuid[:8]}: {total_sessions} total sessions "
          f"({n_true} mem-req + {n_false_placed}/{n_false_generated} one-off placed) "
          f"over {actual_months} months")

    result = {
        "uuid": uuid,
        "persona": persona,
        "domain_skeletons": valid_skeletons,
        "oneoff_sessions": oneoff_sessions,
        "timeline": timeline,
        "validation_warnings": warnings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2.1: Integrate life skeletons into a unified timeline"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to a single life skeleton JSON file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing life skeleton JSON files")
    parser.add_argument("--output_dir", type=str, default="./life_timelines_v5",
                        help="Output directory (default: ./life_timelines)")
    parser.add_argument("--provider", type=str, default="openai",
                        help="LLM provider: openai | claude | together | gemini")
    parser.add_argument("--model", type=str, default='gpt-5.4',
                        help="Model name override (uses provider default if omitted)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("Provide --input_file or --input_dir")

    os.makedirs(args.output_dir, exist_ok=True)
    llm = UnifiedLLM(args.provider, args.model)

    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(glob(os.path.join(args.input_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]

    print(f"Provider : {args.provider} / {args.model or 'default'}")
    print(f"Files    : {len(files)}")
    print(f"Output   : {args.output_dir}\n")

    success, skipped, errors = 0, 0, 0
    for filepath in tqdm(files, desc="Personas"):
        try:
            result = process_skeleton_file(filepath, llm, args.output_dir, args.overwrite)
            if result.get("skipped"):
                skipped += 1
            elif result.get("error"):
                errors += 1
                print(f"  ✗ {result['uuid'][:8]}: {result['error']}")
            else:
                success += 1
        except Exception as e:
            errors += 1
            print(f"  ✗ {filepath}: {e}")

    print(f"\nDone — success: {success}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()
