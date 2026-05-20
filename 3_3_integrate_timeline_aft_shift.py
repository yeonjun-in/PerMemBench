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

WEEKS_PER_MONTH = 4.33
DEFAULT_INTERVAL_WEEKS = 8


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are designing a Phase 2 session timeline for a personalized AI agent memory benchmark.

The user has gone through a major life transition and is now in a new phase of their life.
You will receive their Phase 2 domain skeletons and must arrange all events into a realistic
chronological timeline spanning ~12 months.

## Key Principles
1. Ground placements in the user's post-transition life situation.
2. Respect project sequentiality within each domain (Project N before N+1).
3. Interleave domains realistically — the user uses different domains at different times.
4. Identify anchor life events that trigger multiple domains simultaneously.
5. session_id must be sequential (1, 2, ...) within Phase 2 only
   (final IDs will be offset to continue from Phase 1 by downstream code).
6. Months are relative to Phase 2 start (month 1 = first month after transition).

## Output Format (strict JSON, no trailing commas)
{
  "total_months": <integer, actual Phase 2 span — should be ~12>,
  "anchor_life_events": [
    {
      "event_name": "Short name",
      "description": "1 sentence",
      "month": <integer, relative to Phase 2>,
      "triggered_sessions": [
        {"domain": "...", "project_id": <int>, "event_id": <int>}
      ]
    }
  ],
  "sessions": [
    {
      "session_id": <integer, 1-indexed within Phase 2>,
      "month": <integer, 1-indexed relative to Phase 2 start>,
      "domain": "<domain name>",
      "memory_required": true,
      "project_id": <integer>,
      "event_id": <integer>,
      "event_title": "<copy from skeleton>",
      "event_description": null,
      "anchor_life_event": "<name or null>",
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

## Life Transition Event
{transition_event}

## Phase 2 Domain Skeletons (memory_required=True)
{skeletons}

## Task
Produce a Phase 2 timeline (months 1–{max_month}, relative to the transition).

### Events to place (each must appear exactly once):
{event_inventory}

### Requirements
- Every event listed above must appear exactly once.
- session_id: sequential within Phase 2, starting from 1.
- month: relative to Phase 2 start (1 = first month after transition).
- Respect project ordering within each domain.
- memory_required=true for all events here.
- event_description: set to null.

Return ONLY the JSON object.
"""


# ============================================================
# HELPERS (from 2_1, adapted for Phase 2)
# ============================================================

def format_persona(persona_dict: dict) -> str:
    skip_keys = {"uuid"}
    lines = []
    for key, value in persona_dict.items():
        if key in skip_keys:
            continue
        lines.append(f"[{key}]\n{value}")
    return "\n\n".join(lines)


def format_transition_event(te: dict) -> str:
    return (
        f"Name: {te.get('name', '')}\n"
        f"Description: {te.get('description', '')}\n"
        f"Month offset: Phase 2 starts at Phase 1 month {te.get('transition_month', '?')}"
    )


def build_skeleton_summary(domain_skeletons: list) -> str:
    lines = []
    for ds in domain_skeletons:
        stype = ds.get("skeleton_type", "?")
        lines.append(f"\n### Domain: {ds['domain_name']} (type: {stype}, freq: {ds.get('frequency','?')})")
        for proj in ds["skeleton"].get("projects", []):
            lines.append(
                f"  Project {proj['project_id']}: {proj['title']} "
                f"[{proj.get('approximate_duration', '?')}]"
            )
            lines.append(f"    {proj.get('description', '')}")
            for evt in proj.get("events", []):
                lines.append(f"    Event {evt['event_id']}: {evt['title']}")
    return "\n".join(lines)


def build_event_inventory(domain_skeletons: list) -> str:
    lines = ["domain | project_id | event_id | event_title"]
    lines.append("-" * 70)
    for ds in domain_skeletons:
        for proj in ds["skeleton"].get("projects", []):
            for evt in proj.get("events", []):
                lines.append(
                    f"{ds['domain_name']} | {proj['project_id']} | {evt['event_id']} | {evt['title']}"
                )
    return "\n".join(lines)


def count_events(domain_skeletons: list) -> int:
    return sum(
        len(proj.get("events", []))
        for ds in domain_skeletons
        for proj in ds.get("skeleton", {}).get("projects", [])
    )


def generate_phase2_timeline(
    llm: UnifiedLLM,
    persona: dict,
    transition_event: dict,
    domain_skeletons: list,
    max_month: int = 12,
) -> dict:
    prompt = INTEGRATION_PROMPT.format(
        persona=format_persona(persona),
        transition_event=format_transition_event(transition_event),
        skeletons=build_skeleton_summary(domain_skeletons),
        event_inventory=build_event_inventory(domain_skeletons),
        max_month=max_month,
    )
    raw = llm.chat(prompt, system=SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def merge_oneoff_sessions_phase2(timeline: dict, oneoff_sessions: list) -> dict:
    
    if not oneoff_sessions:
        return timeline

    total_months = timeline.get("total_months", 12)
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
        current_month = interval_months

        for evt in events:
            month = max(1, round(current_month))
            if month > total_months:
                dropped = len(events) - placed
                print(f"    [{domain_name}] placed {placed}, "
                      f"dropped {dropped} (month {month} > {total_months}mo Phase2 timeline)")
                break

            oneoff_flat.append({
                "session_id": -1,
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

    merged = existing_sessions + oneoff_flat
    merged.sort(key=lambda s: (
        s["month"],
        0 if s["memory_required"] else 1,
        s["session_id"] if s["memory_required"] else 0,
    ))

    for idx, session in enumerate(merged, start=1):
        session["session_id"] = idx

    timeline["sessions"] = merged
    return timeline


def validate_phase2_timeline(timeline: dict, domain_skeletons: list) -> list[str]:
    """Validate memory_required=True sessions."""
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
            warnings.append(f"Duplicate: {key}")
        seen.add(key)

    for m in expected - seen:
        warnings.append(f"Missing: {m}")
    for e in seen - expected:
        warnings.append(f"Unknown: {e}")

    # Project order check
    domain_proj_months: dict = {}
    for s in sessions:
        if not s.get("memory_required", True):
            continue
        d, p, m = s["domain"], s["project_id"], s["month"]
        domain_proj_months.setdefault(d, {}).setdefault(p, []).append(m)

    for domain, proj_map in domain_proj_months.items():
        for pid_a, pid_b in zip(sorted(proj_map)[:-1], sorted(proj_map)[1:]):
            if max(proj_map[pid_a]) > min(proj_map[pid_b]):
                warnings.append(
                    f"Project order violation in {domain}: "
                    f"P{pid_a}(max={max(proj_map[pid_a])}) > P{pid_b}(min={min(proj_map[pid_b])})"
                )

    return warnings


def apply_month_offset(sessions: list, offset: int) -> list:
    """Convert Phase 2 relative months to absolute (offset from Phase 1)."""
    for s in sessions:
        s["month"] = s["month"] + offset
    return sessions


def apply_session_id_offset(sessions: list, offset: int) -> list:
    """Continue Phase 2 session_ids after Phase 1 last session_id."""
    for s in sessions:
        s["session_id"] = s["session_id"] + offset
    return sessions


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_files(
    timeline_path: str,
    phase2_path: str,
    llm: UnifiedLLM,
    output_dir: str,
    overwrite: bool = False,
) -> dict:
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)
    with open(phase2_path, "r", encoding="utf-8") as f:
        phase2_data = json.load(f)

    uuid = timeline_data["uuid"]
    output_path = os.path.join(output_dir, f"{uuid}.json")

    if not overwrite and os.path.exists(output_path):
        return {"uuid": uuid, "skipped": True}

    persona = timeline_data["persona"]
    phase1_timeline = timeline_data.get("timeline", {})
    phase1_total_months = phase1_timeline.get("total_months", 24)
    phase1_sessions = phase1_timeline.get("sessions", [])
    phase1_session_count = len(phase1_sessions)

    transition_event = phase2_data.get("transition_event", {})
    domain_skeletons_p2 = phase2_data.get("domain_skeletons", [])
    oneoff_sessions_p2 = phase2_data.get("oneoff_sessions", [])

    valid_skeletons = [
        ds for ds in domain_skeletons_p2
        if ds.get("skeleton") and ds["skeleton"].get("projects")
    ]

    if not valid_skeletons:
        return {"uuid": uuid, "skipped": True, "reason": "no valid Phase 2 skeletons"}

    n_true = count_events(valid_skeletons)
    print(f"  {uuid[:8]}: {len(valid_skeletons)} Phase2 domains ({n_true} events) | "
          f"{sum(len(b.get('events',[])) for b in oneoff_sessions_p2)} oneoff")

    # Step 1: LLM → memory_required=True placement (Phase 2 relative months)
    try:
        p2_timeline = generate_phase2_timeline(
            llm, persona, transition_event, valid_skeletons, max_month=12
        )
    except Exception as e:
        return {"uuid": uuid, "error": str(e)}

    p2_actual_months = p2_timeline.get("total_months", 12)
    print(f"  {uuid[:8]}: Phase2 LLM → {p2_actual_months}-month timeline")

    # Step 2: Validate Phase 2
    warnings = validate_phase2_timeline(p2_timeline, valid_skeletons)
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    - {w}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings)-5} more")

    # Step 3: Merge Phase 2 one-off (Phase 2 relative months)
    p2_timeline = merge_oneoff_sessions_phase2(p2_timeline, oneoff_sessions_p2)

    n_oneoff_placed = sum(
        1 for s in p2_timeline.get("sessions", []) if not s.get("memory_required", True)
    )
    p2_total_sessions = len(p2_timeline.get("sessions", []))
    print(f"  {uuid[:8]}: Phase2 total {p2_total_sessions} sessions "
          f"({n_true} mem-req + {n_oneoff_placed} oneoff)")

    # Step 4: Apply offsets → absolute month & session_id
    p2_sessions = p2_timeline.get("sessions", [])
    p2_sessions = apply_month_offset(p2_sessions, offset=phase1_total_months)
    p2_sessions = apply_session_id_offset(p2_sessions, offset=phase1_session_count)
    p2_timeline["sessions"] = p2_sessions

    # Step 5: all_sessions = Phase 1 + Phase 2
    all_sessions = phase1_sessions + p2_sessions

    result = {
        "uuid": uuid,
        "persona": persona,
        "phase1": {
            "domain_skeletons": timeline_data.get("domain_skeletons", []),
            "oneoff_sessions": timeline_data.get("oneoff_sessions", []),
            "total_months": phase1_total_months,
            "sessions": phase1_sessions,
        },
        "phase2": {
            "transition_event": transition_event,
            "domain_skeletons": domain_skeletons_p2,
            "oneoff_sessions": oneoff_sessions_p2,
            "total_months": p2_actual_months,
            "month_offset": phase1_total_months,
            "sessions": p2_sessions,
        },
        "all_sessions": all_sessions,
        "validation_warnings": warnings,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {
        "uuid": uuid,
        "phase1_sessions": phase1_session_count,
        "phase2_sessions": p2_total_sessions,
        "total_sessions": len(all_sessions),
        "total_months": phase1_total_months + p2_actual_months,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2.4: Integrate Phase 1 + Phase 2 into extended timeline"
    )
    parser.add_argument("--timeline_file", type=str, default=None,
                        help="Single Phase 1 timeline file")
    parser.add_argument("--phase2_file", type=str, default=None,
                        help="Single Phase 2 skeleton file")
    parser.add_argument("--timeline_dir", type=str, default="./life_timelines",
                        help="Phase 1 timeline directory")
    parser.add_argument("--phase2_dir", type=str, default="./phase2_skeletons",
                        help="Phase 2 skeleton directory")
    parser.add_argument("--output_dir", type=str, default="./extended_timelines")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    llm = UnifiedLLM(args.provider, args.model)

    if args.timeline_file and args.phase2_file:
        pairs = [(args.timeline_file, args.phase2_file)]
    else:
        timeline_files = sorted(glob(os.path.join(args.timeline_dir, "*.json")))
        timeline_files = [f for f in timeline_files if not os.path.basename(f).startswith("_")]
        pairs = []
        for tf in timeline_files:
            uuid = os.path.splitext(os.path.basename(tf))[0]
            p2f = os.path.join(args.phase2_dir, f"{uuid}.json")
            if os.path.exists(p2f):
                pairs.append((tf, p2f))
            else:
                print(f"  skip {uuid[:8]}: no Phase 2 skeleton")

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Provider : {args.provider} / {args.model}")
    print(f"Pairs    : {len(pairs)}")
    print(f"Output   : {args.output_dir}\n")

    success = skipped = errors = 0
    for timeline_path, phase2_path in tqdm(pairs, desc="Personas"):
        try:
            result = process_files(timeline_path, phase2_path, llm, args.output_dir, args.overwrite)
            if result.get("skipped"):
                skipped += 1
            elif result.get("error"):
                errors += 1
                print(f"  ✗ {result['uuid'][:8]}: {result['error']}")
            else:
                success += 1
                print(
                    f"  ok {result['uuid'][:8]}: "
                    f"P1={result['phase1_sessions']} + P2={result['phase2_sessions']} "
                    f"= {result['total_sessions']} sessions, "
                    f"{result['total_months']}mo total"
                )
        except Exception as e:
            errors += 1
            print(f"  ✗ {timeline_path}: {e}")

    print(f"\nDone — success: {success}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()
