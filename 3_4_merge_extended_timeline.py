
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _merge_domain_skeletons(phase1: list[dict[str, Any]], phase2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    
    merged_by_domain: dict[str, dict[str, Any]] = {}
    domain_order: list[str] = []

    for block in [*phase1, *phase2]:
        domain_name = block.get("domain_name")
        if not domain_name:
            continue

        if domain_name not in merged_by_domain:
            merged_by_domain[domain_name] = copy.deepcopy(block)
            domain_order.append(domain_name)
            continue

        current = merged_by_domain[domain_name]
        incoming = copy.deepcopy(block)
        incoming_projects = incoming.get("skeleton", {}).get("projects", [])

        if "skeleton" not in current:
            current["skeleton"] = {"domain_name": domain_name, "projects": []}
        if "projects" not in current["skeleton"]:
            current["skeleton"]["projects"] = []

        current["skeleton"]["projects"].extend(incoming_projects)

        # Prefer latest frequency/skeleton_type if present in phase2 block.
        if incoming.get("frequency"):
            current["frequency"] = incoming["frequency"]
        if incoming.get("skeleton_type"):
            current["skeleton_type"] = incoming["skeleton_type"]

        # Preserve any other keys that only exist in incoming block.
        for k, v in incoming.items():
            if k not in current:
                current[k] = v

    return [merged_by_domain[d] for d in domain_order]


def _merge_oneoff_sessions(phase1: list[dict[str, Any]], phase2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    
    merged_by_domain: dict[str, dict[str, Any]] = {}
    domain_order: list[str] = []

    for block in [*phase1, *phase2]:
        domain_name = block.get("domain_name")
        if not domain_name:
            continue

        if domain_name not in merged_by_domain:
            merged_by_domain[domain_name] = copy.deepcopy(block)
            domain_order.append(domain_name)
            continue

        current = merged_by_domain[domain_name]
        incoming = copy.deepcopy(block)

        current.setdefault("events", [])
        current["events"].extend(incoming.get("events", []))

        # Keep updated metadata when incoming has explicit values.
        for key in ["memory_required", "frequency", "interval_weeks", "n_events_expected"]:
            if key in incoming and incoming[key] is not None:
                current[key] = incoming[key]

        # Preserve additional keys if they do not exist yet.
        for k, v in incoming.items():
            if k not in current:
                current[k] = v

    return [merged_by_domain[d] for d in domain_order]


def _build_timeline(data: dict[str, Any], phase1: dict[str, Any], phase2: dict[str, Any]) -> dict[str, Any]:
    sessions = data.get("all_sessions")
    if sessions is None:
        sessions = phase1.get("sessions", []) + phase2.get("sessions", [])

    total_months = phase1.get("total_months", 0) + phase2.get("total_months", 0)
    if not total_months:
        # Fallback: infer from sessions when totals are missing.
        session_months = [s.get("month", 0) for s in sessions if isinstance(s, dict)]
        total_months = max(session_months) if session_months else 0

    return {
        "total_months": total_months,
        "sessions": sessions,
    }


def merge_extended_to_life_style(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})

    merged_domain_skeletons = _merge_domain_skeletons(
        phase1.get("domain_skeletons", []),
        phase2.get("domain_skeletons", []),
    )
    merged_oneoff_sessions = _merge_oneoff_sessions(
        phase1.get("oneoff_sessions", []),
        phase2.get("oneoff_sessions", []),
    )

    output = {
        "uuid": data.get("uuid"),
        "persona": data.get("persona", {}),
        "domain_skeletons": merged_domain_skeletons,
        "oneoff_sessions": merged_oneoff_sessions,
        "timeline": _build_timeline(data, phase1, phase2),
    }

    if phase2.get("transition_event") is not None:
        output["transition_event"] = phase2["transition_event"]
    if data.get("validation_warnings") is not None:
        output["validation_warnings"] = data["validation_warnings"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output

in_dir = Path("extended_timelines")
out_dir = Path("life_timelines_v5_merged")

count = 0
for input_path in sorted(in_dir.glob("*.json")):
    output_path = out_dir / input_path.name
    merge_extended_to_life_style(input_path, output_path)
    count += 1

print(f"converted {count} files -> {out_dir}")


# %%



