"""
2_3_generate_phase2_skeleton.py

Stage 2.3: Phase 2 Life Skeleton 생성.

2_2에서 결정된 패턴 전환을 바탕으로 Phase 2 (~12개월)의 skeleton을 생성한다.

2_2 변화 룰:
  1. mem_to_oneoff  : 기존 memory_required=True 도메인 1개 → oneoff으로 강등
  2. added_mem      : 새 memory_required=True 도메인 1개 추가
  3. added_oneoff   : 새 memory_required=False 도메인 1개 추가 (없을 수 있음)

Phase 2 도메인 구성:
  [memory_required=True skeletons]
    - retained : Phase 1 mem 도메인 중 mem_to_oneoff가 아닌 것 (이어서 새 프로젝트)
    - added    : 2_2의 added_mem 도메인 (완전 새 skeleton)

  [oneoff sessions]
    - demoted  : mem_to_oneoff 도메인 (Phase 1엔 mem이었으나 Phase 2에선 oneoff)
    - retained : Phase 1 기존 oneoff 도메인들 (그대로 유지)
    - added    : 2_2의 added_oneoff 도메인 (있을 경우)

전환 서사(transition_event)는 이 스크립트에서 LLM으로 생성.

Usage:
  python 2_3_generate_phase2_skeleton.py \\
      --timeline_file ./life_timelines_v5/{uuid}.json \\
      --shift_file ./pattern_shifts/{uuid}.json \\
      --output_dir ./phase2_skeletons

  python 2_3_generate_phase2_skeleton.py \\
      --timeline_dir ./life_timelines_v5 \\
      --shift_dir ./pattern_shifts \\
      --output_dir ./phase2_skeletons \\
      --provider openai --model gpt-4o
"""

import json
import os
import argparse
import math
from glob import glob
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM import UnifiedLLM


# ============================================================
# CONSTANTS
# ============================================================

PHASE2_MONTHS = 12
WEEKS_PER_MONTH = 4
WEEKS_PER_SESSION = {"high": 4, "medium": 8, "low": 12}
DEFAULT_WEEKS_PER_SESSION = 8
MAX_ONEOFF_DOMAINS_PHASE2 = 4


# ============================================================
# PROMPT — 전환 서사 생성
# ============================================================

NARRATIVE_SYSTEM_PROMPT = """\
You are writing a life transition narrative for a personalized AI agent memory benchmark.

Domain changes have already been decided. Your job is to write a short, coherent life
transition event that naturally explains all the changes listed.
Keep it specific to this user's circumstances — not generic.
"""

NARRATIVE_PROMPT = """\
## User Persona
{persona}

## Current AI Agent Usage (Phase 1, {total_months} months)
### Memory-Required Domains (longitudinal):
{mem_domains}

### One-Off Domains (single sessions):
{oneoff_domains}

## Phase 2 Domain Changes (already decided)
1. DEMOTED to occasional use : {mem_to_oneoff_name} (was longitudinal, now one-off)
2. NEW longitudinal domain   : {added_mem_name}
3. NEW one-off domain        : {added_oneoff_name}

## Task
Write a single life transition event that coherently explains why:
- {mem_to_oneoff_name} no longer needs sustained engagement
- {added_mem_name} becomes a new long-term focus
- {added_oneoff_name} becomes occasionally needed (if provided)

### Output Format (strict JSON)
{{
  "name": "Short name for the life transition (5-10 words)",
  "description": "2-3 sentences. What changed in this person's life? Be specific to this persona."
}}

Return ONLY the JSON object. No explanation before or after.
"""


# ============================================================
# PROMPTS — memory_required=True skeleton
# ============================================================

SKELETON_SYSTEM_PROMPT = """\
You are designing a Phase 2 Life Skeleton for a personalized AI agent memory benchmark.

A user has just gone through a major life transition. You are generating their AI agent
interaction structure for the NEXT ~12 months (Phase 2).

## Structure
- Project : A self-contained undertaking. Projects are strictly sequential.
- Event   : A specific moment when the user opens the AI agent.
- GT Memory:
    - user_profile  : Stable facts. Can UPDATE Phase 1 facts if life has changed.
                      Only record when something genuinely NEW or CHANGED is revealed.
    - ongoing_state : Current project context. Resets when project ends.

## Rules
1. Ground everything in the user's persona AND the life transition event.
2. user_profile may UPDATE Phase 1 facts if the transition caused genuine changes.
   Do NOT re-record unchanged facts.
3. ongoing_state: MUST be things decided/discovered DURING the conversation.
4. GT memory may be empty [] if an event reveals nothing new.
"""

SKELETON_PROMPT_ADDED = """\
## User Persona
{persona}

## Life Transition Event
{transition_event}

## Domain
- Name      : {domain_name}
- Frequency : {frequency}
- Why used  : {reason}
- Status    : NEW domain (started because of the life transition)

## Task
Generate a Phase 2 Life Skeleton for this user in the "{domain_name}" domain over ~12 months.

### Scale
- Number of projects : {n_projects}
- Events per project : {n_events_min}–{n_events_max}

{already_covered_section}\
### Output Format (strict JSON, no trailing commas)
{{
  "domain_name": "{domain_name}",
  "projects": [
    {{
      "project_id": 1,
      "title": "Short project title",
      "description": "1-2 sentences",
      "approximate_duration": "e.g., Phase2 Month 1-3",
      "events": [
        {{
          "event_id": 1,
          "title": "Short event title",
          "description": "What triggers this agent interaction?",
          "gt_memory": [
            {{
              "type": "user_profile|ongoing_state",
              "fact": "Concise, self-contained statement",
              "probing_question": "Question answerable only if this was stored",
              "answer": "Expected precise answer"
            }}
          ]
        }}
      ]
    }}
  ]
}}

Return ONLY the JSON object. No explanation before or after.
"""

SKELETON_PROMPT_RETAINED = """\
## User Persona
{persona}

## Life Transition Event
{transition_event}

## Domain
- Name      : {domain_name}
- Frequency : {frequency}
- Status    : CONTINUING domain (was active in Phase 1, continues into Phase 2)

## Phase 1 Summary in This Domain
{phase1_projects_summary}

## Task
Generate Phase 2 projects for "{domain_name}" over ~12 months.
These are NEW projects following on from Phase 1 — user has existing skills/context
but starts fresh undertakings appropriate to their new life situation.

### Scale
- Number of projects : {n_projects}
- Events per project : {n_events_min}–{n_events_max}

{already_covered_section}\
### Output Format (strict JSON, no trailing commas)
{{
  "domain_name": "{domain_name}",
  "projects": [
    {{
      "project_id": 1,
      "title": "Short project title",
      "description": "1-2 sentences",
      "approximate_duration": "e.g., Phase2 Month 1-3",
      "events": [
        {{
          "event_id": 1,
          "title": "Short event title",
          "description": "What triggers this agent interaction?",
          "gt_memory": [
            {{
              "type": "user_profile|ongoing_state",
              "fact": "Concise, self-contained statement",
              "probing_question": "Question answerable only if this was stored",
              "answer": "Expected precise answer"
            }}
          ]
        }}
      ]
    }}
  ]
}}

Return ONLY the JSON object. No explanation before or after.
"""

ALREADY_COVERED_TEMPLATE = """\
### Already Covered — DO NOT Duplicate ⚠
The following facts are already recorded from Phase 1 interactions.
Do NOT generate gt_memory items semantically equivalent to any of these.
user_profile facts MAY be updated/refined if the transition caused genuine change.

{covered_facts}

"""


# ============================================================
# PROMPTS — one-off sessions (Phase 2)
# ============================================================

ONEOFF_SYSTEM_PROMPT = """\
You are designing one-off AI agent interaction events for Phase 2 of a personalized memory benchmark.

These are self-contained, single-session interactions — no cross-session continuity.
The user's life has recently changed. Events must reflect the new situation.

Requirements:
- Grounded in THIS user's new life situation after the transition
- Each event covers a DIFFERENT situation or need
- NOT repetitive of the listed Phase 1 events for this domain
"""

ONEOFF_PROMPT = """\
## User Persona
{persona}

## Life Transition Event
{transition_event}

## Domain
- Name      : {domain_name}
- Frequency : {frequency}
- Why used  : {reason}

## Phase 1 Events for This Domain (DO NOT REPEAT these scenarios)
{phase1_events_to_avoid}

## Task
Generate exactly {n_events} Phase 2 one-off interaction events for "{domain_name}".
Span ~{total_months} months, reflecting the user's life AFTER the transition.

### Output Format (strict JSON, no trailing commas)
{{
  "domain_name": "{domain_name}",
  "events": [
    {{
      "event_id": 1,
      "event_title": "Short specific title (4-8 words)",
      "event_description": "2-3 sentences grounding the request in this user's new life."
    }}
  ]
}}

Return ONLY the JSON object. No explanation before or after.
"""


# ============================================================
# HELPERS
# ============================================================

def format_persona(persona_dict: dict) -> str:
    return "\n\n".join(
        f"[{k}]\n{v}" for k, v in persona_dict.items() if k != "uuid"
    )


def format_transition_event(te: dict) -> str:
    return (
        f"Name: {te.get('name', '')}\n"
        f"Description: {te.get('description', '')}\n"
        f"Month: {te.get('transition_month', '?')} (start of Phase 2)"
    )


def get_project_count(frequency: str) -> int:
    return {"high": 4, "medium": 3, "low": 2}.get(frequency, 3)


def get_event_range(frequency: str) -> tuple:
    return {"high": (3, 5), "medium": (2, 4), "low": (2, 3)}.get(frequency, (2, 4))


def normalize_gt_memory_items(raw_gt_memory) -> list:
    """
    Canonicalize gt_memory items to the life_skeletons_v5_clean format:
      [{"type","fact","probing_question","answer"}, ...]

    Handles malformed outputs such as:
      - {"user_profile": [...], "ongoing_state": [...]}
      - [{"user_profile": [...], "ongoing_state": [...]}]
      - mixed canonical + grouped entries
    """
    if raw_gt_memory is None:
        return []

    normalized = []
    seen = set()

    def _append_item(mem_type: str, fact: str, probing_question: str = "", answer: str = ""):
        if mem_type not in ("user_profile", "ongoing_state"):
            return
        if not isinstance(fact, str):
            return
        fact_clean = fact.strip()
        if not fact_clean:
            return

        probing_clean = probing_question.strip() if isinstance(probing_question, str) else ""
        answer_clean = answer.strip() if isinstance(answer, str) else ""
        if not probing_clean:
            probing_clean = "What fact was captured in this memory item?"
        if not answer_clean:
            answer_clean = fact_clean

        key = (mem_type, fact_clean)
        if key in seen:
            return
        seen.add(key)
        normalized.append(
            {
                "type": mem_type,
                "fact": fact_clean,
                "probing_question": probing_clean,
                "answer": answer_clean,
            }
        )

    def _from_grouped_dict(grouped: dict):
        for mem_type in ("user_profile", "ongoing_state"):
            facts = grouped.get(mem_type, [])
            if not isinstance(facts, list):
                continue
            for fact in facts:
                _append_item(mem_type, fact)

    if isinstance(raw_gt_memory, dict):
        _from_grouped_dict(raw_gt_memory)
        return normalized

    if not isinstance(raw_gt_memory, list):
        return normalized

    for item in raw_gt_memory:
        if not isinstance(item, dict):
            continue

        if "type" in item and "fact" in item:
            _append_item(
                item.get("type", ""),
                item.get("fact", ""),
                item.get("probing_question", ""),
                item.get("answer", ""),
            )
        else:
            _from_grouped_dict(item)

    return normalized


def normalize_skeleton_gt_memory(skeleton: dict) -> dict:
    for proj in skeleton.get("projects", []):
        for evt in proj.get("events", []):
            evt["gt_memory"] = normalize_gt_memory_items(evt.get("gt_memory", []))
    return skeleton


def extract_covered_facts(phase1_domain_skeletons: list, phase2_already_generated: list) -> str:
    """Phase 1 + Phase 2 기생성 도메인의 user_profile facts 수집."""
    facts = []
    for ds in phase1_domain_skeletons:
        for proj in ds.get("skeleton", {}).get("projects", []):
            for evt in proj.get("events", []):
                for gm in evt.get("gt_memory", []):
                    if gm.get("type") == "user_profile":
                        facts.append(f"  [Phase1 | {ds['domain_name']}] {gm['fact']}")

    for ds in phase2_already_generated:
        for proj in ds.get("skeleton", {}).get("projects", []):
            for evt in proj.get("events", []):
                for gm in evt.get("gt_memory", []):
                    if gm.get("type") == "user_profile":
                        facts.append(f"  [Phase2 | {ds['domain_name']}] {gm['fact']}")

    return "\n".join(facts) if facts else "  (none)"


def build_already_covered_section(phase1_domain_skeletons, phase2_already_generated) -> str:
    covered = extract_covered_facts(phase1_domain_skeletons, phase2_already_generated)
    if covered.strip() == "(none)":
        return ""
    return ALREADY_COVERED_TEMPLATE.format(covered_facts=covered)


def format_phase1_projects_summary(domain_skeletons: list, domain_name: str) -> str:
    for ds in domain_skeletons:
        if ds["domain_name"] == domain_name:
            lines = []
            for proj in ds.get("skeleton", {}).get("projects", []):
                lines.append(
                    f"  Project {proj['project_id']}: {proj['title']} "
                    f"[{proj.get('approximate_duration', '?')}]"
                )
                for evt in proj.get("events", []):
                    lines.append(f"    Event {evt['event_id']}: {evt['title']}")
            return "\n".join(lines)
    return "  (no Phase 1 history found)"


def format_phase1_events_to_avoid(
    phase1_domain_skeletons: list,
    phase1_oneoff_sessions: list,
    domain_name: str,
    domain_was_mem: bool,
) -> str:
    """
    Phase 2 oneoff 생성 시 피해야 할 Phase 1 이벤트 목록.
    - domain_was_mem=True  : Phase 1에서 memory_required=True였던 도메인 → skeleton events 참조
    - domain_was_mem=False : Phase 1에서 oneoff이었던 도메인 → oneoff events 참조
    """
    if domain_was_mem:
        for ds in phase1_domain_skeletons:
            if ds["domain_name"] == domain_name:
                lines = []
                for proj in ds.get("skeleton", {}).get("projects", []):
                    for evt in proj.get("events", []):
                        lines.append(f"  - {evt.get('title', '')}: {evt.get('description', '')}")
                return "\n".join(lines) if lines else "  (none)"
    else:
        for block in phase1_oneoff_sessions:
            if block["domain_name"] == domain_name:
                events = block.get("events", [])
                lines = [
                    f"  {i+1}. {e.get('event_title', '')}: {e.get('event_description', '')}"
                    for i, e in enumerate(events)
                ]
                return "\n".join(lines) if lines else "  (none)"
    return "  (no Phase 1 events for this domain)"


def format_domain_list(domains, name_key="domain_name") -> str:
    if not domains:
        return "  (none)"
    return "\n".join(
        f"  - {d[name_key]} (frequency: {d.get('frequency','?')})" for d in domains
    )


# ============================================================
# NARRATIVE GENERATION
# ============================================================

def generate_narrative(
    llm: UnifiedLLM,
    persona: dict,
    total_months: int,
    phase1_domain_skeletons: list,
    phase1_oneoff_sessions: list,
    changes: dict,
) -> dict:
    mem_domains = [
        {"domain_name": ds["domain_name"], "frequency": ds.get("frequency")}
        for ds in phase1_domain_skeletons
    ]
    oneoff_domains = [
        {"domain_name": b["domain_name"], "frequency": b.get("frequency")}
        for b in phase1_oneoff_sessions
    ]
    added_oneoff = changes.get("added_oneoff")
    added_oneoff_name = added_oneoff["domain_name"] if added_oneoff else "N/A"

    prompt = NARRATIVE_PROMPT.format(
        persona=format_persona(persona),
        total_months=total_months,
        mem_domains=format_domain_list(mem_domains),
        oneoff_domains=format_domain_list(oneoff_domains),
        mem_to_oneoff_name=changes["mem_to_oneoff"]["domain_name"],
        added_mem_name=changes["added_mem"]["domain_name"],
        added_oneoff_name=added_oneoff_name,
    )
    raw = llm.chat(prompt, system=NARRATIVE_SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# ============================================================
# SKELETON GENERATION
# ============================================================

def generate_skeleton(
    llm: UnifiedLLM,
    persona: dict,
    transition_event: dict,
    domain: dict,
    skeleton_type: str,          # "added" | "retained"
    phase1_domain_skeletons: list,
    phase2_already_generated: list,
) -> dict:
    persona_text = format_persona(persona)
    te_text = format_transition_event(transition_event)
    frequency = domain.get("frequency", "medium")
    n_projects = get_project_count(frequency)
    n_min, n_max = get_event_range(frequency)
    domain_name = domain["domain_name"]

    already_covered_section = build_already_covered_section(
        phase1_domain_skeletons, phase2_already_generated
    )

    if skeleton_type == "added":
        prompt = SKELETON_PROMPT_ADDED.format(
            persona=persona_text,
            transition_event=te_text,
            domain_name=domain_name,
            frequency=frequency,
            reason=domain.get("reason", ""),
            n_projects=n_projects,
            n_events_min=n_min,
            n_events_max=n_max,
            already_covered_section=already_covered_section,
        )
    elif skeleton_type == "retained":
        phase1_summary = format_phase1_projects_summary(phase1_domain_skeletons, domain_name)
        prompt = SKELETON_PROMPT_RETAINED.format(
            persona=persona_text,
            transition_event=te_text,
            domain_name=domain_name,
            frequency=frequency,
            phase1_projects_summary=phase1_summary,
            n_projects=n_projects,
            n_events_min=n_min,
            n_events_max=n_max,
            already_covered_section=already_covered_section,
        )
    else:
        raise ValueError(f"Unknown skeleton_type: {skeleton_type}")

    raw = llm.chat(prompt, system=SKELETON_SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)
    return normalize_skeleton_gt_memory(parsed)


# ============================================================
# ONE-OFF GENERATION
# ============================================================

def get_oneoff_session_count(frequency: str) -> int:
    total_weeks = PHASE2_MONTHS * WEEKS_PER_MONTH
    interval_weeks = WEEKS_PER_SESSION.get(frequency, DEFAULT_WEEKS_PER_SESSION)
    return max(1, math.floor(total_weeks / interval_weeks))


def generate_oneoff_events(
    llm: UnifiedLLM,
    persona: dict,
    transition_event: dict,
    domain: dict,
    phase1_domain_skeletons: list,
    phase1_oneoff_sessions: list,
    domain_was_mem: bool = False,
) -> dict:
    frequency = domain.get("frequency", "medium")
    interval_weeks = WEEKS_PER_SESSION.get(frequency, DEFAULT_WEEKS_PER_SESSION)
    n_events = get_oneoff_session_count(frequency)
    domain_name = domain["domain_name"]

    events_to_avoid = format_phase1_events_to_avoid(
        phase1_domain_skeletons, phase1_oneoff_sessions, domain_name, domain_was_mem
    )

    prompt = ONEOFF_PROMPT.format(
        persona=format_persona(persona),
        transition_event=format_transition_event(transition_event),
        domain_name=domain_name,
        frequency=frequency,
        reason=domain.get("reason", ""),
        phase1_events_to_avoid=events_to_avoid,
        n_events=n_events,
        total_months=PHASE2_MONTHS,
    )

    raw = llm.chat(prompt, system=ONEOFF_SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(cleaned)

    result["memory_required"] = False
    result["frequency"] = frequency
    result["interval_weeks"] = interval_weeks
    result["n_events_expected"] = n_events

    return result


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_files(
    timeline_path: str,
    shift_path: str,
    llm: UnifiedLLM,
    output_dir: str,
    overwrite: bool = False,
) -> dict:
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)
    with open(shift_path, "r", encoding="utf-8") as f:
        shift_data = json.load(f)

    uuid = timeline_data["uuid"]
    output_path = os.path.join(output_dir, f"{uuid}.json")

    if not overwrite and os.path.exists(output_path):
        return {"uuid": uuid, "skipped": True}

    persona               = timeline_data["persona"]
    phase1_domain_skeletons = timeline_data.get("domain_skeletons", [])
    phase1_oneoff_sessions  = timeline_data.get("oneoff_sessions", [])
    total_months          = timeline_data.get("timeline", {}).get("total_months", 24)

    changes               = shift_data["changes"]
    transition_month      = shift_data["transition_month"]
    mem_to_oneoff_name    = changes["mem_to_oneoff"]["domain_name"]
    added_domains         = shift_data.get("added_domains", [])  # [added_mem, added_oneoff?]

    # ── Step 1: 전환 서사 생성 ────────────────────────────────────────────
    try:
        narrative = generate_narrative(
            llm, persona, total_months,
            phase1_domain_skeletons, phase1_oneoff_sessions, changes
        )
    except Exception as e:
        return {"uuid": uuid, "error": f"narrative generation failed: {e}"}

    transition_event = {
        "name": narrative.get("name", ""),
        "description": narrative.get("description", ""),
        "transition_month": transition_month,
    }
    print(f"  {uuid[:8]}: transition='{transition_event['name']}'")

    # ── Step 2: Phase 2 도메인 구성 ──────────────────────────────────────
    # memory_required=True skeletons
    #   retained: Phase 1 mem 도메인 중 mem_to_oneoff가 아닌 것
    #   added:    added_domains 중 memory_required=True
    retained_mem_domains = [
        {
            "domain_name": ds["domain_name"],
            "frequency": ds.get("frequency", "medium"),
            "reason": f"Continuing from Phase 1",
        }
        for ds in phase1_domain_skeletons
        if ds["domain_name"] != mem_to_oneoff_name
    ]
    added_mem_domains = [d for d in added_domains if d.get("memory_required", False)]

    skeleton_queue = (
        [(d, "retained") for d in retained_mem_domains]
        + [(d, "added")    for d in added_mem_domains]
    )

    # oneoff sessions
    #   demoted:  mem_to_oneoff 도메인 (Phase 1 mem → Phase 2 oneoff)
    #   retained: Phase 1 기존 oneoff 도메인 (모두 유지)
    #   added:    added_domains 중 memory_required=False
    demoted_domain = {
        "domain_name": mem_to_oneoff_name,
        "frequency": changes["mem_to_oneoff"].get("frequency", "medium"),
        "reason": "Demoted from longitudinal to occasional use",
    }
    added_oneoff_domains = [d for d in added_domains if not d.get("memory_required", False)]
    retained_oneoff_domains = [
        {
            "domain_name": b["domain_name"],
            "frequency": b.get("frequency", "medium"),
            "reason": "Continuing one-off usage from Phase 1",
        }
        for b in phase1_oneoff_sessions
    ]

    oneoff_targets = (
        [(demoted_domain, True)]                                   # (domain, was_mem)
        + [(d, False) for d in retained_oneoff_domains]
        + [(d, False) for d in added_oneoff_domains]
    )[:MAX_ONEOFF_DOMAINS_PHASE2]

    # ── Step 3: Skeleton 생성 ────────────────────────────────────────────
    domain_skeletons_phase2 = []
    errors = []

    for domain, stype in tqdm(skeleton_queue, desc=f"  {uuid[:8]} skeletons", leave=False):
        try:
            skeleton = generate_skeleton(
                llm=llm,
                persona=persona,
                transition_event=transition_event,
                domain=domain,
                skeleton_type=stype,
                phase1_domain_skeletons=phase1_domain_skeletons,
                phase2_already_generated=[
                    ds["skeleton"] for ds in domain_skeletons_phase2
                ],
            )
            phase1_project_this_domain = [a for a in phase1_domain_skeletons if a['domain_name'] == skeleton['domain_name']]
            if len(phase1_project_this_domain) == 0:
                phase1_project_count = 0
            else:
                phase1_project_count =  max([a['project_id'] for a in phase1_project_this_domain[0]['skeleton']['projects']])

            for a in skeleton['projects']:
                a['project_id'] += phase1_project_count

            domain_skeletons_phase2.append({
                "domain_name": domain["domain_name"],
                "frequency": domain.get("frequency", "medium"),
                "skeleton_type": stype,
                "skeleton": skeleton,
            })
        except Exception as e:
            errors.append({"domain_name": domain["domain_name"], "error": str(e)})
            print(f"    ✗ skeleton [{stype}] {domain['domain_name']}: {e}")

    # ── Step 4: One-off 세션 생성 ────────────────────────────────────────
    oneoff_sessions_phase2 = []

    for domain, was_mem in tqdm(oneoff_targets, desc=f"  {uuid[:8]} oneoff", leave=False):
        try:
            result = generate_oneoff_events(
                llm=llm,
                persona=persona,
                transition_event=transition_event,
                domain=domain,
                phase1_domain_skeletons=phase1_domain_skeletons,
                phase1_oneoff_sessions=phase1_oneoff_sessions,
                domain_was_mem=was_mem,
            )
            oneoff_sessions_phase2.append(result)
        except Exception as e:
            errors.append({"domain_name": domain["domain_name"], "error": str(e)})
            print(f"    ✗ oneoff {domain['domain_name']}: {e}")

    # ── 저장 ────────────────────────────────────────────────────────────
    output = {
        "uuid": uuid,
        "persona": persona,
        "transition_event": transition_event,
        "phase2_months": PHASE2_MONTHS,
        "mem_to_oneoff_domain": mem_to_oneoff_name,
        "domain_skeletons": domain_skeletons_phase2,
        "oneoff_sessions": oneoff_sessions_phase2,
        "errors": errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return {
        "uuid": uuid,
        "n_skeletons": len(domain_skeletons_phase2),
        "n_oneoff": len(oneoff_sessions_phase2),
        "n_errors": len(errors),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2.3: Generate Phase 2 Life Skeletons")
    parser.add_argument("--timeline_file", type=str, default=None)
    parser.add_argument("--shift_file",    type=str, default=None)
    parser.add_argument("--timeline_dir",  type=str, default="./life_timelines_v5")
    parser.add_argument("--shift_dir",     type=str, default="./pattern_shifts")
    parser.add_argument("--output_dir",    type=str, default="./phase2_skeletons")
    parser.add_argument("--provider",      type=str, default="openai")
    parser.add_argument("--model",         type=str, default="gpt-5.4")
    parser.add_argument("--limit",         type=int, default=None)
    parser.add_argument("--overwrite",     action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    llm = UnifiedLLM(args.provider, args.model)

    if args.timeline_file and args.shift_file:
        pairs = [(args.timeline_file, args.shift_file)]
    else:
        timeline_files = sorted(glob(os.path.join(args.timeline_dir, "*.json")))
        timeline_files = [f for f in timeline_files if not os.path.basename(f).startswith("_")]
        pairs = []
        for tf in timeline_files:
            uuid = os.path.splitext(os.path.basename(tf))[0]
            sf = os.path.join(args.shift_dir, f"{uuid}.json")
            if os.path.exists(sf):
                pairs.append((tf, sf))
            else:
                print(f"  skip {uuid[:8]}: no matching shift file")

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Provider      : {args.provider} / {args.model}")
    print(f"Pairs found   : {len(pairs)}")
    print(f"Output        : {args.output_dir}")
    print(f"Phase 2 months: {PHASE2_MONTHS}\n")

    success = skipped = errors = 0
    for timeline_path, shift_path in tqdm(pairs, desc="Personas"):
        try:
            result = process_files(timeline_path, shift_path, llm, args.output_dir, args.overwrite)
            if result.get("skipped"):
                skipped += 1
            elif result.get("error"):
                errors += 1
                print(f"  ✗ {result['uuid'][:8]}: {result['error']}")
            else:
                success += 1
                n_err = result.get("n_errors", 0)
                print(
                    f"  ok {result['uuid'][:8]}: "
                    f"{result['n_skeletons']} skeletons, {result['n_oneoff']} oneoff"
                    + (f"  [{n_err} errors]" if n_err else "")
                )
        except Exception as e:
            errors += 1
            print(f"  ✗ {timeline_path}: {e}")

    print(f"\nDone — success: {success}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()