"""
2_0_generate_life_skeleton.py

For each persona's memory-required domains, generates a Life Skeleton:
  - Project sequence (sequential projects over 1-2 years)
  - Event sequence within each project
  - GT Memory per event:
      - user_profile: stable facts about the user (persists across projects)
      - ongoing_state: current project context (disappears when project ends)
  - Probing question + answer for each GT memory item

Usage:
  # single file
  python 2_0_generate_life_skeleton.py --input_file final_persona_metadata/0a0dcec0.json --output_dir ./life_skeletons

  # full directory
  python 2_0_generate_life_skeleton.py --input_dir ./final_persona_metadata --output_dir ./life_skeletons

  # with specific model
  python 2_0_generate_life_skeleton.py --input_dir ./final_persona_metadata --output_dir ./life_skeletons --provider openai --model gpt-4o

  # sample up to 3 domains per user (reproducible)
  python 2_0_generate_life_skeleton.py --input_dir ./final_persona_metadata --output_dir ./life_skeletons --max_domains_per_user 3 --domain_sample_seed 42
"""

import json
import os
import argparse
import random
from tqdm import tqdm
from glob import glob
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM import UnifiedLLM


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are designing a Life Skeleton for a personalized AI agent memory benchmark.

## What is a Life Skeleton?
A Life Skeleton captures how a specific user engages with an AI agent in a particular domain
over 1-2 years. It defines the sequence of real-life projects and events that cause the user
to interact with an AI agent, and what the memory system MUST retain from those interactions.

## Structure
- Project : A self-contained undertaking (weeks to months). Projects are strictly sequential —
            one must end before the next begins.
- Event   : A specific moment within a project when the user opens the AI agent.
- GT Memory: Ground-truth information an ideal memory system MUST store from the event.
    - user_profile  : Stable facts about the user. Persists across ALL future projects/sessions.
                      Only record when something genuinely NEW is learned through interaction.
    - ongoing_state : Current project context. Is RESET when the project ends.
                      Records decisions made, tools adopted, current progress/blockers.

## Rules
1. Ground everything in the user's specific persona — do NOT invent unrelated interests.
2. Only add a GT memory item when there is a genuine CHANGE or NEW piece of information.
   - Do NOT echo facts already explicit in the persona description.
   - DO record: decisions made, tools chosen, preferences revealed through usage, progress.
3. user_profile items accumulate; ongoing_state items are project-scoped.
4. Each GT memory item needs a crisp probing question and a precise answer.
5. gt_memory may be an empty list [] if an event reveals nothing new worth storing.
"""

SKELETON_PROMPT = """\
## User Persona
{persona}

## Domain
- Name      : {domain_name}
- Frequency : {frequency}
- Why used  : {reason}

## Task
Generate a Life Skeleton for this user in the "{domain_name}" domain over a 1-2 year period.

### Scale
- Number of projects : {n_projects}
- Events per project : {n_events_min}–{n_events_max}

### GT Memory Definitions

**user_profile** — persists forever after it is learned
  • Skills, tools mastered, communication style, revealed preferences
  • Example: "Amanda prefers code examples without theoretical preamble"
  • Example: "Amanda's Python level updated: none → beginner"
  • ⚠ Only record if NOT already in the persona text.

**ongoing_state** — lives only while the project is active
  • Decisions made, tools/options chosen, progress reached, schedule set
  • Example: "Chose pandas for CSV parsing after comparing options with agent"
  • Example: "Decided on Mon/Wed/Fri study schedule"
  • ⚠ CRITICAL: ongoing_state facts MUST be things that get DECIDED or DISCOVERED
    *during* the conversation — not things the user already knows before opening the chat.
  • The user arrives WITHOUT this information — the conversation produces it.

{already_covered_section}\
### Output Format (strict JSON, no trailing commas)

{{
  "domain_name": "{domain_name}",
  "projects": [
    {{
      "project_id": 1,
      "title": "Short project title",
      "description": "1-2 sentences: what is this project and why does it start?",
      "approximate_duration": "e.g., Month 1-3",
      "events": [
        {{
          "event_id": 1,
          "title": "Short event title",
          "description": "What happens? Why does the user turn to the AI agent right now?",
          "gt_memory": [
            {{
              "type": "user_profile",
              "fact": "Concise, self-contained statement to store",
              "probing_question": "Question that can only be answered if this was stored",
              "answer": "Expected precise answer"
            }},
            {{
              "type": "ongoing_state",
              "fact": "Concise statement about what was decided/chosen during this conversation",
              "probing_question": "Question that tests if this decision was stored",
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

# Injected into SKELETON_PROMPT when prior domains exist
ALREADY_COVERED_TEMPLATE = """\
### Domain Role Boundaries ⚠ CRITICAL
Other domains have already been assigned to cover specific angles of this user's life.
Your domain "{domain_name}" must cover ONLY its OWN unique angle.

**Already covered by other domains — DO NOT duplicate these facts:**
{covered_facts}

Rules for this section:
- Do NOT record gt_memory facts that are semantically equivalent to any fact above,
  even if the wording differs.
- If the same real-world activity (e.g. studying for an exam) appears in both your domain
  and a prior domain, cover a DIFFERENT aspect of it.
  Example: if "Career Development" already recorded the study schedule,
  "Academic Study" should focus on content strategy or learning methods instead.
- When in doubt, leave gt_memory empty rather than duplicating.

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


def get_project_count(frequency: str) -> int:
    return {"high": 5, "medium": 3, "low": 2}.get(frequency, 3)


def get_event_range(frequency: str) -> tuple[int, int]:
    return {"high": (3, 5), "medium": (2, 4), "low": (2, 3)}.get(frequency, (2, 4))


def build_already_covered_section(domain_name: str, already_generated: list[dict]) -> str:
    """
    Builds the 'Already covered' prompt section from previously generated domain skeletons.
    Returns an empty string if nothing has been generated yet.
    """
    if not already_generated:
        return ""

    fact_lines = []
    for ds in already_generated:
        prior_domain = ds["domain_name"]
        for proj in ds["skeleton"].get("projects", []):
            for evt in proj.get("events", []):
                for gm in evt.get("gt_memory", []):
                    fact_lines.append(
                        f"  [{prior_domain} | {gm['type']}] {gm['fact']}"
                    )

    if not fact_lines:
        return ""

    covered_facts = "\n".join(fact_lines)
    return ALREADY_COVERED_TEMPLATE.format(
        domain_name=domain_name,
        covered_facts=covered_facts,
    )


FREQUENCY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def sample_domains(
    memory_domains: list[dict],
    max_domains: int | None,
    rng: random.Random,
) -> list[dict]:
    """
    Randomly sample up to max_domains from memory_domains WITHOUT replacement,
    using frequency-based weights (high=3, medium=2, low=1) normalized to probabilities.

    If max_domains is None or >= len(memory_domains), returns all domains as-is.
    Final result is sorted back to the original order so the already_covered_section
    logic remains deterministic.
    """
    if max_domains is None or max_domains >= len(memory_domains):
        return memory_domains

    # Assign raw weights per domain
    weights = [
        FREQUENCY_WEIGHTS.get(d.get("frequency", "medium"), 2)
        for d in memory_domains
    ]

    # Weighted sampling WITHOUT replacement
    pool = list(range(len(memory_domains)))
    selected_indices: list[int] = []
    remaining_weights = weights[:]

    for _ in range(min(max_domains, len(pool))):
        total = sum(remaining_weights[i] for i in pool)
        probs = [remaining_weights[i] / total for i in pool]
        chosen_pos = rng.choices(range(len(pool)), weights=probs, k=1)[0]
        chosen_idx = pool.pop(chosen_pos)
        remaining_weights[chosen_idx] = 0  # neutralize (already removed from pool)
        selected_indices.append(chosen_idx)

    # Restore original relative order
    selected_indices.sort()
    return [memory_domains[i] for i in selected_indices]


def generate_skeleton_for_domain(
    llm: UnifiedLLM,
    persona: dict,
    domain: dict,
    already_generated: list[dict] | None = None,
) -> dict:
    persona_text = format_persona(persona)
    n_projects = get_project_count(domain.get("frequency", "medium"))
    n_min, n_max = get_event_range(domain.get("frequency", "medium"))

    already_covered_section = build_already_covered_section(
        domain["domain_name"],
        already_generated or [],
    )

    prompt = SKELETON_PROMPT.format(
        persona=persona_text,
        domain_name=domain["domain_name"],
        frequency=domain.get("frequency", "medium"),
        reason=domain.get("reason", ""),
        n_projects=n_projects,
        n_events_min=n_min,
        n_events_max=n_max,
        already_covered_section=already_covered_section,
    )

    raw = llm.chat(prompt, system=SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def process_persona_file(
    filepath: str,
    llm: UnifiedLLM,
    output_dir: str,
    overwrite: bool = False,
    max_domains: int | None = None,
    rng: random.Random | None = None,
) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    domain_list = open('domain_list_v3.txt', 'r', encoding='utf-8').read()
    domain_list = [a.split('. ')[1] for a in domain_list.split('\n')]
    uuid = data["uuid"]
    output_path = os.path.join(output_dir, f"{uuid}.json")

    if not overwrite and os.path.exists(output_path):
        return {"uuid": uuid, "skipped": True}

    persona = data["persona"]
    memory_domains = [
        d for d in data.get("domains", [])
        if d.get("memory_required") is True and d['domain_name'] in domain_list
    ]

    # ── Domain sampling ──────────────────────────────────────────────────────
    all_domain_count = len(memory_domains)
    memory_domains = sample_domains(memory_domains, max_domains, rng or random.Random())
    sampled_count = len(memory_domains)
    # ─────────────────────────────────────────────────────────────────────────

    result = {
        "uuid": uuid,
        "persona": persona,
        "domain_skeletons": [],
        "errors": [],
        "sampling_info": {
            "total_memory_required_domains": all_domain_count,
            "sampled_domains": sampled_count,
            "max_domains_per_user": max_domains,
            "selected_domains": [d["domain_name"] for d in memory_domains],
        },
    }

    # Generate domains sequentially, passing accumulated skeletons as context
    for domain in tqdm(memory_domains, desc=f"  {uuid[:8]} domains", leave=False):
        try:
            skeleton = generate_skeleton_for_domain(
                llm,
                persona,
                domain,
                already_generated=result["domain_skeletons"],
            )
            result["domain_skeletons"].append({
                "domain_name": domain["domain_name"],
                "frequency": domain.get("frequency"),
                "skeleton": skeleton,
            })
        except Exception as e:
            result["errors"].append({
                "domain_name": domain["domain_name"],
                "error": str(e),
            })
            print(f"    ✗ {domain['domain_name']}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Life Skeletons for persona domains")
    parser.add_argument("--input_file", type=str, default='final_persona_metadata_v3/0a0dcec0230a4083880dfdee4b46759e.json',
                        help="Path to a single persona JSON file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing persona JSON files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    parser.add_argument("--output_dir", type=str, default="./life_skeletons",
                        help="Output directory (default: ./life_skeletons)")
    parser.add_argument("--provider", type=str, default="openai",
                        help="LLM provider: openai | claude | together | gemini")
    parser.add_argument("--model", type=str, default='gpt-5.4',
                        help="Model name override (uses provider default if omitted)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")

    # ── Domain sampling hyperparameters ──────────────────────────────────────
    parser.add_argument(
        "--max_domains_per_user",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Maximum number of memory-required domains to use per user. "
            "Domains are sampled with frequency-stratified random sampling "
            "(high > medium > low priority). "
            "Default: None (use all memory-required domains)."
        ),
    )
    parser.add_argument(
        "--domain_sample_seed",
        type=int,
        default=42,
        metavar="SEED",
        help=(
            "Random seed for domain sampling. "
            "Each persona gets its own seeded RNG derived from this seed + uuid "
            "so results are reproducible per-persona regardless of processing order. "
            "Default: 42."
        ),
    )
    # ─────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("Provide --input_file or --input_dir")

    os.makedirs(args.output_dir, exist_ok=True)
    llm = UnifiedLLM(args.provider, args.model)

    if args.input_file and args.limit is None:
        files = [args.input_file]
    elif args.input_dir and args.limit is None:
        files = sorted(glob(os.path.join(args.input_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]
    elif args.input_dir and args.limit is not None:
        files = sorted(glob(os.path.join(args.input_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]
        files = files[:args.limit]
    else:
        files = [args.input_file]

    print(f"Provider          : {args.provider} / {args.model or 'default'}")
    print(f"Files             : {len(files)}")
    print(f"Output            : {args.output_dir}")
    print(f"Max domains/user  : {args.max_domains_per_user or 'all'}")
    print(f"Domain seed       : {args.domain_sample_seed}\n")

    success, skipped, errors = 0, 0, 0
    for filepath in tqdm(files, desc="Personas"):
        try:
            # Per-persona RNG: seed derived from global seed + uuid hash
            # → sampling is stable per persona regardless of --limit / ordering
            uuid_str = os.path.splitext(os.path.basename(filepath))[0]
            persona_seed = args.domain_sample_seed ^ (hash(uuid_str) & 0xFFFFFFFF)
            rng = random.Random(persona_seed)

            result = process_persona_file(
                filepath,
                llm,
                args.output_dir,
                args.overwrite,
                max_domains=args.max_domains_per_user,
                rng=rng,
            )
            if result.get("skipped"):
                skipped += 1
            else:
                success += 1
                info = result.get("sampling_info", {})
                n_total = info.get("total_memory_required_domains", "?")
                n_used = info.get("sampled_domains", "?")
                if n_total != n_used:
                    print(f"  ↓ {uuid_str[:8]}: {n_used}/{n_total} domains sampled")
                n_errors = len(result.get("errors", []))
                if n_errors:
                    print(f"  ⚠ {uuid_str[:8]}: {n_errors} domain error(s)")
        except Exception as e:
            errors += 1
            print(f"  ✗ {filepath}: {e}")

    print(f"\nDone — success: {success}, skipped: {skipped}, file errors: {errors}")


if __name__ == "__main__":
    main()