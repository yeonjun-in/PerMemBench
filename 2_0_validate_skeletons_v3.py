"""
2_0_validate_skeletons.py

LLM judge that validates life skeleton files on three dimensions at the PROJECT level,
and optionally removes detected duplicates in the same pass.

  1. Cross-domain duplicates   : same real-world activity in different domains  → removed
  2. Intra-domain duplicates   : overlapping projects within the same domain    → removed
  3. Persona reasonableness    : does the skeleton reflect the persona realistically?

Output:
  - Terminal report (always)
  - --report_dir  : saves validation judgment JSON (what was found)
  - --clean_dir   : saves cleaned skeleton JSON (duplicates removed)
  Both output dirs are independent and optional.

Usage:
  # report only (no files written)
  python 2_0_validate_skeletons.py --input_file ./life_skeletons/0a0dcec0.json

  # report + cleaned skeletons
  python 2_0_validate_skeletons.py --input_dir ./life_skeletons \\
      --report_dir ./skeleton_validation --clean_dir ./life_skeletons_clean

  # skip reasonableness check
  python 2_0_validate_skeletons.py --input_dir ./life_skeletons --skip_reasonableness
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
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are a quality auditor for a personalized AI agent memory benchmark.

You will be given a user persona and a life skeleton — a structured record of how the user
interacts with an AI agent across multiple domains over 1-2 years.

Your job is to audit this life skeleton at the PROJECT level and return a single JSON judgment.
"""

JUDGE_PROMPT = """\
## User Persona
{persona}

## Project Inventory (all domains)
{project_table}

---

## Audit Task

### Dimension 1: Cross-Domain Duplicate Projects
Compare projects from DIFFERENT domains.
Flag pairs where two projects cover the same real-world activity — even if the domain angle
or wording differs.

A duplicate means the user is essentially doing the same thing twice in two different domain
buckets, which is unrealistic and inflates memory coverage.

Example duplicate:
  [Career Development] "CST Exam Preparation Plan" — building a study roadmap for the credential
  [Academic Study]     "CST Exam Study System Setup" — creating a study schedule for the same exam
  → Same real-world activity (preparing for CST exam). HIGH severity.
  → remove "b" because Career Development more naturally owns credential preparation.

NOT a duplicate:
  [Career Development] "Job Application Sprint" — applying to OR tech roles
  [Academic Study]     "Anatomy Refresher" — reviewing anatomy for clinical competency
  → Clearly different activities. Skip.

### Dimension 2: Intra-Domain Duplicate Projects
Within the SAME domain, flag projects that cover the same ground.
(Projects are supposed to be sequential and non-overlapping within a domain.)

### Dimension 3: Persona Reasonableness
Given the user's persona, evaluate whether the set of projects is realistic.

Check for:
- **invented_interest**    : project has no basis in the persona
- **mismatched_scale**     : activity doesn't fit the user's situation or resources
- **implausible_sequence** : project couldn't happen given the user's constraints
- **missing_coverage**     : important life area in the persona is entirely absent
- **generic_filler**       : project feels generic and not tailored to this specific person

Rate overall reasonableness: "good" | "minor_issues" | "major_issues"

---

## Output Format (strict JSON, no trailing commas)

{{
  "total_projects_checked": <int>,

  "duplicate_pairs": [
    {{
      "duplicate_type": "cross_domain" | "intra_domain",
      "severity": "high" | "medium" | "low",
      "domain_a": "<domain name>",
      "project_id_a": <int>,
      "title_a": "<project title>",
      "domain_b": "<domain name>",
      "project_id_b": <int>,
      "title_b": "<project title>",
      "remove": "a" | "b",
      "reason": "1-2 sentences: why duplicate + why this side is removed"
    }}
  ],

  "duplicate_summary": {{
    "cross_domain": {{ "high": <int>, "medium": <int>, "low": <int> }},
    "intra_domain": {{ "high": <int>, "medium": <int>, "low": <int> }}
  }},

  "persona_reasonableness": {{
    "rating": "good" | "minor_issues" | "major_issues",
    "issues": [
      {{
        "issue_type": "invented_interest" | "mismatched_scale" | "implausible_sequence" | "missing_coverage" | "generic_filler" | "other",
        "domain": "<domain name or 'overall'>",
        "project": "<project title or 'overall'>",
        "description": "Specific description of the issue"
      }}
    ],
    "strengths": ["What aspects are well-grounded in the persona"]
  }}
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


def build_project_table(domain_skeletons: list[dict]) -> str:
    lines = ["domain | proj_id | duration | n_events | title | description"]
    lines.append("-" * 110)
    for ds in domain_skeletons:
        for proj in ds["skeleton"].get("projects", []):
            desc = proj.get("description", "")[:120].replace("\n", " ")
            lines.append(
                f"{ds['domain_name'][:40]:<40} | "
                f"{proj['project_id']:7d} | "
                f"{proj.get('approximate_duration','?'):12} | "
                f"{len(proj.get('events', [])):8d} | "
                f"{proj.get('title','')[:40]:<40} | "
                f"{desc}"
            )
    return "\n".join(lines)


def count_projects(domain_skeletons: list[dict]) -> int:
    return sum(len(ds["skeleton"].get("projects", [])) for ds in domain_skeletons)


# ============================================================
# LLM JUDGE
# ============================================================

def run_judge(
    llm: UnifiedLLM,
    persona: dict,
    domain_skeletons: list[dict],
    skip_reasonableness: bool = False,
) -> dict:
    n_projects = count_projects(domain_skeletons)
    if n_projects == 0:
        return _empty_judgment()

    prompt = JUDGE_PROMPT.format(
        persona=format_persona(persona) if not skip_reasonableness else "(omitted)",
        project_table=build_project_table(domain_skeletons),
    )
    if skip_reasonableness:
        prompt = _strip_reasonableness(prompt)

    raw     = llm.chat(prompt, system=SYSTEM_PROMPT)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    result  = json.loads(cleaned)

    # Recount summaries from actual pairs (don't trust LLM counts)
    result["total_projects_checked"] = n_projects
    pairs = result.get("duplicate_pairs", [])
    result["duplicate_summary"] = {
        "cross_domain": {
            sev: sum(1 for p in pairs
                     if p.get("duplicate_type") == "cross_domain" and p.get("severity") == sev)
            for sev in ("high", "medium", "low")
        },
        "intra_domain": {
            sev: sum(1 for p in pairs
                     if p.get("duplicate_type") == "intra_domain" and p.get("severity") == sev)
            for sev in ("high", "medium", "low")
        },
    }

    if skip_reasonableness:
        result["persona_reasonableness"] = None

    return result


def _empty_judgment() -> dict:
    return {
        "total_projects_checked": 0,
        "duplicate_pairs": [],
        "duplicate_summary": {
            "cross_domain": {"high": 0, "medium": 0, "low": 0},
            "intra_domain": {"high": 0, "medium": 0, "low": 0},
        },
        "persona_reasonableness": None,
    }


def _strip_reasonableness(prompt: str) -> str:
    lines = prompt.split("\n")
    out, skip = [], False
    for line in lines:
        if "### Dimension 3" in line:
            skip = True
        if skip and line.strip().startswith("---"):
            skip = False
            continue
        if '"persona_reasonableness"' in line:
            skip = True
        if not skip:
            out.append(line)
    return "\n".join(out)


# ============================================================
# APPLY REMOVALS
# ============================================================

def apply_removals(
    domain_skeletons: list[dict],
    judgment: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Removes projects flagged as duplicates by the LLM judge.
    Returns (cleaned_skeletons, removal_log).
    """
    pairs = judgment.get("duplicate_pairs", [])

    to_remove: set[tuple[str, int]] = set()  # (domain_name, project_id)
    removal_log = []

    for pair in pairs:
        remove_side = pair.get("remove")
        if remove_side not in ("a", "b"):
            continue

        if remove_side == "a":
            loser_domain  = pair["domain_a"]
            loser_pid     = pair["project_id_a"]
            loser_title   = pair["title_a"]
            keeper_domain = pair["domain_b"]
            keeper_title  = pair["title_b"]
        else:
            loser_domain  = pair["domain_b"]
            loser_pid     = pair["project_id_b"]
            loser_title   = pair["title_b"]
            keeper_domain = pair["domain_a"]
            keeper_title  = pair["title_a"]

        key = (loser_domain, loser_pid)
        if key in to_remove:
            continue  # already scheduled by a prior pair

        to_remove.add(key)
        removal_log.append({
            "duplicate_type":     pair.get("duplicate_type"),
            "severity":           pair.get("severity"),
            "kept_domain":        keeper_domain,
            "kept_project":       keeper_title,
            "removed_domain":     loser_domain,
            "removed_project":    loser_title,
            "removed_project_id": loser_pid,
            "reason":             pair.get("reason", ""),
        })

    # Build cleaned skeletons with re-indexed project_ids
    cleaned = []
    for ds in domain_skeletons:
        kept = [
            p for p in ds["skeleton"].get("projects", [])
            if (ds["domain_name"], p["project_id"]) not in to_remove
        ]
        for new_idx, proj in enumerate(kept, 1):
            proj["project_id"] = new_idx

        cleaned.append({**ds, "skeleton": {**ds["skeleton"], "projects": kept}})

    return cleaned, removal_log


# ============================================================
# REPORT PRINTER
# ============================================================

SEVERITY_ICON       = {"high": "🔴", "medium": "🟡", "low": "🟢"}
REASONABLENESS_ICON = {"good": "✅", "minor_issues": "🟡", "major_issues": "🔴"}
ISSUE_LABEL = {
    "invented_interest":    "Invented interest",
    "mismatched_scale":     "Mismatched scale",
    "implausible_sequence": "Implausible sequence",
    "missing_coverage":     "Missing coverage",
    "generic_filler":       "Generic filler",
    "other":                "Other",
}


def print_report(uuid: str, judgment: dict, removal_log: list[dict]) -> None:
    pairs   = judgment.get("duplicate_pairs", [])
    dup_sum = judgment.get("duplicate_summary", {})
    pr      = judgment.get("persona_reasonableness")
    total   = judgment.get("total_projects_checked", 0)

    cd   = dup_sum.get("cross_domain", {})
    id_  = dup_sum.get("intra_domain", {})
    rating = pr.get("rating", "N/A") if pr else "N/A"

    print(f"\n{'='*72}")
    print(f"  {uuid[:8]}  |  {total} projects  |  "
          f"cross 🔴{cd.get('high',0)} 🟡{cd.get('medium',0)} 🟢{cd.get('low',0)}  |  "
          f"intra 🔴{id_.get('high',0)} 🟡{id_.get('medium',0)} 🟢{id_.get('low',0)}  |  "
          f"persona {REASONABLENESS_ICON.get(rating,'—')} {rating}  |  "
          f"removed: {len(removal_log)}")
    print(f"{'='*72}")

    cross = [p for p in pairs if p.get("duplicate_type") == "cross_domain"]
    intra = [p for p in pairs if p.get("duplicate_type") == "intra_domain"]

    if not pairs:
        print("  ✓ No duplicate projects found.")
    else:
        if cross:
            print(f"\n  ── Cross-Domain Duplicates ({len(cross)}) ──")
            for i, p in enumerate(cross, 1):
                icon = SEVERITY_ICON.get(p.get("severity"), "⚪")
                print(f"\n  [{i}] {icon} {p.get('severity','?').upper()}")
                print(f"      A: [{p.get('domain_a')}] p{p.get('project_id_a')} — {p.get('title_a')}")
                print(f"      B: [{p.get('domain_b')}] p{p.get('project_id_b')} — {p.get('title_b')}")
                remove = p.get("remove", "?")
                removed_title = p.get(f"title_{remove}", "?")
                print(f"      → REMOVE {remove.upper()}: {removed_title}")
                print(f"      → {p.get('reason')}")

        if intra:
            print(f"\n  ── Intra-Domain Duplicates ({len(intra)}) ──")
            for i, p in enumerate(intra, 1):
                icon = SEVERITY_ICON.get(p.get("severity"), "⚪")
                print(f"\n  [{i}] {icon} {p.get('severity','?').upper()}")
                print(f"      A: [{p.get('domain_a')}] p{p.get('project_id_a')} — {p.get('title_a')}")
                print(f"      B: p{p.get('project_id_b')} — {p.get('title_b')}")
                remove = p.get("remove", "?")
                removed_title = p.get(f"title_{remove}", "?")
                print(f"      → REMOVE {remove.upper()}: {removed_title}")
                print(f"      → {p.get('reason')}")

    if pr is None:
        return

    print(f"\n  ── Persona Reasonableness: {REASONABLENESS_ICON.get(rating,'?')} {rating.upper()} ──")
    issues = pr.get("issues", [])
    if not issues:
        print("  ✓ No persona-grounding issues found.")
    else:
        for iss in issues:
            label = ISSUE_LABEL.get(iss.get("issue_type", "other"), "Issue")
            print(f"  ⚠  [{label}] ({iss.get('domain','?')} / {iss.get('project','?')})")
            print(f"     {iss.get('description')}")

    strengths = pr.get("strengths", [])
    if strengths:
        print(f"\n  Strengths:")
        for s in strengths:
            print(f"    + {s}")


# ============================================================
# FILE PROCESSOR
# ============================================================

def process_skeleton_file(
    filepath: str,
    llm: UnifiedLLM,
    report_dir: str | None,
    clean_dir: str | None,
    overwrite: bool = False,
    skip_reasonableness: bool = False,
) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    uuid = data["uuid"]

    # Skip check (based on report output)
    if report_dir:
        report_path = os.path.join(report_dir, f"{uuid}.json")
        if not overwrite and os.path.exists(report_path):
            print(f"  skipped: {uuid[:8]}")
            return {"uuid": uuid, "skipped": True}

    domain_skeletons = [
        ds for ds in data.get("domain_skeletons", [])
        if ds.get("skeleton") and ds["skeleton"].get("projects")
    ]
    if not domain_skeletons:
        return {"uuid": uuid, "skipped": True, "reason": "no valid skeletons"}

    # Judge
    try:
        judgment = run_judge(llm, data.get("persona", {}), domain_skeletons, skip_reasonableness)
    except Exception as e:
        return {"uuid": uuid, "error": str(e)}

    # Remove duplicates
    cleaned_skeletons, removal_log = apply_removals(domain_skeletons, judgment)

    print_report(uuid, judgment, removal_log)

    # Save report
    if report_dir:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"uuid": uuid, "judgment": judgment, "removal_log": removal_log},
                      f, ensure_ascii=False, indent=2)

    # Save cleaned skeleton
    if clean_dir:
        clean_path = os.path.join(clean_dir, f"{uuid}.json")
        cleaned_data = {**data, "domain_skeletons": cleaned_skeletons, "validation_log": removal_log}
        with open(clean_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    return {"uuid": uuid, "judgment": judgment, "removal_log": removal_log}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM judge: validate + deduplicate life skeletons at the project level"
    )
    parser.add_argument("--input_file",  type=str, default=None)
    parser.add_argument("--input_dir",   type=str, default=None)
    parser.add_argument("--report_dir",  type=str, default=None,
                        help="Save validation judgment JSON files here")
    parser.add_argument("--clean_dir",   type=str, default=None,
                        help="Save cleaned skeleton JSON files here (duplicates removed)")
    parser.add_argument("--provider",    type=str, default="claude")
    parser.add_argument("--model",       type=str, default="claude-sonnet-4-6")
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--skip_reasonableness", action="store_true")
    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("Provide --input_file or --input_dir")

    if args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)
    if args.clean_dir:
        os.makedirs(args.clean_dir, exist_ok=True)

    llm = UnifiedLLM(args.provider, args.model)

    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(glob(os.path.join(args.input_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]

    print(f"Provider : {args.provider} / {args.model}")
    print(f"Files    : {len(files)}")
    if args.report_dir: print(f"Reports  : {args.report_dir}")
    if args.clean_dir:  print(f"Cleaned  : {args.clean_dir}")

    total_cd_high = total_cd_med = total_cd_low = 0
    total_id_high = total_id_med = total_id_low = 0
    total_removed = 0
    reasonableness_counts = {"good": 0, "minor_issues": 0, "major_issues": 0}
    errors = 0

    for filepath in tqdm(files, desc="Validating"):
        result = process_skeleton_file(
            filepath, llm,
            report_dir=args.report_dir,
            clean_dir=args.clean_dir,
            overwrite=args.overwrite,
            skip_reasonableness=args.skip_reasonableness,
        )
        if result.get("error"):
            errors += 1
            print(f"  ✗ {result.get('uuid','?')[:8]}: {result['error']}")
        elif not result.get("skipped"):
            j   = result.get("judgment", {})
            cd  = j.get("duplicate_summary", {}).get("cross_domain", {})
            id_ = j.get("duplicate_summary", {}).get("intra_domain", {})
            total_cd_high  += cd.get("high", 0)
            total_cd_med   += cd.get("medium", 0)
            total_cd_low   += cd.get("low", 0)
            total_id_high  += id_.get("high", 0)
            total_id_med   += id_.get("medium", 0)
            total_id_low   += id_.get("low", 0)
            total_removed  += len(result.get("removal_log", []))
            pr = (j.get("persona_reasonableness") or {})
            rating = pr.get("rating", "good")
            reasonableness_counts[rating] = reasonableness_counts.get(rating, 0) + 1

    print(f"\n{'='*72}")
    print(f"TOTAL DUPLICATES  (cross 🔴{total_cd_high} 🟡{total_cd_med} 🟢{total_cd_low}"
          f"  |  intra 🔴{total_id_high} 🟡{total_id_med} 🟢{total_id_low})")
    print(f"REMOVED           {total_removed} projects")
    if not args.skip_reasonableness:
        print(f"REASONABLENESS    ✅{reasonableness_counts.get('good',0)}  "
              f"🟡{reasonableness_counts.get('minor_issues',0)}  "
              f"🔴{reasonableness_counts.get('major_issues',0)}")
    print(f"ERRORS            {errors}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
