"""
2_2_generate_pattern_shift.py

Stage 2.2: 사용 패턴 전환 — 순수 룰 기반 샘플링. LLM 없음.

고정 룰 (3~4가지):
  1. mem_to_oneoff  : 기존 memory_required=True 도메인 1개 → memory_required=False (oneoff)
  2. added_mem      : 미사용 풀에서 memory_required=True 도메인 1개 추가
  3. added_oneoff   : 미사용 풀에서 memory_required=False 도메인 1개 추가 (풀 없으면 skip)

전환 서사는 2_3에서 skeleton 생성 시 처리.

Usage:
  python 2_2_generate_pattern_shift.py \\
      --input_dir ./life_timelines_v5 \\
      --persona_dir ./final_persona_metadata_v3 \\
      --output_dir ./pattern_shifts \\
      --seed 42
"""

import json
import os
import argparse
import random
from glob import glob
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


FREQUENCY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def weighted_sample_one(items: list, rng: random.Random):
    if not items:
        return None
    weights = [FREQUENCY_WEIGHTS.get(d.get("frequency", "medium"), 2) for d in items]
    return rng.choices(items, weights=weights, k=1)[0]


def build_available_pool(persona_metadata_path: str, used_domain_names: set) -> dict:
    """미사용 도메인을 memory_required 기준으로 분리해서 반환."""
    if not os.path.exists(persona_metadata_path):
        return {"mem": [], "oneoff": []}

    with open(persona_metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mem_pool, oneoff_pool = [], []
    for d in meta.get("domains", []):
        if not d.get("use"):
            continue
        if d["domain_name"] in used_domain_names:
            continue
        if d.get("frequency") is None:
            continue
        entry = {
            "domain_name": d["domain_name"],
            "memory_required": d.get("memory_required", False),
            "frequency": d["frequency"],
            "reason": d.get("reason", ""),
        }
        if d.get("memory_required"):
            mem_pool.append(entry)
        else:
            oneoff_pool.append(entry)

    return {"mem": mem_pool, "oneoff": oneoff_pool}


def sample_changes(
    domain_skeletons: list,
    oneoff_sessions: list,
    available_pool: dict,
    rng: random.Random,
):
    """
    룰 기반 샘플링.

    1. mem_to_oneoff : domain_skeletons 중 1개 → oneoff으로 강등 (필수)
    2. added_mem     : available_pool["mem"] 중 frequency 가중 1개 (필수)
    3. added_oneoff  : available_pool["oneoff"] 중 frequency 가중 1개 (없으면 None)
    """
    if not domain_skeletons:
        return None, "no domain_skeletons to demote"
    if not available_pool["mem"]:
        return None, "no unused mem domain in pool"

    # 1. mem → oneoff
    ds = rng.choice(domain_skeletons)
    mem_to_oneoff = {
        "domain_name": ds["domain_name"],
        "frequency": ds.get("frequency", "medium"),
        "memory_required": False,
    }

    # 2. 새 mem 도메인 (필수)
    added_mem = weighted_sample_one(available_pool["mem"], rng)

    # 3. 새 oneoff 도메인 (선택)
    added_oneoff = weighted_sample_one(available_pool["oneoff"], rng)  # 없으면 None

    return {
        "mem_to_oneoff": mem_to_oneoff,
        "added_mem":     added_mem,
        "added_oneoff":  added_oneoff,
    }, None


def process_timeline_file(
    filepath: str,
    persona_dir: str,
    output_dir: str,
    seed: int,
    overwrite: bool = False,
) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    uuid = data["uuid"]
    output_path = os.path.join(output_dir, f"{uuid}.json")

    if not overwrite and os.path.exists(output_path):
        return {"uuid": uuid, "skipped": True}

    domain_skeletons = data.get("domain_skeletons", [])
    oneoff_sessions  = data.get("oneoff_sessions", [])
    total_months     = data.get("timeline", {}).get("total_months", 24)

    if not domain_skeletons:
        return {"uuid": uuid, "skipped": True, "reason": "no domain_skeletons"}
    if not oneoff_sessions:
        return {"uuid": uuid, "skipped": True, "reason": "no oneoff_sessions"}

    used_names = (
        {ds["domain_name"] for ds in domain_skeletons}
        | {b["domain_name"] for b in oneoff_sessions}
    )

    persona_path = os.path.join(persona_dir, f"{uuid}.json")
    available_pool = build_available_pool(persona_path, used_names)

    rng = random.Random(seed ^ (hash(uuid) & 0xFFFFFFFF))
    changes, err = sample_changes(domain_skeletons, oneoff_sessions, available_pool, rng)

    if changes is None:
        return {"uuid": uuid, "skipped": True, "reason": err}

    added_domains = [changes["added_mem"]]
    if changes["added_oneoff"] is not None:
        added_domains.append(changes["added_oneoff"])

    result = {
        "uuid": uuid,
        "transition_month": total_months + 1,
        "changes": changes,
        # 2_3 인터페이스
        "mem_to_oneoff_domains": [changes["mem_to_oneoff"]["domain_name"]],
        "added_domains":         added_domains,
        "phase1_domain_summary": {
            "total_months": total_months,
            "memory_required_domains": [
                {"domain_name": ds["domain_name"], "frequency": ds.get("frequency")}
                for ds in domain_skeletons
            ],
            "oneoff_domains": [
                {"domain_name": b["domain_name"], "frequency": b.get("frequency")}
                for b in oneoff_sessions
            ],
        },
        "available_pool_size": {
            "mem":   len(available_pool["mem"]),
            "oneoff": len(available_pool["oneoff"]),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2.2: Rule-based domain change sampling (no LLM)"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input_file", type=str, default=None)
    input_group.add_argument("--input_dir",  type=str, default="./life_timelines_v5")
    parser.add_argument("--persona_dir", type=str, default="./final_persona_metadata_v3")
    parser.add_argument("--output_dir",  type=str, default="./pattern_shifts")
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--seed",        type=int, default=42,
                        help="샘플링 시드 (per-persona: seed ^ hash(uuid)). 기본값: 42")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(glob(os.path.join(args.input_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]

    if args.limit:
        files = files[:args.limit]

    print(f"Files       : {len(files)}")
    print(f"Persona dir : {args.persona_dir}")
    print(f"Output      : {args.output_dir}")
    print(f"Seed        : {args.seed}")
    print(f"Rules       : mem→oneoff | +mem | +oneoff(opt)\n")

    success = skipped = errors = 0
    for filepath in tqdm(files, desc="Timelines"):
        try:
            result = process_timeline_file(
                filepath, args.persona_dir, args.output_dir,
                args.seed, args.overwrite,
            )
            if result.get("skipped"):
                skipped += 1
                print(f"  skip {result['uuid'][:8]}: {result.get('reason', '')}")
            else:
                c = result["changes"]
                oneoff_str = c["added_oneoff"]["domain_name"] if c["added_oneoff"] else "N/A"
                print(
                    f"  ok   {result['uuid'][:8]}  "
                    f"mem→oneoff={c['mem_to_oneoff']['domain_name']!r} | "
                    f"+mem={c['added_mem']['domain_name']!r} | "
                    f"+oneoff={oneoff_str!r}"
                )
                success += 1
        except Exception as e:
            errors += 1
            print(f"  err  {filepath}: {e}")

    print(f"\nDone - success: {success}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()