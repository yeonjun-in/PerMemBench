# %% [markdown]
# # Snapshot Augmentation
# 
# Simulate memory system behavior from existing snapshots without re-running the memory system.
# 
# **Two interventions:**
# 1. **Skip sessions** – given a set of session IDs to skip per user, skip that session's memory writes
# 2. **Entry budget** – cap the memory bank at N entries; when exceeded, evict oldest-first (by `insert_order`)

# %%
import json
import os
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_snapshot_dir", type=str, default="results/run_mem0/sys=mem0__gran=session__bud=-1_10000000000000000__mllm=openai-gpt-5-mini__d=v5/snapshots/mem0_session")
parser.add_argument("--entry_budget", type=int, default=300)
parser.add_argument("--skip_sessions", type=str, default=None)
args = parser.parse_args()

# %% [markdown]
# ## Configuration

# %%
SOURCE_SNAPSHOT_DIR = args.source_snapshot_dir

# Budget: max number of memory entries allowed at any time.
# Set to None for no limit.
ENTRY_BUDGET: Optional[int] = args.entry_budget

# Skip sessions: dict mapping uuid -> set/list of session_ids (1-based int) to skip.
# Use {} to skip nothing.
# Example: {"00aefb8e6cfd47dc939d6d3b30a5aefb": {3, 5, 7}}
if args.skip_sessions is None:
    PTS = 'base'
    skip_sessions = None
elif args.skip_sessions == 'gt':
    skip_sessions = {}
    PTS = 'gt'
    for uuid in os.listdir(SOURCE_SNAPSHOT_DIR):
        temp = []
        for file_ in os.listdir(os.path.join(SOURCE_SNAPSHOT_DIR, uuid)):
            with open(os.path.join(SOURCE_SNAPSHOT_DIR, uuid, file_), 'r') as f:
                data = json.load(f)
                if not data['memory_required']:
                    temp.append(data['session_id'])
        skip_sessions[uuid] = temp

else:
    skip_sessions_dir = os.path.join('results', args.skip_sessions, 'per_uuid')
    PTS = args.skip_sessions.replace('/', '_')
    skip_sessions = {}
    for f in os.listdir(skip_sessions_dir):
        with open(os.path.join(skip_sessions_dir, f), 'r') as f:
            data = json.load(f)
        skip_sessions[data['uuid']] = [sr['session_id'] for sr in data['session_records'] if not sr['pred_required']]


# Output directory
budget_str = str(ENTRY_BUDGET) if ENTRY_BUDGET is not None else "inf"
OUTPUT_SNAPSHOT_DIR = SOURCE_SNAPSHOT_DIR.replace(SOURCE_SNAPSHOT_DIR.split('/')[1], SOURCE_SNAPSHOT_DIR.split('/')[1] + f'_{PTS}').replace('bud=-1_10000000000000000', f'bud=-1_{int(ENTRY_BUDGET)}')

print(f"Source : {SOURCE_SNAPSHOT_DIR}")
print(f"Output : {OUTPUT_SNAPSHOT_DIR}")
print(f"Budget : {ENTRY_BUDGET} entries")
print(f"Skips  : {args.skip_sessions}")

SOURCE_SNAPSHOT_DIR = Path(SOURCE_SNAPSHOT_DIR)
OUTPUT_SNAPSHOT_DIR = Path(OUTPUT_SNAPSHOT_DIR)

# %% [markdown]
# ## Core Simulation

# %%
def simulate_user(
    uuid: str,
    source_dir: Path,
    output_dir: Path,
    entry_budget: Optional[int],
    skip_session_ids: set[int],
) -> dict:
    """
    Simulate memory system for one user.

    Walks session files in order, applying skip and budget logic,
    and writes modified snapshots to output_dir / uuid /.

    Returns summary stats.
    """
    user_src = source_dir / uuid
    user_out = output_dir / uuid
    user_out.mkdir(parents=True, exist_ok=True)

    session_files = sorted(user_src.glob("session_*.json"))
    if not session_files:
        return {"uuid": uuid, "sessions": 0}

    # Current memory bank: list of memory dicts, maintained in insert_order
    current_bank: list[dict] = []
    total_skipped = 0
    total_evicted = 0

    for sf in session_files:
        with open(sf) as f:
            orig = json.load(f)

        session_id: int = orig["session_id"]
        session_file_name: str = sf.name  # e.g. "session_0003.json"
        skipped = session_id in skip_session_ids

        # New memories written by the memory system in this session
        new_memories = [
            m for m in orig["memories"]
            if m["session_file"] == session_file_name
        ]

        if not skipped:
            current_bank.extend(new_memories)
        else:
            total_skipped += 1

        # Apply entry budget: evict oldest (lowest insert_order) first
        n_evicted = 0
        if entry_budget is not None and len(current_bank) > entry_budget:
            current_bank.sort(key=lambda m: m["insert_order"])
            overflow = len(current_bank) - entry_budget
            current_bank = current_bank[overflow:]  # drop oldest
            n_evicted = overflow
            total_evicted += n_evicted

        # Build output snapshot
        out = dict(orig)  # shallow copy; we'll replace mutable fields
        out["memories"] = list(current_bank)  # snapshot of bank now
        out["n_entries_after"] = len(current_bank)
        out["total_tokens_after"] = sum(m["token_count"] for m in current_bank)
        out["n_evicted_by_budget"] = n_evicted

        if skipped:
            out["n_written_this_session"] = 0
            out["written_this_session"] = []
            out["write_evidence"] = ""

        with open(user_out / sf.name, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return {
        "uuid": uuid,
        "sessions": len(session_files),
        "skipped": total_skipped,
        "evicted": total_evicted,
        "final_bank_size": len(current_bank),
    }

# %%
def run_simulation(
    source_dir: Path,
    output_dir: Path,
    entry_budget: Optional[int],
    skip_sessions: Optional[dict[str, set[int]]],
) -> list[dict]:
    """
    Run simulation for all users found under source_dir.
    skip_sessions=None  → base (skip nothing)
    skip_sessions={}    → skip_sessions dict loaded from gt or predictions
    """
    uuid_dirs = sorted([p for p in source_dir.iterdir() if p.is_dir()])
    print(f"Found {len(uuid_dirs)} users")

    results = []
    for user_dir in tqdm(uuid_dirs, desc="Simulating users"):
        uuid = user_dir.name
        skip_ids = set(skip_sessions.get(uuid, [])) if skip_sessions is not None else set()
        stats = simulate_user(
            uuid=uuid,
            source_dir=source_dir,
            output_dir=output_dir,
            entry_budget=entry_budget,
            skip_session_ids=skip_ids,
        )
        results.append(stats)

    return results

# %% [markdown]
# ## Run

# %%
results = run_simulation(
    source_dir=SOURCE_SNAPSHOT_DIR,
    output_dir=OUTPUT_SNAPSHOT_DIR,
    entry_budget=ENTRY_BUDGET,
    skip_sessions=skip_sessions,
)

print(f"\nDone. Output → {OUTPUT_SNAPSHOT_DIR}")
total_evicted = sum(r.get("evicted", 0) for r in results)
total_skipped = sum(r.get("skipped", 0) for r in results)
print(f"Total sessions skipped : {total_skipped}")
print(f"Total entries evicted  : {total_evicted}")

# %% [markdown]
# ## Quick Sanity Check

# %%
# Check one user to verify correctness
CHECK_UUID = results[0]["uuid"]
CHECK_SESSION = 3  # session_id to inspect

src_file = SOURCE_SNAPSHOT_DIR / CHECK_UUID / f"session_{CHECK_SESSION:04d}.json"
out_file = OUTPUT_SNAPSHOT_DIR / CHECK_UUID / f"session_{CHECK_SESSION:04d}.json"

src = json.loads(src_file.read_text())
out = json.loads(out_file.read_text())

print(f"User: {CHECK_UUID}  |  Session: {CHECK_SESSION}")
print(f"  Source → entries: {src['n_entries_after']}, evicted: {src['n_evicted_by_budget']}, written: {src['n_written_this_session']}")
print(f"  Output → entries: {out['n_entries_after']}, evicted: {out['n_evicted_by_budget']}, written: {out['n_written_this_session']}")

if ENTRY_BUDGET is not None:
    assert out["n_entries_after"] <= ENTRY_BUDGET, "Budget violated!"
    print(f"  Budget check OK (≤ {ENTRY_BUDGET})")

# %% [markdown]
# ## Summary Table

# %%
import pandas as pd

df = pd.DataFrame(results)
print(df.describe())
df.head(10)

# %%



