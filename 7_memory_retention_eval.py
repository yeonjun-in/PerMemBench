import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from LLM import UnifiedLLM
from memory_bank import EmbeddingModel, cosine_similarity

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot_dir",  default='')
parser.add_argument("--skeleton_dir",  default="./skeleton_dialogues")
parser.add_argument("--output_dir",    default='memory_retention_results')
parser.add_argument("--uuid",          default='036886affea947b48448038332105e45', help="Process a single UUID (default: all)")
parser.add_argument("--top_k",         type=int, default=5)
parser.add_argument("--write_top_k",   type=int, default=10,
                    help="Top-k fact-based retrieval for write eval "
                         "(pass only top-N fact-similar candidates to judge, not full memory)")
parser.add_argument("--agent_model",       default="gpt-5-nano")
parser.add_argument("--judge_model",       default="gpt-5-nano")
parser.add_argument("--agent_temperature", type=float, default=0.0,
                    help="Agent LLM temperature (default: 0.0)")
parser.add_argument("--judge_temperature", type=float, default=0.0,
                    help="Judge LLM temperature (default: 0.0)")
parser.add_argument("--embedding_provider", default="sentence_transformers",
                    choices=["openai", "sentence_transformers"])
parser.add_argument("--embedding_model",    default="all-MiniLM-L6-v2",
                    choices=["text-embedding-3-small", "all-MiniLM-L6-v2"])
parser.add_argument("--skip_existing", action="store_true", default=False,
                    help="If per_fact JSON exists, skip re-eval and only aggregate summary")
args = parser.parse_args()


# ──────────────────────────────────────────────
# Prompts (same as 6_static_eval_v11)
# ──────────────────────────────────────────────

BINARY_JUDGE_PROMPT = """\
You are checking whether a specific piece of information is contained in a given text.

Fact to find:
"{fact}"

Text to search:
{text}

Does the text above contain the core meaning of this fact?
Answer YES if the fact is clearly expressed (even if worded differently).
Answer NO if the fact is absent or cannot be inferred from the text.

Reply with only YES or NO.
"""

QA_AGENT_PROMPT = """\
You are a helpful AI assistant. Answer the user's question using the provided memory context.
If the memory context contains relevant information, use it to give a specific and accurate answer.
If no relevant memory is available, say so briefly.

## Retrieved Memory Context
{memory_context}

## Question
{question}

Answer concisely and directly.
"""

QA_JUDGE_PROMPT = """\
You are evaluating whether an AI agent correctly answered a question.

## Question
{question}

## Ground Truth Answer
{gt_answer}

## Agent's Response
{agent_response}

Evaluate whether the agent's response contains the exact key terms or specific information present in the ground truth answer.

Rules:
- Score 1.0: the response contains the specific terms, names, numbers, or phrases from the ground truth
- Score 0.0: the response is vague, paraphrased without key terms, incorrect, or missing critical specifics

Respond ONLY with a JSON object:
{{"score": <0.0 | 1.0>}}
"""

def sample_even_with_ends(arr, k):
    n = len(arr)
    if n == k:
        return arr
    if k < 2:
        raise ValueError("k must be at least 2 (includes first and last)")
    if k > n:
        raise ValueError("need k <= len(arr) for sampling without replacement")

    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [arr[i] for i in idx]


# ──────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────

def make_llm(model: str, temperature: float = 0.0) -> UnifiedLLM:
    if "claude" in model.lower():
        return UnifiedLLM(provider="claude", model=model, temperature=temperature)
    elif "/" in model:
        return UnifiedLLM(provider="vllm", model=model, temperature=temperature)
    else:
        return UnifiedLLM(provider="openai", model=model, temperature=temperature)


def binary_judge(llm: UnifiedLLM, fact: str, text: str) -> int:
    if not text.strip():
        return 0
    prompt = BINARY_JUDGE_PROMPT.format(fact=fact, text=text)
    try:
        answer = llm.chat(prompt).strip().upper()
        return 1 if answer.startswith("YES") else 0
    except Exception as e:
        print(f"    [WARN] binary_judge failed: {e}")
        return 0


def build_memory_context(memory_texts: list[str]) -> str:
    if not memory_texts:
        return "(No relevant memory found)"
    lines = []
    for i, txt in enumerate(memory_texts, 1):
        lines.append(f"[Memory {i}]\n{txt}")
    return "\n\n".join(lines)


def run_qa(llm_agent: UnifiedLLM, llm_judge: UnifiedLLM,
           probing_question: str, gt_answer: str,
           retrieved_texts: list[str]) -> tuple[float, str]:
    """Return QA score and agent response."""
    memory_context = build_memory_context(retrieved_texts)
    agent_prompt = QA_AGENT_PROMPT.format(
        memory_context=memory_context, question=probing_question
    )
    try:
        agent_response = llm_agent.chat(agent_prompt)
    except Exception as e:
        print(f"    [WARN] QA agent failed: {e}")
        return 0.0, ""

    if not agent_response.strip():
        return 0.0, ""

    judge_prompt = QA_JUDGE_PROMPT.format(
        question=probing_question,
        gt_answer=gt_answer,
        agent_response=agent_response,
    )
    try:
        raw = llm_judge.chat(judge_prompt)
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return float(result.get("score", 0.0)), agent_response
    except Exception as e:
        print(f"    [WARN] QA judge failed: {e}")
        return 0.0, agent_response


# ──────────────────────────────────────────────
# Snapshot helpers
# ──────────────────────────────────────────────

def load_snapshot(snapshot_path: Path) -> Optional[dict]:
    try:
        with open(snapshot_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot load snapshot {snapshot_path.name}: {e}")
        return None


def has_session_snapshots(dir_path: Path) -> bool:
    return dir_path.is_dir() and any(dir_path.glob("session_*.json"))


def discover_snapshot_uuid_dirs(
    snapshot_root: Path,
    target_uuid: str | None = None,
) -> dict[str, Path]:

    candidates: dict[str, list[Path]] = {}

    # 1) search legacy layout
    if snapshot_root.exists():
        for d in sorted(snapshot_root.iterdir()):
            if not d.is_dir():
                continue
            if has_session_snapshots(d):
                candidates.setdefault(d.name, []).append(d)

    # 2) search new layout
    if snapshot_root.exists():
        for system_dir in sorted(snapshot_root.iterdir()):
            if not system_dir.is_dir():
                continue
            for uuid_dir in sorted(system_dir.iterdir()):
                if not uuid_dir.is_dir():
                    continue
                if has_session_snapshots(uuid_dir):
                    candidates.setdefault(uuid_dir.name, []).append(uuid_dir)

    if target_uuid is not None:
        paths = candidates.get(target_uuid, [])
        if not paths:
            return {}
        if len(paths) == 1:
            return {target_uuid: paths[0]}

        # if duplicate candidates, pick path with most sessions
        scored = sorted(
            paths,
            key=lambda p: (len(list(p.glob("session_*.json"))), str(p)),
            reverse=True,
        )
        chosen = scored[0]
        print(f"[WARN] multiple snapshot dirs for UUID={target_uuid}. Using: {chosen}")
        return {target_uuid: chosen}

    resolved: dict[str, Path] = {}
    for uuid_str, paths in candidates.items():
        if len(paths) == 1:
            resolved[uuid_str] = paths[0]
            continue
        scored = sorted(
            paths,
            key=lambda p: (len(list(p.glob("session_*.json"))), str(p)),
            reverse=True,
        )
        chosen = scored[0]
        print(f"[WARN] multiple snapshot dirs for UUID={uuid_str}. Using: {chosen}")
        resolved[uuid_str] = chosen

    return resolved


def get_memory_texts(snapshot: dict) -> list[str]:
    """Return text list from snapshot memories."""
    texts = []
    for m in snapshot.get("memories", []):
        txt = m.get("memory_text") or m.get("content") or ""
        if txt.strip():
            texts.append(txt.strip())
    return texts


def retrieve_top_k(query: str, memory_texts: list[str],
                   embedder: EmbeddingModel, top_k: int) -> list[str]:
    """Return top-k memories by embedding cosine similarity."""
    if not memory_texts:
        return []
    # return all if fewer memories than top_k
    actual_k = min(top_k, len(memory_texts))
    try:
        query_emb = embedder.embed_one(query)           # (D,)
        mem_embs  = embedder.embed(memory_texts)        # (N, D)
        sims      = cosine_similarity(query_emb, mem_embs)  # (N,)
        indices   = np.argsort(sims)[::-1][:actual_k]
        return [memory_texts[i] for i in indices]
    except Exception as e:
        print(f"    [WARN] retrieve_top_k failed: {e}")
        return memory_texts[:actual_k]


# ──────────────────────────────────────────────
# Skeleton helpers: compute project end sessions
# ──────────────────────────────────────────────

def build_project_end_map(skeleton_sessions: list[dict]) -> dict[tuple, int]:

    project_last: dict[tuple, int] = {}
    for s in skeleton_sessions:
        domain     = s.get("domain", "")
        project_id = s.get("project_id")
        session_id = s.get("session_id", 0)
        if project_id is None:
            continue
        key = (domain, project_id)
        if key not in project_last or session_id > project_last[key]:
            project_last[key] = session_id
    return project_last


# ──────────────────────────────────────────────
# Per-session evaluation
# ──────────────────────────────────────────────

def eval_at_snapshot(
    snapshot: dict,
    fact: str,
    probing_question: str,
    gt_answer: str,
    embedder: EmbeddingModel,
    llm_judge: UnifiedLLM,
    llm_agent: UnifiedLLM,
    top_k: int,
    write_top_k: int,
) -> dict:

    memory_texts = get_memory_texts(snapshot)

    # ── Write Score ──────────────────────────────
    # use fact as query, take top write_top_k candidates, then judge
    

    # Measure time for write_candidates
    t0 = time.time()
    write_candidates = retrieve_top_k(fact, memory_texts, embedder, write_top_k)
    t1 = time.time()
    print(f"[TIMER] retrieve_top_k (write): {t1-t0:.3f} sec")

    # Measure time for join write_evidence
    t2 = time.time()
    write_evidence   = "\n".join(write_candidates)
    t3 = time.time()
    print(f"[TIMER] join write_evidence: {t3-t2:.3f} sec")

    # Measure time for write_score judgment
    t4 = time.time()
    write_score      = binary_judge(llm_judge, fact, write_evidence)
    t5 = time.time()
    print(f"[TIMER] binary_judge (write): {t5-t4:.3f} sec")

    retrieved = write_candidates
    read_score   = 0
    qa_score = 0
    agent_response = ""

    return {
        "session_id":     snapshot["session_id"],
        "write":          write_score,
        "read":           read_score,
        "qa":             round(qa_score, 4),
        'retrieved':      retrieved,
        "agent_response": agent_response,

    }


# ──────────────────────────────────────────────
# Metrics aggregation
# ──────────────────────────────────────────────

def aggregate_metrics(session_scores: list[dict]) -> dict:

    n = len(session_scores)
    if n == 0:
        return {
            "retention_rate":        None,
            "retrieval_rate":        None,
            "avg_qa_score":          None,
            "first_failure_session": None,
            "n_eval_sessions":       0,
        }

    writes = [s["write"] for s in session_scores]
    reads  = [s["read"]  for s in session_scores]
    qas    = [s["qa"]    for s in session_scores]

    retention_rate = sum(writes) / n
    retrieval_rate = sum(reads)  / n
    avg_qa_score   = sum(qas)    / n

    first_failure = None
    for s in session_scores:
        if s["write"] == 0:
            first_failure = s["session_id"]
            break

    return {
        "retention_rate":        round(retention_rate, 4),
        "retrieval_rate":        round(retrieval_rate, 4),
        "avg_qa_score":          round(avg_qa_score, 4),
        "first_failure_session": first_failure,
        "n_eval_sessions":       n,
    }


# ──────────────────────────────────────────────
# Main: process one UUID
# ──────────────────────────────────────────────

def process_uuid(
    uuid_str: str,
    snapshot_uuid_dir: Path,
    skeleton_uuid_dir: Path,
    output_dir: Path,
    embedder: EmbeddingModel,
    llm_judge: UnifiedLLM,
    llm_agent: UnifiedLLM,
    top_k: int,
    write_top_k: int,
    skip_existing: bool,
) -> tuple[list[dict], bool]:

    per_fact_out = output_dir / "per_fact" / f"{uuid_str}.json"

    # ── skip_existing: load completed UUID from per_fact JSON for summary only ──
    if skip_existing and per_fact_out.exists():
        print(f"  [SKIP] already done: {uuid_str}")
        with open(per_fact_out) as f:
            saved = json.load(f)
        rows = []
        for fact_rec in saved:
            row = {
                "uuid":                  uuid_str,
                "source_session_id":     fact_rec["source_session_id"],
                "gt_idx":                fact_rec["gt_idx"],
                "gt_type":               fact_rec["gt_type"],
                "domain":                fact_rec["domain"],
                "project_id":            fact_rec.get("project_id"),
                "fact":                  fact_rec["fact"],
                "valid_through_session": fact_rec["valid_through_session"],
                **fact_rec["metrics"],
            }
            rows.append(row)
        return rows, True   # already_done=True → do not write CSV again

    # ── Load skeleton sessions ──────────────────
    if not skeleton_uuid_dir.exists():
        print(f"  [WARN] skeleton dir not found: {skeleton_uuid_dir}")
        return [], False

    skeleton_sessions = []
    for f in sorted(skeleton_uuid_dir.glob("session_*.json")):
        with open(f) as fp:
            skeleton_sessions.append(json.load(fp))
    skeleton_sessions.sort(key=lambda s: s["session_id"])

    if not skeleton_sessions:
        return [], False

    last_session_id = skeleton_sessions[-1]["session_id"]
    project_end_map = build_project_end_map(skeleton_sessions)

    # ── Load snapshot files ─────────────────────
    snap_files = sorted(snapshot_uuid_dir.glob("session_*.json"))
    snapshots: dict[int, dict] = {}
    for sf in snap_files:
        snap = load_snapshot(sf)
        if snap:
            snapshots[snap["session_id"]] = snap

    if not snapshots:
        print(f"  [WARN] no snapshots found in {snapshot_uuid_dir}")
        return [], False

    # ── Process each fact ───────────────────────
    all_fact_records = []
    aggregate_rows   = []

    for skeleton_session in skeleton_sessions:
        session_id  = skeleton_session["session_id"]
        domain      = skeleton_session.get("domain", "")
        project_id  = skeleton_session.get("project_id")
        gt_memory   = skeleton_session.get("gt_memory", [])

        if not gt_memory:
            continue

        for gt_idx, gt_item in enumerate(gt_memory):
            gt_type          = gt_item.get("type", "")
            fact             = gt_item.get("fact", "").strip()
            probing_question = gt_item.get("probing_question", "").strip()
            gt_answer        = gt_item.get("answer", "").strip()

            if not fact or not probing_question or not gt_answer:
                continue

            # determine validity window
            if gt_type == "user_profile":
                valid_through = last_session_id
            elif gt_type == "ongoing_state":
                key = (domain, project_id)
                valid_through = project_end_map.get(key, last_session_id)
            else:
                continue  # skip unknown type

            # snapshot session ids to evaluate (source ~ valid_through)
            eval_session_ids = sorted(
                sid for sid in snapshots
                if session_id <= sid <= valid_through
            )

            if not eval_session_ids:
                print(f"    [WARN] no snapshots in range "
                      f"[{session_id}, {valid_through}] for fact: {fact[:50]}")
                continue

            print(f"    Fact S{session_id} gt{gt_idx} ({gt_type}): "
                  f"eval T={eval_session_ids[0]}~{eval_session_ids[-1]} "
                  f"({len(eval_session_ids)} sessions) | {fact[:50]}...")

            session_scores = []
            for t in sample_even_with_ends(eval_session_ids, min(20, len(eval_session_ids))):
                scores = eval_at_snapshot(
                    snapshot=snapshots[t],
                    fact=fact,
                    probing_question=probing_question,
                    gt_answer=gt_answer,
                    embedder=embedder,
                    llm_judge=llm_judge,
                    llm_agent=llm_agent,
                    top_k=top_k,
                    write_top_k=write_top_k,
                )
                session_scores.append(scores)
                print(f"      T={t:02d} write={scores['write']} "
                      f"read={scores['read']} qa={scores['qa']:.2f}")

            metrics = aggregate_metrics(session_scores)

            fact_record = {
                "uuid":                  uuid_str,
                "source_session_id":     session_id,
                "gt_idx":                gt_idx,
                "gt_type":               gt_type,
                "domain":                domain,
                "project_id":            project_id,
                "fact":                  fact,
                "probing_question":      probing_question,
                "gt_answer":             gt_answer,
                "valid_through_session": valid_through,
                "session_scores":        session_scores,
                "metrics":               metrics,
                "evaluated_at":          datetime.now().isoformat(),
            }
            all_fact_records.append(fact_record)

            aggregate_rows.append({
                "uuid":                  uuid_str,
                "source_session_id":     session_id,
                "gt_idx":                gt_idx,
                "gt_type":               gt_type,
                "domain":                domain,
                "project_id":            project_id,
                "fact":                  fact,
                "valid_through_session": valid_through,
                **metrics,
            })

    # ── Save per-fact JSON ──────────────────────
    per_fact_out.parent.mkdir(parents=True, exist_ok=True)
    with open(per_fact_out, "w", encoding="utf-8") as f:
        json.dump(all_fact_records, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {per_fact_out.name} ({len(all_fact_records)} facts)")

    return aggregate_rows, False   # already_done=False → write CSV normally


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    snapshot_root = Path(args.snapshot_dir)
    skeleton_root = Path(args.skeleton_dir)
    output_dir    = Path(args.snapshot_dir.replace('snapshots', 'longitudinal_results_approx_20') + '/' + args.embedding_model + '_' + 'top'+str(args.top_k) + '_' + args.judge_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder  = EmbeddingModel(provider=args.embedding_provider,
                               model=args.embedding_model)
    llm_judge = make_llm(args.judge_model, temperature=args.judge_temperature)
    llm_agent = make_llm(args.agent_model, temperature=args.agent_temperature)

    # UUID list + snapshot path mapping (auto legacy/new layout)
    snapshot_uuid_map = discover_snapshot_uuid_dirs(snapshot_root, args.uuid)
    uuid_list = sorted(snapshot_uuid_map.keys())

    print(f"Snapshot dir : {snapshot_root}")
    print(f"Skeleton dir : {skeleton_root}")
    print(f"Output dir   : {output_dir}")
    print(f"UUIDs        : {len(uuid_list)}")
    print(f"top_k        : {args.top_k}  (read/qa)")
    print(f"write_top_k  : {args.write_top_k}  (write judge candidates)")
    print(f"agent_model  : {args.agent_model}")
    print(f"judge_model  : {args.judge_model}\n")

    # CSV setup
    agg_csv_path = output_dir / "aggregate.csv"
    csv_fieldnames = [
        "uuid", "source_session_id", "gt_idx", "gt_type",
        "domain", "project_id", "fact", "valid_through_session",
        "retention_rate", "retrieval_rate", "avg_qa_score",
        "first_failure_session", "n_eval_sessions",
    ]
    write_header = not agg_csv_path.exists()
    agg_csv = open(agg_csv_path, "a", newline="", encoding="utf-8")
    agg_writer = csv.DictWriter(agg_csv, fieldnames=csv_fieldnames)
    if write_header:
        agg_writer.writeheader()

    all_rows = []  # for summary aggregation (includes skipped)

    for i, uuid_str in enumerate(uuid_list, 1):
        print(f"\n=== [{i}/{len(uuid_list)}] UUID: {uuid_str} ===")

        snapshot_uuid_dir = snapshot_uuid_map[uuid_str]
        skeleton_uuid_dir = skeleton_root / uuid_str

        rows, already_done = process_uuid(
            uuid_str=uuid_str,
            snapshot_uuid_dir=snapshot_uuid_dir,
            skeleton_uuid_dir=skeleton_uuid_dir,
            output_dir=output_dir,
            embedder=embedder,
            llm_judge=llm_judge,
            llm_agent=llm_agent,
            top_k=args.top_k,
            write_top_k=args.write_top_k,
            skip_existing=args.skip_existing,
        )

        # if already_done=True, skip duplicate CSV write
        if not already_done:
            for row in rows:
                agg_writer.writerow(row)
            agg_csv.flush()

        all_rows.extend(rows)

    agg_csv.close()

    # ── Summary ──────────────────────────────────
    def mean(lst):
        lst = [x for x in lst if x is not None]
        return round(sum(lst) / len(lst), 4) if lst else None

    summary = {"total_facts": len(all_rows), "by_gt_type": {}}

    for gt_type in ("user_profile", "ongoing_state"):
        subset = [r for r in all_rows if r["gt_type"] == gt_type]
        if not subset:
            continue
        summary["by_gt_type"][gt_type] = {
            "n_facts":        len(subset),
            "retention_rate": mean([r["retention_rate"] for r in subset]),
            "retrieval_rate": mean([r["retrieval_rate"] for r in subset]),
            "avg_qa_score":   mean([r["avg_qa_score"]   for r in subset]),
        }

    summary["overall"] = {
        "retention_rate": mean([r["retention_rate"] for r in all_rows]),
        "retrieval_rate": mean([r["retrieval_rate"] for r in all_rows]),
        "avg_qa_score":   mean([r["avg_qa_score"]   for r in all_rows]),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE | Total facts: {len(all_rows)}")
    for gt_type, m in summary["by_gt_type"].items():
        print(f"\n  [{gt_type}]  n={m['n_facts']}")
        print(f"    retention_rate : {m['retention_rate']}")
        print(f"    retrieval_rate : {m['retrieval_rate']}")
        print(f"    avg_qa_score   : {m['avg_qa_score']}")
    print(f"\n  [overall]")
    print(f"    retention_rate : {summary['overall']['retention_rate']}")
    print(f"    retrieval_rate : {summary['overall']['retrieval_rate']}")
    print(f"    avg_qa_score   : {summary['overall']['avg_qa_score']}")
    print(f"\nResults → {output_dir}")


if __name__ == "__main__":
    main()
