import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from LLM import UnifiedLLM


# ========================
# Args
# ========================

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',      type=str, default='./PerMemBench')
parser.add_argument('--output_dir',    type=str, default='./results/greedy_gating')
parser.add_argument('--llm_model',     type=str, default='gpt-5-mini')
parser.add_argument('--current_chars', type=int, default=2000,
                    help='Max characters of current session dialogue')
parser.add_argument('--uuid',          type=str, default=None)
parser.add_argument('--limit',         type=int, default=None)
args = parser.parse_args()


# ========================
# LLM
# ========================

def make_llm(model: str, temperature: float = 0.0) -> UnifiedLLM:
    if "claude" in model.lower():
        return UnifiedLLM(provider="claude", model=model, temperature=temperature)
    elif "/" in model:
        return UnifiedLLM(provider="vllm", model=model, temperature=temperature)
    else:
        return UnifiedLLM(provider="openai", model=model, temperature=temperature)

# ========================
# Prompt
# ========================

PREDICT_PROMPT = """\
You are a memory policy agent for an AI assistant. Your job is to decide whether the content of a conversation session should be stored in long-term memory.

## Current Session Dialogue
{current_dialogue}

## Your Tasks

1. **memory_required**: Should this session be stored in memory?
   - true  = long-horizon session. This session is part of an ongoing project or recurring goal.
   - false = transient session. This session is standalone and self-contained.

   Signals for true:
   - User is working on a specific ongoing project (mentions goals, previous decisions, long-term plans)
   - The dialogue references something being built or developed over time

   Signals for false:
   - Query is self-contained and informational (one-time lookup, general question)
   - No reference to past decisions or future plans


Respond ONLY with a JSON object:
{{
  "memory_required": <true | false>
}}
"""


# ========================
# Data loading
# ========================

def load_sessions(user_dir: Path) -> list[dict]:
    sessions = []
    for f in sorted(user_dir.glob('session_*.json')):
        try:
            sessions.append(json.load(open(f, encoding='utf-8')))
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")
    sessions.sort(key=lambda s: s.get('session_id', 0))
    return sessions


def dialogue_to_text(dialogue: list[dict], max_chars: int) -> str:
    lines = []
    for turn in dialogue:
        role    = turn.get('role', '').capitalize()
        content = turn.get('content', '').strip()
        lines.append(f"{role}: {content}")
    full = '\n'.join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + ' ...[truncated]'
    return full


# ========================
# LLM prediction
# ========================

def predict_session(
    llm: UnifiedLLM,
    current_session: dict,
    current_chars: int,
) -> dict:
    current_text    = dialogue_to_text(current_session.get('dialogue', []), current_chars)

    prompt = PREDICT_PROMPT.format(
        current_dialogue=current_text,
    )

    try:
        raw   = llm.chat(prompt)
        clean = raw.replace('```json', '').replace('```', '').strip()
        result = json.loads(clean)
        return result
    except Exception as e:
        print(f"    [WARN] LLM predict failed: {e}")
        return {
            'memory_required': True,
            'reasoning':       f'prediction failed: {e}',
        }


# ========================
# Per-UUID processing
# ========================

def process_uuid(
    uuid: str,
    user_dir: Path,
    llm: UnifiedLLM,
    output_dir: Path,
    current_chars: int,
) -> list[dict]:
    sessions  = load_sessions(user_dir)
    if not sessions:
        return []

    session_records: list[dict] = []

    for session in sessions:
        session_id  = session.get('session_id', 0)
        gt_required = session.get('memory_required', True)

        # ── LLM prediction ─────────────────────────────────────────
        # input: current dialogue (session-only)
        # never pass GT field (memory_required)
        import time
        start_time = time.time()
        pred = predict_session(
            llm=llm,
            current_session=session,
            current_chars=current_chars,
        )
        elapsed_time = time.time() - start_time
        print(f"      [TIME] predict_session took {elapsed_time:.3f} seconds.")
   

        pred_required = pred.get('memory_required', True)
        reasoning     = pred.get('reasoning', '')

        required_correct = (pred_required == gt_required)

        record = {
            'uuid':             uuid,
            'session_id':       session_id,
            'n_history':        0,   # session-only greedy
            'gt_required':      gt_required,
            'pred_required':    pred_required,
            'required_correct': required_correct,
            'reasoning':        reasoning,
        }
        session_records.append(record)

        req_icon = 'O' if required_correct else 'X'
        print(f"    S{session_id:02d} (hist={0:2d})"
              f"  req[{req_icon}] pred={str(pred_required):5s}  gt={str(gt_required):5s}")

    # ── aggregate ─────────────────────────────────────────────────
    n       = len(session_records)
    req_acc = sum(r['required_correct'] for r in session_records) / n if n else None

    uuid_result = {
        'uuid':              uuid,
        'n_sessions':        n,
        'required_accuracy': round(req_acc, 4) if req_acc is not None else None,
        'session_records':   session_records,
        'evaluated_at':      datetime.now().isoformat(),
    }

    out_path = output_dir / 'per_uuid' / f"{uuid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(uuid_result, f, ensure_ascii=False, indent=2)

    print(f"  [SAVED] per_uuid/{uuid}.json")
    if req_acc is not None:
        print(f"  req_acc={req_acc:.3f}")

    return session_records


# ========================
# Main
# ========================

def main():
    llm = make_llm(args.llm_model)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) / f"{args.llm_model.replace('/', '-')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.uuid:
        uuids = [args.uuid]
    else:
        uuids = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if args.limit:
        uuids = uuids[:args.limit]

    print(f"data_dir         : {data_dir}")
    print(f"output_dir       : {output_dir}")
    print(f"llm_model        : {args.llm_model}")
    print("history_k        : 0  (session-only: no past sessions)")
    print("history_chars    : 0  (session-only: no past sessions)")
    print(f"UUIDs            : {len(uuids)}\n")

    # aggregate CSV
    agg_csv_path   = output_dir / 'aggregate.csv'
    csv_fieldnames = [
        'uuid', 'session_id', 'n_history',
        'gt_required', 'pred_required', 'required_correct',
        'reasoning',
    ]
    write_header = not agg_csv_path.exists()
    agg_csv    = open(agg_csv_path, 'a', newline='', encoding='utf-8')
    agg_writer = csv.DictWriter(agg_csv, fieldnames=csv_fieldnames, extrasaction='ignore')
    if write_header:
        agg_writer.writeheader()

    all_records = []

    for i, uuid in enumerate(uuids, 1):
        user_dir = data_dir / uuid
        if not user_dir.exists():
            print(f"[{i}/{len(uuids)}] {uuid[:8]}: not found, skip")
            continue
        print(f"\n=== [{i}/{len(uuids)}] UUID: {uuid} ===")

        records = process_uuid(
            uuid=uuid,
            user_dir=user_dir,
            llm=llm,
            output_dir=output_dir,
            current_chars=args.current_chars,
        )
        for r in records:
            agg_writer.writerow(r)
        agg_csv.flush()
        all_records.extend(records)

    agg_csv.close()

    # ── Summary ──────────────────────────────────────────────
    n  = len(all_records)
    tp = sum(1 for r in all_records if     r['gt_required'] and     r['pred_required'])
    fp = sum(1 for r in all_records if not r['gt_required'] and     r['pred_required'])
    fn = sum(1 for r in all_records if     r['gt_required'] and not r['pred_required'])
    tn = sum(1 for r in all_records if not r['gt_required'] and not r['pred_required'])

    precision = tp / (tp + fp) if (tp + fp) else None
    recall    = tp / (tp + fn) if (tp + fn) else None
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision and recall) else None

    summary = {
        'total_sessions':    n,
        'history_k':         0,
        'history_chars':     0,
        'required_accuracy': round(sum(r['required_correct'] for r in all_records) / n, 4) if n else None,
        'confusion':         {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
        'precision':         round(precision, 4) if precision else None,
        'recall':            round(recall, 4)    if recall    else None,
        'f1':                round(f1, 4)        if f1        else None,
        'evaluated_at':      datetime.now().isoformat(),
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 60)
    print(f"DONE | {n} sessions")
    print(f"  required_accuracy : {summary['required_accuracy']}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision={summary['precision']}  recall={summary['recall']}  f1={summary['f1']}")
    print(f"\nResults -> {output_dir}")


if __name__ == '__main__':
    main()
