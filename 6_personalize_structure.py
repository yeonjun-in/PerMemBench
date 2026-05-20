import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from LLM import UnifiedLLM


# ========================
# Args
# ========================

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',      type=str, default='./skeleton_dialogues')
parser.add_argument('--output_dir',    type=str, default='./results/personalize_structure')
parser.add_argument('--llm_model',     type=str, default='gpt-5-mini')
parser.add_argument('--window_k',      type=int, default=10,
                    help='Update personal note every N sessions')
parser.add_argument('--current_chars', type=int, default=2000,
                    help='Max session dialogue characters when building session records')
parser.add_argument('--uuid',          type=str, default=None)
parser.add_argument('--limit',         type=int, default=None)
args = parser.parse_args()


# ========================
# LLM
# ========================

def make_llm(model: str, temperature: float = 0.0) -> UnifiedLLM:
    if "claude" in model.lower():
        return UnifiedLLM(provider="claude", model=model, temperature=temperature)
    if "/" in model:
        return UnifiedLLM(provider="vllm", model=model, temperature=temperature)
    return UnifiedLLM(provider="openai", model=model, temperature=temperature)


# ========================
# Prompts
# ========================

EXTRACT_PROMPT = """\
You are analyzing a single conversation session between a user and an AI assistant.

## Session Dialogue
{dialogue}

## Task
Extract a structured summary of this session. Respond ONLY with a JSON object:
{{
  "purpose": "<1 sentence: what the user wanted to accomplish in this session>",
  "summary": "<1-2 sentences: what was discussed and accomplished>",
  "topic": "<short phrase: the domain or subject area, e.g. 'Python caching', 'fitness routine', 'travel planning'>"
}}
"""

UPDATE_NOTE_PROMPT = """\
You are managing a personal note that tracks a user's ongoing projects and usage patterns across AI assistant sessions.

## Current Personal Note
{current_note}

## New Session Records (session_id {window_start} ~ {window_end})
{session_records}

## Task
Update the personal note by analyzing the new sessions in context of the existing note.

Rules:
1. Group sessions that belong to the same ongoing project or closely related goal.
2. Sessions that are self-contained one-off requests (isolated) should go in isolated_sessions.
3. You MAY reassign previously isolated sessions into a project if new evidence connects them.
4. You MAY add new sessions to existing projects if they are continuations.
5. You MAY create new projects for newly discovered clusters.
6. Assign status "ongoing" if the project seems active, "completed" if it seems finished.
7. Every session_id that has appeared so far must appear in exactly one place:
   either in one project's session_ids, or in isolated_sessions.

Respond ONLY with the full updated personal note as a JSON object:
{{
  "projects": [
    {{
      "project_id": "<string, e.g. P1>",
      "label": "<short project name>",
      "core_topic": "<central topic or technology>",
      "session_ids": [<list of int>],
      "status": "ongoing" | "completed"
    }}
  ],
  "isolated_sessions": [<list of int>]
}}
"""


# ========================
# Helpers
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
    text = '\n'.join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + ' ...[truncated]'
    return text


def empty_note() -> dict:
    return {"projects": [], "isolated_sessions": []}


def note_to_str(note: dict) -> str:
    return json.dumps(note, ensure_ascii=False, indent=2)


def parse_llm_json(raw: str) -> dict:
    clean = raw.replace('```json', '').replace('```', '').strip()
    return json.loads(clean)


# ========================
# LLM calls
# ========================

def extract_session_record(llm: UnifiedLLM, session: dict, current_chars: int) -> dict:
    """Extract {purpose, summary, topic} from one session."""
    dialogue_text = dialogue_to_text(session.get('dialogue', []), current_chars)
    prompt = EXTRACT_PROMPT.format(dialogue=dialogue_text)
    try:
        raw = llm.chat(prompt)
        result = parse_llm_json(raw)
        result['session_id'] = session.get('session_id', 0)
        return result
    except Exception as e:
        print(f"    [WARN] extract_session_record failed (S{session.get('session_id')}): {e}")
        return {
            'session_id': session.get('session_id', 0),
            'purpose': '(extraction failed)',
            'summary': '(extraction failed)',
            'topic':   '(extraction failed)',
        }


def update_personal_note(
    llm: UnifiedLLM,
    current_note: dict,
    window_records: list[dict],
) -> dict:
    """Update personal_note from current note and new window session records."""
    records_text = '\n'.join(
        json.dumps(r, ensure_ascii=False) for r in window_records
    )
    window_ids = [r['session_id'] for r in window_records]
    prompt = UPDATE_NOTE_PROMPT.format(
        current_note=note_to_str(current_note),
        session_records=records_text,
        window_start=min(window_ids),
        window_end=max(window_ids),
    )
    try:
        raw = llm.chat(prompt)
        updated = parse_llm_json(raw)
        # ensure required keys
        updated.setdefault('projects', [])
        updated.setdefault('isolated_sessions', [])
        return updated
    except Exception as e:
        print(f"    [WARN] update_personal_note failed: {e}")
        # on failure, keep existing note; mark new sessions as isolated
        fallback = dict(current_note)
        fallback['isolated_sessions'] = (
            fallback.get('isolated_sessions', []) + window_ids
        )
        return fallback


# ========================
# classification logic
# ========================

def classify_from_note(note: dict) -> dict[int, bool]:
    """
    Return {session_id: memory_required} from personal_note.
    True if in projects; False if in isolated_sessions.
    """
    result: dict[int, bool] = {}
    for project in note.get('projects', []):
        for sid in project.get('session_ids', []):
            result[sid] = True
    for sid in note.get('isolated_sessions', []):
        result[sid] = False
    return result


# ========================
# Per-UUID
# ========================

def process_uuid(
    uuid: str,
    user_dir: Path,
    llm: UnifiedLLM,
    output_dir: Path,
    window_k: int,
    current_chars: int,
) -> list[dict]:
    sessions = load_sessions(user_dir)
    if not sessions:
        return []

    personal_note: dict = empty_note()
    all_session_records: list[dict] = []   # extracted {session_id, purpose, summary, topic}
    window_buffer: list[dict] = []         # records accumulating in current window
    classification: dict[int, bool] = {}   # session_id → memory_required

    session_records_out: list[dict] = []

    for session in sessions:
        session_id  = session.get('session_id', 0)
        gt_required = session.get('memory_required', True)

        # ── Step 1: extract session records ──────────────────────
        t0 = time.time()
        record = extract_session_record(llm, session, current_chars)
        print(f"    [S{session_id:02d}] extract: {time.time()-t0:.2f}s  topic={record.get('topic','')}")

        all_session_records.append(record)
        window_buffer.append(record)

        # ── Step 2: update personal_note when window is full ──
        if len(window_buffer) >= window_k:
            t0 = time.time()
            personal_note = update_personal_note(llm, personal_note, window_buffer)
            print(f"    [NOTE UPDATE] window {window_buffer[0]['session_id']}~{window_buffer[-1]['session_id']}  ({time.time()-t0:.2f}s)")
            print(f"      projects: {[p['project_id'] for p in personal_note.get('projects', [])]}")
            print(f"      isolated: {personal_note.get('isolated_sessions', [])}")
            window_buffer = []

            # refresh classification from updated note (retroactive)
            classification = classify_from_note(personal_note)

    # ── flush remaining buffer (last window smaller than window_k) ──
    if window_buffer:
        t0 = time.time()
        personal_note = update_personal_note(llm, personal_note, window_buffer)
        print(f"    [NOTE UPDATE (final)] window {window_buffer[0]['session_id']}~{window_buffer[-1]['session_id']}  ({time.time()-t0:.2f}s)")
        print(f"      projects: {[p['project_id'] for p in personal_note.get('projects', [])]}")
        print(f"      isolated: {personal_note.get('isolated_sessions', [])}")
        window_buffer = []
        classification = classify_from_note(personal_note)

    # ── build evaluation records ──────────────────────────────────────
    for session in sessions:
        session_id  = session.get('session_id', 0)
        gt_required = session.get('memory_required', True)
        # if not in personal_note yet, default conservatively to True
        pred_required = classification.get(session_id, True)
        required_correct = (pred_required == gt_required)

        req_icon = 'O' if required_correct else 'X'
        print(f"    S{session_id:02d}  req[{req_icon}] pred={str(pred_required):5s}  gt={str(gt_required):5s}")

        session_records_out.append({
            'uuid':             uuid,
            'session_id':       session_id,
            'gt_required':      gt_required,
            'pred_required':    pred_required,
            'required_correct': required_correct,
        })

    n       = len(session_records_out)
    req_acc = sum(r['required_correct'] for r in session_records_out) / n if n else None

    # ── save results ────────────────────────────────────────────
    uuid_result = {
        'uuid':              uuid,
        'n_sessions':        n,
        'window_k':          window_k,
        'required_accuracy': round(req_acc, 4) if req_acc is not None else None,
        'personal_note':     personal_note,
        'session_records':   session_records_out,
        'evaluated_at':      datetime.now().isoformat(),
    }
    out_path = output_dir / 'per_uuid' / f"{uuid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(uuid_result, f, ensure_ascii=False, indent=2)

    print(f"  [SAVED] per_uuid/{uuid}.json  req_acc={req_acc:.3f}" if req_acc else f"  [SAVED] per_uuid/{uuid}.json")
    return session_records_out


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
    print(f"window_k         : {args.window_k}")
    print(f"current_chars    : {args.current_chars}")
    print(f"UUIDs            : {len(uuids)}\n")

    agg_csv_path = output_dir / 'aggregate.csv'
    csv_fieldnames = [
        'uuid', 'session_id',
        'gt_required', 'pred_required', 'required_correct',
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
            window_k=args.window_k,
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
        'window_k':          args.window_k,
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
