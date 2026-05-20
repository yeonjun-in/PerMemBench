#!/usr/bin/env python3
"""
7_personalize_summary.py

Adaptive Personalized Memory Policy 평가 스크립트.

LLM이 아래 정보를 보고 매 세션마다 예측:
  1. 현재 세션의 raw dialogue
  2. 과거 K개 세션의 요약(summary) 히스토리

  → 예측: memory_required 여부 (long-horizon vs transient)

또한 각 세션 종료 후, 해당 세션을 1~2문장으로 요약해 저장하고
다음 세션들의 히스토리 컨텍스트로 사용한다.

평가 Metric:
  - required_accuracy : memory_required 예측 정확도 (per session)

Usage:
    python 7_personalize_summary.py \
        --data_dir ./skeleton_dialogues_v5 \
        --output_dir ./results/personalize_summary \
        --llm_model gpt-5-mini \
        --history_k 3 \
        --uuid 00aefb8e6cfd47dc939d6d3b30a5aefb
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from LLM import UnifiedLLM


# ========================
# Args
# ========================

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./skeleton_dialogues_v5')
parser.add_argument('--output_dir', type=str, default='./results/personalize_summary')
parser.add_argument('--llm_model', type=str, default='gpt-5-mini')
parser.add_argument('--current_chars', type=int, default=2000, help='현재 세션 dialogue 최대 문자수')
parser.add_argument('--history_k', type=int, default=5, help='참조할 과거 세션 summary 개수')
parser.add_argument('--uuid', type=str, default=None)
parser.add_argument('--limit', type=int, default=None)
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
# Prompt
# ========================

PREDICT_PROMPT = """\
You are a memory policy agent for an AI assistant.
Decide whether the current session should be stored in long-term memory.

## Recent Session Summaries (Past Sessions Only)
{history_summaries}

## Current Session Dialogue
{current_dialogue}

## Task
Predict **memory_required**:
- true  = long-horizon session (ongoing project, recurring goal, dependency on prior context)
- false = transient session (standalone request, one-off information)

Respond ONLY with a JSON object:
{{
  "memory_required": <true | false>
}}
"""


SUMMARY_PROMPT = """\
You are summarizing one conversation session between a user and an AI assistant.

## Session Dialogue
{current_dialogue}

## Task
Write a short summary in 1-2 sentences.
Focus on the main topic and the user's intent/progress.
Avoid unnecessary detail.
"""


# ========================
# Data loading / formatting
# ========================

def load_sessions(user_dir: Path) -> list[dict]:
    sessions = []
    for session_file in sorted(user_dir.glob('session_*.json')):
        try:
            sessions.append(json.load(open(session_file, encoding='utf-8')))
        except Exception as exc:
            print(f"  [WARN] {session_file.name}: {exc}")
    sessions.sort(key=lambda s: s.get('session_id', 0))
    return sessions


def dialogue_to_text(dialogue: list[dict], max_chars: int) -> str:
    lines = []
    for turn in dialogue:
        role = turn.get('role', '').capitalize()
        content = turn.get('content', '').strip()
        lines.append(f"{role}: {content}")
    full = '\n'.join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + ' ...[truncated]'
    return full


def history_to_text(history_summaries: list[dict]) -> str:
    if not history_summaries:
        return "(none)"

    lines = []
    for item in history_summaries:
        sid = item.get('session_id', '?')
        summary = item.get('summary', '').strip()
        lines.append(f"- Session {sid}: {summary}")
    return '\n'.join(lines)


# ========================
# LLM calls
# ========================

def predict_session(
    llm: UnifiedLLM,
    current_session: dict,
    history_summaries: list[dict],
    current_chars: int,
) -> dict:
    current_text = dialogue_to_text(current_session.get('dialogue', []), current_chars)
    history_text = history_to_text(history_summaries)

    prompt = PREDICT_PROMPT.format(
        history_summaries=history_text,
        current_dialogue=current_text,
    )

    try:
        raw = llm.chat(prompt)
        clean = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(clean)
    except Exception as exc:
        print(f"    [WARN] LLM predict failed: {exc}")
        return {
            'memory_required': True,
            'reasoning': f'prediction failed: {exc}',
        }


def summarize_session(
    llm: UnifiedLLM,
    current_session: dict,
    current_chars: int,
) -> str:
    current_text = dialogue_to_text(current_session.get('dialogue', []), current_chars)
    prompt = SUMMARY_PROMPT.format(current_dialogue=current_text)

    try:
        summary = llm.chat(prompt).strip()
        summary = summary.replace('```', '').strip()
        return summary if summary else "(empty summary)"
    except Exception as exc:
        print(f"    [WARN] LLM summary failed: {exc}")
        return f"(summary failed: {exc})"


# ========================
# Per-UUID processing
# ========================

def process_uuid(
    uuid: str,
    user_dir: Path,
    llm: UnifiedLLM,
    output_dir: Path,
    current_chars: int,
    history_k: int,
) -> list[dict]:
    sessions = load_sessions(user_dir)
    if not sessions:
        return []

    session_records: list[dict] = []
    summary_memory: list[dict] = []

    for session in sessions:
        session_id = session.get('session_id', 0)
        gt_required = session.get('memory_required', True)

        # 현재 세션 예측 시에는 과거 요약만 사용한다.
        history_for_prompt = summary_memory[-history_k:] if history_k > 0 else []

        pred = predict_session(
            llm=llm,
            current_session=session,
            history_summaries=history_for_prompt,
            current_chars=current_chars,
        )

        pred_required = pred.get('memory_required', True)
        reasoning = pred.get('reasoning', '')
        required_correct = (pred_required == gt_required)

        session_summary = summarize_session(
            llm=llm,
            current_session=session,
            current_chars=current_chars,
        )
        summary_memory.append({
            'session_id': session_id,
            'summary': session_summary,
        })

        record = {
            'uuid': uuid,
            'session_id': session_id,
            'n_history': len(history_for_prompt),
            'history_session_ids': [h.get('session_id') for h in history_for_prompt],
            'history_summaries': [h.get('summary', '') for h in history_for_prompt],
            'current_summary': session_summary,
            'gt_required': gt_required,
            'pred_required': pred_required,
            'required_correct': required_correct,
            'reasoning': reasoning,
        }
        session_records.append(record)

        req_icon = 'O' if required_correct else 'X'
        print(
            f"    S{session_id:02d} (hist={len(history_for_prompt):2d})"
            f"  req[{req_icon}] pred={str(pred_required):5s}  gt={str(gt_required):5s}"
        )

    n = len(session_records)
    req_acc = sum(r['required_correct'] for r in session_records) / n if n else None

    uuid_result = {
        'uuid': uuid,
        'n_sessions': n,
        'history_k': history_k,
        'required_accuracy': round(req_acc, 4) if req_acc is not None else None,
        'session_records': session_records,
        'evaluated_at': datetime.now().isoformat(),
    }

    out_path = output_dir / 'per_uuid' / f"{uuid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as file_obj:
        json.dump(uuid_result, file_obj, ensure_ascii=False, indent=2)

    print(f"  [SAVED] per_uuid/{uuid}.json")
    if req_acc is not None:
        print(f"  req_acc={req_acc:.3f}")

    return session_records


# ========================
# Main
# ========================

def main() -> None:
    llm = make_llm(args.llm_model)

    data_dir = Path(args.data_dir)
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
    print(f"history_k        : {args.history_k}")
    print(f"current_chars    : {args.current_chars}")
    print(f"UUIDs            : {len(uuids)}\n")

    agg_csv_path = output_dir / 'aggregate.csv'
    csv_fieldnames = [
        'uuid',
        'session_id',
        'n_history',
        'history_session_ids',
        'history_summaries',
        'current_summary',
        'gt_required',
        'pred_required',
        'required_correct',
        'reasoning',
    ]
    write_header = not agg_csv_path.exists()
    agg_csv = open(agg_csv_path, 'a', newline='', encoding='utf-8')
    agg_writer = csv.DictWriter(agg_csv, fieldnames=csv_fieldnames, extrasaction='ignore')
    if write_header:
        agg_writer.writeheader()

    all_records = []

    for index, uuid in enumerate(uuids, 1):
        user_dir = data_dir / uuid
        if not user_dir.exists():
            print(f"[{index}/{len(uuids)}] {uuid[:8]}: not found, skip")
            continue

        print(f"\n=== [{index}/{len(uuids)}] UUID: {uuid} ===")
        records = process_uuid(
            uuid=uuid,
            user_dir=user_dir,
            llm=llm,
            output_dir=output_dir,
            current_chars=args.current_chars,
            history_k=args.history_k,
        )
        for record in records:
            agg_writer.writerow(record)
        agg_csv.flush()
        all_records.extend(records)

    agg_csv.close()

    n = len(all_records)
    tp = sum(1 for r in all_records if r['gt_required'] and r['pred_required'])
    fp = sum(1 for r in all_records if not r['gt_required'] and r['pred_required'])
    fn = sum(1 for r in all_records if r['gt_required'] and not r['pred_required'])
    tn = sum(1 for r in all_records if not r['gt_required'] and not r['pred_required'])

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    summary = {
        'total_sessions': n,
        'history_k': args.history_k,
        'current_chars': args.current_chars,
        'required_accuracy': round(sum(r['required_correct'] for r in all_records) / n, 4) if n else None,
        'confusion': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
        'precision': round(precision, 4) if precision else None,
        'recall': round(recall, 4) if recall else None,
        'f1': round(f1, 4) if f1 else None,
        'evaluated_at': datetime.now().isoformat(),
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print('\n' + '=' * 60)
    print(f"DONE | {n} sessions")
    print(f"  required_accuracy : {summary['required_accuracy']}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision={summary['precision']}  recall={summary['recall']}  f1={summary['f1']}")
    print(f"\nResults -> {output_dir}")


if __name__ == '__main__':
    main()
