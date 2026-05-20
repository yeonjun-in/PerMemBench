"""
run_utils.py

6_run_*.py 스크립트들이 공유하는 공통 유틸리티.
- session loading / normalization
- snapshot saving
- OracleFilter 체크
"""

import json
from datetime import datetime
from pathlib import Path

from memory_systems import BaseMemorySystem, OracleFilter


# ─────────────────────────────────────────────────
# Session loading
# ─────────────────────────────────────────────────

def load_sessions_ordered(user_dir: Path) -> list[dict]:
    sessions = []
    for fpath in sorted(user_dir.glob("session_*.json")):
        try:
            session = json.load(open(fpath, encoding='utf-8'))
        except Exception as e:
            print(f"  [WARN] Failed to load {fpath.name}: {e}")
            continue
        session['_filename'] = fpath.name
        sessions.append(session)
    sessions.sort(key=lambda s: s.get('session_id', 0))
    return sessions


def normalize_session(session: dict) -> dict:
    """base.py write()가 읽는 필드명으로 매핑."""
    if 'domain_name' not in session:
        session['domain_name'] = session.get('domain', '')
    if 'session_idx' not in session:
        session['session_idx'] = session.get('session_id', -1)
    return session


def should_skip_session(session: dict, oracle: bool) -> bool:
    """oracle=True 일 때 memory_required=False 세션 skip."""
    if oracle and not session.get('memory_required', True):
        return True
    return False


# ─────────────────────────────────────────────────
# Snapshot saving
# ─────────────────────────────────────────────────

def save_snapshot(
    snapshot_dir: Path,
    system_label: str,
    uuid: str,
    session_id: int,
    session_file: str,
    domain: str,
    month: int,
    memory_required: bool,
    written_keys: list[tuple[str, int]],
    n_evicted: int,
    system: BaseMemorySystem,
    write_evidence: str = "",
    gt_memory: list[dict] | None = None,
) -> None:
    """
    세션 처리 후 snapshot 저장.
    평가 점수 없이 메모리 상태 + gt_memory 정보만 포함.
    """
    out_dir = snapshot_dir / system_label / uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_{session_id:04d}.json"

    snapshot = {
        "uuid":          uuid,
        "session_id":    session_id,
        "session_file":  session_file,
        "domain":        domain,
        "month":         month,
        "memory_required": memory_required,
        "written_this_session": [
            {"session_file": sf, "turn_idx": ti}
            for sf, ti in written_keys
        ],
        "n_written_this_session": len(written_keys),
        "n_evicted_by_budget":    n_evicted,
        "total_tokens_after":     system.total_tokens,
        "n_entries_after":        system.n_entries,
        "write_evidence":         write_evidence,
        "memories":               system.dump_memories(),
        # gt_memory: 평가 시점에 필요한 정보 보존
        "gt_memory": [
            {
                "gt_idx":           i,
                "gt_type":          g.get("type",              ""),
                "fact":             g.get("fact",              ""),
                "probing_question": g.get("probing_question",  ""),
                "gt_answer":        g.get("answer",            ""),
            }
            for i, g in enumerate(gt_memory or [])
            if g.get("fact") and g.get("probing_question")
        ],
        "saved_at": datetime.now().isoformat(),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def save_final_memory(
    final_memory_dir: Path,
    system_label: str,
    uuid: str,
    system: BaseMemorySystem,
    n_sessions_processed: int,
) -> None:
    out_dir = final_memory_dir / system_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid}_final.json"

    final = {
        "uuid":                uuid,
        "system":              system_label,
        "n_sessions_processed": n_sessions_processed,
        "total_tokens":        system.total_tokens,
        "n_entries":           system.n_entries,
        "memories":            system.dump_memories(),
        "saved_at":            datetime.now().isoformat(),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────
# 공통 실행 루프
# ─────────────────────────────────────────────────

def run_memory_pipeline(
    system: BaseMemorySystem,
    system_label: str,
    data_dir: Path,
    snapshot_dir: Path | None,
    final_memory_dir: Path | None,
    storage_unit: str,          # 'turn' | 'session'
    oracle: bool,
    mem_budget_tokens: int,
    mem_budget_entries: int | None,
    uuid_filter: str | None,
    limit: int | None,
    save_checkpoint_dir: Path | None = None,
    load_checkpoint_dir: Path | None = None,
) -> None:
    """
    공통 실행 루프.
    세션을 순서대로 처리하며 메모리 저장 + snapshot 저장.
    평가(write/read/QA score)는 하지 않음.

    save_checkpoint_dir : 모든 세션 처리 후 메모리 상태를 체크포인트로 저장.
                          {save_checkpoint_dir}/{uuid}/ 에 budget_state.json + backend_state.json.
    load_checkpoint_dir : 세션 처리 시작 전 체크포인트를 로드하여 해당 시점 메모리부터 시작.
                          {load_checkpoint_dir}/{uuid}/ 에서 읽음.
    """
    if uuid_filter:
        uuids = [uuid_filter]
    else:
        uuids = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if limit:
        uuids = uuids[:limit]

    print(f"System      : {system_label}")
    print(f"storage_unit: {storage_unit}")
    print(f"oracle      : {oracle}")
    print(f"mem_budget  : {mem_budget_tokens}")
    print(f"entry_budget: {mem_budget_entries if mem_budget_entries is not None else 'unlimited'}")
    print(f"UUIDs       : {len(uuids)}")
    print(f"snapshot_dir: {snapshot_dir or 'disabled'}")
    print(f"load_ckpt   : {load_checkpoint_dir or 'disabled'}")
    print(f"save_ckpt   : {save_checkpoint_dir or 'disabled'}")
    print()

    for user_idx, uuid in enumerate(uuids, 1):
        user_dir = data_dir / uuid
        if not user_dir.exists():
            print(f"[{user_idx}/{len(uuids)}] {uuid[:8]}: not found, skip")
            continue

        print(f"\n=== [{user_idx}/{len(uuids)}] UUID: {uuid} ===")

        sessions = load_sessions_ordered(user_dir)
        if not sessions:
            print("  [SKIP] No sessions found.")
            continue

        # UUID 기반으로 system reset
        system.reset(user_id=uuid)

        # 체크포인트 로드 (reset 이후 — backend(Memory 객체)는 재생성, user_id만 복원)
        if load_checkpoint_dir is not None:
            ckpt_path = load_checkpoint_dir / uuid
            if ckpt_path.exists() and (ckpt_path / "budget_state.json").exists():
                try:
                    system.load_checkpoint(str(ckpt_path))
                    print(f"  [CKPT] loaded from {ckpt_path}"
                          f" | entries={system.n_entries}"
                          f" | tokens={system.total_tokens}")
                except Exception as e:
                    print(f"  [WARN] checkpoint load failed: {e} → fresh start")

        n_sessions = len(sessions)
        n_sessions_processed = 0

        for s_idx, session in enumerate(sessions):
            normalize_session(session)

            session_id      = session.get('session_id', s_idx + 1)
            domain          = session.get('domain', '')
            month           = session.get('month', -1)
            memory_required = session.get('memory_required', True)
            gt_memory       = session.get('gt_memory', [])
            session_file    = session.get('_filename', 'unknown')

            # oracle: memory_required=False 세션 skip
            if should_skip_session(session, oracle):
                print(f"  [{s_idx+1}/{n_sessions}] {session_file} | [ORACLE SKIP]")
                # snapshot은 빈 상태로 저장 (일관성 유지)
                if snapshot_dir:
                    save_snapshot(
                        snapshot_dir=snapshot_dir,
                        system_label=system_label,
                        uuid=uuid,
                        session_id=session_id,
                        session_file=session_file,
                        domain=domain,
                        month=month,
                        memory_required=memory_required,
                        written_keys=[],
                        n_evicted=0,
                        system=system,
                        write_evidence="",
                        gt_memory=[],
                    )
                n_sessions_processed += 1
                continue

            print(f"  [{s_idx+1}/{n_sessions}] {session_file}"
                  f" | mem_req={memory_required}"
                  f" | gt_facts={len(gt_memory)}"
                  f" | tokens={system.total_tokens}/{mem_budget_tokens}"
                  f" | entries={system.n_entries}/{mem_budget_entries}"
                  )

            # ── Step 1: Write ──────────────────────────────────────
            written_keys = []
            try:
                if storage_unit == 'session':
                    written_keys = system.write_session(session)
                else:
                    written_keys = system.write(session)
            except Exception as e:
                print(f"    [ERROR] write failed: {e}")

            # ── Step 2: Write evidence ─────────────────────────────
            write_evidence = ""
            try:
                write_evidence = system.get_write_evidence(session, written_keys)
            except Exception as e:
                print(f"    [WARN] get_write_evidence failed: {e}")

            # ── Step 3: post_session (token budget) ────────────────
            n_evicted = 0
            try:
                post_result = system.post_session()
                deleted_for_token_limit = post_result.get('deleted_for_token_limit', [])
                deleted_for_entry_limit = post_result.get('deleted_for_entry_limit', [])
                n_evicted = len(deleted_for_token_limit) + len(deleted_for_entry_limit)
            except Exception as e:
                print(f"    [ERROR] post_session failed: {e}")

            n_sessions_processed += 1

            print(f"    [MEM] written={len(written_keys)}"
                  f" | evicted={n_evicted}"
                  f" | total_tokens={system.total_tokens}"
                  f" | entries={system.n_entries}")

            # ── Step 4: Snapshot 저장 ──────────────────────────────
            if snapshot_dir is not None:
                try:
                    save_snapshot(
                        snapshot_dir=snapshot_dir,
                        system_label=system_label,
                        uuid=uuid,
                        session_id=session_id,
                        session_file=session_file,
                        domain=domain,
                        month=month,
                        memory_required=memory_required,
                        written_keys=written_keys,
                        n_evicted=n_evicted,
                        system=system,
                        write_evidence=write_evidence,
                        gt_memory=gt_memory,
                    )
                    print(f"    [SNAP] saved session_{session_id:04d}.json")
                except Exception as e:
                    print(f"    [WARN] snapshot failed: {e}")

        print(f"  → {n_sessions_processed} sessions processed")

        # ── Final memory dump ──────────────────────────────────────
        if final_memory_dir is not None:
            try:
                save_final_memory(
                    final_memory_dir=final_memory_dir,
                    system_label=system_label,
                    uuid=uuid,
                    system=system,
                    n_sessions_processed=n_sessions_processed,
                )
                print(f"  [FINAL] saved {uuid}_final.json")
            except Exception as e:
                print(f"  [WARN] final memory dump failed: {e}")

        # ── Checkpoint 저장 ────────────────────────────────────────
        if save_checkpoint_dir is not None:
            ckpt_path = save_checkpoint_dir / uuid
            try:
                system.save_checkpoint(str(ckpt_path))
                print(f"  [CKPT] saved to {ckpt_path}"
                      f" | entries={system.n_entries}"
                      f" | tokens={system.total_tokens}")
            except Exception as e:
                print(f"  [WARN] checkpoint save failed: {e}")

    print("\n" + "=" * 60)
    print(f"DONE | System: {system_label}")
