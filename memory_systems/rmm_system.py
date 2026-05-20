"""
memory_systems/rmm_system.py

RMM (Reflective Memory Management) — ACL 2025
"In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents"

논문의 Prospective Reflection만 구현 (Retrospective Reflection은 learnable reranker + RL이 필요하므로 제외).

동작:
  write_session() 전용 (turn 단위 write 없음).

  세션 끝난 후 두 단계:
    1) Memory Extraction (LLM 1회):
       전체 세션 대화 → topic 단위 summary 리스트 추출
       출력: [{"summary": "...", "reference": [turn_id, ...]}, ...]

    2) Memory Update (LLM 1회 — batch):
       모든 new summaries + candidate pool(embedding top-k union) → 한 번에 처리
       LLM이 각 new summary에 대해 Add 또는 Merge 결정
       JSON 배열로 반환: [{"new_idx": i, "action": "Add"|"Merge", ...}, ...]

메모리 저장 형식:
  각 entry = summary 문자열 (embedding key + context 모두 summary)

Token budget:
  entry 단위 oldest-first eviction (BaseMemorySystem 공통).

Write check:
  get_write_evidence() → 저장된 summary 텍스트 반환.
"""

import json
import re
import uuid as uuid_lib
from dataclasses import dataclass

import numpy as np

from memory_bank import EmbeddingModel, cosine_similarity
from .base import BaseMemorySystem, MemoryChunk, count_tokens

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM import UnifiedLLM


# ─────────────────────────────────────────────────
# 내부 레코드
# ─────────────────────────────────────────────────

@dataclass
class _MemoryEntry:
    """메모리 뱅크의 단일 entry — topic summary."""
    entry_id: str
    summary: str       # retrieval key + context (embedding 대상, LLM이 갱신 가능)
    embedding: object  # np.ndarray | None
    session_file: str
    turn_idx: int      # -1 sentinel (session 단위)
    insert_order: int


# ─────────────────────────────────────────────────
# Prompt: Memory Extraction (논문 Appendix D.1.1)
# ─────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a memory assistant for a personalized dialogue agent.
Given a session of dialogue, extract personal summaries of the USER, with references to the corresponding turn IDs.

Rules:
- Each summary should capture a distinct topic: personal attributes, preferences, life events, goals, ongoing projects, relationships, decisions, or significant state changes.
- A topic may span multiple turns; group related turns together.
- Each summary must be concise (under 40 words).
- Do NOT include agent/assistant information — focus only on what the USER revealed.
- If nothing personal can be extracted, return an empty list [].

Return ONLY a JSON array — no explanation, no markdown fences:
[
  {"summary": "<concise personal summary>", "reference": [<turn_id>, ...]},
  ...
]"""

EXTRACTION_PROMPT = """\
Dialogue session (turn IDs start at 0):
{session_text}

Extract personal summaries for the USER and return as a JSON array."""


# ─────────────────────────────────────────────────
# Prompt: Memory Update — batch (LLM 1회)
# ─────────────────────────────────────────────────

UPDATE_SYSTEM = """\
You are a memory manager for a personalized dialogue agent.
Given a list of existing memories and a list of new summaries extracted from a session,
decide how to update the memory bank for each new summary.

For each new summary, choose one action:
  - "Add": the new summary covers a topic NOT in any existing memory.
  - "Merge": the new summary is relevant to an existing memory -> merge into one updated summary.

Two summaries are relevant if they discuss the same aspect of the user's personal life.
Prefer Merge over Add when the new summary refines or extends an existing memory.
Merged summaries must be concise (under 50 words).

Return ONLY a JSON array — one object per new summary, in order. No explanation, no markdown fences:
[
  {"new_idx": 0, "action": "Add"},
  {"new_idx": 1, "action": "Merge", "existing_idx": 2, "merged_summary": "<merged text>"},
  ...
]"""

UPDATE_PROMPT = """\
Existing memories:
{existing_summaries}

New summaries:
{new_summaries}

Return a JSON array of operations, one per new summary:"""


# ─────────────────────────────────────────────────
# System
# ─────────────────────────────────────────────────

class RMMSystem(BaseMemorySystem):
    """
    RMM (Reflective Memory Management) — Prospective Reflection 구현.

    write_session() 전용 시스템. write()는 NotImplementedError.

    Args:
        max_tokens         : 토큰 예산 (BaseMemorySystem 공통)
        llm_provider       : LLM provider
        llm_model          : LLM 모델명
        llm_base_url       : vLLM 등 커스텀 base URL (선택)
        embedding_provider : EmbeddingModel provider
        embedding_model    : 임베딩 모델명
        update_top_k       : Memory Update 시 비교할 기존 메모리 수 (embedding top-k per summary)
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4.1-mini',
        llm_base_url: str | None = None,
        embedding_provider: str = 'openai',
        embedding_model: str = 'text-embedding-3-small',
        update_top_k: int = 5,
    ):
        super().__init__(max_tokens=max_tokens)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model
        self.update_top_k = update_top_k

        self._embedder = EmbeddingModel(provider=embedding_provider, model=embedding_model)
        self._llm: UnifiedLLM | None = None
        self._entries: dict[str, _MemoryEntry] = {}  # entry_id → _MemoryEntry

    # ─────────────────────────────────────
    # LLM lazy init
    # ─────────────────────────────────────

    def _get_llm(self) -> UnifiedLLM:
        if self._llm is None:
            self._llm = UnifiedLLM(
                provider=self.llm_provider,
                model=self.llm_model,
                base_url=self.llm_base_url,
                temperature=0.0,
            )
        return self._llm

    # ─────────────────────────────────────
    # BaseMemorySystem 추상 메서드
    # ─────────────────────────────────────

    def _write_turn(self, *args, **kwargs):
        raise NotImplementedError(
            "RMMSystem은 session 단위만 지원합니다. write_session()을 사용하세요."
        )

    def _delete_entry(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def _reset_backend(self, user_id: str | None = None) -> None:
        self._entries = {}
        self._llm = None

    # ─────────────────────────────────────
    # write() — 미지원
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """RMMSystem은 turn 단위 write를 지원하지 않습니다."""
        raise NotImplementedError(
            "RMMSystem은 세션 종료 후 write_session()으로만 메모리를 저장합니다. "
            "storage_unit='session'으로 실행하세요."
        )

    # ─────────────────────────────────────
    # write_session() — 핵심 메서드 (LLM 2회)
    # ─────────────────────────────────────

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        세션 전체 대화를 처리:
          1) Memory Extraction (LLM 1회): topic 단위 summary 리스트 추출
          2) Memory Update   (LLM 1회): 모든 new summaries + candidate pool → batch 처리
        written_keys: [(session_file, -1)] sentinel
        """
        dialogue     = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        # ── Step 1: Memory Extraction ──────────────────────────────────
        session_text = self._build_session_text(turns)
        extracted    = self._extract_memories(session_text)
        if not extracted:
            return []

        # ── Step 2: Memory Update (모든 summaries → LLM 1회) ──────────
        new_summaries = [item.get("summary", "").strip() for item in extracted]
        new_summaries = [s for s in new_summaries if s]
        if not new_summaries:
            return []

        any_stored = self._batch_update_memories(new_summaries, session_file)

        if any_stored:
            self._written_turns.add((session_file, -1))
            return [(session_file, -1)]
        return []

    # ─────────────────────────────────────
    # Memory Extraction (LLM 1회)
    # ─────────────────────────────────────

    def _extract_memories(self, session_text: str) -> list[dict]:
        """
        세션 전체 텍스트 → topic 단위 memory 리스트 추출 (LLM 1회).
        반환: [{"summary": str, "reference": [int, ...]}, ...]
        """
        prompt = EXTRACTION_PROMPT.format(session_text=session_text)
        try:
            raw = self._get_llm().chat(prompt, system=EXTRACTION_SYSTEM).strip()
        except Exception as e:
            print(f"  [RMM] extraction failed: {e}")
            return []
        return self._parse_json_list(raw)

    # ─────────────────────────────────────
    # Memory Update — batch (LLM 1회)
    # ─────────────────────────────────────

    def _batch_update_memories(
        self,
        new_summaries: list[str],
        session_file: str,
    ) -> bool:
        """
        모든 new_summaries를 LLM 1회로 한꺼번에 처리.

        1) 각 new_summary마다 top-k candidates → union으로 candidate pool 구성
        2) LLM: existing pool([E0],[E1]...) + new summaries([N0],[N1]...) → JSON operations
        3) 각 operation 적용 (Add / Merge)

        저장/갱신이 하나라도 발생하면 True 반환.
        """
        # candidate pool: 각 new_summary의 top-k union (순서 유지, 중복 제거)
        seen: set[str] = set()
        candidate_ids: list[str] = []
        for s in new_summaries:
            for eid in self._get_candidates(s, top_k=self.update_top_k):
                if eid not in seen:
                    seen.add(eid)
                    candidate_ids.append(eid)

        # 프롬프트 구성
        existing_text = (
            "\n".join(
                f"[E{i}] {self._entries[eid].summary}"
                for i, eid in enumerate(candidate_ids)
            )
            if candidate_ids else "(none)"
        )
        new_text = "\n".join(f"[N{i}] {s}" for i, s in enumerate(new_summaries))

        prompt = UPDATE_PROMPT.format(
            existing_summaries=existing_text,
            new_summaries=new_text,
        )

        try:
            raw = self._get_llm().chat(prompt, system=UPDATE_SYSTEM).strip()
        except Exception as e:
            print(f"  [RMM] batch_update failed: {e}")
            for s in new_summaries:
                self._do_add(s, session_file)
            return True

        operations = self._parse_json_list(raw)
        if not operations:
            print(f"  [RMM] operations parse failed, raw={raw[:120]}, fallback to Add all")
            for s in new_summaries:
                self._do_add(s, session_file)
            return True

        stored = False
        merged_existing: set[int] = set()  # 이미 Merge된 existing_idx (중복 방지)

        for op in operations:
            new_idx = op.get("new_idx")
            action  = (op.get("action") or "Add").strip()

            # new_idx 유효성 검사
            if new_idx is None or not (0 <= new_idx < len(new_summaries)):
                continue
            new_summary = new_summaries[new_idx]

            if action == "Add":
                self._do_add(new_summary, session_file)
                stored = True

            elif action == "Merge":
                existing_idx   = op.get("existing_idx")
                merged_summary = (op.get("merged_summary") or "").strip()

                if (
                    existing_idx is None
                    or not (0 <= existing_idx < len(candidate_ids))
                    or not merged_summary
                ):
                    print(f"  [RMM] invalid Merge op={op}, fallback to Add")
                    self._do_add(new_summary, session_file)
                    stored = True
                    continue

                target_id = candidate_ids[existing_idx]
                if existing_idx not in merged_existing:
                    self._do_merge(target_id, merged_summary)
                    merged_existing.add(existing_idx)
                    stored = True
                else:
                    # 이미 Merge된 entry에 또 Merge 시도 → Add fallback
                    self._do_add(new_summary, session_file)
                    stored = True

        return stored

    def _do_add(self, summary: str, session_file: str) -> None:
        """새 entry를 메모리 뱅크에 추가."""
        new_id    = str(uuid_lib.uuid4())
        embedding = self._embedder.embed_one(summary)
        self._entries[new_id] = _MemoryEntry(
            entry_id=new_id,
            summary=summary,
            embedding=embedding,
            session_file=session_file,
            turn_idx=-1,
            insert_order=self._insert_counter,
        )
        self._register_entry(new_id, session_file, -1, count_tokens(summary), summary)

    def _do_merge(self, entry_id: str, merged_summary: str) -> None:
        """기존 entry의 summary를 갱신."""
        if entry_id not in self._entries:
            return
        entry           = self._entries[entry_id]
        entry.summary   = merged_summary
        entry.embedding = self._embedder.embed_one(merged_summary)

        new_tokens = count_tokens(merged_summary)
        for rec in self._entry_records:
            if rec.entry_id == entry_id:
                self._total_tokens = max(0, self._total_tokens - rec.token_count + new_tokens)
                rec.token_count    = new_tokens
                rec.content        = merged_summary[:500]
                break

    # ─────────────────────────────────────
    # 유사 메모리 검색
    # ─────────────────────────────────────

    def _get_candidates(self, query_text: str, top_k: int) -> list[str]:
        """query_text와 코사인 유사도 높은 상위 top_k entry_id 리스트 반환."""
        if not self._entries:
            return []
        entries = [e for e in self._entries.values() if e.embedding is not None]
        if not entries:
            return []

        top_k      = min(top_k, len(entries))
        query_emb  = self._embedder.embed_one(query_text)
        emb_matrix = np.stack([e.embedding for e in entries])
        scores     = cosine_similarity(query_emb, emb_matrix)
        top_idx    = np.argsort(scores)[::-1][:top_k]
        return [entries[i].entry_id for i in top_idx]

    # ─────────────────────────────────────
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        if not self._entries:
            return []
        entries = [e for e in self._entries.values() if e.embedding is not None]
        if not entries:
            return []

        query_emb  = self._embedder.embed_one(query)
        emb_matrix = np.stack([e.embedding for e in entries])
        scores     = cosine_similarity(query_emb, emb_matrix)
        top_k      = min(top_k, len(entries))
        top_idx    = np.argsort(scores)[::-1][:top_k]

        return [
            MemoryChunk(
                content=entries[i].summary,
                session_file=entries[i].session_file,
                turn_idx=entries[i].turn_idx,
                keywords=[],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    # ─────────────────────────────────────
    # Write check override
    # ─────────────────────────────────────

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """저장된 summary 텍스트 반환 (Mem0System과 동일한 패턴)."""
        if not written_keys:
            return ""
        written_set = {(sf, ti) for sf, ti in written_keys}
        summaries = [
            rec.content
            for rec in self._entry_records
            if (rec.session_file, rec.turn_idx) in written_set and rec.content
        ]
        return "\n\n".join(summaries)

    # ─────────────────────────────────────
    # dump_memories override
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        """_entries의 최신 summary로 snapshot 구성."""
        result = []
        for rec in self._entry_records:
            entry = self._entries.get(rec.entry_id)
            if entry:
                result.append({
                    "memory_id":    rec.entry_id,
                    "memory_text":  entry.summary,
                    "session_file": rec.session_file,
                    "turn_idx":     rec.turn_idx,
                    "token_count":  rec.token_count,
                    "insert_order": rec.insert_order,
                    "content":      entry.summary,
                })
            else:
                result.append({
                    "memory_id":    rec.entry_id,
                    "memory_text":  rec.content,
                    "session_file": rec.session_file,
                    "turn_idx":     rec.turn_idx,
                    "token_count":  rec.token_count,
                    "insert_order": rec.insert_order,
                    "content":      rec.content,
                })
        return result

    # ─────────────────────────────────────
    # 파싱 유틸
    # ─────────────────────────────────────

    @staticmethod
    def _parse_json_list(raw: str) -> list:
        """LLM 응답에서 JSON array 파싱 (extraction / operations 공용)."""
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            m = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group())
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
        return []

    # ─────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────

    @staticmethod
    def _build_session_text(turns: list[tuple[int, str, str]]) -> str:
        """turns 리스트 → turn ID 포함 대화 텍스트."""
        parts = []
        for idx, user_content, agent_content in turns:
            parts.append(f"Turn {idx}:")
            if user_content:
                parts.append(f"  User: {user_content}")
            if agent_content:
                parts.append(f"  Agent: {agent_content}")
        return "\n".join(parts)

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        entries_meta = []
        embeddings   = {}
        for eid, entry in self._entries.items():
            entries_meta.append({
                "entry_id":     eid,
                "summary":      entry.summary,
                "session_file": entry.session_file,
                "turn_idx":     entry.turn_idx,
                "insert_order": entry.insert_order,
            })
            if entry.embedding is not None:
                embeddings[eid] = entry.embedding

        state = {
            "system":             "RMMSystem",
            "llm_provider":       self.llm_provider,
            "llm_model":          self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model":    self.embedding_model_name,
            "entries":            entries_meta,
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        if embeddings:
            np.savez(str(ckpt_path / "embeddings.npz"), **embeddings)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        backend_path = ckpt_path / "backend_state.json"
        emb_path     = ckpt_path / "embeddings.npz"
        if not backend_path.exists():
            return

        with open(backend_path, encoding="utf-8") as f:
            state = json.load(f)

        emb_data = {}
        if emb_path.exists():
            npz      = np.load(str(emb_path))
            emb_data = {k: npz[k] for k in npz.files}

        self._entries = {}
        for e in state.get("entries", []):
            self._entries[e["entry_id"]] = _MemoryEntry(
                entry_id=e["entry_id"],
                summary=e["summary"],
                embedding=emb_data.get(e["entry_id"]),
                session_file=e["session_file"],
                turn_idx=e["turn_idx"],
                insert_order=e["insert_order"],
            )
