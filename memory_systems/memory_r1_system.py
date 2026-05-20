"""
memory_systems/memory_r1_system.py

Memory-R1 (논문: "Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL")
의 Memory Manager 프롬프트와 알고리즘을 base LLM으로 구현한 시스템.

RL fine-tuning 없이 논문의 프롬프트(Appendix C.1, Figures 9, 10)와
info extraction → memory manager 파이프라인을 그대로 사용.

LLM 호출 구조 (논문 설계 그대로):
  write()         : LLM 2회
                      1) turn에서 facts 리스트 추출
                      2) Memory Manager: 기존 메모리 전체 + 새 facts 전체 → 한 번에 처리
  write_session() : LLM 2회
                      1) 세션 전체 대화에서 facts 리스트 추출
                      2) Memory Manager: 기존 메모리 전체 + 새 facts 전체 → 한 번에 처리

  Memory Manager 출력 형식 (논문 Figure 9/10):
    {"memory": [
        {"id": "existing_id", "text": "...", "event": "NONE"},
        {"id": "existing_id", "text": "updated", "event": "UPDATE", "old_memory": "..."},
        {"id": null,          "text": "new fact", "event": "ADD"}
    ]}

Token budget:
  BaseMemorySystem의 oldest-first eviction 사용.
  fact 단위로 _entry_records 등록 (Mem0System과 동일).

Write check:
  get_write_evidence() → _entry_records에 저장된 fact 텍스트 반환 (raw 대화 아님).
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
class _FactRecord:
    """메모리 뱅크 내 단일 fact 레코드."""
    entry_id: str
    text: str
    embedding: object  # np.ndarray | None
    session_file: str
    turn_idx: int      # write(): 실제 turn index / write_session(): -1 sentinel
    insert_order: int


# ─────────────────────────────────────────────────
# Prompt: Info Extraction (turn 단위)
# ─────────────────────────────────────────────────

TURN_EXTRACTION_SYSTEM = (
    "You are a memory assistant. Extract all important personal facts "
    "worth remembering from the conversation turn. "
    "Focus on: personal attributes, preferences, life events, goals, ongoing projects, "
    "relationships, important decisions, or significant state changes.\n"
    "Return a JSON array of concise fact strings (each under 30 words). "
    "If there is nothing worth remembering, return an empty array []. "
    "Return ONLY the JSON array — no explanation, no markdown fences."
)

TURN_EXTRACTION_PROMPT = """\
Conversation turn:
User: {user_content}
Agent: {agent_content}

Extract all key facts worth remembering and return as a JSON array of strings."""


# ─────────────────────────────────────────────────
# Prompt: Info Extraction (session 단위)
# ─────────────────────────────────────────────────

SESSION_EXTRACTION_SYSTEM = (
    "You are a memory assistant. Extract all important personal facts "
    "worth remembering from the full conversation session below. "
    "Focus on: personal attributes, preferences, life events, goals, ongoing projects, "
    "relationships, important decisions, or significant state changes.\n"
    "Return a JSON array of concise fact strings (each under 30 words). "
    "If there is nothing worth remembering, return an empty array []. "
    "Return ONLY the JSON array — no explanation, no markdown fences."
)

SESSION_EXTRACTION_PROMPT = """\
Full conversation session:
{session_text}

Extract all key facts worth remembering and return as a JSON array of strings."""


# ─────────────────────────────────────────────────
# Prompt: Memory Manager (논문 Appendix C.1, Figures 9/10 기반)
# ─────────────────────────────────────────────────

MEMORY_MANAGER_SYSTEM = """\
You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, \
(3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.
Compare newly retrieved facts with the existing memory. For each new fact,
decide whether to ADD, UPDATE, DELETE, or NONE.

1. ADD: If the retrieved fact contains new information not present in the memory,
   add it by generating a new entry (id: null).
   Example:
     Old Memory: [{"id": "0", "text": "User is a software engineer"}]
     Retrieved facts: ["Name is John"]
     -> {"id": null, "text": "Name is John", "event": "ADD"}

2. UPDATE: If the retrieved fact contains information already present in memory
   but the information is different or more detailed, update it (keep the same id).
   If the retrieved fact is additive (e.g., adopting a second pet), UPDATE to consolidate
   rather than DELETE + ADD.
   Examples:
   (a) Memory "User likes to play cricket", fact "User loves to play cricket with friends"
       -> UPDATE (more detail)
   (b) Memory "User has a dog named Buddy", fact "User adopted another dog named Scout"
       -> UPDATE to "User has two dogs: Buddy and Scout" (additive, do NOT delete)
   If a retrieved fact conveys the same information as the memory, keep the version
   with more detail (do NOT update if no new information is added).

3. DELETE: If the retrieved fact directly contradicts existing memory.
   Example:
     Memory "User likes cheese pizza", fact "User dislikes pizza"
     -> DELETE (direct contradiction)

4. NONE (no change): If the fact is already present or irrelevant.
   Example:
     Memory "User's name is John", fact "User is called John" -> NONE

Return a JSON object with a "memory" array covering ALL existing entries plus any new ADDs.
For existing entries: include their original id and mark event as NONE/UPDATE/DELETE.
For new entries: use null for id and mark event as ADD.
When updating, include "old_memory" with the previous text.

Return ONLY the JSON object — no explanation, no markdown fences:
{
    "memory": [
        {"id": "<existing_id>", "text": "...", "event": "NONE"},
        {"id": "<existing_id>", "text": "<updated text>", "event": "UPDATE", "old_memory": "<old text>"},
        {"id": "<existing_id>", "text": "...", "event": "DELETE"},
        {"id": null,            "text": "<new fact>",    "event": "ADD"}
    ]
}"""

MEMORY_MANAGER_PROMPT = """\
Old Memory:
{old_memory}

Retrieved facts:
{retrieved_facts}

Update the memory and return JSON."""


# ─────────────────────────────────────────────────
# System
# ─────────────────────────────────────────────────

class MemoryR1System(BaseMemorySystem):
    """
    Memory-R1 스타일 메모리 시스템 (base LLM 사용, RL fine-tuning 없음).

    Args:
        max_tokens         : 토큰 예산 (BaseMemorySystem 공통)
        llm_provider       : UnifiedLLM provider ('openai' | 'claude' | 'vllm' | ...)
        llm_model          : LLM 모델명
        llm_base_url       : vLLM 등 커스텀 base URL (선택)
        embedding_provider : EmbeddingModel provider
        embedding_model    : 임베딩 모델명
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4.1-mini',
        llm_base_url: str | None = None,
        embedding_provider: str = 'openai',
        embedding_model: str = 'text-embedding-3-small',
        manager_top_k: int = 20,
    ):
        super().__init__(max_tokens=max_tokens)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model

        self._embedder = EmbeddingModel(provider=embedding_provider, model=embedding_model)
        self._llm: UnifiedLLM | None = None
        self._facts: dict[str, _FactRecord] = {}  # entry_id → _FactRecord
        self.manager_top_k = manager_top_k  # Memory Manager에 넘길 기존 메모리 수

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

    def _write_turn(self, *args, **kwargs) -> str | None:
        raise NotImplementedError("MemoryR1System uses overridden write() directly")

    def _delete_entry(self, entry_id: str) -> None:
        self._facts.pop(entry_id, None)

    def _reset_backend(self, user_id: str | None = None) -> None:
        self._facts = {}
        self._llm = None

    # ─────────────────────────────────────
    # write() — turn 단위 (LLM 2회)
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """
        Turn 단위 저장. 각 turn에서:
          1) facts 리스트 추출 (LLM 1회)
          2) Memory Manager: 기존 메모리 전체 + 새 facts → 한 번에 처리 (LLM 1회)
        written_keys: [(session_file, turn_idx), ...]
        """
        dialogue     = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        written: list[tuple[str, int]] = []

        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue

            # Step 1: turn에서 facts 리스트 추출 (LLM 1회)
            facts = self._extract_turn_facts(user_content, agent_content)
            if not facts:
                continue

            # Step 2: Memory Manager — 기존 메모리 전체 + 새 facts 한 번에 (LLM 1회)
            stored = self._run_and_apply_manager(
                new_facts=facts,
                session_file=session_file,
                turn_idx=turn_idx,
            )
            if stored:
                self._written_turns.add((session_file, turn_idx))
                written.append((session_file, turn_idx))

        return written

    # ─────────────────────────────────────
    # write_session() — session 단위 (LLM 2회)
    # ─────────────────────────────────────

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        Session 단위 저장. 세션 전체 대화를 한 번에:
          1) 전체 대화에서 facts 리스트 추출 (LLM 1회)
          2) Memory Manager: 기존 메모리 전체 + 새 facts → 한 번에 처리 (LLM 1회)
        written_keys: [(session_file, -1)] sentinel (fact가 하나라도 저장되면).
        """
        dialogue     = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        # Step 1: 세션 전체 텍스트에서 facts 리스트 추출 (LLM 1회)
        session_text = self._build_session_text(turns)
        facts = self._extract_session_facts(session_text)
        if not facts:
            return []

        # Step 2: Memory Manager — 기존 메모리 전체 + 새 facts 한 번에 (LLM 1회)
        stored = self._run_and_apply_manager(
            new_facts=facts,
            session_file=session_file,
            turn_idx=-1,  # session sentinel
        )
        if stored:
            self._written_turns.add((session_file, -1))
            return [(session_file, -1)]
        return []

    # ─────────────────────────────────────
    # Memory Manager 실행 + 적용 (공통)
    # ─────────────────────────────────────

    def _run_and_apply_manager(
        self,
        new_facts: list[str],
        session_file: str,
        turn_idx: int,
    ) -> bool:
        """
        Memory Manager를 실행하고 반환된 operations를 메모리에 적용.
        ADD 또는 UPDATE가 하나라도 발생하면 True 반환.
        """
        # 새 facts 전체를 하나의 쿼리로 합쳐서 top-k 기존 메모리만 검색
        query_text = " ".join(new_facts)
        candidate_ids = self._get_candidate_ids(query_text, top_k=self.manager_top_k)
        operations = self._run_memory_manager(new_facts, candidate_ids)
        if not operations:
            return False

        candidate_set = set(candidate_ids)  # 유효한 id 집합

        any_stored = False
        for op_item in operations:
            event      = (op_item.get("event") or "NONE").upper()
            entry_id   = op_item.get("id") or None
            final_text = (op_item.get("memory") or op_item.get("text") or "").strip()

            if event == "ADD" and final_text:
                new_id = str(uuid_lib.uuid4())
                embedding = self._embedder.embed_one(final_text)
                self._facts[new_id] = _FactRecord(
                    entry_id=new_id,
                    text=final_text,
                    embedding=embedding,
                    session_file=session_file,
                    turn_idx=turn_idx,
                    insert_order=self._insert_counter,
                )
                self._register_entry(
                    new_id, session_file, turn_idx, count_tokens(final_text), final_text
                )
                any_stored = True

            elif event == "UPDATE" and entry_id and final_text:
                # id 유효성 검증: candidate로 넘겨준 id + 실제 _facts에 존재해야 함
                if entry_id in candidate_set and entry_id in self._facts:
                    fact_rec   = self._facts[entry_id]
                    new_tokens = count_tokens(final_text)
                    fact_rec.text      = final_text
                    fact_rec.embedding = self._embedder.embed_one(final_text)

                    matched = False
                    for budget_rec in self._entry_records:
                        if budget_rec.entry_id == entry_id:
                            self._total_tokens = max(
                                0, self._total_tokens - budget_rec.token_count + new_tokens
                            )
                            budget_rec.token_count = new_tokens
                            budget_rec.content     = final_text[:500]
                            matched = True
                            break
                    if not matched:
                        self._register_entry(
                            entry_id, session_file, turn_idx, new_tokens, final_text
                        )
                    any_stored = True
                else:
                    # hallucinated id → ADD로 fallback
                    print(f"  [R1] UPDATE id '{entry_id}' not in candidates, fallback to ADD")
                    new_id = str(uuid_lib.uuid4())
                    embedding = self._embedder.embed_one(final_text)
                    self._facts[new_id] = _FactRecord(
                        entry_id=new_id,
                        text=final_text,
                        embedding=embedding,
                        session_file=session_file,
                        turn_idx=turn_idx,
                        insert_order=self._insert_counter,
                    )
                    self._register_entry(
                        new_id, session_file, turn_idx, count_tokens(final_text), final_text
                    )
                    any_stored = True

            elif event == "DELETE" and entry_id:
                # id 유효성 검증: candidate로 넘겨준 id + 실제 _facts에 존재해야 함
                if entry_id in candidate_set and entry_id in self._facts:
                    self._delete_entry(entry_id)
                    self._remove_budget_entry(entry_id)
                else:
                    print(f"  [R1] DELETE id '{entry_id}' not in candidates, skipping")
                # DELETE는 written으로 카운트하지 않음

            # NONE → 스킵

        return any_stored

    # ─────────────────────────────────────
    # LLM 호출 — Info Extraction
    # ─────────────────────────────────────

    def _extract_turn_facts(self, user_content: str, agent_content: str) -> list[str]:
        """단일 turn에서 facts 리스트 추출 (LLM 1회)."""
        prompt = TURN_EXTRACTION_PROMPT.format(
            user_content=user_content,
            agent_content=agent_content,
        )
        try:
            raw = self._get_llm().chat(prompt, system=TURN_EXTRACTION_SYSTEM).strip()
        except Exception as e:
            print(f"  [R1] turn_extraction failed: {e}")
            return []
        return self._parse_facts_list(raw)

    def _extract_session_facts(self, session_text: str) -> list[str]:
        """세션 전체 텍스트에서 facts 리스트 추출 (LLM 1회)."""
        prompt = SESSION_EXTRACTION_PROMPT.format(session_text=session_text)
        try:
            raw = self._get_llm().chat(prompt, system=SESSION_EXTRACTION_SYSTEM).strip()
        except Exception as e:
            print(f"  [R1] session_extraction failed: {e}")
            return []
        return self._parse_facts_list(raw)

    # ─────────────────────────────────────
    # LLM 호출 — Memory Manager
    # ─────────────────────────────────────

    def _get_candidate_ids(self, query_text: str, top_k: int) -> list[str]:
        """
        query_text와 코사인 유사도 높은 상위 top_k fact의 entry_id 리스트 반환.
        전체 메모리가 top_k 이하면 전부 반환.
        """
        if not self._facts:
            return []

        facts = [f for f in self._facts.values() if f.embedding is not None]
        if not facts:
            return []

        top_k = min(top_k, len(facts))
        query_emb = self._embedder.embed_one(query_text)
        emb_matrix = np.stack([f.embedding for f in facts])
        scores = cosine_similarity(query_emb, emb_matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [facts[i].entry_id for i in top_indices]

    def _run_memory_manager(self, new_facts: list[str], candidate_ids: list[str]) -> list[dict]:
        """
        embedding으로 추린 candidate 기존 메모리 + 새 facts를 Memory Manager에 입력.
        반환: [{"id": ..., "text": ..., "event": ...}, ...] operations 리스트.
        논문 Figure 9/10의 프롬프트 형식 그대로 사용.
        """
        # candidate 메모리만 포매팅 (전체가 아님)
        if candidate_ids:
            old_memory_json = json.dumps(
                [{"id": eid, "text": self._facts[eid].text}
                 for eid in candidate_ids if eid in self._facts],
                ensure_ascii=False,
                indent=2,
            )
        else:
            old_memory_json = "[]"

        # 새 facts 포매팅
        retrieved_facts_json = json.dumps(new_facts, ensure_ascii=False)

        prompt = MEMORY_MANAGER_PROMPT.format(
            old_memory=old_memory_json,
            retrieved_facts=retrieved_facts_json,
        )

        try:
            raw = self._get_llm().chat(prompt, system=MEMORY_MANAGER_SYSTEM).strip()
        except Exception as e:
            print(f"  [R1] memory_manager failed: {e}")
            # fallback: 모든 new_facts를 ADD로 처리
            return [{"id": None, "text": f, "event": "ADD"} for f in new_facts]

        parsed = self._parse_json(raw)
        if parsed is None:
            print(f"  [R1] memory_manager parse failed, raw={raw[:120]}")
            return [{"id": None, "text": f, "event": "ADD"} for f in new_facts]

        operations = parsed.get("memory", [])
        if not isinstance(operations, list):
            return [{"id": None, "text": f, "event": "ADD"} for f in new_facts]

        return operations

    # ─────────────────────────────────────
    # 파싱 유틸
    # ─────────────────────────────────────

    @staticmethod
    def _parse_facts_list(raw: str) -> list[str]:
        """LLM 응답에서 JSON array 파싱."""
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return [str(f).strip() for f in result if str(f).strip()]
        except json.JSONDecodeError:
            m = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group())
                    if isinstance(result, list):
                        return [str(f).strip() for f in result if str(f).strip()]
                except json.JSONDecodeError:
                    pass
        return []

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        """LLM 응답에서 JSON 객체 파싱. 마크다운 펜스 제거 후 시도."""
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return None

    # ─────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────

    @staticmethod
    def _build_session_text(turns: list[tuple[int, str, str]]) -> str:
        """turns 리스트를 하나의 대화 텍스트로 병합."""
        parts = []
        for _, user_content, agent_content in turns:
            if user_content:
                parts.append(f"User: {user_content}")
            if agent_content:
                parts.append(f"Agent: {agent_content}")
        return "\n".join(parts)

    # ─────────────────────────────────────
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        if not self._facts:
            return []

        query_emb = self._embedder.embed_one(query)
        facts = [f for f in self._facts.values() if f.embedding is not None]
        if not facts:
            return []

        emb_matrix = np.stack([f.embedding for f in facts])
        scores = cosine_similarity(query_emb, emb_matrix)
        top_k = min(top_k, len(facts))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            MemoryChunk(
                content=facts[i].text,
                session_file=facts[i].session_file,
                turn_idx=facts[i].turn_idx,
                keywords=[],
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    # ─────────────────────────────────────
    # Write check override
    # ─────────────────────────────────────

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """_entry_records에 등록된 fact 텍스트 반환 (Mem0System과 동일 패턴)."""
        if not written_keys:
            return ""

        written_set = {(sf, ti) for sf, ti in written_keys}
        facts = [
            rec.content
            for rec in self._entry_records
            if (rec.session_file, rec.turn_idx) in written_set and rec.content
        ]
        return "\n\n".join(facts)

    # ─────────────────────────────────────
    # dump_memories override
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        """_facts의 최신 텍스트로 snapshot 구성 (UPDATE 반영)."""
        result = []
        for rec in self._entry_records:
            fact = self._facts.get(rec.entry_id)
            if fact:
                result.append({
                    "memory_id":    rec.entry_id,
                    "memory_text":  fact.text,
                    "session_file": rec.session_file,
                    "turn_idx":     rec.turn_idx,
                    "token_count":  rec.token_count,
                    "insert_order": rec.insert_order,
                    "content":      fact.text,
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
    # Budget 내부 헬퍼
    # ─────────────────────────────────────

    def _remove_budget_entry(self, entry_id: str) -> None:
        """_entry_records에서 entry_id 레코드 제거 후 토큰 차감."""
        for rec in self._entry_records:
            if rec.entry_id == entry_id:
                self._total_tokens = max(0, self._total_tokens - rec.token_count)
                self._entry_records = type(self._entry_records)(
                    r for r in self._entry_records if r.entry_id != entry_id
                )
                return

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        """facts 메타 → JSON, embeddings → npz (heuristic.py와 동일 패턴)."""
        facts_meta = []
        embeddings = {}
        for entry_id, fact in self._facts.items():
            facts_meta.append({
                "entry_id":     entry_id,
                "text":         fact.text,
                "session_file": fact.session_file,
                "turn_idx":     fact.turn_idx,
                "insert_order": fact.insert_order,
            })
            if fact.embedding is not None:
                embeddings[entry_id] = fact.embedding

        state = {
            "system":             "MemoryR1System",
            "llm_provider":       self.llm_provider,
            "llm_model":          self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model":    self.embedding_model_name,
            "facts":              facts_meta,
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        if embeddings:
            np.savez(str(ckpt_path / "embeddings.npz"), **embeddings)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        backend_path = ckpt_path / "backend_state.json"
        emb_path = ckpt_path / "embeddings.npz"
        if not backend_path.exists():
            return

        with open(backend_path, encoding="utf-8") as f:
            state = json.load(f)

        emb_data = {}
        if emb_path.exists():
            npz = np.load(str(emb_path))
            emb_data = {k: npz[k] for k in npz.files}

        self._facts = {}
        for f in state.get("facts", []):
            self._facts[f["entry_id"]] = _FactRecord(
                entry_id=f["entry_id"],
                text=f["text"],
                embedding=emb_data.get(f["entry_id"]),
                session_file=f["session_file"],
                turn_idx=f["turn_idx"],
                insert_order=f["insert_order"],
            )
