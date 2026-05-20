"""
memory_systems/mem0_system.py

Mem0 wrapper.
pip install mem0ai

Token budget:
    heuristic과 달리 raw turn 텍스트가 아닌 Mem0가 실제로 추출한 fact 단위로
    _entry_records에 등록한다.
    - turn 1개 → fact N개 (N >= 0)
    - _entry_records 1 row = fact 1개 (fact 텍스트 + 토큰 수)
    - post_session()의 oldest-first eviction도 fact 단위로 동작

Write check:
    get_write_evidence()는 _entry_records에 저장된 fact 텍스트를 반환한다.
    (raw 대화 텍스트 아님)

주의:
    mem0.add() 결과의 "event" 필드:
        ADD    → 새 fact 생성 → _entry_records에 등록
        UPDATE → 기존 fact 갱신 → 기존 record token_count 갱신
        DELETE → 기존 fact 삭제 → _entry_records에서 제거
        NONE   → 변화 없음 → 무시
"""

import re
import uuid as uuid_lib

from .base import BaseMemorySystem, MemoryChunk, count_tokens


class Mem0System(BaseMemorySystem):
    """
    Mem0 기반 memory system.
    install: pip install mem0ai
    """
    SESSION_ADD_MAX_TOKENS = 6000

    def __init__(
        self,
        max_tokens: int = 8000,
        config: dict | None = None,
        user_id_prefix: str = "u_mem_eval",
    ):
        super().__init__(max_tokens=max_tokens)
        self.config = config
        self.user_id_prefix = user_id_prefix
        self._memory = None
        self._user_id: str = ""

    def _reset_backend(self, user_id: str | None = None) -> None:
        from mem0 import Memory

        # Qdrant local backend uses a filesystem lock. When resetting backend
        # in the same process, release previous handles first.
        if self._memory is not None:
            try:
                if hasattr(self._memory, "close"):
                    self._memory.close()
            except Exception:
                pass

            def _close_store_client(store_obj) -> None:
                try:
                    client = getattr(store_obj, "client", None)
                    if client is not None and hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass

            try:
                _close_store_client(getattr(self._memory, "vector_store", None))
                _close_store_client(getattr(self._memory, "_telemetry_vector_store", None))
                _close_store_client(getattr(self._memory, "_entity_store", None))
            except Exception:
                pass

        if self.config:
            from mem0.configs.base import MemoryConfig
            self._memory = Memory(config=MemoryConfig(**self.config))
        else:
            self._memory = Memory()
        self._user_id = f"{self.user_id_prefix}_{uuid_lib.uuid4().hex[:8]}"

    # ─────────────────────────────────────
    # 추상 메서드 구현 (base 요구사항)
    # write()를 override하므로 실제로 호출되지 않지만 abstract이라 구현 필요
    # ─────────────────────────────────────

    def _write_turn(
        self,
        session_file: str,
        turn_idx: int,
        session_idx: int,
        domain_name: str,
        user_content: str,
        agent_content: str,
    ) -> str | None:
        raise NotImplementedError("Mem0System uses overridden write() directly")

    def _delete_entry(self, entry_id: str) -> None:
        try:
            self._memory.delete(entry_id)
        except Exception as e:
            print(f"  [MEM0] delete failed (id={entry_id}): {e}")

    # ─────────────────────────────────────
    # write() / write_session() override
    # fact 단위로 _entry_records 등록
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """
        Turn 단위 저장. 각 turn에서 mem0가 추출한 fact들을 _entry_records에 등록.
        turn 1개 → fact N개가 별도 _EntryRecord로 등록됨.
        """
        dialogue    = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')
        session_idx  = session.get('session_idx', -1)
        domain_name  = session.get('domain_name', '')

        turns = self.extract_turns(dialogue)
        written = []

        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue

            had_fact = self._add_and_register(
                session_file=session_file,
                turn_idx=turn_idx,
                session_idx=session_idx,
                domain_name=domain_name,
                messages=[
                    {"role": "user",
                     "content": user_content + f" [sf:{session_file}|ti:{turn_idx}]"},
                    {"role": "assistant", "content": agent_content},
                ],
            )

            if had_fact:
                self._written_turns.add((session_file, turn_idx))
                written.append((session_file, turn_idx))

        return written

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        Session 단위 저장. 세션 전체 메시지를 한 번의 add()로 전달.
        추출된 fact들을 turn_idx=-1 sentinel로 _entry_records에 등록.
        """
        dialogue     = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')
        session_idx  = session.get('session_idx', -1)
        domain_name  = session.get('domain_name', '')

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        chunks = self._chunk_turns_for_session(
            turns=turns,
            max_tokens=self.SESSION_ADD_MAX_TOKENS,
        )
        if len(chunks) > 1:
            print(
                f"  [MEM0] session input chunked for token limit "
                f"(session={session_file}, chunks={len(chunks)})"
            )

        had_fact = False
        for chunk in chunks:
            messages = []
            for turn_idx, user_content, agent_content in chunk:
                messages.append({
                    "role": "user",
                    "content": user_content + f" [sf:{session_file}|ti:{turn_idx}]",
                })
                if agent_content:
                    messages.append({"role": "assistant", "content": agent_content})

            had_fact = (
                self._add_and_register(
                    session_file=session_file,
                    turn_idx=-1,            # sentinel: 세션 전체
                    session_idx=session_idx,
                    domain_name=domain_name,
                    messages=messages,
                )
                or had_fact
            )

        if had_fact:
            self._written_turns.add((session_file, -1))
            return [(session_file, -1)]
        return []

    @staticmethod
    def _chunk_turns_for_session(
        turns: list[tuple[int, str, str]],
        max_tokens: int,
    ) -> list[list[tuple[int, str, str]]]:
        """
        Session 모드를 유지하면서 too-long 입력을 피하기 위해 turn 묶음을 분할.
        각 chunk는 대략 max_tokens 이하가 되도록 구성한다.
        """
        if not turns:
            return []

        chunks: list[list[tuple[int, str, str]]] = []
        current: list[tuple[int, str, str]] = []
        current_tokens = 0

        for t in turns:
            _, user_content, agent_content = t
            turn_tokens = count_tokens(f"User: {user_content}\nAssistant: {agent_content}")

            if current and current_tokens + turn_tokens > max_tokens:
                chunks.append(current)
                current = []
                current_tokens = 0

            current.append(t)
            current_tokens += turn_tokens

            # 단일 turn이 임계치를 넘는 경우라도 최소 1 turn chunk는 유지한다.
            if current_tokens > max_tokens:
                chunks.append(current)
                current = []
                current_tokens = 0

        if current:
            chunks.append(current)

        return chunks

    def _add_and_register(
        self,
        session_file: str,
        turn_idx: int,
        session_idx: int,
        domain_name: str,
        messages: list[dict],
    ) -> bool:
        """
        memory.add() 호출 후 반환된 fact 결과들을 _entry_records에 등록.

        mem0 결과의 event 처리:
            ADD    → _register_entry() (새 fact)
            UPDATE → 기존 record token_count 갱신 (fact 텍스트 변경)
            DELETE → _entry_records에서 제거 (mem0가 자체 판단으로 삭제)
            NONE / 기타 → 무시
        """
        try:
            result  = self._memory.add(
                messages,
                user_id=self._user_id,
                metadata={
                    "session_file": session_file,
                    "turn_idx": turn_idx,
                    "session_idx": session_idx,
                    "domain_name": domain_name,
                },
            )
            results = result.get("results", []) if isinstance(result, dict) else []
        except Exception as e:
            print(f"  [MEM0] add failed "
                  f"(session={session_file}, turn={turn_idx}): {e}")
            return False

        had_fact = False
        for item in results:
            event     = (item.get("event") or "ADD").upper()
            fact_id   = item.get("id", "")
            fact_text = item.get("memory", "")

            if event == "ADD":
                if fact_id and fact_text:
                    self._register_entry(
                        entry_id=fact_id,
                        session_file=session_file,
                        turn_idx=turn_idx,
                        token_count=count_tokens(fact_text),
                        content=fact_text,
                    )
                    had_fact = True

            elif event == "UPDATE":
                if fact_id and fact_text:
                    prev_text     = item.get("previous_memory", "")
                    new_tokens    = count_tokens(fact_text)
                    matched       = False
                    for rec in self._entry_records:
                        if rec.entry_id == fact_id:
                            self._total_tokens = max(
                                0, self._total_tokens - rec.token_count + new_tokens
                            )
                            rec.token_count = new_tokens
                            rec.content     = fact_text
                            matched = True
                            print(f"  [MEM0] UPDATE  prev='{prev_text[:60]}'"
                                  f"  →  new='{fact_text[:60]}'")
                            break
                    if not matched:
                        # 이전 세션에서 만들어진 fact가 갱신되는 경우 — 새로 등록
                        self._register_entry(
                            entry_id=fact_id,
                            session_file=session_file,
                            turn_idx=turn_idx,
                            token_count=new_tokens,
                            content=fact_text,
                        )
                    had_fact = True

            elif event == "DELETE":
                # mem0가 자체 판단으로 fact 삭제 → budget에서도 제거
                if fact_id:
                    self._remove_entry_by_id(fact_id)

            # NONE → 무시

        return had_fact

    def _remove_entry_by_id(self, entry_id: str) -> None:
        """_entry_records에서 entry_id 해당 레코드 제거."""
        for i, rec in enumerate(self._entry_records):
            if rec.entry_id == entry_id:
                self._total_tokens = max(0, self._total_tokens - rec.token_count)
                # deque는 중간 삭제가 없으므로 rebuild
                self._entry_records = type(self._entry_records)(
                    r for r in self._entry_records if r.entry_id != entry_id
                )
                return

    # ─────────────────────────────────────
    # Write check override
    # ─────────────────────────────────────

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """
        _entry_records에 등록된 fact 텍스트를 반환.
        (raw 대화 텍스트 아님 — mem0가 실제로 저장한 fact들)
        """
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
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        try:
            result = self._memory.search(
                query=query,
                user_id=self._user_id,
                limit=top_k,
            )
            results = result.get("results", []) if isinstance(result, dict) else result
        except Exception as e:
            print(f"  [MEM0] search failed: {e}")
            return []

        chunks = []
        for r in results:
            memory_text  = r.get("memory", "")
            meta         = r.get("metadata", {}) or {}
            session_file = meta.get("session_file", "")
            turn_idx     = meta.get("turn_idx", -1)
            if not session_file or turn_idx == -1:
                m = re.search(r'\[sf:([^\|]+)\|ti:(\d+)\]', memory_text)
                if m:
                    session_file = m.group(1)
                    turn_idx     = int(m.group(2))

            clean_content = re.sub(r'\s*\[sf:[^\]]+\]', '', memory_text).strip()
            chunks.append(MemoryChunk(
                content=clean_content,
                session_file=session_file,
                turn_idx=turn_idx,
                keywords=[],
                score=r.get("score", 0.0),
                raw=r,
            ))
        return chunks

    # ─────────────────────────────────────
    # dump_memories override
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        """
        _entry_records 기반 fact 목록 반환.
        mem0 내부 DB와 budget tracker가 일치하지 않을 수 있으므로
        budget tracker 기준(= 실제 추적 중인 fact)을 사용.
        """
        return [
            {
                "memory_id":    rec.entry_id,
                "memory_text":  rec.content,
                "session_file": rec.session_file,
                "turn_idx":     rec.turn_idx,
                "token_count":  rec.token_count,
                "insert_order": rec.insert_order,
                "content":      rec.content,
            }
            for rec in self._entry_records
        ]

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        import json
        state = {
            "system":          "Mem0System",
            "user_id":         self._user_id,
            "user_id_prefix":  self.user_id_prefix,
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        import json
        backend_path = ckpt_path / "backend_state.json"
        if not backend_path.exists():
            return
        with open(backend_path, encoding="utf-8") as f:
            state = json.load(f)
        self._user_id = state["user_id"]
