"""
memory_systems/supermemory_system.py

Supermemory wrapper — Cloud Memory API.
https://github.com/supermemoryai/supermemory

설치:
    pip install supermemory

API 키 설정:
    export SUPERMEMORY_API_KEY="sm_your_api_key_here"
    (https://console.supermemory.ai 에서 발급)

핵심 동작:
    - client.add(content, container_tag) 로 대화 저장
    - client.search.memories(q, container_tag) 로 검색
    - container_tag = user별 격리 ID
    - write_evidence: 저장한 raw content 텍스트 직접 반환

주의:
    - 클라우드 전용 API (로컬 실행 불가)
    - Supermemory가 자체적으로 fact 추출/dedup 수행
    - 개별 memory 삭제 API 있으나 token budget enforcement는 근사치
    - retrieve 결과 구조: result.content 또는 result.memory 필드
"""

import os
import uuid as uuid_lib
from .base import BaseMemorySystem, MemoryChunk, count_tokens


class SupermemorySystem(BaseMemorySystem):
    """
    Supermemory 기반 memory system.
    install: pip install supermemory
    env: SUPERMEMORY_API_KEY
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        api_key: str | None = None,
        user_id_prefix: str = "u_mem_eval",
    ):
        super().__init__(max_tokens=max_tokens)
        self.api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "SUPERMEMORY_API_KEY가 설정되지 않았습니다. "
                "export SUPERMEMORY_API_KEY=... 또는 api_key 인자 사용."
            )
        self.user_id_prefix = user_id_prefix

        self._client = None
        self._container_tag: str = ""
        self._session_contents: list[str] = []  # 이번 세션에 저장한 raw content
        self._reset_backend()

    def _reset_backend(self, user_id: str | None = None) -> None:
        from supermemory import Supermemory
        self._client = Supermemory(api_key=self.api_key)
        self._container_tag = f"{self.user_id_prefix}_{uuid_lib.uuid4().hex[:8]}"
        self._session_contents = []

    # ─────────────────────────────────────
    # 추상 메서드 (write() 전체를 override)
    # ─────────────────────────────────────

    def _write_turn(self, session_file, turn_idx, session_idx, domain_name,
                    user_content, agent_content) -> str | None:
        raise NotImplementedError("SupermemorySystem uses overridden write() directly")

    def _delete_entry(self, entry_id: str) -> None:
        # Supermemory는 cloud 관리 — token budget 초과 시 no-op
        # (필요 시 client.memories.delete(memory_id) 구현 가능)
        pass

    # ─────────────────────────────────────
    # write() / write_session() override
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """
        Turn 단위 저장.
        각 turn을 개별 content로 add() 호출.
        """
        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        written = []
        self._session_contents = []

        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue
            content = self.build_raw_text(user_content, agent_content)
            try:
                self._client.add(
                    content=content,
                    container_tag=self._container_tag,
                )
                entry_id = f"{session_file}__t{turn_idx}"
                self._register_entry(
                    entry_id, session_file, turn_idx,
                    count_tokens(content), content,
                )
                self._written_turns.add((session_file, turn_idx))
                written.append((session_file, turn_idx))
                self._session_contents.append(content)
            except Exception as e:
                print(f"  [SUPERMEMORY] add failed (turn={turn_idx}): {e}")

        return written

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        Session 단위 저장.
        전체 turn을 하나의 content로 묶어 add() 1회 호출.
        """
        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        all_content = "\n\n".join(
            self.build_raw_text(uc, ac) for _, uc, ac in turns
        )

        self._session_contents = []
        try:
            self._client.add(
                content=all_content,
                container_tag=self._container_tag,
            )
            entry_id = f"{session_file}__session"
            self._register_entry(
                entry_id, session_file, -1,
                count_tokens(all_content), all_content[:500],
            )
            self._written_turns.add((session_file, -1))
            self._session_contents = [all_content]
            return [(session_file, -1)]
        except Exception as e:
            print(f"  [SUPERMEMORY] add failed (session): {e}")
            return []

    # ─────────────────────────────────────
    # Write check override
    # ─────────────────────────────────────

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """
        이번 세션에 add()한 raw content를 write evidence로 반환.
        Supermemory가 내부적으로 추출한 fact가 아닌 입력 텍스트 기준.
        """
        if not written_keys:
            return ""
        return "\n\n".join(self._session_contents)

    # ─────────────────────────────────────
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        """
        Supermemory semantic search.
        client.search.memories(q, container_tag) 사용.
        """
        try:
            results = self._client.search.memories(
                q=query,
                container_tag=self._container_tag,
            )
        except Exception as e:
            print(f"  [SUPERMEMORY] search failed: {e}")
            return []

        chunks = []
        # results는 list 또는 SearchResponse 객체
        items = results if isinstance(results, list) else getattr(results, "results", [])
        for item in items[:top_k]:
            # 필드명은 SDK 버전에 따라 다를 수 있음
            content = (
                getattr(item, "content", None)
                or getattr(item, "memory",  None)
                or getattr(item, "text",    None)
                or str(item)
            )
            score = float(getattr(item, "score", 1.0))
            chunks.append(MemoryChunk(
                content=content,
                session_file="",
                turn_idx=-1,
                score=score,
            ))
        return chunks

    # ─────────────────────────────────────
    # dump_memories override
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        """
        현재 저장된 memory 목록.
        _entry_records 기반 (Supermemory 서버 직접 조회 아님).
        """
        return [
            {
                "session_file": r.session_file,
                "turn_idx":     r.turn_idx,
                "token_count":  r.token_count,
                "content":      r.content,
                "insert_order": r.insert_order,
            }
            for r in self._entry_records
        ]

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        import json
        state = {
            "system":        "SupermemorySystem",
            "container_tag": self._container_tag,
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        import json
        p = ckpt_path / "backend_state.json"
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            state = json.load(f)
        self._container_tag = state["container_tag"]
