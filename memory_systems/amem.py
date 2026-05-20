"""
memory_systems/amem.py

A-MEM (Agentic Memory) wrapper.
pip install git+https://github.com/agiresearch/A-mem.git

Token budget: BaseMemorySystem의 oldest-first 공통 로직 사용.
A-MEM 내부 delete API: memory_system.delete(memory_id)
"""

import re

from .base import BaseMemorySystem, MemoryChunk, count_tokens


_META_TAG_PREFIX = "__meta__"


def _make_meta_tag(session_file: str, turn_idx: int) -> str:
    safe = re.sub(r'[^\w]', '_', session_file)
    return f"{_META_TAG_PREFIX}sf_{safe}_ti_{turn_idx}"


def _parse_meta_from_content(content: str) -> tuple[str, int] | None:
    m = re.match(r'\[META:([^:]+):(\d+)\]', content)
    if m:
        return m.group(1), int(m.group(2))
    return None


class AmemSystem(BaseMemorySystem):
    """
    A-MEM 기반 memory system.
    install: pip install git+https://github.com/agiresearch/A-mem.git
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        embedding_model: str = 'all-MiniLM-L6-v2',
        llm_backend: str = 'openai',
        llm_model: str = 'gpt-4o-mini',
    ):
        super().__init__(max_tokens=max_tokens)
        self.embedding_model = embedding_model
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self._memory = None
        self._reset_backend()

    def _reset_backend(self, user_id: str | None = None) -> None:
        from agentic_memory.memory_system import AgenticMemorySystem
        self._memory = AgenticMemorySystem(
            model_name=self.embedding_model,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
        )

    def _write_turn(
        self,
        session_file: str,
        turn_idx: int,
        session_idx: int,
        domain_name: str,
        user_content: str,
        agent_content: str,
    ) -> str | None:
        raw_text = self.build_raw_text(user_content, agent_content)
        meta_header = f"[META:{session_file}:{turn_idx}]"
        content = f"{meta_header}\n{raw_text}"
        meta_tag = _make_meta_tag(session_file, turn_idx)
        tags = [meta_tag, domain_name, f"session_idx_{session_idx}"]

        try:
            memory_id = self._memory.add_note(
                content=content,
                tags=tags,
                category=domain_name,
            )
            return memory_id if memory_id else None
        except Exception as e:
            print(f"  [AMEM] add_note failed (session={session_file}, turn={turn_idx}): {e}")
            return None

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        Session 단위 저장: 세션 전체 대화를 하나의 note로 A-MEM에 전달.

        turn_idx=-1 sentinel을 사용 (Mem0System과 동일 규칙).
        A-MEM은 세션 전체 텍스트를 한 번의 add_note()로 처리하므로
        LLM 호출 횟수가 turn 단위 대비 크게 줄어든다.
        """
        dialogue     = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')
        session_idx  = session.get('session_idx', -1)
        domain_name  = session.get('domain_name', '')

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        # 세션 전체 텍스트를 하나로 합침
        turn_parts = []
        for turn_idx, user_content, agent_content in turns:
            turn_parts.append(self.build_raw_text(user_content, agent_content))
        full_text = "\n\n".join(turn_parts)

        meta_header = f"[META:{session_file}:-1]"
        content = f"{meta_header}\n{full_text}"
        meta_tag = _make_meta_tag(session_file, -1)
        tags = [meta_tag, domain_name, f"session_idx_{session_idx}"]

        try:
            memory_id = self._memory.add_note(
                content=content,
                tags=tags,
                category=domain_name,
            )
        except Exception as e:
            print(f"  [AMEM] add_note failed (session={session_file}, session-level): {e}")
            return []

        if not memory_id:
            return []

        token_count = count_tokens(full_text)
        self._register_entry(memory_id, session_file, -1, token_count, full_text[:500])
        self._written_turns.add((session_file, -1))
        return [(session_file, -1)]

    def _delete_entry(self, entry_id: str) -> None:
        try:
            self._memory.delete(entry_id)
        except Exception as e:
            print(f"  [AMEM] delete failed (id={entry_id}): {e}")

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        try:
            results = self._memory.search_agentic(query, k=top_k)
        except Exception as e:
            print(f"  [AMEM] search_agentic failed: {e}")
            return []

        chunks = []
        for r in results:
            content = r.get('content', '')
            keywords = r.get('keywords', [])
            if isinstance(keywords, str):
                keywords = [keywords]

            meta = _parse_meta_from_content(content)
            clean_content = re.sub(r'^\[META:[^\]]+\]\n?', '', content)
            session_file = meta[0] if meta else ""
            turn_idx = meta[1] if meta else -1

            chunks.append(MemoryChunk(
                content=clean_content,
                session_file=session_file,
                turn_idx=turn_idx,
                keywords=keywords if isinstance(keywords, list) else [],
                score=r.get('score', 0.0),
                raw=r,
            ))
        return chunks

    def dump_memories(self) -> list[dict]:
        """
        A-MEM이 진화(evolution)시킨 실제 note 내용으로 snapshot을 구성.

        base class는 _entry_records의 raw input text(500자 truncated)를 반환하지만,
        A-MEM은 add_note 이후 LLM이 keywords/context/tags를 재구성하고
        인접 메모리와 연결(evolution)하여 note 내용이 변할 수 있다.
        따라서 self._memory.memories에서 최신 note를 읽어 반환한다.

        7_longitudinal_eval_v2.py의 get_memory_texts()가
        m.get("content")를 참조하므로 content 키에 clean text를 넣는다.
        """
        result = []
        for r in self._entry_records:
            note = self._memory.memories.get(r.entry_id) if self._memory else None
            if note:
                clean_content = re.sub(r'^\[META:[^\]]+\]\n?', '', note.content)
                result.append({
                    "session_file": r.session_file,
                    "turn_idx":     r.turn_idx,
                    "token_count":  r.token_count,
                    "insert_order": r.insert_order,
                    "content":      clean_content,
                    "keywords":     note.keywords if isinstance(note.keywords, list) else [],
                    "context":      note.context or "",
                    "tags":         note.tags if isinstance(note.tags, list) else [],
                })
            else:
                # note가 eviction 등으로 이미 삭제된 경우 fallback
                result.append({
                    "session_file": r.session_file,
                    "turn_idx":     r.turn_idx,
                    "token_count":  r.token_count,
                    "insert_order": r.insert_order,
                    "content":      r.content,
                })
        return result

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        """
        A-MEM은 ChromaDB가 디스크에 자동으로 persist.
        collection 이름과 설정만 저장하면 복원 가능.
        """
        import json
        state = {
            "system": "AmemSystem",
            "embedding_model": self.embedding_model,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            # ChromaDB persist_directory (A-MEM 기본값)
            "note": "ChromaDB persists to disk automatically. Re-init with same config to restore.",
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        """
        동일 설정으로 AgenticMemorySystem을 재초기화하면
        ChromaDB가 기존 데이터를 자동으로 로드.
        _reset_backend()를 호출하지 않고 그대로 둠 — 이미 init된 상태 유지.
        """
        pass  # ChromaDB가 디스크에서 자동 복원