"""
memory_systems/memgpt.py

MemGPT (Letta) wrapper.
pip install letta-client

Token budget: BaseMemorySystem의 oldest-first 공통 로직 사용.
Letta delete API: client.agents.archival_memory.delete(agent_id, memory_id)

서버 실행: letta server (기본 포트 8283)
"""

import re

from .base import BaseMemorySystem, MemoryChunk


class MemGPTSystem(BaseMemorySystem):
    """
    MemGPT (Letta) 기반 memory system.
    install: pip install letta-client
    server:  letta server
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        base_url: str = 'http://localhost:8283',
        model: str = 'openai/gpt-4o-mini',
        embedding: str = 'openai/text-embedding-3-small',
        agent_name_prefix: str = 'mem_eval_agent',
    ):
        super().__init__(max_tokens=max_tokens)
        self.base_url = base_url
        self.model = model
        self.embedding = embedding
        self.agent_name_prefix = agent_name_prefix
        self._client = None
        self._agent_id: str | None = None
        self._agent_counter = 0
        self._init_client()
        self._reset_backend()

    def _init_client(self):
        from letta_client import Letta
        self._client = Letta(base_url=self.base_url)

    def _reset_backend(self, user_id: str | None = None) -> None:
        """기존 agent 삭제 후 새 agent 생성."""
        if self._agent_id:
            try:
                self._client.agents.delete(self._agent_id)
            except Exception:
                pass
        self._agent_counter += 1
        name = f"{self.agent_name_prefix}_{self._agent_counter}"
        try:
            agent = self._client.agents.create(
                name=name,
                model=self.model,
                embedding=self.embedding,
            )
            self._agent_id = agent.id
        except Exception as e:
            raise RuntimeError(f"[MEMGPT] agent 생성 실패: {e}")

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
        content = f"{meta_header} {domain_name}\n{raw_text}"
        try:
            result = self._client.agents.archival_memory.insert(
                agent_id=self._agent_id,
                text=content,
            )
            # Letta는 삽입된 memory 객체 반환
            memory_id = getattr(result, 'id', None) or str(result)
            return memory_id
        except Exception as e:
            print(f"  [MEMGPT] insert failed (session={session_file}, turn={turn_idx}): {e}")
            return None

    def _delete_entry(self, entry_id: str) -> None:
        try:
            self._client.agents.archival_memory.delete(
                agent_id=self._agent_id,
                memory_id=entry_id,
            )
        except Exception as e:
            print(f"  [MEMGPT] delete failed (id={entry_id}): {e}")

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        if not self._agent_id:
            return []
        try:
            results = self._client.agents.archival_memory.list(
                agent_id=self._agent_id,
                query=query,
                limit=top_k,
            )
        except Exception as e:
            print(f"  [MEMGPT] archival_memory.list failed: {e}")
            return []

        chunks = []
        for r in results:
            text = getattr(r, 'text', '') or ''
            m = re.match(r'\[META:([^:]+):(\d+)\]', text)
            session_file = m.group(1) if m else ""
            turn_idx = int(m.group(2)) if m else -1
            clean_content = re.sub(r'^\[META:[^\]]+\]\s*\S*\n?', '', text).strip()
            chunks.append(MemoryChunk(
                content=clean_content,
                session_file=session_file,
                turn_idx=turn_idx,
                keywords=[],
                score=getattr(r, 'score', 0.0),
                raw={"id": getattr(r, 'id', ''), "text": text},
            ))
        return chunks

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        """
        Letta server가 agent 상태를 영속적으로 관리.
        agent_id만 저장하면 서버에서 그대로 복원 가능.
        """
        import json
        state = {
            "system": "MemGPTSystem",
            "agent_id": self._agent_id,
            "base_url": self.base_url,
            "model": self.model,
            "embedding": self.embedding,
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        """
        저장된 agent_id로 복원.
        _reset_backend()에서 새 agent를 만들었을 수 있으므로
        그 agent를 삭제하고 checkpoint의 agent_id로 교체.
        """
        import json
        backend_path = ckpt_path / "backend_state.json"
        if not backend_path.exists():
            return
        with open(backend_path, encoding="utf-8") as f:
            state = json.load(f)
        restored_agent_id = state.get("agent_id")
        if not restored_agent_id:
            return
        # reset_backend에서 만든 새 agent 삭제
        if self._agent_id and self._agent_id != restored_agent_id:
            try:
                self._client.agents.delete(self._agent_id)
            except Exception:
                pass
        self._agent_id = restored_agent_id