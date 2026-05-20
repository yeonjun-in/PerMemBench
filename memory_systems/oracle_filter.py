"""
memory_systems/oracle_filter.py

Oracle Filter — BaseMemorySystem을 감싸는 decorator.
memory_required=True인 도메인 세션만 write()로 전달.
token budget, retrieve, reset 등은 내부 system에 위임.
"""

from .base import BaseMemorySystem, MemoryChunk


class OracleFilter(BaseMemorySystem):
    """
    memory_required=True인 세션만 통과시키는 필터 decorator.
    """

    def __init__(self, system: BaseMemorySystem):
        # max_tokens는 inner system이 관리하므로 여기선 dummy
        super().__init__(max_tokens=system.max_tokens)
        self.system = system

    # Oracle filter는 write()만 오버라이드하고 나머지는 inner에 위임
    def write(self, session: dict) -> list[tuple[str, int]]:
        if not session.get('memory_required', False):
            return []
        return self.system.write(session)

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        if not session.get('memory_required', False):
            return []
        return self.system.write_session(session)

    def post_session(self) -> dict:
        return self.system.post_session()

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        return self.system.retrieve(query, top_k=top_k)

    def get_written_turns(self) -> set[tuple[str, int]]:
        return self.system.get_written_turns()

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        return self.system.get_write_evidence(session, written_keys)

    def reset(self, user_id: str | None = None) -> None:
        # BaseMemorySystem.reset(user_id=...) 호출 규약을 그대로 따른다.
        self.system.reset(user_id=user_id)

    # BaseMemorySystem 추상 메서드 — OracleFilter는 직접 쓰지 않음
    def _write_turn(self, *args, **kwargs) -> str | None:
        return self.system._write_turn(*args, **kwargs)

    def _delete_entry(self, entry_id: str) -> None:
        self.system._delete_entry(entry_id)

    def _reset_backend(self, user_id: str | None = None) -> None:
        self.system._reset_backend(user_id=user_id)

    def dump_memories(self) -> list[dict]:
        return self.system.dump_memories()

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        self.system.save_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        self.system.load_checkpoint(checkpoint_dir)

    @property
    def inner(self) -> BaseMemorySystem:
        return self.system

    @property
    def total_tokens(self) -> int:
        return self.system.total_tokens

    @property
    def n_entries(self) -> int:
        return self.system.n_entries