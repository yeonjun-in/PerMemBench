"""
memory_systems/base.py

모든 memory system의 공통 추상 인터페이스.

Token Budget 관리:
  - 모든 시스템에 동일한 max_tokens 제한 적용 (공정한 비교)
  - write() 후 post_session()에서 _enforce_token_budget() 자동 호출
  - oldest-first 삭제 전략: 저장 순서 기준으로 초과분 제거
  - 각 시스템은 _write_turn() / _delete_entry() / _reset_backend() 구현 필요

Storage granularity:
  - turn  : write()로 매 turn 개별 처리 (기본)
  - session: write_session()으로 세션 전체를 한 번에 처리
             기본 구현은 write()에 위임; Mem0 등은 override하여
             전체 conversation을 한 번의 add()로 처리

Write check support:
  - get_write_evidence(session, written_keys) → str
    평가 파이프라인이 Write check에 사용할 텍스트를 반환.
    기본: dialogue의 written turn 원본 텍스트 (heuristic용)
    Mem0 override: Mem0가 실제로 추출·저장한 fact 텍스트 반환
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3

# backward-compat alias
_count_tokens = count_tokens


@dataclass
class MemoryChunk:
    """retrieve() 반환 단위."""
    content: str
    session_file: str = ""
    turn_idx: int = -1
    keywords: list = field(default_factory=list)
    score: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class _EntryRecord:
    """Token budget 관리용 내부 레코드."""
    entry_id: str
    session_file: str
    turn_idx: int
    token_count: int
    insert_order: int
    content: str = ""        # snapshot/error analysis용 (첫 500자만 저장)


class BaseMemorySystem(ABC):

    def __init__(self, max_tokens: int = 8000, max_entries: int | None = None):
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self._entry_records: deque[_EntryRecord] = deque()
        self._total_tokens: int = 0
        self._insert_counter: int = 0
        self._written_turns: set[tuple[str, int]] = set()

    # ─────────────────────────────────────
    # 추상 메서드 (각 시스템 구현 필요)
    # ─────────────────────────────────────

    @abstractmethod
    def _write_turn(
        self,
        session_file: str,
        turn_idx: int,
        session_idx: int,
        domain_name: str,
        user_content: str,
        agent_content: str,
    ) -> str | None:
        """단일 turn을 실제 memory system에 저장. entry_id 반환, 저장 안 했으면 None."""
        pass

    @abstractmethod
    def _delete_entry(self, entry_id: str) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        pass

    @abstractmethod
    def _reset_backend(self, user_id: str | None = None) -> None:
        pass

    # ─────────────────────────────────────
    # 공통 구현 (override 가능)
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """Turn 단위 저장 (storage_unit='turn' 기본 경로)."""
        dialogue = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')
        session_idx = session.get('session_idx', -1)
        domain_name = session.get('domain_name', '')

        turns = self.extract_turns(dialogue)
        written = []

        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue

            entry_id = self._write_turn(
                session_file=session_file,
                turn_idx=turn_idx,
                session_idx=session_idx,
                domain_name=domain_name,
                user_content=user_content,
                agent_content=agent_content,
            )

            if entry_id is not None:
                content = self.build_raw_text(user_content, agent_content)
                token_count = count_tokens(content)
                self._register_entry(entry_id, session_file, turn_idx, token_count, content)
                self._written_turns.add((session_file, turn_idx))
                written.append((session_file, turn_idx))

        return written

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        Session 단위 저장 (storage_unit='session' 경로).

        기본 구현은 turn 단위 write()에 위임.
        Mem0처럼 전체 conversation context를 한 번에 처리하는 시스템은
        이 메서드를 override하여 memory.add(all_messages)를 한 번만 호출.

        반환값 규칙:
          - turn 단위: [(session_file, turn_idx), ...]
          - session 단위 override: [(session_file, -1)] 을 sentinel로 사용
            (turn_idx=-1 은 세션 전체를 하나의 단위로 저장했음을 의미)
        """
        return self.write(session)

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """
        Write check에 사용할 텍스트를 반환.

        기본 구현 (heuristic 등 raw text 저장 시스템):
            dialogue에서 written turn의 원본 텍스트를 직접 추출.
            DB 조회 없음 — eviction/변환 노이즈 없는 순수 입력 텍스트.

        Mem0 등 fact 추출 시스템은 이 메서드를 override하여
        실제로 메모리에 저장된 fact 텍스트를 반환해야 함.
        그래야 "raw input에 fact가 있었는데 Mem0가 저장 안 한" 케이스와
        "Mem0가 fact를 추출해서 저장한" 케이스를 올바르게 구분 가능.

        Args:
            session: 현재 세션 dict (dialogue 포함)
            written_keys: write() 또는 write_session()이 반환한
                          [(session_file, turn_idx), ...] 목록.
                          session 단위 sentinel (turn_idx=-1)도 처리.

        Returns:
            Write check용 텍스트 (빈 문자열이면 write_score=0)
        """
        if not written_keys:
            return ""

        dialogue = session.get('dialogue', [])

        # sentinel (-1) = 세션 전체가 하나의 단위로 저장됨 → 전체 dialogue 반환
        if any(ti == -1 for (_, ti) in written_keys):
            parts = []
            for turn_idx, user_content, agent_content in self.extract_turns(dialogue):
                parts.append(f"User: {user_content}")
                if agent_content:
                    parts.append(f"Agent: {agent_content}")
            return "\n\n".join(parts)

        # 일반 turn 단위: written turn만 추출
        written_set = {ti for (_, ti) in written_keys}
        parts = []
        for turn_idx, user_content, agent_content in self.extract_turns(dialogue):
            if turn_idx in written_set:
                parts.append(f"User: {user_content}")
                if agent_content:
                    parts.append(f"Agent: {agent_content}")
        return "\n\n".join(parts)

    def post_session(self) -> dict:
        """세션 종료 후 budget 적용. 초과분 oldest-first 삭제."""
        deleted_for_token_limit = self._enforce_token_budget()
        deleted_for_entry_limit = self._enforce_entry_budget()
        return {
            "deleted_for_token_limit": deleted_for_token_limit,
            "deleted_for_entry_limit": deleted_for_entry_limit,
        }

    def reset(self, user_id: str | None = None) -> None:
        self._entry_records = deque()
        self._total_tokens = 0
        self._insert_counter = 0
        self._written_turns = set()
        self._reset_backend(user_id=user_id)

    def get_written_turns(self) -> set[tuple[str, int]]:
        return self._written_turns

    # ─────────────────────────────────────
    # Token Budget 내부 관리
    # ─────────────────────────────────────

    def _register_entry(
        self,
        entry_id: str,
        session_file: str,
        turn_idx: int,
        token_count: int,
        content: str = "",
    ) -> None:
        record = _EntryRecord(
            entry_id=entry_id,
            session_file=session_file,
            turn_idx=turn_idx,
            token_count=token_count,
            insert_order=self._insert_counter,
            content=content[:500],
        )
        self._entry_records.append(record)
        self._total_tokens += token_count
        self._insert_counter += 1

    def dump_memories(self) -> list[dict]:
        """
        현재 bank에 존재하는 모든 entry 목록 반환 (snapshot/error analysis용).
        Mem0 등은 override하여 실제 저장된 fact 텍스트를 반환.
        """
        return [
            {
                "session_file": r.session_file,
                "turn_idx": r.turn_idx,
                "token_count": r.token_count,
                "insert_order": r.insert_order,
                "content": r.content,
            }
            for r in self._entry_records
        ]

    def _enforce_token_budget(self) -> list[str]:
        deleted_ids = []
        while self._total_tokens > self.max_tokens and self._entry_records:
            oldest = self._entry_records.popleft()
            try:
                self._delete_entry(oldest.entry_id)
            except Exception as e:
                print(f"  [BUDGET] delete failed (id={oldest.entry_id}): {e}")
            self._total_tokens = max(0, self._total_tokens - oldest.token_count)
            self._written_turns.discard((oldest.session_file, oldest.turn_idx))
            deleted_ids.append(oldest.entry_id)
        return deleted_ids

    def _enforce_entry_budget(self) -> list[str]:
        if self.max_entries is None:
            return []

        deleted_ids = []
        while len(self._entry_records) > self.max_entries and self._entry_records:
            oldest = self._entry_records.popleft()
            try:
                self._delete_entry(oldest.entry_id)
            except Exception as e:
                print(f"  [BUDGET] delete failed (id={oldest.entry_id}): {e}")
            self._total_tokens = max(0, self._total_tokens - oldest.token_count)
            self._written_turns.discard((oldest.session_file, oldest.turn_idx))
            deleted_ids.append(oldest.entry_id)
        return deleted_ids

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def n_entries(self) -> int:
        return len(self._entry_records)

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        import json
        from pathlib import Path

        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)

        budget_state = {
            "max_tokens": self.max_tokens,
            "max_entries": self.max_entries,
            "total_tokens": self._total_tokens,
            "insert_counter": self._insert_counter,
            "written_turns": [[sf, ti] for sf, ti in self._written_turns],
            "entry_records": [
                {
                    "entry_id": r.entry_id,
                    "session_file": r.session_file,
                    "turn_idx": r.turn_idx,
                    "token_count": r.token_count,
                    "insert_order": r.insert_order,
                    "content": r.content,
                }
                for r in self._entry_records
            ],
        }
        with open(ckpt / "budget_state.json", "w", encoding="utf-8") as f:
            json.dump(budget_state, f, ensure_ascii=False, indent=2)

        self._save_backend_checkpoint(ckpt)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        import json
        from pathlib import Path
        from collections import deque

        ckpt = Path(checkpoint_dir)
        budget_path = ckpt / "budget_state.json"
        if not budget_path.exists():
            raise FileNotFoundError(f"budget_state.json not found in {checkpoint_dir}")

        with open(budget_path, encoding="utf-8") as f:
            state = json.load(f)

        self.max_tokens = state.get("max_tokens", self.max_tokens)
        self.max_entries = state.get("max_entries", self.max_entries)
        self._total_tokens = state["total_tokens"]
        self._insert_counter = state["insert_counter"]
        self._written_turns = {(sf, ti) for sf, ti in state["written_turns"]}
        self._entry_records = deque(
            _EntryRecord(
                entry_id=r["entry_id"],
                session_file=r["session_file"],
                turn_idx=r["turn_idx"],
                token_count=r["token_count"],
                insert_order=r["insert_order"],
                content=r.get("content", ""),
            )
            for r in state["entry_records"]
        )
        self._load_backend_checkpoint(ckpt)

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        import json
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump({"system": self.__class__.__name__}, f)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        pass

    # ─────────────────────────────────────
    # 공통 유틸
    # ─────────────────────────────────────

    @staticmethod
    def extract_turns(dialogue: list[dict]) -> list[tuple[int, str, str]]:
        turns = []
        pair_idx = 0
        i = 0
        while i < len(dialogue):
            if dialogue[i].get('role') == 'user':
                user_content = dialogue[i].get('content', '')
                agent_content = ''
                if i + 1 < len(dialogue) and dialogue[i + 1].get('role') == 'assistant':
                    agent_content = dialogue[i + 1].get('content', '')
                    i += 2
                else:
                    i += 1
                turns.append((pair_idx, user_content, agent_content))
                pair_idx += 1
            else:
                i += 1
        return turns

    @staticmethod
    def build_raw_text(user_content: str, agent_content: str) -> str:
        return f"User: {user_content}\nAgent: {agent_content}"
