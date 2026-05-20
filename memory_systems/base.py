"""
memory_systems/base.py

Common abstract interface for all memory systems.

Token budget:
  - Same max_tokens limit on every system (fair comparison)
  - _enforce_token_budget() runs automatically in post_session() after write()
  - oldest-first eviction by insertion order
  - Each system implements _write_turn() / _delete_entry() / _reset_backend()

Storage granularity:
  - turn  : write() processes each turn individually (default)
  - session: write_session() processes the whole session at once
             Default delegates to write(); Mem0 etc. override to call
             memory.add() once on the full conversation

Write check support:
  - get_write_evidence(session, written_keys) → str
    Text used by the eval pipeline for write checks.
    Default: raw text of written turns from dialogue (heuristic)
    Mem0 override: facts actually extracted and stored by Mem0
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
    """Unit returned by retrieve()."""
    content: str
    session_file: str = ""
    turn_idx: int = -1
    keywords: list = field(default_factory=list)
    score: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class _EntryRecord:
    """Internal record for token budget tracking."""
    entry_id: str
    session_file: str
    turn_idx: int
    token_count: int
    insert_order: int
    content: str = ""        # for snapshot/error analysis (first 500 chars only)


class BaseMemorySystem(ABC):

    def __init__(self, max_tokens: int = 8000, max_entries: int | None = None):
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self._entry_records: deque[_EntryRecord] = deque()
        self._total_tokens: int = 0
        self._insert_counter: int = 0
        self._written_turns: set[tuple[str, int]] = set()

    # ─────────────────────────────────────
    # Abstract methods (implemented by each system)
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
        """Store one turn in the backend. Returns entry_id, or None if not stored."""
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
    # Shared implementation (overridable)
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """Per-turn storage (default path when storage_unit='turn')."""
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
        Per-session storage (path when storage_unit='session').

        Default delegates to per-turn write().
        Systems that handle full conversation context at once (e.g. Mem0)
        override this to call memory.add(all_messages) once.

        Return value rules:
          - per-turn: [(session_file, turn_idx), ...]
          - session override: use [(session_file, -1)] as sentinel
            (turn_idx=-1 means the whole session was stored as one unit)
        """
        return self.write(session)

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """
        Return text for write checks.

        Default (heuristic and other raw-text systems):
            Extract original text of written turns from dialogue.
            No DB lookup — pure input text without eviction/transform noise.

        Fact-extraction systems (e.g. Mem0) must override this to return
        facts actually stored in memory, so we can distinguish
        "fact in raw input but not stored" vs "fact extracted and stored".

        Args:
            session: current session dict (includes dialogue)
            written_keys: [(session_file, turn_idx), ...] from write() or write_session().
                          Also handles session sentinel (turn_idx=-1).

        Returns:
            Text for write check (empty string → write_score=0)
        """
        if not written_keys:
            return ""

        dialogue = session.get('dialogue', [])

        # sentinel (-1) = whole session stored as one unit → return full dialogue
        if any(ti == -1 for (_, ti) in written_keys):
            parts = []
            for turn_idx, user_content, agent_content in self.extract_turns(dialogue):
                parts.append(f"User: {user_content}")
                if agent_content:
                    parts.append(f"Agent: {agent_content}")
            return "\n\n".join(parts)

        # per-turn: extract only written turns
        written_set = {ti for (_, ti) in written_keys}
        parts = []
        for turn_idx, user_content, agent_content in self.extract_turns(dialogue):
            if turn_idx in written_set:
                parts.append(f"User: {user_content}")
                if agent_content:
                    parts.append(f"Agent: {agent_content}")
        return "\n\n".join(parts)

    def post_session(self) -> dict:
        """Apply budget after session; evict excess oldest-first."""
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
    # Token budget internals
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
        Return all entries currently in the bank (snapshot/error analysis).
        Mem0 etc. override to return actually stored fact text.
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
    # Shared utilities
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
