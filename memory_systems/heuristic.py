"""
memory_systems/heuristic.py

Heuristic Memory System.
- 모든 turn 무조건 raw text 저장
- Token budget: BaseMemorySystem의 oldest-first 공통 로직 사용
- consolidation 없음
"""

import re
import time
import uuid as uuid_lib
from collections import Counter

import numpy as np

from memory_bank import MemoryBank, MemoryEntry, EmbeddingModel, cosine_similarity
from .base import BaseMemorySystem, MemoryChunk


def _simple_keywords(text: str, n: int = 5) -> list[str]:
    STOPWORDS = {
        'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'and', 'or', 'but',
        'not', 'it', 'this', 'that', 'you', 'your', 'we', 'our', 'they', 'user', 'agent',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(n)]


class HeuristicSystem(BaseMemorySystem):
    """
    모든 turn을 무조건 raw text로 저장.
    Token budget은 BaseMemorySystem의 공통 oldest-first 로직으로 처리.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        embedding_provider: str = 'openai',
        embedding_model: str = 'text-embedding-3-small',
    ):
        super().__init__(max_tokens=max_tokens)
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model
        self._embedder = EmbeddingModel(provider=embedding_provider, model=embedding_model)
        self._entries: dict[str, MemoryEntry] = {}  # entry_id → MemoryEntry
        self._reset_backend()

    def _reset_backend(self, user_id: str | None = None) -> None:
        self._entries = {}

    def _write_turn(
        self,
        session_file: str,
        turn_idx: int,
        session_idx: int,
        domain_name: str,
        user_content: str,
        agent_content: str,
    ) -> str | None:
        content = self.build_raw_text(user_content, agent_content)
        keywords = _simple_keywords(user_content, n=5)
        entry_id = str(uuid_lib.uuid4())
        embedding = self._embedder.embed_one(content)

        entry = MemoryEntry(
            entry_id=entry_id,
            session_idx=session_idx,
            session_file=session_file,
            turn_idx=turn_idx,
            domain_name=domain_name,
            content=content,
            keywords=keywords,
            importance_score=0.5,
            timestamp=time.time(),
            embedding=embedding,
        )
        self._entries[entry_id] = entry
        return entry_id

    def _delete_entry(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        if not self._entries:
            return []
        query_emb = self._embedder.embed_one(query)
        entries = list(self._entries.values())
        emb_matrix = np.stack([e.embedding for e in entries])
        scores = cosine_similarity(query_emb, emb_matrix)
        top_k = min(top_k, len(entries))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            MemoryChunk(
                content=entries[i].content,
                session_file=entries[i].session_file,
                turn_idx=entries[i].turn_idx,
                keywords=entries[i].keywords,
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        """
        각 entry의 content, keywords, metadata + embedding을 저장.
        embedding은 numpy npz 형식.
        """
        import json
        entries_meta = []
        embeddings = {}
        for entry_id, entry in self._entries.items():
            entries_meta.append({
                "entry_id": entry_id,
                "session_idx": entry.session_idx,
                "session_file": entry.session_file,
                "turn_idx": entry.turn_idx,
                "domain_name": entry.domain_name,
                "content": entry.content,
                "keywords": entry.keywords,
                "importance_score": entry.importance_score,
                "timestamp": entry.timestamp,
            })
            if entry.embedding is not None:
                embeddings[entry_id] = entry.embedding

        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump({"system": "HeuristicSystem", "entries": entries_meta}, f, ensure_ascii=False, indent=2)

        if embeddings:
            np.savez(str(ckpt_path / "embeddings.npz"), **embeddings)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        import json
        import time as time_lib

        backend_path = ckpt_path / "backend_state.json"
        emb_path = ckpt_path / "embeddings.npz"
        if not backend_path.exists():
            return

        with open(backend_path, encoding="utf-8") as f:
            state = json.load(f)

        # embeddings 로드
        emb_data = {}
        if emb_path.exists():
            npz = np.load(str(emb_path))
            emb_data = {k: npz[k] for k in npz.files}

        self._entries = {}
        for e in state.get("entries", []):
            entry = MemoryEntry(
                entry_id=e["entry_id"],
                session_idx=e["session_idx"],
                session_file=e["session_file"],
                turn_idx=e["turn_idx"],
                domain_name=e["domain_name"],
                content=e["content"],
                keywords=e["keywords"],
                importance_score=e["importance_score"],
                timestamp=e.get("timestamp", time_lib.time()),
                embedding=emb_data.get(e["entry_id"]),
            )
            self._entries[e["entry_id"]] = entry