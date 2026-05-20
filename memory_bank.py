import json
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3  # rough fallback


# ========================
# Entry Schema
# ========================

@dataclass
class MemoryEntry:
    entry_id: str
    session_idx: int        # -1 for cold_start
    session_file: str       # source filename
    turn_idx: int           # turn index within session (0-based, per user-agent pair)
    domain_name: str
    content: str            # raw text or key fact/summary
    keywords: list[str]
    importance_score: float # 0~1
    timestamp: float        # time.time()
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop('embedding', None)
        return d

    @property
    def token_count(self) -> int:
        return count_tokens(self.content)


# ========================
# Embedding
# ========================

class EmbeddingModel:
    def __init__(self, provider: str = 'openai', model: str = 'text-embedding-3-small'):
        self.provider = provider
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.provider == 'openai':
                import openai, os
                self._client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            elif self.provider == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
                self._client = SentenceTransformer(self.model)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        client = self._get_client()
        if self.provider == 'openai':
            response = client.embeddings.create(input=texts, model=self.model)
            return np.array([e.embedding for e in response.data], dtype=np.float32)
        elif self.provider == 'sentence_transformers':
            return client.encode(texts, convert_to_numpy=True).astype(np.float32)
        raise ValueError(f"Unknown provider: {self.provider}")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (D,), b: (N, D) → (N,)"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    if norm_a == 0 or np.any(norm_b == 0):
        return np.zeros(len(b))
    return (b @ a) / (norm_b * norm_a)


# ========================
# MemoryBank
# ========================

class MemoryBank:
    def __init__(
        self,
        max_tokens: int = 8000,
        embedding_provider: str = 'openai',
        embedding_model: str = 'text-embedding-3-small',
    ):
        self.max_tokens = max_tokens
        self.embedder = EmbeddingModel(provider=embedding_provider, model=embedding_model)
        self.entries: list[MemoryEntry] = []

    # ────────────────────────────────
    # Write
    # ────────────────────────────────

    def add(self, entry: MemoryEntry) -> None:
        """Add entry to bank; auto-generate embedding if missing."""
        if entry.embedding is None:
            entry.embedding = self.embedder.embed_one(entry.content)
        self.entries.append(entry)

    def add_batch(self, entries: list[MemoryEntry]) -> None:
        """Batch-embed entries without embeddings, then add."""
        no_emb = [e for e in entries if e.embedding is None]
        if no_emb:
            texts = [e.content for e in no_emb]
            embeddings = self.embedder.embed(texts)
            for e, emb in zip(no_emb, embeddings):
                e.embedding = emb
        self.entries.extend(entries)

    # ────────────────────────────────
    # Retrieve
    # ────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Semantic retrieval; return top_k entries."""
        if not self.entries:
            return []
        query_emb = self.embedder.embed_one(query)
        emb_matrix = np.stack([e.embedding for e in self.entries])
        scores = cosine_similarity(query_emb, emb_matrix)
        top_k = min(top_k, len(self.entries))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.entries[i] for i in top_indices]

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        """Return (entry, score) pairs."""
        if not self.entries:
            return []
        query_emb = self.embedder.embed_one(query)
        emb_matrix = np.stack([e.embedding for e in self.entries])
        scores = cosine_similarity(query_emb, emb_matrix)
        top_k = min(top_k, len(self.entries))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.entries[i], float(scores[i])) for i in top_indices]

    # ────────────────────────────────
    # Delete
    # ────────────────────────────────

    def delete(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.entry_id != entry_id]
        return len(self.entries) < before

    def enforce_token_limit(self, strategy: str = 'oldest_first') -> list[str]:
        
        deleted = []
        while self.total_tokens > self.max_tokens and self.entries:
            if strategy == 'oldest_first':
                # delete oldest entry
                target = min(self.entries, key=lambda e: e.timestamp)
            elif strategy == 'importance_based':
                # delete oldest among lowest-importance entries
                sorted_entries = sorted(
                    self.entries,
                    key=lambda e: (e.importance_score, e.timestamp)
                )
                target = sorted_entries[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            self.delete(target.entry_id)
            deleted.append(target.entry_id)

        return deleted

    # ────────────────────────────────
    # Update / Consolidation
    # ────────────────────────────────

    def update_entry(self, entry_id: str, new_content: str, new_keywords: list[str] = None) -> bool:
        """Update existing entry content; recompute embedding."""
        for e in self.entries:
            if e.entry_id == entry_id:
                e.content = new_content
                if new_keywords is not None:
                    e.keywords = new_keywords
                e.embedding = self.embedder.embed_one(new_content)
                return True
        return False

    def merge_entries(self, entry_ids: list[str], merged_content: str,
                      merged_keywords: list[str], importance_score: float) -> MemoryEntry:
        """Merge multiple entries into one and delete originals."""
        sources = [e for e in self.entries if e.entry_id in entry_ids]
        if not sources:
            raise ValueError("No matching entries found")

        new_entry = MemoryEntry(
            entry_id=str(uuid.uuid4()),
            session_idx=sources[0].session_idx,
            session_file=sources[0].session_file,
            turn_idx=sources[0].turn_idx,
            domain_name=sources[0].domain_name,
            content=merged_content,
            keywords=merged_keywords,
            importance_score=importance_score,
            timestamp=time.time(),
        )
        new_entry.embedding = self.embedder.embed_one(merged_content)

        for eid in entry_ids:
            self.delete(eid)
        self.entries.append(new_entry)
        return new_entry

    # ────────────────────────────────
    # Lookup (for evaluation)
    # ────────────────────────────────

    def find_by_session_turn(self, session_file: str, turn_idx: int) -> Optional[MemoryEntry]:
        """Check whether a given session turn is stored (for evaluation)."""
        for e in self.entries:
            if e.session_file == session_file and e.turn_idx == turn_idx:
                return e
        return None

    def exists(self, session_file: str, turn_idx: int) -> bool:
        return self.find_by_session_turn(session_file, turn_idx) is not None

    # ────────────────────────────────
    # Stats
    # ────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return sum(e.token_count for e in self.entries)

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict:
        return {
            "n_entries": self.size,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0,
        }

    # ────────────────────────────────
    # Serialization
    # ────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
