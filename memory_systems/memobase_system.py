"""
memory_systems/memobase_system.py

Memobase wrapper — User Profile-Based Long-Term Memory.
https://github.com/memodb-io/memobase

설치:
    pip install memobase

로컬 서버:
    cd memobase/src/server && docker compose up -d
    → project_url="http://localhost:8019", api_key="secret"

클라우드:
    https://www.memobase.io 에서 API 키 발급
    → project_url="https://api.memobase.dev", api_key="sk-proj-xxx"

핵심 동작:
    - eval UUID를 Memobase user_id로 직접 사용 (UUID별 격리)
    - 대화를 ChatBlob으로 insert() → flush()로 profile 추출
    - flush 후 _profile_cache에 실제 profile 캐싱
    - dump_memories() / total_tokens / n_entries 모두 캐시 기반 → 항상 실제 상태 반영

Token Budget:
    - _entry_records: flush 후 profile entries를 등록 (eviction용)
    - _get_profile_id(): 다양한 SDK 버전의 ID 필드명 대응
    - _delete_entry(profile_id): Memobase API로 실제 삭제 시도

User ID:
    - reset(user_id=uuid) 호출 시 해당 UUID를 Memobase user_id로 사용
"""

from __future__ import annotations

import re
from collections import deque

from .base import BaseMemorySystem, MemoryChunk, _EntryRecord, count_tokens


class MemobaseSystem(BaseMemorySystem):
    """
    Memobase 기반 memory system.
    install: pip install memobase
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        project_url: str = "http://localhost:8019",
        api_key: str = "secret",
        user_id_prefix: str = "umem",
    ):
        self.project_url = project_url
        self.api_key = api_key
        self.user_id_prefix = user_id_prefix

        self._client = None
        self._user = None
        self._uid: str = ""
        self._session_facts: list[str] = []  # 이번 세션 write_evidence용
        self._profile_cache: list = []        # 마지막 flush 후 profile 캐시

        # super().__init__()이 _reset_backend()를 호출하므로
        # client 먼저 초기화 후 호출
        self._init_client()
        super().__init__(max_tokens=max_tokens)

    def _init_client(self) -> None:
        from memobase import MemoBaseClient
        self._client = MemoBaseClient(
            project_url=self.project_url,
            api_key=self.api_key,
        )

    def _reset_backend(self, user_id: str | None = None) -> None:
        """
        user_id(eval UUID)를 Memobase user_id로 사용.
        - 이미 존재하면 재사용
        - 없으면 새로 생성
        user_id가 None이면 임시 ID 사용.
        """
        if self._client is None:
            self._init_client()

        if user_id:
            safe_id = re.sub(r'[^a-zA-Z0-9\-]', '-', user_id)
            self._uid = f"{self.user_id_prefix}-{safe_id}"
        else:
            import uuid as uuid_lib
            self._uid = f"{self.user_id_prefix}-{uuid_lib.uuid4().hex[:8]}"

        # 기존 user 재사용 or 새로 생성
        try:
            self._user = self._client.get_user(self._uid)
            print(f"  [MEMOBASE] 기존 user 재사용: {self._uid}")
        except Exception:
            try:
                self._uid = self._client.add_user(
                    {"eval_id": self._uid},
                    user_id=self._uid,
                )
                self._user = self._client.get_user(self._uid)
                print(f"  [MEMOBASE] 새 user 생성: {self._uid}")
            except TypeError:
                self._uid = self._client.add_user({"eval_id": self._uid})
                self._user = self._client.get_user(self._uid)
                print(f"  [MEMOBASE] 새 user 생성 (auto-id): {self._uid}")

        self._session_facts = []
        self._profile_cache = []

    # ─────────────────────────────────────
    # 추상 메서드 (write() 전체를 override)
    # ─────────────────────────────────────

    def _write_turn(self, session_file, turn_idx, session_idx, domain_name,
                    user_content, agent_content) -> str | None:
        raise NotImplementedError("MemobaseSystem uses overridden write() directly")

    def _delete_entry(self, entry_id: str) -> None:
        """Memobase profile entry 실제 삭제."""
        try:
            self._user.delete_profile(entry_id)
            return
        except AttributeError:
            pass
        except Exception as e:
            print(f"  [MEMOBASE] delete_profile failed (id={entry_id}): {e}")
            return

        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.delete(
                f"{self.project_url}/api/v1/users/{self._uid}/profiles/{entry_id}",
                headers=headers,
                timeout=10,
            )
            if resp.status_code not in (200, 204):
                print(f"  [MEMOBASE] delete HTTP {resp.status_code} (id={entry_id})")
        except Exception as e:
            print(f"  [MEMOBASE] delete HTTP failed (id={entry_id}): {e}")

    # ─────────────────────────────────────
    # write() / write_session() override
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        """
        Turn 단위 저장.
        각 turn을 개별 ChatBlob으로 insert → 세션 끝에 한 번 flush.
        """
        from memobase import ChatBlob

        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        written = []
        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue
            try:
                self._user.insert(ChatBlob(messages=[
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": agent_content},
                ]))
                self._written_turns.add((session_file, turn_idx))
                written.append((session_file, turn_idx))
            except Exception as e:
                print(f"  [MEMOBASE] insert failed (turn={turn_idx}): {e}")

        self._flush_and_sync(session_file=session_file, sentinel_turn=-99)
        return written

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        Session 단위 저장.
        전체 turn을 하나의 ChatBlob으로 묶어 insert → flush.
        """
        from memobase import ChatBlob

        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")

        turns = self.extract_turns(dialogue)
        if not turns:
            return []

        messages = []
        for _, user_content, agent_content in turns:
            messages.append({"role": "user",      "content": user_content})
            if agent_content:
                messages.append({"role": "assistant", "content": agent_content})

        try:
            self._user.insert(ChatBlob(messages=messages))
            self._written_turns.add((session_file, -1))
        except Exception as e:
            print(f"  [MEMOBASE] insert failed (session): {e}")
            return []

        self._flush_and_sync(session_file=session_file, sentinel_turn=-1)
        return [(session_file, -1)]

    def _flush_and_sync(self, session_file: str, sentinel_turn: int) -> None:
        """
        1. Memobase buffer flush
        2. flush 전후 profile diff → session_facts 추출
        3. _profile_cache 업데이트
        4. _entry_records를 실제 profile entries 기준으로 동기화 (eviction용)
        """
        # flush 전 profile ID 스냅샷
        before_ids: set[str] = set()
        try:
            before_profiles = self._user.profile()
            before_ids = {self._get_profile_id(p) for p in before_profiles} - {""}
        except Exception:
            pass

        # flush
        try:
            self._user.flush(sync=True)
        except Exception as e:
            print(f"  [MEMOBASE] flush failed: {e}")
            self._session_facts = []
            return

        # flush 후 profile 조회
        try:
            after_profiles = self._user.profile()
        except Exception as e:
            print(f"  [MEMOBASE] profile fetch failed: {e}")
            self._session_facts = []
            return

        # ── _profile_cache 업데이트 ────────────────────────────────────────
        self._profile_cache = list(after_profiles)

        # ── session write evidence: 이번 flush에서 새로 생긴 profile entries ──
        new_profiles = [
            p for p in after_profiles
            if self._get_profile_id(p) not in before_ids
        ]
        self._session_facts = [self._profile_to_text(p) for p in new_profiles]
        if not self._session_facts:
            self._session_facts = [self._profile_to_text(p) for p in after_profiles]

        # ── _entry_records를 실제 profile entries와 동기화 (eviction용) ───────
        tracked_ids = {r.entry_id for r in self._entry_records}
        after_ids   = {self._get_profile_id(p) for p in after_profiles} - {""}

        # Memobase에서 사라진 entries 제거
        self._entry_records = deque(
            r for r in self._entry_records if r.entry_id in after_ids
        )
        self._total_tokens = sum(r.token_count for r in self._entry_records)

        # 새로 생긴 profile entries 등록
        for p in after_profiles:
            pid = self._get_profile_id(p)
            if not pid or pid in tracked_ids:
                continue
            text = self._profile_to_text(p)
            tc   = count_tokens(text)
            record = _EntryRecord(
                entry_id=pid,
                session_file=session_file,
                turn_idx=sentinel_turn,
                token_count=tc,
                insert_order=self._insert_counter,
                content=text,
            )
            self._entry_records.append(record)
            self._total_tokens += tc
            self._insert_counter += 1

        print(f"  [MEMOBASE] flushed"
              f" | profiles={len(after_profiles)}"
              f" | new={len(new_profiles)}"
              f" | total_tokens={self._total_tokens}")

    # ─────────────────────────────────────
    # ID 추출 / 텍스트 변환 유틸
    # ─────────────────────────────────────

    @staticmethod
    def _get_profile_id(p) -> str:
        """
        profile 객체에서 ID 추출.
        Memobase SDK 버전에 따라 필드명이 다를 수 있어 여러 필드명 시도.
        """
        for attr in ('id', 'profile_id', 'uid', 'key', 'memory_id'):
            val = getattr(p, attr, None)
            if val is not None and str(val).strip():
                return str(val)
        if isinstance(p, dict):
            for key in ('id', 'profile_id', 'uid', 'key', 'memory_id'):
                if p.get(key):
                    return str(p[key])
        return ""

    @staticmethod
    def _profile_to_text(p) -> str:
        topic     = getattr(p, "topic",     "")
        sub_topic = getattr(p, "sub_topic", "")
        content   = getattr(p, "content",   str(p))
        if topic and sub_topic:
            return f"{topic}/{sub_topic}: {content}"
        elif topic:
            return f"{topic}: {content}"
        return content

    # ─────────────────────────────────────
    # Write check override
    # ─────────────────────────────────────

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """이번 세션 flush에서 새로 생긴 profile entry 텍스트 반환."""
        if not written_keys:
            return ""
        return "\n\n".join(self._session_facts)

    # ─────────────────────────────────────
    # post_session override
    # ─────────────────────────────────────

    def post_session(self) -> dict:
        """
        flush는 write()에서 이미 완료.
        token budget 초과 시 oldest profile entry부터 실제 삭제.
        """
        deleted = self._enforce_token_budget()
        self._session_facts = []
        return {"deleted_for_token_limit": deleted}

    # ─────────────────────────────────────
    # total_tokens / n_entries override
    # → _profile_cache 기반으로 항상 실제 상태 반영
    # ─────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        """실제 profile entries 기반 token count."""
        if self._profile_cache:
            return sum(
                count_tokens(self._profile_to_text(p))
                for p in self._profile_cache
            )
        return self._total_tokens  # fallback

    @property
    def n_entries(self) -> int:
        """실제 profile entries 개수."""
        if self._profile_cache:
            return len(self._profile_cache)
        return len(self._entry_records)  # fallback

    # ─────────────────────────────────────
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        """
        Memobase profile + event 기반 retrieval.
        context API: profile + event를 모두 포함한 컨텍스트 반환.
        fallback: profile + event_gist 검색 결과 합산.
        """
        # ── 1순위: context API (profile + event 통합) ──────────────────
        try:
            context = self._user.context(
                query=query,
                max_token_size=top_k * 300,
            )
            if context:
                return [MemoryChunk(
                    content=str(context),
                    session_file="",
                    turn_idx=-1,
                    score=1.0,
                )]
        except Exception:
            pass

        chunks = []

        # ── 2순위 fallback: profile ────────────────────────────────────
        profiles = self._profile_cache
        if not profiles:
            try:
                profiles = self._user.profile()
            except Exception as e:
                print(f"  [MEMOBASE] profile fetch failed: {e}")
        for p in profiles[:top_k]:
            content = self._profile_to_text(p)
            if content:
                chunks.append(MemoryChunk(
                    content=content,
                    session_file="",
                    turn_idx=-1,
                    score=1.0,
                ))

        # ── 2순위 fallback: event_gist 검색 ───────────────────────────
        try:
            event_gists = self._user.search_event_gist(query)
            for eg in (event_gists or [])[:top_k]:
                content = getattr(eg, 'content', None) or str(eg)
                if content:
                    chunks.append(MemoryChunk(
                        content=content,
                        session_file="",
                        turn_idx=-1,
                        score=getattr(eg, 'score', 0.9),
                    ))
        except Exception:
            # event_gist 미지원 시 search_event fallback
            try:
                events = self._user.search_event(query)
                for ev in (events or [])[:top_k]:
                    content = getattr(ev, 'content', None) or str(ev)
                    if content:
                        chunks.append(MemoryChunk(
                            content=content,
                            session_file="",
                            turn_idx=-1,
                            score=0.8,
                        ))
            except Exception:
                pass

        return chunks[:top_k]

    # ─────────────────────────────────────
    # dump_memories override
    # → _profile_cache 기반으로 실제 Memobase 데이터 반환
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        """
        실제 Memobase profile + event 반환.
        - profiles: _profile_cache 기반 (없으면 직접 API 호출)
        - events: 최근 events (search_event 또는 get_events)
        """
        result = []

        # ── Profile ───────────────────────────────────────────────────
        profiles = self._profile_cache
        if not profiles:
            try:
                profiles = self._user.profile()
                self._profile_cache = list(profiles)
            except Exception as e:
                print(f"  [MEMOBASE] dump_memories: profile fetch failed: {e}")

        for p in profiles:
            text = self._profile_to_text(p)
            result.append({
                "type":        "profile",
                "memory_id":   self._get_profile_id(p),
                "topic":       getattr(p, "topic",     ""),
                "sub_topic":   getattr(p, "sub_topic", ""),
                "content":     getattr(p, "content",   ""),
                "memory_text": text,
                "token_count": count_tokens(text),
            })

        # ── Events ────────────────────────────────────────────────────
        try:
            events = self._user.get_events()
            for ev in (events or []):
                content = getattr(ev, 'content', None) or str(ev)
                result.append({
                    "type":        "event",
                    "memory_id":   str(getattr(ev, 'id', '')),
                    "created_at":  str(getattr(ev, 'created_at', '')),
                    "tags":        getattr(ev, 'tags', []),
                    "memory_text": content,
                    "token_count": count_tokens(content),
                })
        except AttributeError:
            # get_events 없으면 skip (events는 선택적 기능)
            pass
        except Exception as e:
            print(f"  [MEMOBASE] dump_memories: event fetch failed: {e}")

        return result

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        import json
        state = {
            "system":      "MemobaseSystem",
            "uid":         self._uid,
            "project_url": self.project_url,
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
        self._uid  = state["uid"]
        self._user = self._client.get_user(self._uid)
