"""
memory_systems/memoryos_system.py

BAI-LAB/MemoryOS wrapper.
Paper: "Memory OS of AI Agent" (EMNLP 2025 Oral)
GitHub: https://github.com/BAI-LAB/MemoryOS

설치:
    pip install memoryos-pro
    # 또는
    cd MemoryOS/memoryos-pypi && pip install -r requirements.txt

아키텍처:
    - 단기(Short-term) / 중기(Mid-term) / 장기(Long-term) 3계층
    - add_memory(user_input, agent_response) 로 turn 단위 저장
    - retriever.retrieve_context(...) 또는 retrieve_memory(...) 로 검색
    - data_storage_path로 user별 파일 격리

핵심 파라미터:
    openai_base_url     : vLLM 서버 URL (예: http://localhost:8000/v1)
    llm_model           : LLM 모델명 (예: Qwen/Qwen2.5-7B-Instruct)
    embedding_model_name: 로컬 임베딩 모델 (예: BAAI/bge-m3)
    mid_term_capacity   : 중기 메모리 최대 항목 수 (기본 1000)
    short_term_capacity : 단기 메모리 최대 항목 수 (기본 2)
"""

import os
import json
import re
import inspect
import uuid as uuid_lib
from pathlib import Path
from .base import BaseMemorySystem, MemoryChunk, count_tokens


class MemoryOSSystem(BaseMemorySystem):
    """
    BAI-LAB/MemoryOS 기반 memory system.

    Args:
        max_tokens            : token budget (base class용)
        openai_api_key        : OpenAI 호환 API 키
        openai_base_url       : API base URL (vLLM: http://localhost:8000/v1)
        llm_model             : LLM 모델명
        embedding_model_name  : 임베딩 모델 (로컬 지원: BAAI/bge-m3)
        data_storage_root     : 데이터 저장 root 경로 (user별 하위 디렉터리 생성)
        mid_term_capacity     : 중기 메모리 최대 항목 수
        mid_term_heat_threshold     : 중기 → 장기 승격 heat threshold
        mid_term_similarity_threshold: 중기 메모리 유사도 threshold
        short_term_capacity   : 단기 메모리 최대 항목 수
        user_id_prefix        : 평가용 user_id prefix
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model_name: str = "BAAI/bge-m3",
        data_storage_root: str = "./.memoryos_data",
        mid_term_capacity: int = 1000,
        mid_term_heat_threshold: float = 13.0,
        mid_term_similarity_threshold: float = 0.7,
        short_term_capacity: int = 2,
        user_id_prefix: str = "u_mem_eval",
    ):
        super().__init__(max_tokens=max_tokens)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.openai_base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name
        self.data_storage_root = data_storage_root
        self.mid_term_capacity = mid_term_capacity
        self.mid_term_heat_threshold = mid_term_heat_threshold
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.short_term_capacity = short_term_capacity
        self.user_id_prefix = user_id_prefix

        self._mos = None
        self._user_id: str = ""
        self._data_path: str = ""
        self._session_memories: list[str] = []

    # ─────────────────────────────────────
    # 초기화
    # ─────────────────────────────────────

    def _reset_backend(self, user_id: str | None = None) -> None:
        """새 UUID 기반 user_id로 Memoryos 인스턴스 초기화."""
        try:
            from memoryos import Memoryos
        except ImportError:
            raise ImportError(
                "MemoryOS SDK를 찾을 수 없습니다.\n"
                "설치: pip install memoryos-pro\n"
                "또는: cd MemoryOS/memoryos-pypi && pip install -r requirements.txt"
            )

        self._user_id = f"{self.user_id_prefix}_{uuid_lib.uuid4().hex[:8]}"

        # user별 독립 저장 경로
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]", "_", self._user_id)
        self._data_path = str(Path(self.data_storage_root) / safe_uid)
        Path(self._data_path).mkdir(parents=True, exist_ok=True)

        init_kwargs = dict(
            user_id=self._user_id,
            assistant_id="u_mem_assistant",
            openai_api_key=self.openai_api_key,
            data_storage_path=self._data_path,
            llm_model=self.llm_model,
            embedding_model_name=self.embedding_model_name,
            mid_term_capacity=self.mid_term_capacity,
            mid_term_heat_threshold=self.mid_term_heat_threshold,
            mid_term_similarity_threshold=self.mid_term_similarity_threshold,
            short_term_capacity=self.short_term_capacity,
        )
        if self.openai_base_url:
            init_kwargs["openai_base_url"] = self.openai_base_url

        # memoryos 패키지 버전에 따라 __init__ 인자셋이 다르므로
        # 현재 설치된 버전이 지원하는 키만 전달한다.
        supported_params = set(inspect.signature(Memoryos.__init__).parameters.keys())
        supported_params.discard("self")
        filtered_kwargs = {
            k: v for k, v in init_kwargs.items()
            if k in supported_params
        }
        self._mos = Memoryos(**filtered_kwargs)
        self._session_memories = []

    # ─────────────────────────────────────
    # write (turn 단위)
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
        """단일 turn을 MemoryOS에 저장."""
        try:
            self._mos.add_memory(
                user_input=user_content,
                agent_response=agent_content,
            )
            content = self.build_raw_text(user_content, agent_content)
            self._session_memories.append(content)
            return f"{session_file}__t{turn_idx}"
        except Exception as e:
            print(f"  [MemoryOS] add_memory failed (turn={turn_idx}): {e}")
            return None

    # ─────────────────────────────────────
    # write_session (session 단위 override)
    # ─────────────────────────────────────

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        """
        세션 전체를 turn별로 순차 add_memory() 호출.
        MemoryOS SDK가 QA쌍 단위 API만 지원하므로 내부적으로는 turn 루프를 돌지만,
        반환 키는 session 단위 sentinel (session_file, -1) 을 사용한다.
        """
        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")
        turns        = self.extract_turns(dialogue)
        if not turns:
            return []

        self._session_memories = []
        n_written = 0

        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue
            try:
                self._mos.add_memory(
                    user_input=user_content,
                    agent_response=agent_content,
                )
                self._session_memories.append(
                    self.build_raw_text(user_content, agent_content)
                )
                n_written += 1
            except Exception as e:
                print(f"  [MemoryOS] add_memory failed (turn={turn_idx}): {e}")

        if n_written == 0:
            return []

        # 세션 전체를 하나의 entry로 등록 (sentinel: turn_idx=-1)
        all_content = "\n\n".join(self._session_memories)
        entry_id = f"{session_file}__session"
        self._register_entry(
            entry_id, session_file, -1,
            count_tokens(all_content), all_content[:500],
        )
        self._written_turns.add((session_file, -1))
        return [(session_file, -1)]

    # ─────────────────────────────────────
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        """MemoryOS 버전별 retrieval API 차이를 흡수하여 검색."""
        if self._mos is None:
            return []
        try:
            # 구버전/커스텀 포크: retrieve_memory(query)
            if hasattr(self._mos, "retrieve_memory"):
                result = self._mos.retrieve_memory(query)
            # 현재 설치된 memoryos-pro: self._mos.retriever.retrieve_context(...)
            elif hasattr(self._mos, "retriever") and hasattr(self._mos.retriever, "retrieve_context"):
                mos_user_id = getattr(self._mos, "user_id", self._user_id)
                # ctx = self._mos.retriever.retrieve_context(
                #     user_query=query,
                #     user_id=mos_user_id,
                # )
                ctx = self._mos.retriever.retrieve_context_seq(
                    user_query=query,
                    user_id=mos_user_id,
                )
                # if ctx['retrieved_user_knowledge'] or ctx['retrieved_assistant_knowledge']:
                #     import pdb; pdb.set_trace()
                result = ctx.get("retrieved_pages", []) if isinstance(ctx, dict) else ctx
            else:
                print("  [MemoryOS] No supported retrieval API found on SDK instance.")
                return []
        except Exception as e:
            print(f"  [MemoryOS] retrieve failed: {e}")
            return []

        return self._parse_retrieve_result(result, top_k)

    def _parse_retrieve_result(self, result, top_k: int) -> list[MemoryChunk]:
        """retrieve_memory() 반환값 파싱 (str / list / dict 모두 처리)."""
        chunks = []

        def _qa_raw_text(item: dict) -> str:
            """
            retrieval 결과 dict에서 user/assistant 원문 QA를 우선 추출.
            page_data 안에 중첩된 형태도 지원.
            """
            if not isinstance(item, dict):
                return ""

            # Retriever 결과는 page dict가 직접 오거나 {"page_data": {...}} 형태일 수 있음
            page = item.get("page_data") if isinstance(item.get("page_data"), dict) else item
            user_text = page.get("user_input", "")
            agent_text = page.get("agent_response", "")
            if user_text or agent_text:
                return f"User: {user_text}\nAssistant: {agent_text}".strip()
            return ""

        if isinstance(result, str) and result.strip():
            # 포맷된 문자열인 경우 그대로 하나의 chunk로
            chunks.append(MemoryChunk(
                content=result.strip(),
                session_file="",
                turn_idx=-1,
                score=1.0,
            ))
        elif isinstance(result, list):
            for item in result[:top_k]:
                if isinstance(item, dict):
                    content = (_qa_raw_text(item)
                                or item.get("content")
                                or item.get("memory")
                                or item.get("text")
                                or str(item))
                    score = float(item.get("score", 1.0))
                elif isinstance(item, str):
                    content = item
                    score = 1.0
                else:
                    content = str(item)
                    score = 1.0
                if content:
                    chunks.append(MemoryChunk(
                        content=content,
                        session_file="",
                        turn_idx=-1,
                        score=score,
                    ))
        elif isinstance(result, dict):
            # {memories: [...]} 형태 등
            for key in ("memories", "results", "data"):
                if key in result and isinstance(result[key], list):
                    return self._parse_retrieve_result(result[key], top_k)
            # 그 외 dict → str 변환
            qa_text = _qa_raw_text(result)
            chunks.append(MemoryChunk(
                content=qa_text or str(result),
                session_file="",
                turn_idx=-1,
                score=1.0,
            ))

        return chunks[:top_k]

    # ─────────────────────────────────────
    # Write evidence
    # ─────────────────────────────────────

    def get_write_evidence(
        self,
        session: dict,
        written_keys: list[tuple[str, int]],
    ) -> str:
        """이번 세션에 add_memory한 turn 내용을 write evidence로 반환."""
        if not written_keys:
            return ""
        return "\n\n".join(self._session_memories)

    # ─────────────────────────────────────
    # delete (MemoryOS는 직접 삭제 API 미제공 → no-op)
    # ─────────────────────────────────────

    def _delete_entry(self, entry_id: str) -> None:
        # MemoryOS는 내부 heat/eviction으로 자체 관리 → 외부 삭제 no-op
        pass

    # ─────────────────────────────────────
    # dump_memories
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        """현재 저장된 메모리 조회 (short/mid/long 파일 기반 포함)."""
        if self._mos is None:
            return []
        memories = []

        # 1) SDK 제공 조회 API가 있으면 우선 사용
        try:
            for attr in ("get_memory", "get_all_memories", "retrieve_all"):
                if hasattr(self._mos, attr):
                    raw = getattr(self._mos, attr)()
                    if isinstance(raw, list):
                        for item in raw:
                            memories.append({
                                "memory_text": item if isinstance(item, str)
                                               else item.get("content", str(item)),
                                "raw": item,
                            })
                    elif isinstance(raw, str) and raw.strip():
                        memories.append({"memory_text": raw, "raw": raw})
                    break
        except Exception:
            pass

        # 2) 파일 기반 조회: short/mid/long 메모리 파싱
        user_id = getattr(self._mos, "user_id", self._user_id)
        assistant_id = getattr(self._mos, "assistant_id", "u_mem_assistant")
        base_path = Path(self._data_path)

        def _load_json(path: Path):
            if not path.exists():
                return None
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        short_path = base_path / "users" / user_id / "short_term.json"
        short_data = _load_json(short_path)
        if isinstance(short_data, list):
            for i, qa in enumerate(short_data):
                memories.append({
                    "source": "short_term",
                    "memory_text": (
                        f"User: {qa.get('user_input', '')}\n"
                        f"Assistant: {qa.get('agent_response', '')}"
                    ).strip(),
                    "timestamp": qa.get("timestamp"),
                    "index": i,
                })

        mid_path = base_path / "users" / user_id / "mid_term.json"
        mid_data = _load_json(mid_path)
        if isinstance(mid_data, dict):
            sessions = mid_data.get("sessions", {})
            if isinstance(sessions, dict):
                for sid, session in sessions.items():
                    details = session.get("details", [])
                    for page in details if isinstance(details, list) else []:
                        memories.append({
                            "source": "mid_term",
                            "session_id": sid,
                            "memory_text": (
                                f"User: {page.get('user_input', '')}\n"
                                f"Assistant: {page.get('agent_response', '')}"
                            ).strip(),
                            "timestamp": page.get("timestamp"),
                            "page_id": page.get("page_id"),
                            "meta_info": page.get("meta_info"),
                        })

        long_user_path = base_path / "users" / user_id / "long_term_user.json"
        long_user_data = _load_json(long_user_path)
        if isinstance(long_user_data, dict):
            user_profiles = long_user_data.get("user_profiles", {})
            if isinstance(user_profiles, dict):
                for uid, profile in user_profiles.items():
                    memories.append({
                        "source": "long_term_user_profile",
                        "user_id": uid,
                        "memory_text": profile.get("data", "") if isinstance(profile, dict) else str(profile),
                        "timestamp": profile.get("last_updated") if isinstance(profile, dict) else None,
                    })
            for i, k in enumerate(long_user_data.get("knowledge_base", []) or []):
                memories.append({
                    "source": "long_term_user_knowledge",
                    "memory_text": k.get("knowledge", "") if isinstance(k, dict) else str(k),
                    "timestamp": k.get("timestamp") if isinstance(k, dict) else None,
                    "index": i,
                })

        long_assist_path = base_path / "assistants" / assistant_id / "long_term_assistant.json"
        long_assist_data = _load_json(long_assist_path)
        if isinstance(long_assist_data, dict):
            for i, k in enumerate(long_assist_data.get("assistant_knowledge", []) or []):
                memories.append({
                    "source": "long_term_assistant_knowledge",
                    "memory_text": k.get("knowledge", "") if isinstance(k, dict) else str(k),
                    "timestamp": k.get("timestamp") if isinstance(k, dict) else None,
                    "index": i,
                })

        # 3) 그래도 비어 있으면 budget tracker fallback
        if not memories:
            for r in self._entry_records:
                memories.append(
                    {
                        "source": "entry_records_fallback",
                        "memory_text": r.content,
                        "session_file": r.session_file,
                        "turn_idx": r.turn_idx,
                    }
                )
        return memories

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        import json
        state = {
            "system":               "MemoryOSSystem",
            "user_id":              self._user_id,
            "data_path":            self._data_path,
            "llm_model":            self.llm_model,
            "embedding_model_name": self.embedding_model_name,
            "openai_base_url":      self.openai_base_url,
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
        self._user_id  = state.get("user_id", "")
        self._data_path = state.get("data_path", "")
        # _mos는 재생성 필요 → reset()으로 재초기화
