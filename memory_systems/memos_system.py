"""
memory_systems/memos_system.py  (v3)

핵심 수정:
- _register_cube_for_user():
    MOS의 mem_reader 설정을 cube의 text_mem config로 재사용.
    cube 디렉터리에 config.json을 직접 써서 init_from_dir이 성공하도록 함.
    (GeneralMemCubeConfig.text_mem 타입 == mem_reader 타입)
"""

import gc
import importlib
import json
import os
import re
import uuid as uuid_lib
from pathlib import Path
from .base import BaseMemorySystem, MemoryChunk, count_tokens


class MemOSSystem(BaseMemorySystem):
    def __init__(
        self,
        max_tokens: int = 8000,
        config_path: str | None = None,
        user_id_prefix: str = "u_mem_eval",
    ):
        super().__init__(max_tokens=max_tokens)
        self.config_path = config_path
        self.user_id_prefix = user_id_prefix

        self._mos = None
        self._user_id: str = ""
        self._cube_id: str | None = None
        self._session_memories: list[str] = []
        self._active_memos_base_path: str | None = None

    # ─────────────────────────────────────
    # 초기화
    # ─────────────────────────────────────

    def _reset_backend(self, user_id: str | None = None) -> None:
        self._prepare_memos_storage(user_id=user_id)
        self._teardown_mos()
        try:
            self._init_mos()
        except ImportError as e:
            raise ImportError(f"MemOS SDK 없음: {e}")
        except Exception as e:
            raise RuntimeError(f"MemOS 초기화 실패: {e}")
        self._session_memories = []

    def _prepare_memos_storage(self, user_id: str | None = None) -> None:
        existing_base = os.environ.get("MEMOS_BASE_PATH")
        if existing_base:
            base_path = Path(existing_base)
        else:
            safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id or "default")
            base_path = Path.cwd() / ".memos_runtime" / f"{safe_user}_pid{os.getpid()}"
            os.environ["MEMOS_BASE_PATH"] = str(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        (base_path / ".memos").mkdir(parents=True, exist_ok=True)
        self._active_memos_base_path = str(base_path)
        try:
            memos_settings = importlib.import_module("memos.settings")
            memos_settings.MEMOS_DIR = base_path / ".memos"
        except Exception:
            pass

    def _teardown_mos(self) -> None:
        if self._mos is None:
            return
        try:
            if getattr(self._mos, "enable_mem_scheduler", False) and hasattr(self._mos, "mem_scheduler_off"):
                self._mos.mem_scheduler_off()
        except Exception:
            pass
        try:
            if hasattr(self._mos, "mem_reorganizer_off"):
                self._mos.mem_reorganizer_off()
        except Exception:
            pass
        self._mos = None
        gc.collect()

    def _init_mos(self) -> None:
        """MOS 초기화 → user 생성 → cube 등록."""
        try:
            from memos.configs.mem_os import MOSConfig
            from memos.mem_os.main import MOS
            _sdk = "memos"
        except ImportError:
            try:
                from memoryos import MemoryOS as MOS
                MOSConfig = None
                _sdk = "memoryos"
            except ImportError:
                raise ImportError("memos 또는 memoryos 패키지 없음")

        self._user_id = f"{self.user_id_prefix}_{uuid_lib.uuid4().hex[:8]}"

        if _sdk == "memos":
            if self.config_path and MOSConfig is not None:
                config = MOSConfig.from_yaml(self.config_path)
                self._mos = MOS(config)
            else:
                self._mos = MOS()

            # user 생성 (create_user 우선, add_user fallback)
            try:
                self._mos.create_user(user_id=self._user_id)
            except AttributeError:
                try:
                    self._mos.add_user(user_id=self._user_id)
                except Exception:
                    pass
            except Exception:
                pass

            # cube 등록 (add() 전에 반드시)
            self._cube_id = self._register_cube_for_user(self._user_id)
            if self._cube_id is None:
                print("  [MEMOS] WARNING: cube 등록 실패.")

        elif _sdk == "memoryos":
            self._mos = MOS(
                user_id=self._user_id,
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                assistant_id="assistant",
            )

        self._sdk_type = _sdk

    def _register_cube_for_user(self, user_id: str) -> str | None:
        """
        MOS의 mem_reader config를 text_mem으로 재사용해 cube config.json을 생성,
        init_from_dir로 cube 객체를 만들어 등록한다.

        MOSConfig 구조:
          mos.config.mem_reader = {"backend": "simple_struct", "config": {...}}
        GeneralMemCubeConfig 구조:
          {"cube_id": ..., "text_mem": {"backend": "simple_struct", "config": {...}}}
        → mem_reader 필드를 그대로 text_mem으로 사용.
        """
        from memos.mem_cube.general import GeneralMemCube
        from memos.configs.mem_cube import GeneralMemCubeConfig

        cube_id   = f"cube_{user_id}"
        cube_path = Path(self._active_memos_base_path) / "cubes" / user_id
        cube_path.mkdir(parents=True, exist_ok=True)

        # ── config.json 생성 ──────────────────────────────────────────────
        config_json_path = cube_path / "config.json"
        if not config_json_path.exists():
            cube_cfg_dict = self._build_cube_config_dict(cube_id)
            with open(config_json_path, "w", encoding="utf-8") as f:
                json.dump(cube_cfg_dict, f, ensure_ascii=False, indent=2)

        # ── init_from_dir으로 cube 객체 생성 ─────────────────────────────
        try:
            # default_config를 MOS config 기반으로 제공 (merge 시 안전망)
            default_cfg = self._build_default_cube_config(cube_id)
            cube = GeneralMemCube.init_from_dir(
                str(cube_path),
                memory_types=["text_mem"],
                default_config=default_cfg,
            )
        except Exception as e:
            print(f"  [MEMOS] init_from_dir 실패({e}), default_config 없이 재시도")
            try:
                cube = GeneralMemCube.init_from_dir(str(cube_path))
            except Exception as e2:
                print(f"  [MEMOS] init_from_dir 완전 실패: {e2}")
                return None

        # ── register_mem_cube에 cube 객체 직접 전달 ──────────────────────
        try:
            self._mos.register_mem_cube(cube, mem_cube_id=cube_id, user_id=user_id)
            print(f"  [MEMOS] cube 등록 성공: {cube_id}")
            return cube_id
        except Exception as e:
            print(f"  [MEMOS] cube 등록 실패(객체): {e}")
            # fallback: 경로 전달
            try:
                self._mos.register_mem_cube(str(cube_path), mem_cube_id=cube_id, user_id=user_id)
                print(f"  [MEMOS] cube 등록 성공(경로): {cube_id}")
                return cube_id
            except Exception as e2:
                print(f"  [MEMOS] cube 등록 완전 실패: {e2}")
                return None

    def _build_cube_config_dict(self, cube_id: str) -> dict:
        """
        MOS의 mem_reader 설정을 text_mem으로 재사용한 cube config dict 생성.
        실패하면 최소한의 빈 config 반환.
        """
        base = {"cube_id": cube_id}
        try:
            mos_cfg_json = self._mos.config.model_dump_json()
            mos_cfg_dict = json.loads(mos_cfg_json)
            mem_reader   = mos_cfg_dict.get("mem_reader")
            if mem_reader:
                base["text_mem"] = mem_reader
        except Exception as e:
            print(f"  [MEMOS] config 추출 실패({e}), 빈 cube config 사용")
        return base

    def _build_default_cube_config(self, cube_id: str):
        """GeneralMemCubeConfig 객체를 생성. 실패하면 None."""
        try:
            from memos.configs.mem_cube import GeneralMemCubeConfig
            cfg_dict = self._build_cube_config_dict(cube_id)
            return GeneralMemCubeConfig(**cfg_dict)
        except Exception:
            try:
                from memos.configs.mem_cube import GeneralMemCubeConfig
                return GeneralMemCubeConfig()
            except Exception:
                return None

    # ─────────────────────────────────────
    # 추상 메서드
    # ─────────────────────────────────────

    def _write_turn(self, session_file, turn_idx, session_idx, domain_name,
                    user_content, agent_content) -> str | None:
        raise NotImplementedError("MemOSSystem uses overridden write() directly")

    def _delete_entry(self, entry_id: str) -> None:
        pass

    # ─────────────────────────────────────
    # write / write_session
    # ─────────────────────────────────────

    def write(self, session: dict) -> list[tuple[str, int]]:
        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")
        turns        = self.extract_turns(dialogue)
        if not turns:
            return []

        written = []
        self._session_memories = []
        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue
            messages = [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": agent_content},
            ]
            try:
                self._add_memory(messages=messages)
                content  = self.build_raw_text(user_content, agent_content)
                entry_id = f"{session_file}__t{turn_idx}"
                self._register_entry(entry_id, session_file, turn_idx,
                                     count_tokens(content), content)
                self._written_turns.add((session_file, turn_idx))
                written.append((session_file, turn_idx))
                self._session_memories.append(content)
            except Exception as e:
                print(f"  [MEMOS] add_memory failed (turn={turn_idx}): {e}")
        return written

    def write_session(self, session: dict) -> list[tuple[str, int]]:
        dialogue     = session.get("dialogue", [])
        session_file = session.get("_filename", "unknown")
        turns        = self.extract_turns(dialogue)
        if not turns:
            return []

        messages = []
        for _, user_content, agent_content in turns:
            messages.append({"role": "user", "content": user_content})
            if agent_content:
                messages.append({"role": "assistant", "content": agent_content})

        all_content = "\n\n".join(self.build_raw_text(uc, ac) for _, uc, ac in turns)
        self._session_memories = []
        try:
            self._add_memory(messages=messages)
            entry_id = f"{session_file}__session"
            self._register_entry(entry_id, session_file, -1,
                                 count_tokens(all_content), all_content[:500])
            self._written_turns.add((session_file, -1))
            self._session_memories = [all_content]
            return [(session_file, -1)]
        except Exception as e:
            print(f"  [MEMOS] add_memory failed (session): {e}")
            return []

    # ─────────────────────────────────────
    # _add_memory
    # ─────────────────────────────────────

    def _add_memory(self, messages: list[dict]) -> None:
        if self._sdk_type == "memos":
            self._add_memory_memos(messages)
        elif self._sdk_type == "memoryos":
            self._add_memory_memoryos(messages)

    def _add_memory_memos(self, messages: list[dict]) -> None:
        def _try_add():
            try:
                self._mos.add(messages=messages, user_id=self._user_id)
            except TypeError:
                self._mos.add(messages, user_id=self._user_id)

        try:
            _try_add()
        except ValueError as e:
            if "No accessible cubes found" not in str(e):
                raise
            print("  [MEMOS] cube 미등록 → 재등록 후 재시도")
            self._cube_id = self._register_cube_for_user(self._user_id)
            _try_add()
        except AttributeError:
            try:
                self._mos.add_memory(messages=messages, user_id=self._user_id)
            except TypeError:
                self._mos.add_memory(messages, user_id=self._user_id)

    def _add_memory_memoryos(self, messages: list[dict]) -> None:
        user_input = ""
        for msg in messages:
            if msg["role"] == "user":
                user_input = msg["content"]
            elif msg["role"] == "assistant":
                try:
                    self._mos.add_memory(user_input=user_input, agent_response=msg["content"])
                except Exception:
                    pass

    # ─────────────────────────────────────
    # Write evidence
    # ─────────────────────────────────────

    def get_write_evidence(self, session: dict, written_keys: list[tuple[str, int]]) -> str:
        if not written_keys:
            return ""
        return "\n\n".join(self._session_memories)

    # ─────────────────────────────────────
    # retrieve
    # ─────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        try:
            results = self._search_memory(query=query, top_k=top_k)
        except Exception as e:
            print(f"  [MEMOS] search_memory failed: {e}")
            return []

        chunks = []
        if isinstance(results, list):
            for item in results[:top_k]:
                content = (
                    item.get("content") or item.get("memory") or item.get("text")
                    if isinstance(item, dict) else str(item)
                )
                score = float(item.get("score", 1.0)) if isinstance(item, dict) else 1.0
                if content:
                    chunks.append(MemoryChunk(content=content, session_file="",
                                              turn_idx=-1, score=score))
        elif isinstance(results, str) and results.strip():
            chunks.append(MemoryChunk(content=results, session_file="", turn_idx=-1, score=1.0))
        return chunks

    def _search_memory(self, query: str, top_k: int) -> list | str:
        if self._sdk_type == "memos":
            try:
                result = self._mos.search(query=query, user_id=self._user_id, top_k=top_k)
            except (ValueError, AttributeError):
                return []
            try:
                memories = []
                for cube_result in result.get("text_mem", []):
                    for item in cube_result.get("memories", []):
                        if hasattr(item, "memory"):
                            memories.append({"content": item.memory,
                                             "score": getattr(item, "score", 1.0)})
                        elif isinstance(item, dict):
                            memories.append(item)
                        else:
                            memories.append({"content": str(item)})
                return memories
            except Exception:
                return []
        elif self._sdk_type == "memoryos":
            result = self._mos.retrieve_memory(query)
            return result if isinstance(result, list) else [{"content": str(result)}]
        return []

    # ─────────────────────────────────────
    # dump_memories
    # ─────────────────────────────────────

    def dump_memories(self) -> list[dict]:
        try:
            result = self._mos.get_all(user_id=self._user_id)
            out = []
            for cube_result in result.get("text_mem", []):
                for item in cube_result.get("memories", []):
                    text = item.memory if hasattr(item, "memory") else (
                        item.get("content", str(item)) if isinstance(item, dict) else str(item)
                    )
                    out.append({"memory_text": text, "raw": item})
            return out
        except Exception:
            return []

    # ─────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────

    def _save_backend_checkpoint(self, ckpt_path) -> None:
        state = {
            "system":      "MemOSSystem",
            "user_id":     self._user_id,
            "sdk_type":    getattr(self, "_sdk_type", ""),
            "config_path": self.config_path,
            "cube_id":     self._cube_id,
        }
        with open(ckpt_path / "backend_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_backend_checkpoint(self, ckpt_path) -> None:
        p = ckpt_path / "backend_state.json"
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            state = json.load(f)
        self._user_id = state["user_id"]
        self._cube_id = state.get("cube_id")