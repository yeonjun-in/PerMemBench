"""memory_systems package"""

import os

from .base import BaseMemorySystem, MemoryChunk, count_tokens
from .heuristic import HeuristicSystem
from .mem0_system import Mem0System
from .memory_r1_system import MemoryR1System
from .rmm_system import RMMSystem
from .memobase_system import MemobaseSystem
from .supermemory_system import SupermemorySystem
from .memos_system import MemOSSystem
from .memoryos_system import MemoryOSSystem
from .amem import AmemSystem
from .oracle_filter import OracleFilter
import dotenv

dotenv.load_dotenv()

def build_system(
    system_name: str,
    oracle: bool = False,
    # heuristic / 공통
    max_tokens: int = 8000,
    max_entries: int | None = None,
    embedding_provider: str = 'openai',
    embedding_model: str = 'text-embedding-3-small',
    # mem0 LLM
    mem0_config: dict | None = None,
    mem0_llm_provider: str | None = None,
    mem0_llm_model: str | None = None,
    mem0_llm_temperature: float | None = None,
    mem0_vllm_base_url: str | None = None,
    # mem0 embedder
    mem0_embedder_provider: str | None = None,
    mem0_embedder_model: str | None = None,
    # memobase
    memobase_project_url: str = os.getenv('MEMOBASE_PROJECT_URL'),
    memobase_api_key: str = os.getenv('MEMOBASE_API_KEY'),
    # supermemory
    supermemory_api_key: str | None = None,
    # memos (MemOS)
    memos_config_path: str | None = None,
    # memoryos (BAI-LAB)
    memoryos_openai_api_key: str | None = None,
    memoryos_openai_base_url: str | None = None,
    memoryos_llm_model: str = 'gpt-4o-mini',
    memoryos_embedding_model: str = 'BAAI/bge-m3',
    memoryos_data_storage_root: str = './.memoryos_data',
    memoryos_mid_term_capacity: int = 1000,
    memoryos_mid_term_heat_threshold: float = 13.0,
    memoryos_mid_term_similarity_threshold: float = 0.7,
    memoryos_short_term_capacity: int = 2,
    # amem (A-MEM)
    amem_embedding_model: str = 'all-MiniLM-L6-v2',
    amem_llm_backend: str = 'openai',
    amem_llm_model: str = 'gpt-4o-mini',
    # rmm
    rmm_llm_provider: str = 'openai',
    rmm_llm_model: str = 'gpt-4.1-mini',
    rmm_llm_base_url: str | None = None,
    rmm_embedding_provider: str = 'openai',
    rmm_embedding_model: str = 'text-embedding-3-small',
    rmm_update_top_k: int = 5,
    # memory_r1
    r1_llm_provider: str = 'openai',
    r1_llm_model: str = 'gpt-4.1-mini',
    r1_llm_base_url: str | None = None,
    r1_embedding_provider: str = 'openai',
    r1_embedding_model: str = 'text-embedding-3-small',
    r1_manager_top_k: int = 20,
) -> BaseMemorySystem:
    """
    system_name과 oracle flag를 받아 적절한 system 인스턴스 반환.

    Args:
        system_name : 'heuristic' | 'mem0' | 'memory_r1' | 'memobase' |
                      'supermemory' | 'memos' | 'memoryos' | 'amem'
        oracle      : True면 OracleFilter로 래핑

        [memory_r1 설정]
        r1_llm_provider      : LLM provider (openai | claude | vllm | together | gemini)
        r1_llm_model         : 모델명 (예: gpt-4.1-mini)
        r1_llm_base_url      : vLLM 등 커스텀 base URL
        r1_embedding_provider: 임베딩 provider
        r1_embedding_model   : 임베딩 모델명
        r1_retrieval_top_k   : Memory Manager에 넘길 유사 메모리 수

        [mem0 설정]
        mem0_llm_provider    : mem0 내부 LLM provider
        mem0_llm_model       : mem0 내부 LLM 모델명
        mem0_vllm_base_url   : vLLM 서버 URL (provider=vllm 일 때)

        [memobase 설정]
        memobase_project_url : Memobase 서버 URL
        memobase_api_key     : Memobase API 키

        [supermemory 설정]
        supermemory_api_key  : Supermemory API 키 (없으면 env 사용)

        [memos 설정]
        memos_config_path    : MemOS yaml 설정 파일 경로
    """
    system_name = system_name.lower()

    if system_name == 'heuristic':
        system = HeuristicSystem(
            max_tokens=max_tokens,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )

    elif system_name == 'rmm':
        system = RMMSystem(
            max_tokens=max_tokens,
            llm_provider=rmm_llm_provider,
            llm_model=rmm_llm_model,
            llm_base_url=rmm_llm_base_url,
            embedding_provider=rmm_embedding_provider,
            embedding_model=rmm_embedding_model,
            update_top_k=rmm_update_top_k,
        )

    elif system_name == 'memory_r1':
        system = MemoryR1System(
            max_tokens=max_tokens,
            llm_provider=r1_llm_provider,
            llm_model=r1_llm_model,
            llm_base_url=r1_llm_base_url,
            embedding_provider=r1_embedding_provider,
            embedding_model=r1_embedding_model,
            manager_top_k=r1_manager_top_k,
        )

    elif system_name == 'mem0':
        resolved_config = dict(mem0_config or {})

        mem0_root = os.environ.get("MEM0_DIR")
        if mem0_root:
            resolved_config.setdefault(
                "history_db_path",
                os.path.join(mem0_root, "history.db"),
            )
            vector_cfg = dict(resolved_config.get("vector_store", {}))
            vector_provider = vector_cfg.get("provider", "qdrant")
            vector_inner = dict(vector_cfg.get("config", {}))
            if vector_provider in {"qdrant", "chroma", "faiss"}:
                vector_inner.setdefault(
                    "path",
                    os.path.join(mem0_root, vector_provider),
                )
                vector_cfg["provider"] = vector_provider
                vector_cfg["config"] = vector_inner
                resolved_config["vector_store"] = vector_cfg

        if mem0_llm_provider or mem0_llm_model or mem0_llm_temperature is not None:
            llm_cfg = dict(resolved_config.get("llm", {}))
            if mem0_llm_provider:
                llm_cfg["provider"] = mem0_llm_provider
            inner_cfg = dict(llm_cfg.get("config", {}))
            if mem0_llm_model:
                inner_cfg["model"] = mem0_llm_model
            if mem0_llm_temperature is not None:
                inner_cfg["temperature"] = mem0_llm_temperature
            if mem0_vllm_base_url and mem0_llm_provider == "vllm":
                inner_cfg["vllm_base_url"] = mem0_vllm_base_url
            llm_cfg["config"] = inner_cfg
            resolved_config["llm"] = llm_cfg

        if mem0_embedder_provider or mem0_embedder_model:
            emb_cfg = dict(resolved_config.get("embedder", {}))
            if mem0_embedder_provider:
                emb_cfg["provider"] = mem0_embedder_provider
            inner_emb = dict(emb_cfg.get("config", {}))
            if mem0_embedder_model:
                inner_emb["model"] = mem0_embedder_model
            emb_cfg["config"] = inner_emb
            resolved_config["embedder"] = emb_cfg

        system = Mem0System(
            max_tokens=max_tokens,
            config=resolved_config if resolved_config else None,
        )

    elif system_name == 'memobase':
        system = MemobaseSystem(
            max_tokens=max_tokens,
            project_url=memobase_project_url,
            api_key=memobase_api_key,
        )

    elif system_name == 'supermemory':
        system = SupermemorySystem(
            max_tokens=max_tokens,
            api_key=supermemory_api_key,
        )

    elif system_name == 'memos':
        system = MemOSSystem(
            max_tokens=max_tokens,
            config_path=memos_config_path,
        )

    elif system_name == 'memoryos':
        system = MemoryOSSystem(
            max_tokens=max_tokens,
            openai_api_key=memoryos_openai_api_key,
            openai_base_url=memoryos_openai_base_url,
            llm_model=memoryos_llm_model,
            embedding_model_name=memoryos_embedding_model,
            data_storage_root=memoryos_data_storage_root,
            mid_term_capacity=memoryos_mid_term_capacity,
            mid_term_heat_threshold=memoryos_mid_term_heat_threshold,
            mid_term_similarity_threshold=memoryos_mid_term_similarity_threshold,
            short_term_capacity=memoryos_short_term_capacity,
        )

    elif system_name == 'amem':
        system = AmemSystem(
            max_tokens=max_tokens,
            embedding_model=amem_embedding_model,
            llm_backend=amem_llm_backend,
            llm_model=amem_llm_model,
        )

    else:
        raise ValueError(
            f"Unknown system: '{system_name}'. "
            "Choose from: heuristic, mem0, memory_r1, memobase, supermemory, memos, memoryos, amem"
        )

    system.max_entries = max_entries

    if oracle:
        return OracleFilter(system)
    return system


__all__ = [
    'BaseMemorySystem',
    'MemoryChunk',
    'HeuristicSystem',
    'Mem0System',
    'MemoryR1System',
    'RMMSystem',
    'MemobaseSystem',
    'SupermemorySystem',
    'MemOSSystem',
    'MemoryOSSystem',
    'AmemSystem',
    'OracleFilter',
    'build_system',
]
