#!/usr/bin/env python3
"""
6_run_mem0.py

Mem0 memory system 실행 스크립트.
대화를 읽어 Mem0에 저장하고 snapshot을 남긴다.
평가(write/read/QA score)는 하지 않음.

Usage:
    # OpenAI (기본)
    python 6_run_mem0.py \\
        --data_dir ./skeleton_dialogues_v4 \\
        --snapshot_dir ./snapshots \\
        --mem0_llm_model gpt-4o-mini

    # vLLM
    python 6_run_mem0.py \\
        --data_dir ./skeleton_dialogues_v4 \\
        --snapshot_dir ./snapshots \\
        --mem0_llm_provider vllm \\
        --mem0_llm_model Qwen/Qwen2.5-7B-Instruct \\
        --mem0_vllm_base_url http://localhost:8000

    # Oracle (memory_required=False 세션 skip)
    python 6_run_mem0.py \\
        --data_dir ./skeleton_dialogues_v4 \\
        --snapshot_dir ./snapshots \\
        --oracle

    # Phase1 처리 후 체크포인트 저장
    python 6_run_mem0.py \\
        --data_dir ./skeleton_dialogues_v5 \\
        --snapshot_dir ./snapshots/mem0_phase1 \\
        --save_checkpoint_dir ./checkpoints/mem0_after_phase1

    # Phase2: 저장된 메모리 상태에서 이어서 실행
    python 6_run_mem0.py \\
        --data_dir ./phase2_skeletons \\
        --snapshot_dir ./snapshots/mem0_phase2 \\
        --load_checkpoint_dir ./checkpoints/mem0_after_phase1
"""

import argparse
from pathlib import Path

from memory_systems import build_system
from run_utils import run_memory_pipeline
import warnings, os
warnings.filterwarnings("ignore")

# ========================
# Args
# ========================

parser = argparse.ArgumentParser(description="Mem0 memory system runner")

# Data
parser.add_argument('--data_dir',      type=str, default='./skeleton_dialogues_v5',
                    help='skeleton_dialogues_vX 디렉터리 경로')
parser.add_argument('--snapshot_dir',  type=str, default='./snapshots/mem0',
                    help='snapshot 저장 디렉터리')
parser.add_argument('--final_memory_dir', type=str, default='./final_memories/mem0',
                    help='최종 메모리 상태 저장 디렉터리')
parser.add_argument('--uuid',          type=str, default=None,
                    help='특정 UUID만 실행 (미설정 시 전체)')
parser.add_argument('--limit',         type=int, default=None,
                    help='처리할 UUID 최대 수')

# Storage
parser.add_argument('--storage_unit',  type=str, default='session',
                    choices=['turn', 'session'],
                    help='turn: 매 turn마다 Mem0 호출 / session: 세션 전체를 한 번에')
parser.add_argument('--oracle',        action='store_true', default=False,
                    help='memory_required=False 세션 skip')

# Memory budget
parser.add_argument('--mem_budget_tokens', type=int, default=-1,
                    help='최대 token 수. 초과 시 oldest fact부터 삭제')
parser.add_argument('--mem_budget_entries', type=int, default=20,
                    help='최대 memory entry 수. 초과 시 oldest fact부터 삭제 (-1: unlimited)')

# Mem0 LLM
parser.add_argument('--mem0_llm_provider', type=str, default='openai',
                    help='Mem0 내부 LLM provider (openai | vllm | anthropic)')
parser.add_argument('--mem0_llm_model',    type=str, default='gpt-5-mini',
                    help='Mem0 내부 LLM 모델명')
parser.add_argument('--mem0_llm_temperature', type=float, default=0.0,
                    help='Mem0 내부 LLM temperature (기본값: 0.0)')
parser.add_argument('--mem0_vllm_base_url', type=str, default='http://localhost:8000/v1',
                    help='vLLM 서버 URL (예: http://localhost:8000). '
                         'mem0_llm_provider=vllm 일 때 사용')

# Mem0 Embedder
parser.add_argument('--mem0_embedder_provider', type=str, default=None,
                    help='Mem0 embedder provider (예: openai). 미설정 시 기본값 사용')
parser.add_argument('--mem0_embedder_model',    type=str, default=None,
                    help='Mem0 embedder 모델명. 미설정 시 기본값 사용')

# Checkpoint
parser.add_argument('--save_checkpoint_dir', type=str, default=None,
                    help='전체 세션 처리 후 메모리 상태를 체크포인트로 저장할 디렉터리. '
                         '{dir}/{uuid}/ 아래에 budget_state.json + backend_state.json 저장.')
parser.add_argument('--load_checkpoint_dir', type=str, default=None,
                    help='세션 처리 시작 전 불러올 체크포인트 디렉터리 (phase2 이어받기용). '
                         '{dir}/{uuid}/ 에서 읽음. MEM0_DIR 설정 필요 (벡터 DB 영속성).')

args = parser.parse_args()

# ========================
# Main
# ========================

def main():
    system_label = f"mem0_{args.storage_unit}{'_oracle' if args.oracle else ''}"
    
    if args.mem_budget_tokens == -1:
        args.mem_budget_tokens = 10000000000000000
    if args.mem_budget_entries == -1:
        args.mem_budget_entries = 10000000000000000
    elif args.mem_budget_entries < -1:
        raise ValueError("--mem_budget_entries must be -1 or >= 0")

    system = build_system(
        system_name='mem0',
        oracle=args.oracle,
        max_tokens=args.mem_budget_tokens,
        max_entries=args.mem_budget_entries,
        mem0_llm_provider=args.mem0_llm_provider,
        mem0_llm_model=args.mem0_llm_model,
        mem0_llm_temperature=args.mem0_llm_temperature,
        mem0_vllm_base_url=args.mem0_vllm_base_url,
        mem0_embedder_provider=args.mem0_embedder_provider,
        mem0_embedder_model=args.mem0_embedder_model,
    )

    
    session_num = len(os.listdir(os.path.join('skeleton_dialogues_v5', args.uuid.replace('.json', ''))))
    if os.path.exists(os.path.join(args.snapshot_dir, system_label, args.uuid.replace('.json', ''))):
        snapshot_num = len(os.listdir(os.path.join(args.snapshot_dir, system_label, args.uuid.replace('.json', ''))))
        if session_num == snapshot_num:
            return

    run_memory_pipeline(
        system=system,
        system_label=system_label,
        data_dir=Path(args.data_dir),
        snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
        final_memory_dir=Path(args.final_memory_dir) if args.final_memory_dir else None,
        storage_unit=args.storage_unit,
        oracle=args.oracle,
        mem_budget_tokens=args.mem_budget_tokens,
        mem_budget_entries=args.mem_budget_entries,
        uuid_filter=args.uuid,
        limit=args.limit,
        save_checkpoint_dir=Path(args.save_checkpoint_dir) if args.save_checkpoint_dir else None,
        load_checkpoint_dir=Path(args.load_checkpoint_dir) if args.load_checkpoint_dir else None,
    )


if __name__ == '__main__':
    main()
