
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
parser.add_argument('--data_dir',      type=str, default='./skeleton_dialogues',
                    help='Path to skeleton_dialogues_vX directory')
parser.add_argument('--snapshot_dir',  type=str, default='./snapshots/mem0',
                    help='Directory to save snapshots')
parser.add_argument('--final_memory_dir', type=str, default='./final_memories/mem0',
                    help='Directory to save final memory state')
parser.add_argument('--uuid',          type=str, default=None,
                    help='Run a single UUID only (default: all)')
parser.add_argument('--limit',         type=int, default=None,
                    help='Maximum number of UUIDs to process')

# Storage
parser.add_argument('--storage_unit',  type=str, default='session',
                    choices=['turn', 'session'],
                    help='turn: call Mem0 per turn / session: process whole session at once')
parser.add_argument('--oracle',        action='store_true', default=False,
                    help='Skip sessions with memory_required=False')

# Memory budget
parser.add_argument('--mem_budget_tokens', type=int, default=-1,
                    help='Max tokens; evict oldest facts when exceeded')
parser.add_argument('--mem_budget_entries', type=int, default=20,
                    help='Max memory entries; evict oldest when exceeded (-1: unlimited)')

# Mem0 LLM
parser.add_argument('--mem0_llm_provider', type=str, default='openai',
                    help='Mem0 internal LLM provider (openai | vllm | anthropic)')
parser.add_argument('--mem0_llm_model',    type=str, default='gpt-5-mini',
                    help='Mem0 internal LLM model name')
parser.add_argument('--mem0_llm_temperature', type=float, default=0.0,
                    help='Mem0 internal LLM temperature (default: 0.0)')
parser.add_argument('--mem0_vllm_base_url', type=str, default='http://localhost:8000/v1',
                    help='vLLM server URL (e.g. http://localhost:8000). '
                         'Used when mem0_llm_provider=vllm')

# Mem0 Embedder
parser.add_argument('--mem0_embedder_provider', type=str, default=None,
                    help='Mem0 embedder provider (e.g. openai). Uses default if unset')
parser.add_argument('--mem0_embedder_model',    type=str, default=None,
                    help='Mem0 embedder model. Uses default if unset')

# Checkpoint
parser.add_argument('--save_checkpoint_dir', type=str, default=None,
                    help='Directory to save memory checkpoint after all sessions. '
                         'Writes budget_state.json + backend_state.json under {dir}/{uuid}/.')
parser.add_argument('--load_checkpoint_dir', type=str, default=None,
                    help='Checkpoint directory to load before sessions (resume phase2). '
                         'Reads from {dir}/{uuid}/. Requires MEM0_DIR for vector DB persistence.')

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
