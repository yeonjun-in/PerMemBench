#!/usr/bin/env bash
set -euo pipefail


if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <granularity(turn|session)> <budget> <dialogue_path> <cuda_visible_devices> [mem0_llm_provider] [mem0_llm_model] [mem0_vllm_base_url] [oracle(true|false)] [uuid] [experiment_name]"
  exit 1
fi

granularity=$1
budget_tokens=-1
budget=$2
dialogue_path=$3
export CUDA_VISIBLE_DEVICES=$4
mem0_llm_provider=${5:-openai}
mem0_llm_model=${6:-gpt-4o-mini}
mem0_vllm_base_url=http://localhost:8000/v1
oracle=${7:-false}
experiment_name=${8:-run_mem0}
if [ "$oracle" = "true" ]; then
  experiment_name="${experiment_name}_oracle"
fi
uuid=${9:-}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

timestamp=$(date +%Y%m%d-%H%M%S)
git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")

sanitize() {
  echo "$1" | tr '/: ' '---'
}

base_id="sys=mem0__gran=$(sanitize "$granularity")__bud=${budget_tokens}_${budget}__mllm=$(sanitize "${mem0_llm_provider}-${mem0_llm_model}")__d=$(sanitize "$dialogue_path")"
if [ -n "$uuid" ]; then
  run_leaf="$(sanitize "$uuid")"
else
  run_leaf="all__git=${git_sha}__ts=${timestamp}__pid=$$"
fi
run_dir="./results/${experiment_name}/${base_id}"
log_dir="${run_dir}/logs"
snapshot_dir="${run_dir}/snapshots"
final_memory_dir="${run_dir}/final_memories"
config_json="${run_dir}/config.json"

if [ "$oracle" = "true" ]; then
    export MEM0_DIR="/tmp/mem0_oracle_${USER}/${base_id}/${run_leaf}"
else
    export MEM0_DIR="/tmp/mem0_${USER}/${base_id}/${run_leaf}"
fi

mkdir -p "$MEM0_DIR" "$log_dir" "$snapshot_dir" "$final_memory_dir"

cat > "$config_json" <<EOF
{
  "timestamp": "${timestamp}",
  "git_sha": "${git_sha}",
  "system": "mem0",
  "granularity": "${granularity}",
  "mem_budget_entries": ${budget},
  "mem_budget_tokens": ${budget_tokens},
  "dialogue_path": "${dialogue_path}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "mem0_llm_provider": "${mem0_llm_provider}",
  "mem0_llm_model": "${mem0_llm_model}",
  "mem0_vllm_base_url": "${mem0_vllm_base_url}",
  "oracle": ${oracle},
  "uuid": "${uuid}",
  "mem0_dir": "${MEM0_DIR}",
  "data_dir": "${dialogue_path}",
  "snapshot_dir": "${snapshot_dir}",
  "final_memory_dir": "${final_memory_dir}"
}
EOF

echo "[RUN] ${base_id}"
echo "[DIR] ${run_dir}"

cmd=(
  python 6_run_mem0.py
  --data_dir "$dialogue_path"
  --snapshot_dir "$snapshot_dir"
  --final_memory_dir "$final_memory_dir"
  --storage_unit "$granularity"
  --mem_budget_entries "$budget"
  --mem_budget_tokens "$budget_tokens"
  --mem0_llm_provider "$mem0_llm_provider"
  --mem0_llm_model "$mem0_llm_model"
)

if [ -n "$mem0_vllm_base_url" ]; then
  cmd+=(--mem0_vllm_base_url "$mem0_vllm_base_url")
fi

if [ "$oracle" = "true" ]; then
  cmd+=(--oracle)
fi

if [ -n "$uuid" ]; then
  cmd+=(--uuid "$uuid")
fi
"${cmd[@]}"
echo "[DONE] ${run_dir}"
