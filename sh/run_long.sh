export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

snapshot_dir=$1
uuid=$2
workers=${3:-16}  # Default is 16; override with the third argument.
skeleton_dir="skeleton_dialogues_v5"
judge_model=gpt-5-nano

python 7_memory_retention_eval.py \
    --snapshot_dir $snapshot_dir --skeleton_dir $skeleton_dir \
    --judge_model $judge_model --write_top_k 10 --uuid $uuid \
    --embedding_provider sentence_transformers --embedding_model all-MiniLM-L6-v2 \
    --skip_existing
