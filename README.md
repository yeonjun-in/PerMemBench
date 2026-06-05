# Personalize-then-Store / PerMemBench

This repository contains the benchmark construction and evaluation pipeline from
**"Personalize-then-Store: Benchmarking and Learning Personalized Memory for Long-horizon Agents"**
([arXiv:2605.25535](https://arxiv.org/abs/2605.25535)).

The project builds long-horizon personalized dialogue trajectories and evaluates memory systems under different storage policies, including session-level gating.

## What this repo includes

- **Benchmark construction pipeline**:
  - Persona metadata generation and filtering (`1_*`)
  - Pre/post-shift life skeleton and timeline generation (`2_*`, `3_*`)
  - Dialogue generation (`4_dialogue_gen.py`)
- **Memory-system runs**:
  - Base Mem0 runs (`5_run_mem0.py`, `sh/run_mem0.sh`)
  - Gating baselines (`6_gating_*.py`)
  - Snapshot augmentation for budgeted post-hoc simulation (`6_snapshot_augmentation.py`)
- **Evaluation and analysis**:
  - Memory retention evaluation (`7_memory_retention_eval.py`, `sh/run_long.sh`)
  - Notebook-based analysis (`7_MRR_display.ipynb`)

## Quick start

### 1) Environment

Use Python 3.10+ (recommended) and install dependencies used by your run.

At minimum, you will typically need packages such as:

- `python-dotenv`
- `openai`
- `anthropic` (if using Claude)
- `google-genai` (if using Gemini)
- `sentence-transformers`

If you use local/vLLM inference, make sure your vLLM endpoint is running and reachable.

### 2) API keys

Set the provider key(s) in your environment or `.env` file (loaded automatically):

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `TOGETHER_API_KEY`
- `GOOGLE_API_KEY`

### 3) Download/use the released PerMemBench dataset

The prebuilt benchmark dataset is available in this repository under:

- `PerMemBench/PerMemBench.zip`

If you only want to run memory-system experiments and evaluation, you can use this dataset directly instead of running the full data-generation pipeline.

```bash
unzip PerMemBench/PerMemBench.zip -d .
```

After extraction, pass `PerMemBench` (or your extracted dataset path) as `dialogue_path` for `sh/run_mem0.sh`.

## End-to-end pipeline

The main orchestration script is:

```bash
bash sh/run_dial.sh <uuid>
```

This script executes:

1. Persona metadata generation and validation
2. Long-horizon timeline construction (before and after pattern shift)
3. Dialogue generation into `PerMemBench/`
4. Memory system runs (base + gating variants)
5. Snapshot augmentation hook (configure paths/options as needed)
6. Retention evaluation through `sh/run_long.sh`

## Running only memory system experiments

To run Mem0 directly:

```bash
bash sh/run_mem0.sh <granularity(turn|session)> <budget> <dialogue_path> <cuda_visible_devices> [mem0_llm_provider] [mem0_llm_model] [mem0_vllm_base_url] [oracle(true|false)] [uuid] [experiment_name]
```

Example:

```bash
bash sh/run_mem0.sh session -1 PerMemBench 0 openai gpt-5-mini "" false "" exp_mem0
```

## Evaluation and analysis

- Retention evaluation:
  ```bash
  bash sh/run_long.sh <snapshot_dir> <uuid> [workers]
  ```
- Result analysis:
  - `7_MRR_display.ipynb`
  - `7_gating_performance_display.ipynb`

## Notes

- Output artifacts are typically written to `results/`.
- Intermediate benchmark assets are generated under directories such as `life_*`, `pattern_shifts/`, and `PerMemBench/`.
- `run_dial.sh` includes comments and defaults for common experiments; adapt model/provider/budget values to your setup.

## Citation

If you use this repository, please cite:

```bibtex
@article{in2026personalize,
  title={Personalize-then-Store: Benchmarking and Learning Personalized Memory for Long-horizon Agents},
  author={In, Yeonjun and Kim, Wonjoong and Park, Sangwu and Yoon, Kanghoon and Park, Chanyoung},
  journal={arXiv preprint arXiv:2605.25535},
  year={2026}
}
```
