export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
uuid=$1


# 1. User-Specific Agent Use Profiling
python 1_1_persona_meta_data_generation.py
python 1_2_judge_meta_data.py
python 1_3_make_final_meta_data.py

# 2. Life Skeleton Generation (Before Shift)
python 2_1_generate_life_skeleton_bf_shift.py --input_dir ./final_persona_metadata --max_domains_per_user 3 --domain_sample_seed 1995 --output_dir ./life_skeletons --limit 20
python 2_2_generate_oneoff_sessions.py --persona_dir ./final_persona_metadata --skeleton_dir ./life_skeletons --provider openai --model gpt-5.4
python 2_3_integrate_timeline.py --input_dir ./life_skeletons --output_dir ./life_timelines --provider openai --model gpt-5.4

# 3. Life Skeleton Generation (After Shift)
python 3_1_generate_pattern_shift.py --input_dir ./life_timelines --output_dir ./pattern_shifts
python 3_2_generate_life_skeleton_aft_shift.py --shift_dir ./pattern_shifts --timeline_dir ./life_timelines --output_dir ./phase2_skeletons
python 3_3_integrate_timeline_aft_shift.py --phase2_dir ./phase2_skeletons --timeline_dir ./life_timelines --output_dir ./extended_timelines
python 3_4_merge_extended_timeline.py

# 4. Dialogue Generation
python 4_dialogue_gen.py --input_dir ./life_timelines_merged --output_dir ./PerMemBench

# 5-1. Run Base Memory Systems
bash sh/run_mem0.sh session -1 PerMemBench 0 openai gpt-5-mini "" false "" exp_mem0

# 5-2. Run Greedy Session Gating Memory Systems
python 6_gating_greedy.py --data_dir ./PerMemBench --output_dir ./results/greedy_gating

# 5-3. Run Context-based Session Gating Memory Systems
python 6_gating_context.py --data_dir ./PerMemBench --output_dir ./results/context_gating

# 5-4. Run Structure-based Session Gating Memory Systems
python 6_gating_structure.py --data_dir ./PerMemBench --output_dir ./results/structure_gating

# 5-5. Implement Snapshot Augmentation (Run once without strict budgets, save snapshots, then apply post-hoc budget/gating/eviction augmentation to reduce repeated memory-system costs.)
python 6_snapshot_augmentation.py --source_snapshot_dir SNAPSHOT_PATH --entry_budget 300 --skip_sessions gt
# Example: ./results/run_mem0/sys=mem0__gran=session__bud=-1_10000000000000000__mllm=openai-gpt-5-mini__d=v5/snapshots/mem0_session
# skip_session can be one of: None, gt, results/context_gating, results/greedy_gating, results/structure_gating.

# 6. Run Evaluation
sh sh/run_long.sh snapshot_path $uuid # Set uuid to run multiple users in parallel.




