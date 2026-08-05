# Test-suite duration reference (tubingen (RTX 3090))

Measured at commit `5f502c9` on tubingen (RTX 3090), single serial run of the canonical suite invocation (`python -m pytest -q --durations=0`, `dataflow` conda env, box otherwise idle). Use it as the expectation baseline for how long suite tasks take at this commit; re-measure after structural suite changes.

Stack: torch 2.13.0+cu130 / triton 3.7.1 (upgraded from 2.12.1+cu130 immediately before measuring). Two consecutive full runs measured 19:02 (first post-upgrade) and 19:00 (this run, the table source) — cache warmth is negligible at suite scale and the wall is reproducible.

## Latest validation

Commit `fd0c24d` was validated on 2026-08-05 with the same serial command plus
`--junitxml` so the result and per-test totals were retained. It completed in
**19:22** (1162.65s): **1365 passed, 1 skipped, 58 deselected**. That is 22.65s
(2.0%) above the 19:00 baseline while selecting 31 additional tests.

- No test approached the timeout threshold; the slowest was 52.96s.
- The largest structural increase is
  `training/planning/test_planning.py`: 9 to 13 tests and 3.6s to 23.8s. Its
  two deliberate large-geometry planner checks account for 20.54s.
- `runtime/test_parity_vs_sim.py` rose from 2.0s to 11.3s; its starved-PCIe
  recompute case accounts for 7.28s.
- The largest existing-file fluctuation was the 22-test engine-family parity
  file (298.8s to 341.2s). Faster model-family, engine-reference, and kernel
  audit files offset most of that variance, leaving the whole-suite movement
  at 2.0%.
- The run left no pytest, torchrun, test daemon, or test worker processes
  behind.

The detailed tables below remain the stable `5f502c9` baseline. The retained
JUnit report for this validation was `/tmp/dataflow_full_fd0c24d.xml` on
Tübingen; it is a transient test artifact rather than repository input.

## Summary

- **Wall time: 19:00** (1140s) — 1334 passed, 1 skipped.
- Time accounted to individual tests below: 1126s (99% of wall; the rest is collection, session setup, and the many sub-5ms phases pytest omits).
- 118 test files with measurable time; distribution by per-file total: 5 files over 60s, 5 in 10-60s, 69 in 1-10s, 39 under 1s.
- Concentration: the top 10 files hold 895s (80% of accounted time) — they are the levers if the suite ever needs to get faster.

### Top 10 files

| # | file | tests | total | share of accounted |
|---|---|---|---|---|
| 1 | `tests/dataflow_training/pretrain/test_engine_parity_families.py` | 22 | 298.8s | 26.5% |
| 2 | `tests/dataflow_training/models/test_model_families.py` | 110 | 184.6s | 16.4% |
| 3 | `tests/dataflow_training/models/test_engine_vs_reference.py` | 20 | 124.7s | 11.1% |
| 4 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` | 3 | 91.8s | 8.2% |
| 5 | `tests/dataflow_training/tasks/test_kernel_audit.py` | 371 | 84.7s | 7.5% |
| 6 | `tests/examples/test_rl_training.py` | 6 | 36.7s | 3.3% |
| 7 | `tests/dataflow_training/pretrain/test_client_model_step.py` | 4 | 30.0s | 2.7% |
| 8 | `tests/dataflow_training/data/test_data_pipeline.py` | 22 | 19.0s | 1.7% |
| 9 | `tests/dataflow_training/training/surfaces/test_replicate_load.py` | 1 | 14.0s | 1.2% |
| 10 | `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` | 2 | 11.3s | 1.0% |

### Top 10 individual tests

| # | test | total |
|---|---|---|
| 1 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 50.8s |
| 2 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[qwen35_smoke_preset]` | 48.8s |
| 3 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_qwen35moe_engine_vs_reference` | 47.4s |
| 4 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_moe_persistent_set_round_trips` | 31.1s |
| 5 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_remapped_resume_bitwise` | 30.5s |
| 6 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_resume_reproduces_tail_bitwise` | 30.2s |
| 7 | `tests/dataflow_training/models/test_engine_vs_reference.py::test_engine_matches_reference_uniform[qwen35moe]` | 20.4s |
| 8 | `tests/dataflow_training/models/test_engine_vs_reference.py::test_engine_matches_reference_uniform[qwen35]` | 19.9s |
| 9 | `tests/dataflow_training/training/surfaces/test_replicate_load.py::test_world1_replicate_steps_once_bitwise` | 14.0s |
| 10 | `tests/dataflow_training/models/test_engine_vs_reference.py::test_engine_matches_reference_ragged[qwen35moe]` | 12.0s |

## Per-file breakdown

Sorted by total attributed time (call + setup + teardown of every test in the file).

| file | tests | total | slowest test | its total |
|---|---|---|---|---|
| `tests/dataflow_training/pretrain/test_engine_parity_families.py` | 22 | 298.8s | `test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 50.8s |
| `tests/dataflow_training/models/test_model_families.py` | 110 | 184.6s | `test_poison_on_free_changes_nothing[qwen35moe]` | 8.0s |
| `tests/dataflow_training/models/test_engine_vs_reference.py` | 20 | 124.7s | `test_engine_matches_reference_uniform[qwen35moe]` | 20.4s |
| `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` | 3 | 91.8s | `test_world2_moe_persistent_set_round_trips` | 31.1s |
| `tests/dataflow_training/tasks/test_kernel_audit.py` | 371 | 84.7s | `test_degenerate_finite[moe_grouped_mm_dgrad:eager:empty_experts]` | 0.3s |
| `tests/examples/test_rl_training.py` | 6 | 36.7s | `test_rl_training_parity_ppo[qwen35]` | 7.1s |
| `tests/dataflow_training/pretrain/test_client_model_step.py` | 4 | 30.0s | `test_client_model_step_matches_in_process_olmoe` | 9.8s |
| `tests/dataflow_training/data/test_data_pipeline.py` | 22 | 19.0s | `test_checkpoint_resume_tail_matches_uninterrupted_run` | 9.4s |
| `tests/dataflow_training/training/surfaces/test_replicate_load.py` | 1 | 14.0s | `test_world1_replicate_steps_once_bitwise` | 14.0s |
| `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` | 2 | 11.3s | `test_kill_escalates_past_sigterm` | 10.8s |
| `tests/dataflow_training/tasks/test_kernels.py` | 21 | 7.9s | `test_swiglu_fused[4099-14336]` | 1.7s |
| `tests/dataflow/runtime/test_profiling_memory.py` | 1 | 7.8s | `test_tables_leave_no_reserved_memory` | 7.8s |
| `tests/dataflow/service/test_shared_server_self_heal.py` | 1 | 7.8s | `test_self_heal_respawns_after_illegal_access` | 7.8s |
| `tests/dataflow/service/test_daemon_relaunch.py` | 1 | 6.9s | `test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses` | 6.9s |
| `tests/dataflow_sim/engine/test_simulator.py` | 38 | 6.6s | `test_missing_input_raises` | 0.2s |
| `tests/dataflow_sim/planning/policies/test_auto_policy.py` | 36 | 6.2s | `test_auto_policy_L10_works_down_to_cap_500[500]` | 0.2s |
| `tests/dataflow/service/test_slice_snapshots.py` | 12 | 6.0s | `test_restore_runs_in_background_and_parks_writers` | 0.8s |
| `tests/dataflow_training/pretrain/test_client_fetch_surface.py` | 2 | 6.0s | `test_client_fetch_surface_dense` | 3.2s |
| `tests/dataflow_sim/core/test_validate_chain.py` | 29 | 5.0s | `test_invalid_chain_rejected[make_invalid_capacity_initial_backing_overflow-make_invalid_capacity_initial_backing_overflow]` | 0.2s |
| `tests/dataflow/service/test_service_store.py` | 16 | 4.7s | `test_real_boot_family_init_byte_identity` | 1.5s |
| `tests/dataflow_sim/app/test_server.py` | 16 | 4.7s | `test_simulate_large_chain_uses_snapshot_free_response` | 2.1s |
| `tests/dataflow_training/training/e2e/test_varlen_e2e.py` | 11 | 4.6s | `test_model_step_ragged_matches_golden_all_families[qwen35moe]` | 0.8s |
| `tests/dataflow_training/training/lowering/test_layout_registry.py` | 21 | 4.5s | `test_registry_covers_external_family` | 0.2s |
| `tests/dataflow_training/modules/test_moe.py` | 24 | 4.5s | `test_moe_tail_fwd_bwd_vs_reference[False-0.0-softmax_then_topk]` | 0.2s |
| `tests/dataflow_training/training/surfaces/test_solo_resume_bitwise.py` | 1 | 4.5s | `test_solo_resume_reproduces_tail_bitwise` | 4.5s |
| `tests/dataflow/service/test_service_skeleton.py` | 10 | 4.4s | `test_fast_path_answers_while_dispatcher_held` | 0.9s |
| `tests/dataflow_training/training/e2e/test_freeze_plan.py` | 17 | 4.2s | `test_model_step_truncated_olmoe` | 0.3s |
| `tests/dataflow_sim/workloads/test_modular_workload_builder.py` | 24 | 4.2s | `test_constrained_memory_recompute_planning_selects_useful_variants` | 0.4s |
| `tests/dataflow_sim/planning/policies/test_min_grow.py` | 22 | 3.7s | `test_respects_static_cap_passes_with_unlimited` | 0.2s |
| `tests/dataflow_training/pretrain/test_parity_smoke.py` | 1 | 3.7s | `test_reference_vs_service_parity_smoke` | 3.7s |
| `tests/dataflow_training/training/planning/test_planning.py` | 9 | 3.6s | `test_backing_capacity_drives_recompute` | 0.9s |
| `tests/dataflow_training/training/surfaces/test_checkpoint_record.py` | 4 | 3.4s | `test_load_checkpoint_targets` | 1.2s |
| `tests/dataflow_training/pretrain/test_flops.py` | 15 | 3.3s | `test_gpt2_walker_matches_hand_formula` | 0.2s |
| `tests/dataflow_training/training/e2e/test_lbl_modes.py` | 8 | 3.2s | `test_retained_router_delta_is_ga_invariant_per_round_is_not` | 0.8s |
| `tests/dataflow_training/pretrain/test_presets.py` | 16 | 3.2s | `test_preset_lowers[l3_125m]` | 0.2s |
| `tests/dataflow/service/test_pinned_slab.py` | 3 | 3.2s | `test_slab_costs_what_it_asks_for` | 1.5s |
| `tests/dataflow_sim/workloads/test_dataflow_schema.py` | 18 | 3.1s | `test_inline_cost_normalizes_to_one_off_compute_block` | 0.2s |
| `tests/dataflow_training/training/e2e/test_dtype_policy_e2e.py` | 7 | 3.0s | `test_qwen35_model_step_depth_dependent` | 0.8s |
| `tests/dataflow_training/pretrain/test_sharding.py` | 13 | 2.9s | `test_world4_world8_plans_balance_cover_and_comm_identity` | 0.2s |
| `tests/dataflow/service/test_engine_determinism.py` | 1 | 2.9s | `test_same_daemon_rerun_bitwise` | 2.9s |
| `tests/dataflow/service/test_peer_protocol.py` | 17 | 2.8s | `test_overwrite_matrix` | 0.2s |
| `tests/dataflow_training/tasks/test_optim.py` | 11 | 2.7s | `test_hyper_overrides_and_schedule_model_step` | 0.3s |
| `tests/dataflow_training/models/test_block_isolation.py` | 5 | 2.6s | `test_isolated_block_at_floor[glm52-isolate0-6]` | 0.8s |
| `tests/dataflow/service/test_service_runs.py` | 5 | 2.4s | `test_rebind_two_token_slabs` | 1.3s |
| `tests/dataflow_training/modules/test_dsa.py` | 10 | 2.4s | `test_index_scores_vs_hand_loop` | 0.4s |
| `tests/dataflow_training/models/test_glm52.py` | 7 | 2.4s | `test_glm52_frozen_indexer_ablation` | 0.5s |
| `tests/dataflow_training/models/test_dsv32.py` | 7 | 2.4s | `test_dsv32_frozen_indexer_ablation` | 0.4s |
| `tests/test_import_boundaries.py` | 7 | 2.4s | `test_sim_required_only_under_lowering` | 0.6s |
| `tests/dataflow_training/models/test_gpt2.py` | 10 | 2.3s | `test_model_step_ragged` | 0.3s |
| `tests/dataflow/service/test_service_snapshot.py` | 3 | 2.3s | `test_checkpoint_roundtrip_bit_continuity` | 1.5s |
| `tests/dataflow_training/training/lowering/test_shaped_program.py` | 9 | 2.1s | `test_tied_embeddings_chain_structure` | 0.2s |
| `tests/dataflow_training/training/e2e/test_packed_args_e2e.py` | 7 | 2.0s | `test_packed_args_match_golden` | 0.4s |
| `tests/dataflow/runtime/test_parity_vs_sim.py` | 9 | 2.0s | `test_parity_8b_starved_pcie_recompute` | 0.6s |
| `tests/dataflow_training/models/test_qwen35.py` | 6 | 2.0s | `test_qwen35_tied_model_step_vs_golden` | 0.8s |
| `tests/dataflow/runtime/test_engine_semantics.py` | 12 | 1.9s | `test_capacity_deadlock_raises` | 0.2s |
| `tests/dataflow_sim/planning/policies/test_pressurefit.py` | 12 | 1.9s | `test_pressurefit_extends_prefetch_intervals_under_strict_cap` | 0.2s |
| `tests/dataflow_training/data/test_packing.py` | 10 | 1.9s | `test_pack_batch_deterministic_for_same_input` | 0.2s |
| `tests/dataflow/runtime/test_engine_stress.py` | 3 | 1.9s | `test_measured_costs_replan_still_golden` | 1.0s |
| `tests/dataflow_training/models/test_llama3.py` | 9 | 1.8s | `test_model_step_muon_policy_golden_parity` | 0.3s |
| `tests/dataflow_training/training/surfaces/test_source_policy_drills.py` | 3 | 1.7s | `test_simple_policy_round_trip_world2` | 0.8s |
| `tests/dataflow/core/test_ir_validate.py` | 11 | 1.7s | `test_dual_location_initial_size_mismatch_rejected` | 0.2s |
| `tests/dataflow/runtime/test_cuda_backend.py` | 5 | 1.6s | `test_mini_program_execution_matches_plan` | 0.8s |
| `tests/dataflow_training/training/e2e/test_ga_invariance.py` | 4 | 1.5s | `test_sgd_rounds_are_memory_optimization` | 0.5s |
| `tests/dataflow/runtime/test_run_contract.py` | 10 | 1.5s | `test_inv2_drain_runs_on_failure` | 0.2s |
| `tests/dataflow_sim/planning/test_recompute.py` | 8 | 1.5s | `test_recompute_loop_converts_under_pressure_and_improves` | 0.2s |
| `tests/dataflow/checkpoint/test_record_layer.py` | 9 | 1.4s | `test_replica_twins_must_hash_equal` | 0.2s |
| `tests/dataflow_training/tasks/test_varlen_attention.py` | 6 | 1.4s | `test_fwd_matches_ragged_fallback` | 0.2s |
| `tests/dataflow_training/modules/test_mla.py` | 7 | 1.4s | `test_dsv3_block_fwd_recompute_bwd_accum_match_autograd_golden[dense]` | 0.2s |
| `tests/dataflow_training/pretrain/test_reference_muon.py` | 5 | 1.3s | `test_tiny_muon_reference_trains` | 0.4s |
| `tests/dataflow/runtime/test_placement.py` | 7 | 1.3s | `test_parity_with_placement_8b` | 0.3s |
| `tests/dataflow_training/training/lowering/test_responsibility.py` | 6 | 1.3s | `test_run_lock_refuses_second_same_name` | 0.2s |
| `tests/dataflow_training/models/test_qwen3moe.py` | 5 | 1.3s | `test_qwen3moe_aux_zero_model_step_vs_golden` | 0.3s |
| `tests/dataflow_training/pretrain/test_sharding_lowering.py` | 5 | 1.2s | `test_shard_block_params_consistent_across_ranks` | 0.2s |
| `tests/test_program_hashes.py` | 1 | 1.2s | `test_lowered_program_hashes_stable` | 1.2s |
| `tests/dataflow_training/models/test_dsv3.py` | 4 | 1.2s | `test_dsv3_aux_zero_model_step_vs_golden` | 0.4s |
| `tests/dataflow_training/tasks/test_dtype_policy.py` | 5 | 1.2s | `test_default_policy_is_all_bf16` | 0.2s |
| `tests/dataflow/service/test_service_packed_args.py` | 1 | 1.1s | `test_daemon_packed_args_bit_equal` | 1.1s |
| `tests/dataflow/runtime/test_vmm.py` | 7 | 1.0s | `test_e2e_mini_vmm_matches_static` | 0.2s |
| `tests/dataflow/service/test_nccl_binding.py` | 2 | 1.0s | `test_binding_world1_roundtrip` | 0.9s |
| `tests/dataflow_training/models/test_olmoe.py` | 4 | 1.0s | `test_olmoe_aux_zero_model_step_vs_golden` | 0.3s |
| `tests/dataflow_training/training/lowering/test_parallelism_scheme.py` | 4 | 0.9s | `test_validate_refusals` | 0.2s |
| `tests/dataflow_training/pretrain/test_schedule.py` | 4 | 0.9s | `test_warmup_then_cosine_shape` | 0.2s |
| `tests/dataflow_training/models/test_glm52_lowering.py` | 5 | 0.9s | `test_full_scale_presets_lower` | 0.2s |
| `tests/dataflow_training/pretrain/test_topology.py` | 3 | 0.9s | `test_daemonize_detach_and_group_kill` | 0.5s |
| `tests/dataflow_sim/core/test_reference_stream.py` | 5 | 0.9s | `test_compute_reference_stream_returns_only_first_use` | 0.2s |
| `tests/dataflow_training/tasks/test_ignore_index_ce.py` | 4 | 0.9s | `test_no_ignore_rows_matches_torch_ce_and_rerun_is_bitwise[triton]` | 0.2s |
| `tests/dataflow/core/test_sim_convert.py` | 6 | 0.9s | `test_to_sim_chain_preserves_ids_and_sizes` | 0.2s |
| `tests/test_docstring_index.py` | 2 | 0.8s | `test_index_matches_test_functions` | 0.5s |
| `tests/dataflow_training/training/surfaces/test_plugins.py` | 4 | 0.8s | `test_explicit_plugin_load_end_to_end` | 0.2s |
| `tests/dataflow/checkpoint/test_record_targets.py` | 5 | 0.8s | `test_unknown_target_id_refuses` | 0.2s |
| `tests/dataflow/core/test_json_roundtrip.py` | 5 | 0.7s | `test_comm_groups_roundtrip_and_validation` | 0.2s |
| `tests/dataflow_training/models/test_qwen3.py` | 4 | 0.7s | `test_qwen3_block_backward` | 0.2s |
| `tests/dataflow_training/training/lowering/test_round_prologue.py` | 3 | 0.7s | `test_round_prologue_publishes_round_index_via_run_values_and_object` | 0.3s |
| `tests/dataflow_training/training/lowering/test_group_annotation.py` | 3 | 0.6s | `test_dp_annotation_matches_builder` | 0.2s |
| `tests/dataflow/service/test_peer_groups.py` | 2 | 0.6s | `test_group_lifecycle_and_error_fanout` | 0.4s |
| `tests/dataflow/service/test_error_codes.py` | 1 | 0.6s | `test_every_raised_error_code_is_registered` | 0.6s |
| `tests/dataflow_training/training/e2e/test_batch_ga.py` | 2 | 0.6s | `test_batch_ga_model_step_matches_reference` | 0.4s |
| `tests/dataflow/checkpoint/test_persistent_targets.py` | 4 | 0.6s | `test_program_targets_identity_and_keyed` | 0.1s |
| `tests/dataflow_training/data/test_shard_corpus.py` | 3 | 0.5s | `test_header_parse` | 0.2s |
| `tests/dataflow_training/training/surfaces/test_webapp_upload.py` | 2 | 0.5s | `test_simulate_schema_upload` | 0.3s |
| `tests/dataflow/service/test_service_events.py` | 2 | 0.5s | `test_event_coverage_and_reattach` | 0.3s |
| `tests/dataflow_training/tasks/test_staged_blocks.py` | 2 | 0.5s | `test_derived_recompute_excludes_boundary_work` | 0.2s |
| `tests/dataflow/runtime/test_backing_free.py` | 3 | 0.5s | `test_tight_backing_capacity_feasible_via_dead_free` | 0.2s |
| `tests/dataflow_training/training/lowering/test_persist_marker.py` | 2 | 0.5s | `test_llama3_marks_state_not_data` | 0.2s |
| `tests/dataflow_training/pretrain/test_tp_layouts.py` | 2 | 0.5s | `test_init_parity_shards_are_single_gpu_slices` | 0.2s |
| `tests/test_external_family.py` | 2 | 0.4s | `test_external_family_composes_with_service_path` | 0.2s |
| `tests/dataflow/runtime/test_reserve_inversion.py` | 2 | 0.4s | `test_caller_priority_prevents_poke_starvation` | 0.2s |
| `tests/dataflow/service/test_hostbw.py` | 2 | 0.4s | `test_probe_reports_positive_lanes` | 0.2s |
| `tests/dataflow_training/test_client_only.py` | 1 | 0.3s | `test_workload_tests_are_client_only` | 0.3s |
| `tests/dataflow/service/test_registration.py` | 2 | 0.3s | `test_register_all_resolves_every_family` | 0.2s |
| `tests/dataflow/runtime/test_view_lifetime.py` | 2 | 0.3s | `test_invalidate_evicts_cached_views` | 0.2s |
| `tests/dataflow/service/test_active_pools.py` | 2 | 0.3s | `test_active_pools_reports_live_pools_scoped_to_a_daemon` | 0.1s |
| `tests/dataflow_training/training/lowering/test_lowering_stability.py` | 1 | 0.3s | `test_lowered_programs_bit_identical` | 0.3s |
| `tests/dataflow/test_workload_blind.py` | 1 | 0.2s | `test_engine_tests_are_workload_blind` | 0.2s |
| `tests/dataflow_training/models/test_qwen35moe.py` | 1 | 0.2s | `test_qwen35moe_stage_context_completeness` | 0.2s |
| `tests/dataflow/service/test_store_allocator_concurrency.py` | 1 | 0.2s | `test_two_writer_allocator_invariants` | 0.2s |
| `tests/dataflow/runtime/test_trace_dict.py` | 1 | 0.2s | `test_trace_to_dict_covers_every_task_interval` | 0.2s |
| `tests/dataflow/checkpoint/test_record_boundary.py` | 1 | 0.2s | `test_checkpoint_package_is_workload_blind` | 0.2s |

## Individual tests at or above 1.0s

Grouped by file, slowest first within each. Tests under 1.0s are covered by the per-file totals above.

### `tests/dataflow_training/pretrain/test_engine_parity_families.py` — 298.8s total, 22 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 50.57s | 0.00s | 0.23s | 50.80s |
| `test_underfull_engine_vs_reference[qwen35_smoke_preset]` | 48.56s | 0.00s | 0.21s | 48.77s |
| `test_qwen35moe_engine_vs_reference` | 47.13s | 0.00s | 0.22s | 47.35s |
| `test_underfull_execute_padding_equivalence` | 11.29s | 0.00s | 0.25s | 11.54s |
| `test_underfull_engine_vs_reference[glm52_smoke_preset]` | 11.23s | 0.00s | 0.24s | 11.47s |
| `test_underfull_poisoned_tail_is_dead_bytes` | 10.46s | 0.00s | 0.25s | 10.71s |
| `test_underfull_engine_vs_reference[dsv32_smoke_preset]` | 10.08s | 0.00s | 0.22s | 10.30s |
| `test_underfull_engine_vs_reference[olmoe_smoke_preset]` | 9.79s | 0.00s | 0.23s | 10.02s |
| `test_underfull_engine_vs_reference[dsv3_smoke_preset]` | 9.70s | 0.00s | 0.24s | 9.94s |
| `test_underfull_engine_vs_reference[qwen3moe_smoke_preset]` | 9.69s | 0.00s | 0.23s | 9.92s |
| `test_underfull_engine_vs_reference[qwen3_smoke_preset]` | 8.92s | 0.00s | 0.23s | 9.15s |
| `test_underfull_engine_vs_reference[smoke_preset]` | 8.90s | 0.00s | 0.22s | 9.12s |
| `test_underfull_engine_vs_reference[gpt2_smoke_preset]` | 8.81s | 0.00s | 0.22s | 9.03s |
| `test_gpt2_docaware_engine_vs_reference` | 6.64s | 0.00s | 0.22s | 6.86s |
| `test_glm52_engine_vs_reference` | 6.54s | 0.00s | 0.25s | 6.79s |
| `test_dsv32_engine_vs_reference` | 5.47s | 0.00s | 0.24s | 5.71s |
| `test_olmoe_engine_vs_reference` | 5.40s | 0.00s | 0.21s | 5.61s |
| `test_olmoe_engine_vs_reference_lbl_on` | 5.35s | 0.00s | 0.22s | 5.57s |
| `test_qwen3moe_engine_vs_reference` | 5.20s | 0.00s | 0.22s | 5.42s |
| `test_dsv3_engine_vs_reference` | 5.15s | 0.00s | 0.22s | 5.37s |
| `test_qwen3_engine_vs_reference` | 4.51s | 0.00s | 0.20s | 4.71s |
| `test_gpt2_engine_vs_reference` | 4.41s | 0.00s | 0.22s | 4.63s |

### `tests/dataflow_training/models/test_model_families.py` — 184.6s total, 110 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_poison_on_free_changes_nothing[qwen35moe]` | 7.81s | 0.00s | 0.20s | 8.01s |
| `test_poison_on_free_changes_nothing[glm52]` | 7.77s | 0.00s | 0.20s | 7.97s |
| `test_poison_on_free_changes_nothing[qwen35]` | 7.70s | 0.00s | 0.19s | 7.89s |
| `test_poison_on_free_changes_nothing[olmoe]` | 7.66s | 0.00s | 0.18s | 7.84s |
| `test_poison_on_free_changes_nothing[dsv3]` | 7.63s | 0.00s | 0.20s | 7.83s |
| `test_poison_on_free_changes_nothing[qwen3moe]` | 7.61s | 0.00s | 0.20s | 7.81s |
| `test_poison_on_free_changes_nothing[dsv32]` | 7.59s | 0.00s | 0.21s | 7.80s |
| `test_poison_on_free_changes_nothing[qwen3]` | 7.48s | 0.01s | 0.21s | 7.70s |
| `test_poison_on_free_changes_nothing[gpt2]` | 7.45s | 0.01s | 0.21s | 7.67s |
| `test_golden_model_step[dsv3]` | 4.39s | 2.98s | 0.23s | 7.60s |
| `test_poison_on_free_changes_nothing[llama3]` | 7.36s | 0.01s | 0.20s | 7.57s |
| `test_golden_model_step[qwen35moe]` | 2.58s | 0.01s | 0.23s | 2.82s |
| `test_golden_model_step[qwen35]` | 2.40s | 0.00s | 0.22s | 2.62s |
| `test_reference_twin_build_is_stateless[qwen35moe]` | 2.22s | 0.00s | 0.23s | 2.45s |
| `test_grad_accum_two_rounds[qwen35moe]` | 2.08s | 0.00s | 0.21s | 2.29s |
| `test_reference_twin_build_is_stateless[qwen35]` | 2.04s | 0.00s | 0.20s | 2.24s |
| `test_grad_accum_two_rounds[qwen35]` | 1.90s | 0.00s | 0.20s | 2.10s |
| `test_golden_model_step_batch2_packed[qwen35moe]` | 1.58s | 0.00s | 0.23s | 1.81s |
| `test_golden_model_step_batch2_packed[olmoe]` | 1.39s | 0.01s | 0.23s | 1.63s |
| `test_golden_model_step_batch2_packed[glm52]` | 1.35s | 0.00s | 0.21s | 1.56s |
| `test_golden_model_step[olmoe]` | 1.32s | 0.00s | 0.21s | 1.53s |
| `test_golden_model_step_batch2_packed[qwen35]` | 1.29s | 0.00s | 0.21s | 1.50s |
| `test_reseed_restores_pristine_init[olmoe]` | 1.29s | 0.01s | 0.19s | 1.49s |
| `test_golden_model_step_batch2_packed[dsv3]` | 1.21s | 0.00s | 0.23s | 1.44s |
| `test_golden_model_step[qwen3moe]` | 1.21s | 0.00s | 0.23s | 1.44s |
| `test_golden_model_step_batch2_packed[dsv32]` | 1.21s | 0.00s | 0.22s | 1.43s |
| `test_golden_model_step_batch2_packed[qwen3moe]` | 1.22s | 0.00s | 0.21s | 1.43s |
| `test_golden_model_step[glm52]` | 1.21s | 0.00s | 0.22s | 1.43s |
| `test_golden_model_step_batch2_packed[gpt2]` | 1.16s | 0.00s | 0.23s | 1.39s |
| `test_golden_model_step[dsv32]` | 1.13s | 0.00s | 0.22s | 1.35s |
| `test_golden_model_step[gpt2]` | 1.11s | 0.00s | 0.21s | 1.32s |
| `test_reseed_restores_pristine_init[glm52]` | 1.11s | 0.00s | 0.20s | 1.31s |
| `test_golden_model_step_batch2_packed[llama3]` | 1.07s | 0.01s | 0.21s | 1.29s |
| `test_reseed_restores_pristine_init[qwen35moe]` | 1.08s | 0.00s | 0.20s | 1.28s |
| `test_reseed_restores_pristine_init[qwen3moe]` | 1.06s | 0.01s | 0.20s | 1.27s |
| `test_golden_model_step_batch2_packed[qwen3]` | 1.06s | 0.00s | 0.20s | 1.26s |
| `test_reseed_restores_pristine_init[llama3]` | 1.03s | 0.01s | 0.21s | 1.25s |
| `test_golden_model_step[qwen3]` | 1.01s | 0.00s | 0.21s | 1.22s |
| `test_reseed_restores_pristine_init[qwen35]` | 0.97s | 0.00s | 0.20s | 1.17s |
| `test_reseed_restores_pristine_init[dsv32]` | 0.97s | 0.00s | 0.20s | 1.17s |
| `test_reseed_restores_pristine_init[dsv3]` | 0.95s | 0.01s | 0.20s | 1.16s |
| `test_golden_model_step[llama3]` | 0.94s | 0.00s | 0.22s | 1.16s |
| `test_reseed_restores_pristine_init[qwen3]` | 0.90s | 0.02s | 0.21s | 1.13s |
| `test_reference_twin_build_is_stateless[olmoe]` | 0.92s | 0.00s | 0.20s | 1.12s |
| `test_reference_twin_build_is_stateless[glm52]` | 0.91s | 0.00s | 0.20s | 1.11s |
| `test_reseed_restores_pristine_init[gpt2]` | 0.88s | 0.01s | 0.20s | 1.09s |
| `test_measured_costs_replan_still_golden[olmoe]` | 0.87s | 0.01s | 0.20s | 1.08s |
| `test_reference_twin_build_is_stateless[qwen3moe]` | 0.84s | 0.00s | 0.22s | 1.06s |
| `test_grad_accum_two_rounds[olmoe]` | 0.79s | 0.00s | 0.23s | 1.02s |
| `test_reference_twin_build_is_stateless[dsv3]` | 0.78s | 0.01s | 0.22s | 1.01s |
| `test_grad_accum_two_rounds[glm52]` | 0.80s | 0.00s | 0.21s | 1.01s |
| `test_reference_twin_build_is_stateless[dsv32]` | 0.79s | 0.00s | 0.22s | 1.01s |

### `tests/dataflow_training/models/test_engine_vs_reference.py` — 124.7s total, 20 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_engine_matches_reference_uniform[qwen35moe]` | 20.19s | 0.00s | 0.19s | 20.38s |
| `test_engine_matches_reference_uniform[qwen35]` | 19.69s | 0.00s | 0.19s | 19.88s |
| `test_engine_matches_reference_ragged[qwen35moe]` | 11.73s | 0.00s | 0.22s | 11.95s |
| `test_engine_matches_reference_ragged[qwen35]` | 11.21s | 0.00s | 0.21s | 11.42s |
| `test_engine_matches_reference_ragged[glm52]` | 4.36s | 0.00s | 0.23s | 4.59s |
| `test_engine_matches_reference_uniform[glm52]` | 4.15s | 0.00s | 0.21s | 4.36s |
| `test_engine_matches_reference_ragged[olmoe]` | 3.90s | 0.00s | 0.22s | 4.12s |
| `test_engine_matches_reference_uniform[dsv32]` | 3.75s | 0.00s | 0.21s | 3.96s |
| `test_engine_matches_reference_ragged[dsv3]` | 3.74s | 0.00s | 0.20s | 3.94s |
| `test_engine_matches_reference_ragged[dsv32]` | 3.73s | 0.00s | 0.21s | 3.94s |
| `test_engine_matches_reference_uniform[qwen3moe]` | 3.66s | 0.00s | 0.22s | 3.88s |
| `test_engine_matches_reference_ragged[qwen3moe]` | 3.67s | 0.00s | 0.21s | 3.88s |
| `test_engine_matches_reference_uniform[dsv3]` | 3.68s | 0.00s | 0.19s | 3.87s |
| `test_engine_matches_reference_uniform[olmoe]` | 3.66s | 0.00s | 0.20s | 3.86s |
| `test_engine_matches_reference_uniform[gpt2]` | 3.28s | 0.00s | 0.21s | 3.49s |
| `test_engine_matches_reference_uniform[qwen3]` | 3.28s | 0.00s | 0.21s | 3.49s |
| `test_engine_matches_reference_ragged[qwen3]` | 3.26s | 0.00s | 0.23s | 3.49s |
| `test_engine_matches_reference_uniform[llama3]` | 3.23s | 0.00s | 0.21s | 3.44s |
| `test_engine_matches_reference_ragged[llama3]` | 3.18s | 0.00s | 0.21s | 3.39s |
| `test_engine_matches_reference_ragged[gpt2]` | 3.15s | 0.00s | 0.21s | 3.36s |

### `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` — 91.8s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_world2_moe_persistent_set_round_trips` | 30.82s | 0.00s | 0.24s | 31.06s |
| `test_world2_remapped_resume_bitwise` | 30.26s | 0.00s | 0.24s | 30.50s |
| `test_world2_resume_reproduces_tail_bitwise` | 29.98s | 0.00s | 0.23s | 30.21s |

### `tests/examples/test_rl_training.py` — 36.7s total, 6 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_rl_training_parity_ppo[qwen35]` | 6.96s | 0.00s | 0.16s | 7.12s |
| `test_rl_training_parity_ppo[glm52]` | 5.95s | 0.00s | 0.17s | 6.12s |
| `test_rl_training_parity_reinforce` | 5.95s | 0.00s | 0.16s | 6.11s |
| `test_rl_training_parity_ppo[dsv32]` | 5.74s | 0.00s | 0.17s | 5.91s |
| `test_rl_training_parity_ppo[qwen3moe]` | 5.60s | 0.00s | 0.16s | 5.76s |
| `test_rl_training_parity_ppo[llama3]` | 5.49s | 0.00s | 0.16s | 5.65s |

### `tests/dataflow_training/pretrain/test_client_model_step.py` — 30.0s total, 4 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_client_model_step_matches_in_process_olmoe` | 9.58s | 0.00s | 0.24s | 9.82s |
| `test_client_model_step_matches_in_process_llama3` | 8.87s | 0.00s | 0.23s | 9.10s |
| `test_client_model_step_llama3_passes` | 7.94s | 0.00s | 0.23s | 8.17s |
| `test_out_of_process_daemon_boots_and_reaps` | 2.68s | 0.00s | 0.21s | 2.89s |

### `tests/dataflow_training/data/test_data_pipeline.py` — 19.0s total, 22 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_checkpoint_resume_tail_matches_uninterrupted_run` | 9.25s | 0.00s | 0.19s | 9.44s |
| `test_shard_doc_mode_tokens_in_range_and_cursor_resume` | 2.65s | 0.00s | 0.16s | 2.81s |
| `test_legacy_doc_configuration_pinned` | 2.53s | 0.00s | 0.16s | 2.69s |

### `tests/dataflow_training/training/surfaces/test_replicate_load.py` — 14.0s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_world1_replicate_steps_once_bitwise` | 13.77s | 0.00s | 0.24s | 14.01s |

### `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` — 11.3s total, 2 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_kill_escalates_past_sigterm` | 10.56s | 0.00s | 0.23s | 10.79s |

### `tests/dataflow_training/tasks/test_kernels.py` — 7.9s total, 21 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_swiglu_fused[4099-14336]` | 1.46s | 0.00s | 0.28s | 1.74s |

### `tests/dataflow/runtime/test_profiling_memory.py` — 7.8s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_tables_leave_no_reserved_memory` | 7.65s | 0.00s | 0.15s | 7.80s |

### `tests/dataflow/service/test_shared_server_self_heal.py` — 7.8s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_self_heal_respawns_after_illegal_access` | 7.64s | 0.00s | 0.16s | 7.80s |

### `tests/dataflow/service/test_daemon_relaunch.py` — 6.9s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses` | 6.72s | 0.00s | 0.19s | 6.91s |

### `tests/dataflow_training/pretrain/test_client_fetch_surface.py` — 6.0s total, 2 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_client_fetch_surface_dense` | 3.01s | 0.00s | 0.22s | 3.23s |
| `test_client_fetch_surface_moe_aux` | 2.53s | 0.00s | 0.21s | 2.74s |

### `tests/dataflow/service/test_service_store.py` — 4.7s total, 16 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_real_boot_family_init_byte_identity` | 1.28s | 0.00s | 0.18s | 1.46s |

### `tests/dataflow_sim/app/test_server.py` — 4.7s total, 16 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_simulate_large_chain_uses_snapshot_free_response` | 1.87s | 0.00s | 0.18s | 2.05s |

### `tests/dataflow_training/training/surfaces/test_solo_resume_bitwise.py` — 4.5s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_solo_resume_reproduces_tail_bitwise` | 4.23s | 0.00s | 0.23s | 4.46s |

### `tests/dataflow_training/pretrain/test_parity_smoke.py` — 3.7s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_reference_vs_service_parity_smoke` | 3.47s | 0.00s | 0.23s | 3.70s |

### `tests/dataflow_training/training/surfaces/test_checkpoint_record.py` — 3.4s total, 4 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_load_checkpoint_targets` | 0.93s | 0.00s | 0.24s | 1.17s |

### `tests/dataflow/service/test_pinned_slab.py` — 3.2s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_slab_costs_what_it_asks_for` | 1.37s | 0.00s | 0.17s | 1.54s |
| `test_slab_frees_what_it_pinned` | 0.94s | 0.00s | 0.15s | 1.09s |

### `tests/dataflow/service/test_engine_determinism.py` — 2.9s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_same_daemon_rerun_bitwise` | 2.71s | 0.00s | 0.18s | 2.89s |

### `tests/dataflow/service/test_service_runs.py` — 2.4s total, 5 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_rebind_two_token_slabs` | 0.03s | 1.06s | 0.18s | 1.27s |

### `tests/dataflow/service/test_service_snapshot.py` — 2.3s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_checkpoint_roundtrip_bit_continuity` | 0.22s | 1.06s | 0.18s | 1.46s |

### `tests/dataflow/runtime/test_engine_stress.py` — 1.9s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_measured_costs_replan_still_golden` | 0.82s | 0.00s | 0.19s | 1.01s |

### `tests/test_program_hashes.py` — 1.2s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_lowered_program_hashes_stable` | 0.10s | 0.00s | 1.09s | 1.19s |

### `tests/dataflow/service/test_service_packed_args.py` — 1.1s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_daemon_packed_args_bit_equal` | 0.91s | 0.00s | 0.18s | 1.09s |

## Cross-box expectations

Banked full-suite wall times for the same suite (gate runs, not per-test measurements): chicago (RTX 5090) ~12.5-13.5 min; tubingen (RTX 3090) ~15-18 min; della H100 node ~17-20 min (slower CPU and GPFS dominate the non-GPU phases). Per-test ratios are NOT uniform across boxes — GPU-heavy tests track the device, data/IO-heavy tests track the filesystem.

## Methodology and caveats

- Source: `--durations=0` phase report; pytest omits phases under 5ms (`--durations-min` default), so per-file sums are floors and very cheap tests may be absent entirely.
- Session/module-scoped fixture cost lands in the setup phase of the first test that triggers it — a file's first test can look artificially heavy.
- Serial single-run measurement under the serial-battery rule: concurrent GPU work on the box invalidates comparisons (contention reds are not regressions).
- The deselected count is the opt-in lanes (fleet); their cost is not in this document.
