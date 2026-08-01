# Test-suite duration reference (chicago (RTX 5090))

Measured at the 2026-07-31 working tree (886c041 + uncommitted: lazy cursor, pack-ahead, sampling-floor opt-in, under-sampled guard) on chicago (RTX 5090), single serial run of the canonical suite invocation (`python -m pytest -q --durations=0`, `dataflow` conda env, box otherwise idle). Use it as the expectation baseline for how long suite tasks take at this point; re-measure after structural suite changes.

Stack: torch 2.13.0+cu130 / triton 3.7.1.

## Summary

- **Wall time: 14:16** (856s) — 1338 passed, 1 skipped.
- Time accounted to individual tests below: 845s (99% of wall; the rest is collection, session setup, and the many sub-5ms phases pytest omits).
- 118 test files with measurable time; distribution by per-file total: 5 files over 60s, 7 in 10-60s, 48 in 1-10s, 58 under 1s.
- Concentration: the top 10 files hold 691s (82% of accounted time) — they are the levers if the suite ever needs to get faster.

### Top 10 files

| # | file | tests | total | share of accounted |
|---|---|---|---|---|
| 1 | `tests/dataflow_training/pretrain/test_engine_parity_families.py` | 22 | 235.0s | 27.8% |
| 2 | `tests/dataflow_training/models/test_model_families.py` | 110 | 110.4s | 13.1% |
| 3 | `tests/dataflow/runtime/test_profiling_memory.py` | 1 | 77.7s | 9.2% |
| 4 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` | 3 | 65.0s | 7.7% |
| 5 | `tests/dataflow_training/models/test_engine_vs_reference.py` | 20 | 60.9s | 7.2% |
| 6 | `tests/dataflow_training/tasks/test_kernel_audit.py` | 371 | 55.0s | 6.5% |
| 7 | `tests/examples/test_rl_training.py` | 6 | 26.0s | 3.1% |
| 8 | `tests/dataflow_training/data/test_data_pipeline.py` | 23 | 21.6s | 2.6% |
| 9 | `tests/dataflow/runtime/test_engine_stress.py` | 3 | 20.7s | 2.5% |
| 10 | `tests/dataflow_training/pretrain/test_client_model_step.py` | 4 | 18.8s | 2.2% |

### Top 10 individual tests

| # | test | total |
|---|---|---|
| 1 | `tests/dataflow/runtime/test_profiling_memory.py::test_tables_leave_no_reserved_memory` | 77.7s |
| 2 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 31.4s |
| 3 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[qwen35_smoke_preset]` | 30.8s |
| 4 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_moe_persistent_set_round_trips` | 22.1s |
| 5 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_qwen35moe_engine_vs_reference` | 22.0s |
| 6 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_remapped_resume_bitwise` | 21.5s |
| 7 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_resume_reproduces_tail_bitwise` | 21.4s |
| 8 | `tests/dataflow/runtime/test_engine_stress.py::test_measured_costs_replan_still_golden` | 20.2s |
| 9 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_execute_padding_equivalence` | 13.0s |
| 10 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[glm52_smoke_preset]` | 12.8s |

## Per-file breakdown

Sorted by total attributed time (call + setup + teardown of every test in the file).

| file | tests | total | slowest test | its total |
|---|---|---|---|---|
| `tests/dataflow_training/pretrain/test_engine_parity_families.py` | 22 | 235.0s | `test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 31.4s |
| `tests/dataflow_training/models/test_model_families.py` | 110 | 110.4s | `test_poison_on_free_changes_nothing[glm52]` | 5.2s |
| `tests/dataflow/runtime/test_profiling_memory.py` | 1 | 77.7s | `test_tables_leave_no_reserved_memory` | 77.7s |
| `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` | 3 | 65.0s | `test_world2_moe_persistent_set_round_trips` | 22.1s |
| `tests/dataflow_training/models/test_engine_vs_reference.py` | 20 | 60.9s | `test_engine_matches_reference_uniform[qwen35moe]` | 9.5s |
| `tests/dataflow_training/tasks/test_kernel_audit.py` | 371 | 55.0s | `test_write_coverage_poison_invariance[rope_fwd:eager:huge_positions]` | 0.2s |
| `tests/examples/test_rl_training.py` | 6 | 26.0s | `test_rl_training_parity_ppo[qwen35]` | 4.8s |
| `tests/dataflow_training/data/test_data_pipeline.py` | 23 | 21.6s | `test_checkpoint_resume_tail_matches_uninterrupted_run` | 8.9s |
| `tests/dataflow/runtime/test_engine_stress.py` | 3 | 20.7s | `test_measured_costs_replan_still_golden` | 20.2s |
| `tests/dataflow_training/pretrain/test_client_model_step.py` | 4 | 18.8s | `test_client_model_step_matches_in_process_olmoe` | 6.0s |
| `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` | 2 | 11.1s | `test_kill_escalates_past_sigterm` | 10.7s |
| `tests/dataflow_training/training/surfaces/test_replicate_load.py` | 1 | 10.1s | `test_world1_replicate_steps_once_bitwise` | 10.1s |
| `tests/dataflow/service/test_shared_server_self_heal.py` | 1 | 4.9s | `test_self_heal_respawns_after_illegal_access` | 4.9s |
| `tests/dataflow_training/tasks/test_kernels.py` | 21 | 4.7s | `test_swiglu_fused[4099-14336]` | 1.0s |
| `tests/dataflow_sim/engine/test_simulator.py` | 38 | 3.8s | `test_single_task_emits_start_then_end` | 0.1s |
| `tests/dataflow/service/test_slice_snapshots.py` | 12 | 3.8s | `test_slice_roundtrip_and_compose` | 0.6s |
| `tests/dataflow_sim/planning/policies/test_auto_policy.py` | 36 | 3.7s | `test_auto_policy_L10_works_down_to_cap_500[500]` | 0.1s |
| `tests/dataflow/service/test_daemon_relaunch.py` | 1 | 3.5s | `test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses` | 3.5s |
| `tests/dataflow_training/training/lowering/test_layout_registry.py` | 21 | 3.0s | `test_registry_covers_external_family` | 0.2s |
| `tests/dataflow_training/modules/test_moe.py` | 24 | 3.0s | `test_grouped_mm_vs_dense_loop[False]` | 0.1s |
| `tests/dataflow/service/test_service_skeleton.py` | 10 | 3.0s | `test_fast_path_answers_while_dispatcher_held` | 0.8s |
| `tests/dataflow_sim/core/test_validate_chain.py` | 29 | 2.9s | `test_invalid_chain_rejected[make_invalid_release_mutation_release_then_later_reference-make_invalid_release_mutation_release_then_later_reference]` | 0.1s |
| `tests/dataflow_training/pretrain/test_client_fetch_surface.py` | 2 | 2.8s | `test_client_fetch_surface_dense` | 1.5s |
| `tests/dataflow/service/test_service_store.py` | 16 | 2.8s | `test_real_boot_family_init_byte_identity` | 0.8s |
| `tests/dataflow_training/training/e2e/test_varlen_e2e.py` | 11 | 2.7s | `test_model_step_ragged_matches_golden_all_families[qwen35moe]` | 0.5s |
| `tests/dataflow_sim/app/test_server.py` | 16 | 2.7s | `test_simulate_large_chain_uses_snapshot_free_response` | 1.1s |
| `tests/dataflow_training/training/e2e/test_freeze_plan.py` | 17 | 2.6s | `test_model_step_truncated_olmoe` | 0.2s |
| `tests/dataflow_sim/workloads/test_modular_workload_builder.py` | 24 | 2.6s | `test_constrained_memory_recompute_planning_selects_useful_variants` | 0.2s |
| `tests/dataflow_training/training/planning/test_planning.py` | 10 | 2.5s | `test_backing_capacity_drives_recompute` | 0.5s |
| `tests/dataflow_training/training/surfaces/test_solo_resume_bitwise.py` | 1 | 2.3s | `test_solo_resume_reproduces_tail_bitwise` | 2.3s |
| `tests/dataflow_training/pretrain/test_presets.py` | 16 | 2.3s | `test_preset_lowers[l3_350m]` | 0.2s |
| `tests/dataflow_sim/planning/policies/test_min_grow.py` | 22 | 2.2s | `test_derive_schedule_mutated_backing_init_offloads_at_mutator` | 0.1s |
| `tests/dataflow_training/training/surfaces/test_checkpoint_record.py` | 4 | 2.2s | `test_load_checkpoint_targets` | 0.6s |
| `tests/dataflow_training/pretrain/test_flops.py` | 15 | 2.1s | `test_every_family_walks[llama3]` | 0.1s |
| `tests/dataflow_training/pretrain/test_sharding.py` | 13 | 1.9s | `test_serialization_roundtrip_and_required_groups` | 0.1s |
| `tests/dataflow_training/pretrain/test_parity_smoke.py` | 1 | 1.8s | `test_reference_vs_service_parity_smoke` | 1.8s |
| `tests/dataflow_training/training/e2e/test_lbl_modes.py` | 8 | 1.8s | `test_retained_router_delta_is_ga_invariant_per_round_is_not` | 0.4s |
| `tests/dataflow/service/test_pinned_slab.py` | 3 | 1.7s | `test_slab_costs_what_it_asks_for` | 0.8s |
| `tests/dataflow/service/test_peer_protocol.py` | 17 | 1.7s | `test_exact_chunk_multiple_boundary_byte_identity` | 0.1s |
| `tests/dataflow_training/tasks/test_optim.py` | 11 | 1.7s | `test_muon_recipe_string_model_step_vs_hand_replica` | 0.2s |
| `tests/dataflow_training/training/e2e/test_dtype_policy_e2e.py` | 7 | 1.7s | `test_qwen35_model_step_mixed_policy` | 0.4s |
| `tests/test_import_boundaries.py` | 7 | 1.6s | `test_sim_required_only_under_lowering` | 0.4s |
| `tests/dataflow/service/test_engine_determinism.py` | 1 | 1.5s | `test_same_daemon_rerun_bitwise` | 1.5s |
| `tests/dataflow_training/modules/test_dsa.py` | 10 | 1.4s | `test_index_scores_vs_hand_loop` | 0.2s |
| `tests/dataflow_training/models/test_gpt2.py` | 10 | 1.4s | `test_model_step_nobias` | 0.2s |
| `tests/dataflow/service/test_service_snapshot.py` | 3 | 1.4s | `test_checkpoint_roundtrip_bit_continuity` | 0.8s |
| `tests/dataflow_training/models/test_glm52.py` | 7 | 1.4s | `test_glm52_grad_accum_two_rounds_matches_reference` | 0.3s |
| `tests/dataflow_training/models/test_block_isolation.py` | 5 | 1.4s | `test_isolated_block_at_floor[glm52-isolate0-6]` | 0.5s |
| `tests/dataflow/service/test_service_runs.py` | 5 | 1.3s | `test_rebind_two_token_slabs` | 0.7s |
| `tests/dataflow_training/training/lowering/test_shaped_program.py` | 9 | 1.3s | `test_8b_shape_totals` | 0.1s |
| `tests/dataflow_training/models/test_dsv32.py` | 7 | 1.3s | `test_dsv32_frozen_indexer_ablation` | 0.2s |
| `tests/dataflow/runtime/test_parity_vs_sim.py` | 9 | 1.2s | `test_parity_8b_starved_pcie_recompute` | 0.3s |
| `tests/dataflow_sim/planning/policies/test_pressurefit.py` | 12 | 1.2s | `test_pressurefit_diagnostics_describe_selected_candidate` | 0.1s |
| `tests/dataflow_training/training/e2e/test_packed_args_e2e.py` | 7 | 1.2s | `test_packed_args_with_forced_recompute` | 0.2s |
| `tests/dataflow_training/training/surfaces/test_source_policy_drills.py` | 3 | 1.2s | `test_simple_policy_round_trip_world2` | 0.5s |
| `tests/dataflow/runtime/test_engine_semantics.py` | 13 | 1.2s | `test_ledger_inversion_without_valve_deadlocks` | 0.1s |
| `tests/dataflow_training/models/test_qwen35.py` | 6 | 1.2s | `test_qwen35_tied_model_step_vs_golden` | 0.4s |
| `tests/dataflow_training/models/test_llama3.py` | 9 | 1.1s | `test_model_step_muon_policy_golden_parity` | 0.2s |
| `tests/dataflow_training/data/test_packing.py` | 10 | 1.1s | `test_token_conservation_multiset_identity` | 0.1s |
| `tests/test_program_hashes.py` | 1 | 1.0s | `test_lowered_program_hashes_stable` | 1.0s |
| `tests/dataflow/core/test_ir_validate.py` | 11 | 1.0s | `test_valid_minimal_program` | 0.1s |
| `tests/dataflow/runtime/test_cuda_backend.py` | 5 | 0.9s | `test_mini_program_execution_matches_plan` | 0.4s |
| `tests/dataflow/runtime/test_run_contract.py` | 10 | 0.9s | `test_outcome_serializes_uniformly` | 0.1s |
| `tests/dataflow_sim/workloads/test_dataflow_schema.py` | 9 | 0.9s | `test_compute_block_summary_reports_total_effective_tflops` | 0.1s |
| `tests/dataflow_training/training/e2e/test_ga_invariance.py` | 4 | 0.9s | `test_sgd_rounds_are_memory_optimization` | 0.3s |
| `tests/dataflow_training/tasks/test_varlen_attention.py` | 6 | 0.9s | `test_bwd_bitlevel_segment_isolation` | 0.1s |
| `tests/dataflow/service/test_nccl_binding.py` | 2 | 0.9s | `test_binding_world1_roundtrip` | 0.8s |
| `tests/dataflow_training/modules/test_mla.py` | 7 | 0.9s | `test_dsv3_block_fwd_recompute_bwd_accum_match_autograd_golden[moe]` | 0.1s |
| `tests/dataflow/checkpoint/test_record_layer.py` | 9 | 0.8s | `test_read_refuses_incomplete_and_foreign` | 0.1s |
| `tests/dataflow_training/training/lowering/test_responsibility.py` | 6 | 0.8s | `test_dedup_policy_projection` | 0.1s |
| `tests/dataflow_sim/planning/test_recompute.py` | 8 | 0.8s | `test_recompute_loop_converts_under_pressure_and_improves` | 0.1s |
| `tests/dataflow_training/pretrain/test_reference_muon.py` | 5 | 0.8s | `test_tiny_muon_reference_trains` | 0.2s |
| `tests/dataflow_training/pretrain/test_sharding_lowering.py` | 5 | 0.8s | `test_tp_lowering_params_sizes_and_serialization` | 0.2s |
| `tests/dataflow/runtime/test_placement.py` | 7 | 0.8s | `test_parity_with_placement_8b` | 0.2s |
| `tests/dataflow_training/models/test_qwen3moe.py` | 5 | 0.7s | `test_qwen3moe_aux_zero_model_step_vs_golden` | 0.2s |
| `tests/dataflow_training/tasks/test_dtype_policy.py` | 5 | 0.7s | `test_default_policy_is_all_bf16` | 0.1s |
| `tests/dataflow/runtime/test_vmm.py` | 7 | 0.7s | `test_e2e_mini_vmm_matches_static` | 0.1s |
| `tests/dataflow_training/pretrain/test_topology.py` | 3 | 0.7s | `test_daemonize_detach_and_group_kill` | 0.4s |
| `tests/dataflow_training/models/test_dsv3.py` | 4 | 0.7s | `test_dsv3_aux_zero_model_step_vs_golden` | 0.2s |
| `tests/dataflow_training/models/test_glm52_lowering.py` | 5 | 0.6s | `test_full_scale_presets_lower` | 0.1s |
| `tests/dataflow_training/models/test_olmoe.py` | 4 | 0.6s | `test_olmoe_grad_accum_two_rounds_matches_reference` | 0.2s |
| `tests/dataflow_training/pretrain/test_schedule.py` | 4 | 0.6s | `test_recipe_hyper_spec_matches_base_hyper` | 0.1s |
| `tests/dataflow_training/tasks/test_ignore_index_ce.py` | 4 | 0.6s | `test_no_ignore_rows_matches_torch_ce_and_rerun_is_bitwise[eager]` | 0.1s |
| `tests/dataflow_training/training/surfaces/test_plugins.py` | 4 | 0.6s | `test_entry_point_discovery` | 0.1s |
| `tests/dataflow/service/test_service_packed_args.py` | 1 | 0.6s | `test_daemon_packed_args_bit_equal` | 0.6s |
| `tests/dataflow_training/training/lowering/test_parallelism_scheme.py` | 4 | 0.6s | `test_solo_is_the_empty_mesh` | 0.1s |
| `tests/test_docstring_index.py` | 2 | 0.6s | `test_index_matches_test_functions` | 0.3s |
| `tests/dataflow/core/test_sim_convert.py` | 6 | 0.5s | `test_annotated_chain_validates` | 0.1s |
| `tests/dataflow_sim/core/test_reference_stream.py` | 5 | 0.5s | `test_compute_reference_stream_includes_outputs_at_producer_time` | 0.1s |
| `tests/dataflow_training/models/test_qwen3.py` | 4 | 0.5s | `test_qwen3_block_backward` | 0.1s |
| `tests/dataflow_training/training/lowering/test_round_prologue.py` | 3 | 0.5s | `test_round_prologue_publishes_round_index_via_run_values_and_object` | 0.2s |
| `tests/dataflow/checkpoint/test_record_targets.py` | 5 | 0.4s | `test_id_targets_subset` | 0.1s |
| `tests/dataflow/core/test_json_roundtrip.py` | 5 | 0.4s | `test_grad_accum_variant_roundtrips` | 0.1s |
| `tests/dataflow_training/training/lowering/test_group_annotation.py` | 3 | 0.4s | `test_zero1rs_annotation_matches_builder` | 0.1s |
| `tests/dataflow/service/test_service_events.py` | 2 | 0.4s | `test_event_coverage_and_reattach` | 0.2s |
| `tests/dataflow/service/test_peer_groups.py` | 2 | 0.4s | `test_group_lifecycle_and_error_fanout` | 0.2s |
| `tests/dataflow/checkpoint/test_persistent_targets.py` | 4 | 0.4s | `test_persisted_objects_filter` | 0.1s |
| `tests/dataflow/service/test_error_codes.py` | 1 | 0.3s | `test_every_raised_error_code_is_registered` | 0.3s |
| `tests/dataflow_training/training/e2e/test_batch_ga.py` | 2 | 0.3s | `test_batch_ga_model_step_matches_reference` | 0.2s |
| `tests/dataflow_training/data/test_shard_corpus.py` | 3 | 0.3s | `test_read_spans_shard_boundary` | 0.1s |
| `tests/dataflow_training/training/surfaces/test_webapp_upload.py` | 2 | 0.3s | `test_simulate_schema_upload` | 0.2s |
| `tests/dataflow/runtime/test_reserve_inversion.py` | 2 | 0.3s | `test_caller_priority_prevents_poke_starvation` | 0.2s |
| `tests/test_external_family.py` | 2 | 0.3s | `test_external_family_composes_with_service_path` | 0.1s |
| `tests/dataflow_training/pretrain/test_tp_layouts.py` | 2 | 0.3s | `test_init_parity_shards_are_single_gpu_slices` | 0.1s |
| `tests/dataflow_training/training/lowering/test_persist_marker.py` | 2 | 0.3s | `test_llama3_marks_state_not_data` | 0.1s |
| `tests/dataflow_training/tasks/test_staged_blocks.py` | 2 | 0.3s | `test_derived_recompute_excludes_boundary_work` | 0.1s |
| `tests/dataflow/runtime/test_backing_free.py` | 3 | 0.3s | `test_final_location_backing_survives` | 0.1s |
| `tests/dataflow/service/test_hostbw.py` | 2 | 0.2s | `test_probe_reports_positive_lanes` | 0.1s |
| `tests/dataflow/service/test_registration.py` | 2 | 0.2s | `test_register_all_resolves_every_family` | 0.1s |
| `tests/dataflow_training/test_client_only.py` | 1 | 0.2s | `test_workload_tests_are_client_only` | 0.2s |
| `tests/dataflow/runtime/test_view_lifetime.py` | 2 | 0.2s | `test_free_evicts_cache_no_stale_view` | 0.1s |
| `tests/dataflow/runtime/test_trace_dict.py` | 2 | 0.2s | `test_dispatch_records_cover_every_task_in_order` | 0.1s |
| `tests/dataflow/service/test_active_pools.py` | 2 | 0.2s | `test_active_pools_reports_live_pools_scoped_to_a_daemon` | 0.1s |
| `tests/dataflow_training/training/lowering/test_lowering_stability.py` | 1 | 0.2s | `test_lowered_programs_bit_identical` | 0.2s |
| `tests/dataflow/test_workload_blind.py` | 1 | 0.1s | `test_engine_tests_are_workload_blind` | 0.1s |
| `tests/dataflow_training/models/test_qwen35moe.py` | 1 | 0.1s | `test_qwen35moe_stage_context_completeness` | 0.1s |
| `tests/dataflow/service/test_store_allocator_concurrency.py` | 1 | 0.1s | `test_two_writer_allocator_invariants` | 0.1s |
| `tests/dataflow/checkpoint/test_record_boundary.py` | 1 | 0.1s | `test_checkpoint_package_is_workload_blind` | 0.1s |

## Per-test detail

### `tests/dataflow_training/pretrain/test_engine_parity_families.py` — 235.0s total, 22 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 31.27s | 0.00s | 0.14s | 31.41s |
| `test_underfull_engine_vs_reference[qwen35_smoke_preset]` | 30.64s | 0.00s | 0.14s | 30.78s |
| `test_qwen35moe_engine_vs_reference` | 21.87s | 0.00s | 0.14s | 22.01s |
| `test_underfull_execute_padding_equivalence` | 12.84s | 0.00s | 0.15s | 12.99s |
| `test_underfull_engine_vs_reference[glm52_smoke_preset]` | 12.70s | 0.00s | 0.15s | 12.85s |
| `test_underfull_poisoned_tail_is_dead_bytes` | 12.56s | 0.00s | 0.15s | 12.71s |
| `test_underfull_engine_vs_reference[dsv32_smoke_preset]` | 12.20s | 0.00s | 0.16s | 12.36s |
| `test_underfull_engine_vs_reference[dsv3_smoke_preset]` | 12.06s | 0.00s | 0.15s | 12.21s |
| `test_underfull_engine_vs_reference[olmoe_smoke_preset]` | 12.05s | 0.00s | 0.14s | 12.19s |
| `test_underfull_engine_vs_reference[qwen3moe_smoke_preset]` | 11.99s | 0.00s | 0.15s | 12.14s |
| `test_underfull_engine_vs_reference[qwen3_smoke_preset]` | 11.64s | 0.00s | 0.14s | 11.78s |
| `test_underfull_engine_vs_reference[smoke_preset]` | 11.62s | 0.00s | 0.14s | 11.76s |
| `test_underfull_engine_vs_reference[gpt2_smoke_preset]` | 11.58s | 0.00s | 0.14s | 11.72s |
| `test_gpt2_docaware_engine_vs_reference` | 6.81s | 0.00s | 0.13s | 6.94s |
| `test_glm52_engine_vs_reference` | 3.08s | 0.00s | 0.14s | 3.22s |
| `test_dsv32_engine_vs_reference` | 2.63s | 0.00s | 0.14s | 2.77s |
| `test_olmoe_engine_vs_reference_lbl_on` | 2.57s | 0.00s | 0.14s | 2.71s |
| `test_olmoe_engine_vs_reference` | 2.57s | 0.00s | 0.13s | 2.70s |
| `test_qwen3moe_engine_vs_reference` | 2.50s | 0.00s | 0.13s | 2.63s |
| `test_dsv3_engine_vs_reference` | 2.48s | 0.00s | 0.13s | 2.61s |
| `test_qwen3_engine_vs_reference` | 2.15s | 0.00s | 0.14s | 2.29s |
| `test_gpt2_engine_vs_reference` | 2.11s | 0.00s | 0.13s | 2.24s |

### `tests/dataflow_training/models/test_model_families.py` — 110.4s total, 110 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_poison_on_free_changes_nothing[glm52]` | 5.07s | 0.00s | 0.12s | 5.19s |
| `test_poison_on_free_changes_nothing[qwen35moe]` | 5.00s | 0.00s | 0.13s | 5.13s |
| `test_poison_on_free_changes_nothing[dsv32]` | 4.98s | 0.00s | 0.13s | 5.11s |
| `test_poison_on_free_changes_nothing[dsv3]` | 4.96s | 0.01s | 0.13s | 5.10s |
| `test_poison_on_free_changes_nothing[qwen35]` | 4.93s | 0.00s | 0.13s | 5.06s |
| `test_poison_on_free_changes_nothing[olmoe]` | 4.91s | 0.00s | 0.13s | 5.04s |
| `test_poison_on_free_changes_nothing[qwen3moe]` | 4.80s | 0.00s | 0.13s | 4.93s |
| `test_poison_on_free_changes_nothing[gpt2]` | 4.79s | 0.00s | 0.13s | 4.92s |
| `test_poison_on_free_changes_nothing[qwen3]` | 4.78s | 0.00s | 0.13s | 4.91s |
| `test_poison_on_free_changes_nothing[llama3]` | 4.77s | 0.00s | 0.13s | 4.90s |
| `test_golden_model_step[dsv3]` | 2.75s | 1.94s | 0.14s | 4.83s |
| `test_golden_model_step[qwen35moe]` | 1.20s | 0.00s | 0.13s | 1.33s |
| `test_golden_model_step[qwen35]` | 1.17s | 0.00s | 0.13s | 1.30s |
| `test_reference_twin_build_is_stateless[qwen35moe]` | 1.09s | 0.00s | 0.13s | 1.22s |
| `test_grad_accum_two_rounds[qwen35moe]` | 1.04s | 0.00s | 0.13s | 1.17s |
| `test_reference_twin_build_is_stateless[qwen35]` | 1.02s | 0.00s | 0.13s | 1.15s |
| `test_grad_accum_two_rounds[qwen35]` | 0.98s | 0.00s | 0.13s | 1.11s |
| `test_golden_model_step_batch2_packed[qwen35moe]` | 0.80s | 0.00s | 0.13s | 0.93s |
| `test_reseed_restores_pristine_init[olmoe]` | 0.74s | 0.01s | 0.13s | 0.88s |
| `test_golden_model_step_batch2_packed[qwen35]` | 0.66s | 0.00s | 0.13s | 0.79s |
| `test_golden_model_step[glm52]` | 0.66s | 0.00s | 0.13s | 0.79s |
| `test_golden_model_step[olmoe]` | 0.66s | 0.00s | 0.13s | 0.79s |
| `test_golden_model_step_batch2_packed[olmoe]` | 0.65s | 0.00s | 0.13s | 0.78s |
| `test_golden_model_step_batch2_packed[glm52]` | 0.65s | 0.00s | 0.13s | 0.78s |
| `test_golden_model_step[dsv32]` | 0.64s | 0.00s | 0.13s | 0.77s |
| `test_reseed_restores_pristine_init[glm52]` | 0.61s | 0.01s | 0.12s | 0.74s |
| `test_reseed_restores_pristine_init[qwen35moe]` | 0.61s | 0.01s | 0.12s | 0.74s |
| `test_reseed_restores_pristine_init[qwen3moe]` | 0.59s | 0.01s | 0.13s | 0.73s |
| `test_golden_model_step_batch2_packed[qwen3moe]` | 0.58s | 0.00s | 0.13s | 0.71s |
| `test_reseed_restores_pristine_init[dsv32]` | 0.56s | 0.01s | 0.13s | 0.70s |
| `test_golden_model_step[qwen3moe]` | 0.55s | 0.00s | 0.13s | 0.68s |
| `test_golden_model_step_batch2_packed[dsv32]` | 0.55s | 0.00s | 0.13s | 0.68s |
| `test_reseed_restores_pristine_init[dsv3]` | 0.54s | 0.01s | 0.13s | 0.68s |
| `test_golden_model_step_batch2_packed[dsv3]` | 0.54s | 0.00s | 0.13s | 0.67s |
| `test_reseed_restores_pristine_init[llama3]` | 0.53s | 0.01s | 0.13s | 0.67s |
| `test_reseed_restores_pristine_init[gpt2]` | 0.53s | 0.01s | 0.13s | 0.67s |
| `test_reseed_restores_pristine_init[qwen35]` | 0.52s | 0.01s | 0.13s | 0.66s |
| `test_reseed_restores_pristine_init[qwen3]` | 0.52s | 0.01s | 0.13s | 0.66s |
| `test_golden_model_step_batch2_packed[qwen3]` | 0.52s | 0.00s | 0.13s | 0.65s |
| `test_golden_model_step[gpt2]` | 0.52s | 0.00s | 0.13s | 0.65s |
| `test_golden_model_step_batch2_packed[gpt2]` | 0.52s | 0.00s | 0.12s | 0.64s |
| `test_reference_twin_build_is_stateless[olmoe]` | 0.51s | 0.00s | 0.13s | 0.64s |
| `test_golden_model_step_batch2_packed[llama3]` | 0.50s | 0.00s | 0.13s | 0.63s |
| `test_golden_model_step[qwen3]` | 0.50s | 0.00s | 0.13s | 0.63s |
| `test_golden_model_step[llama3]` | 0.49s | 0.00s | 0.13s | 0.62s |
| `test_reference_twin_build_is_stateless[glm52]` | 0.49s | 0.00s | 0.13s | 0.62s |
| `test_grad_accum_two_rounds[glm52]` | 0.46s | 0.00s | 0.13s | 0.59s |
| `test_grad_accum_two_rounds[olmoe]` | 0.46s | 0.00s | 0.13s | 0.59s |
| `test_reference_twin_build_is_stateless[qwen3moe]` | 0.45s | 0.00s | 0.13s | 0.58s |
| `test_reference_twin_build_is_stateless[dsv32]` | 0.44s | 0.00s | 0.13s | 0.57s |
| `test_reference_twin_build_is_stateless[llama3]` | 0.43s | 0.00s | 0.13s | 0.56s |
| `test_reference_twin_build_is_stateless[dsv3]` | 0.42s | 0.01s | 0.13s | 0.56s |
| `test_grad_accum_two_rounds[dsv32]` | 0.41s | 0.00s | 0.13s | 0.54s |
| `test_reference_twin_build_is_stateless[gpt2]` | 0.41s | 0.00s | 0.13s | 0.54s |
| `test_grad_accum_two_rounds[qwen3moe]` | 0.41s | 0.00s | 0.13s | 0.54s |
| `test_reference_twin_build_is_stateless[qwen3]` | 0.41s | 0.00s | 0.12s | 0.53s |
| `test_measured_costs_replan_still_golden[olmoe]` | 0.40s | 0.01s | 0.12s | 0.53s |
| `test_plan_invariance[glm52]` | 0.39s | 0.01s | 0.13s | 0.53s |
| `test_result_invariant_to_runtime_jitter[qwen35moe]` | 0.38s | 0.00s | 0.13s | 0.51s |
| `test_fixed_seed_bitwise_deterministic[olmoe]` | 0.38s | 0.00s | 0.13s | 0.51s |
| `test_measured_costs_replan_still_golden[qwen3moe]` | 0.37s | 0.01s | 0.13s | 0.51s |
| `test_plan_invariance[olmoe]` | 0.37s | 0.01s | 0.13s | 0.51s |
| `test_result_invariant_to_runtime_jitter[olmoe]` | 0.37s | 0.00s | 0.13s | 0.50s |
| `test_grad_accum_two_rounds[dsv3]` | 0.37s | 0.00s | 0.13s | 0.50s |
| `test_fixed_seed_bitwise_deterministic[glm52]` | 0.36s | 0.01s | 0.13s | 0.50s |
| `test_measured_costs_replan_still_golden[glm52]` | 0.36s | 0.01s | 0.13s | 0.50s |
| `test_fixed_seed_bitwise_deterministic[qwen3moe]` | 0.34s | 0.01s | 0.14s | 0.49s |
| `test_result_invariant_to_runtime_jitter[glm52]` | 0.36s | 0.00s | 0.13s | 0.49s |
| `test_fixed_seed_bitwise_deterministic[qwen35moe]` | 0.36s | 0.00s | 0.13s | 0.49s |
| `test_measured_costs_replan_still_golden[qwen35moe]` | 0.35s | 0.01s | 0.13s | 0.49s |
| `test_plan_invariance[qwen35moe]` | 0.35s | 0.01s | 0.13s | 0.49s |
| `test_measured_costs_replan_still_golden[qwen3]` | 0.34s | 0.01s | 0.13s | 0.48s |
| `test_plan_invariance[dsv32]` | 0.34s | 0.01s | 0.13s | 0.48s |
| `test_grad_accum_two_rounds[gpt2]` | 0.35s | 0.00s | 0.13s | 0.48s |
| `test_fixed_seed_bitwise_deterministic[dsv32]` | 0.35s | 0.00s | 0.13s | 0.48s |
| `test_grad_accum_two_rounds[llama3]` | 0.35s | 0.00s | 0.13s | 0.48s |
| `test_grad_accum_two_rounds[qwen3]` | 0.34s | 0.00s | 0.13s | 0.47s |
| `test_plan_invariance[qwen3moe]` | 0.33s | 0.01s | 0.13s | 0.47s |
| `test_plan_invariance[dsv3]` | 0.33s | 0.01s | 0.13s | 0.47s |
| `test_result_invariant_to_runtime_jitter[qwen3moe]` | 0.33s | 0.00s | 0.13s | 0.46s |
| `test_result_invariant_to_runtime_jitter[qwen35]` | 0.33s | 0.00s | 0.13s | 0.46s |
| `test_result_invariant_to_runtime_jitter[gpt2]` | 0.32s | 0.01s | 0.13s | 0.46s |
| `test_measured_costs_replan_still_golden[llama3]` | 0.32s | 0.01s | 0.13s | 0.46s |
| `test_measured_costs_replan_still_golden[dsv32]` | 0.31s | 0.01s | 0.13s | 0.45s |
| `test_plan_invariance[qwen3]` | 0.31s | 0.01s | 0.13s | 0.45s |
| `test_measured_costs_replan_still_golden[dsv3]` | 0.30s | 0.01s | 0.14s | 0.45s |
| `test_result_invariant_to_runtime_jitter[dsv32]` | 0.32s | 0.00s | 0.12s | 0.44s |
| `test_result_invariant_to_runtime_jitter[dsv3]` | 0.31s | 0.00s | 0.13s | 0.44s |
| `test_fixed_seed_bitwise_deterministic[qwen35]` | 0.31s | 0.00s | 0.13s | 0.44s |
| `test_result_invariant_to_runtime_jitter[llama3]` | 0.31s | 0.00s | 0.13s | 0.44s |
| `test_measured_costs_replan_still_golden[qwen35]` | 0.31s | 0.00s | 0.13s | 0.44s |
| `test_plan_invariance[qwen35]` | 0.30s | 0.01s | 0.13s | 0.44s |
| `test_plan_invariance[gpt2]` | 0.28s | 0.01s | 0.13s | 0.42s |
| `test_plan_invariance[llama3]` | 0.28s | 0.01s | 0.13s | 0.42s |
| `test_result_invariant_to_runtime_jitter[qwen3]` | 0.28s | 0.01s | 0.13s | 0.42s |
| `test_fixed_seed_bitwise_deterministic[qwen3]` | 0.29s | 0.01s | 0.12s | 0.42s |
| `test_fixed_seed_bitwise_deterministic[dsv3]` | 0.29s | 0.00s | 0.13s | 0.42s |
| `test_fixed_seed_bitwise_deterministic[llama3]` | 0.29s | 0.00s | 0.13s | 0.42s |
| `test_measured_costs_replan_still_golden[gpt2]` | 0.29s | 0.01s | 0.12s | 0.42s |
| `test_fixed_seed_bitwise_deterministic[gpt2]` | 0.28s | 0.00s | 0.13s | 0.41s |
| `test_lowering_validates_and_plans[qwen35]` | 0.00s | 0.00s | 0.13s | 0.13s |
| `test_lowering_validates_and_plans[dsv32]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_lowering_validates_and_plans[dsv3]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_lowering_validates_and_plans[qwen35moe]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_lowering_validates_and_plans[glm52]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_lowering_validates_and_plans[qwen3moe]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_lowering_validates_and_plans[olmoe]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_lowering_validates_and_plans[gpt2]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_lowering_validates_and_plans[qwen3]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_lowering_validates_and_plans[llama3]` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow/runtime/test_profiling_memory.py` — 77.7s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_tables_leave_no_reserved_memory` | 77.61s | 0.00s | 0.10s | 77.71s |

### `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` — 65.0s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_world2_moe_persistent_set_round_trips` | 21.91s | 0.00s | 0.15s | 22.06s |
| `test_world2_remapped_resume_bitwise` | 21.31s | 0.00s | 0.16s | 21.47s |
| `test_world2_resume_reproduces_tail_bitwise` | 21.29s | 0.00s | 0.15s | 21.44s |

### `tests/dataflow_training/models/test_engine_vs_reference.py` — 60.9s total, 20 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_engine_matches_reference_uniform[qwen35moe]` | 9.40s | 0.00s | 0.14s | 9.54s |
| `test_engine_matches_reference_uniform[qwen35]` | 9.34s | 0.00s | 0.13s | 9.47s |
| `test_engine_matches_reference_ragged[qwen35moe]` | 5.53s | 0.00s | 0.13s | 5.66s |
| `test_engine_matches_reference_ragged[qwen35]` | 5.20s | 0.00s | 0.13s | 5.33s |
| `test_engine_matches_reference_ragged[glm52]` | 2.15s | 0.00s | 0.13s | 2.28s |
| `test_engine_matches_reference_uniform[glm52]` | 2.07s | 0.00s | 0.13s | 2.20s |
| `test_engine_matches_reference_ragged[dsv32]` | 1.94s | 0.00s | 0.13s | 2.07s |
| `test_engine_matches_reference_ragged[olmoe]` | 1.89s | 0.00s | 0.13s | 2.02s |
| `test_engine_matches_reference_uniform[olmoe]` | 1.88s | 0.00s | 0.13s | 2.01s |
| `test_engine_matches_reference_uniform[dsv32]` | 1.88s | 0.00s | 0.12s | 2.00s |
| `test_engine_matches_reference_ragged[qwen3moe]` | 1.86s | 0.00s | 0.13s | 1.99s |
| `test_engine_matches_reference_uniform[qwen3moe]` | 1.85s | 0.00s | 0.13s | 1.98s |
| `test_engine_matches_reference_ragged[dsv3]` | 1.82s | 0.00s | 0.13s | 1.95s |
| `test_engine_matches_reference_uniform[dsv3]` | 1.80s | 0.00s | 0.13s | 1.93s |
| `test_engine_matches_reference_uniform[qwen3]` | 1.65s | 0.00s | 0.13s | 1.78s |
| `test_engine_matches_reference_uniform[gpt2]` | 1.63s | 0.00s | 0.12s | 1.75s |
| `test_engine_matches_reference_ragged[qwen3]` | 1.62s | 0.00s | 0.13s | 1.75s |
| `test_engine_matches_reference_ragged[llama3]` | 1.60s | 0.00s | 0.13s | 1.73s |
| `test_engine_matches_reference_uniform[llama3]` | 1.60s | 0.00s | 0.13s | 1.73s |
| `test_engine_matches_reference_ragged[gpt2]` | 1.57s | 0.00s | 0.13s | 1.70s |

### `tests/dataflow_training/tasks/test_kernel_audit.py` — 55.0s total, 371 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_write_coverage_poison_invariance[rope_fwd:eager:huge_positions]` | 0.00s | 0.00s | 0.18s | 0.18s |
| `test_degenerate_finite[embed_bwd_accum:triton:fresh]` | 0.00s | 0.00s | 0.16s | 0.16s |
| `test_write_coverage_poison_invariance[dsa_sparse_attn_bwd:triton:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_write_coverage_poison_invariance[dsa_probs_sum:triton:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_degenerate_cross_impl[dsa_probs_sum:triton-vs-eager:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_write_coverage_poison_invariance[dsa_sparse_attn_fwd:triton:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_degenerate_cross_impl[dsa_sparse_attn_fwd:triton-vs-eager:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_write_coverage_poison_invariance[dsa_sparse_attn_bwd:eager:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_degenerate_cross_impl[dsa_sparse_attn_bwd:triton-vs-eager:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_write_coverage_poison_invariance[dsa_probs_sum:eager:ragged_with_len1]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_degenerate_finite[muon_step:aten:zero_grad]` | 0.00s | 0.01s | 0.15s | 0.16s |
| `test_degenerate_finite[moe_grouped_mm_fwd:aten-grouped:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_bwd:eager:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_fwd_out:triton:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:eager:k_equals_e]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[embed_bwd_accum:triton:all_same_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_pack_bits:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:eager:total_ties]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_fwd_out:triton:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_router_bwd:eager:topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[layernorm_apply:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_sparse_attn_bwd:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_dispatch_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_sigmoid_noaux:eager:extreme_bias]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[adamw_step:triton-vs-eager:first_step]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_router_bwd:triton-vs-eager:softmax_then_topk]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gated_rmsnorm_fwd:eager:saturated_gate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_dgrad:triton:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[dsa_index_scores:triton-vs-eager:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_bwd:eager:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_topk:eager:k_exceeds_short_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_packed_fwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rope_fwd:triton-vs-eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_dispatch_bwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_wgrad:triton-vs-eager:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gated_rmsnorm_fwd:fla-fused:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_aux_lb_grad:eager:max_imbalance]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[adamw_step:triton:huge_grad]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_bwd:triton:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_fwd_out:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[causal_conv1d_silu_bwd:fla-triton-vs-eager:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_sparse_attn_fwd:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:triton:total_ties]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[layernorm_bwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[gated_rmsnorm_fwd:fla-fused-vs-eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_index_scores:eager:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_wgrad:aten-grouped:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_apply:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[causal_conv1d_silu_fwd:fla-triton:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_apply:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[ce_loss_fwd_bwd:triton-vs-eager:all_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_sort:aten:uniform]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_fwd:triton:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_topk_softmax:triton-vs-eager:k_equals_e]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rmsnorm_apply:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_wgrad:eager:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_sparse_attn_fwd:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_index_scores:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rope_bwd:eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[layernorm_fwd:triton-vs-eager:constant_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_bwd:triton:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_fwd:aten-grouped-vs-eager:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gelu_fwd_out:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_noweight:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rmsnorm_bwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_scale_rows:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_fwd:aten-grouped-vs-eager:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_apply:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rope_fwd:eager:huge_positions]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_packed_fwd:triton:single]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_bwd:eager:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rope_bwd:triton-vs-eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:triton:mode_topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:eager:odd_vocab]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rmsnorm_noweight:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_packed_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[embed_bwd_accum:triton:accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_dgrad:eager:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_topk_softmax:triton-vs-eager:mode_topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_combine_fwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gelu_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:eager:extreme_logits]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[ce_loss_fwd_bwd:triton-vs-eager:extreme_logits]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gelu_bwd:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_aux_lb_grad:eager:single_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_sort:aten:all_one_expert_rest_empty]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_packed_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[adamw_step:eager:huge_grad]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_probs_sum:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_packed_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_fwd:triton-vs-eager:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[dsa_index_bwd:triton-vs-eager:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[embed_bwd_accum:triton:fresh]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_scale_rows:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[dsa_sparse_attn_fwd:triton-vs-eager:self_only_early_tiles_dead]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[causal_conv1d_silu_fwd:fla-triton:seqs_shorter_than_kernel]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_fwd:triton:zero_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gated_rmsnorm_bwd:eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_fwd:eager:zero_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_wgrad:triton:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_wgrad:eager:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_fwd_out:triton-vs-eager:single]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_fwd_out:eager:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:triton:boundary_targets]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_fwd:triton:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rmsnorm_bwd:eager:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_fwd:aten-grouped:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_dgrad:aten-grouped:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_wgrad:triton:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_fwd_out:eager:single]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[embed_bwd_accum:eager:fresh]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_combine_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[flash_bwd:aten:ragged]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[flash_fwd:aten:unit_segments]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:triton:odd_vocab]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_index_scores:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[embed_bwd_accum:eager:all_same_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_sparse_attn_fwd:triton:self_only_early_tiles_dead]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_bwd:triton:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_sigmoid_noaux:eager:basic]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_dgrad:triton-vs-eager:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:triton:some_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_index_bwd:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:triton:all_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[causal_conv1d_silu_bwd:eager:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[embed_bwd_accum:triton-vs-eager:accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_index_scores:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_packed_fwd:eager:single]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_fwd:triton:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[causal_conv1d_silu_fwd:eager:seqs_shorter_than_kernel]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_bwd:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[flash_bwd:aten:unit_segments]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_wgrad:triton-vs-eager:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[causal_conv1d_silu_fwd:fla-triton-vs-eager:seqs_shorter_than_kernel]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gated_rmsnorm_fwd:fla-fused:saturated_gate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_fwd:eager:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_fwd:eager:constant_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_wgrad:aten-grouped-vs-eager:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[gated_rmsnorm_bwd:fla-fused-vs-eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_fwd:eager:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_packed_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_rowdot:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:eager:mode_topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:eager:total_ties]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_wgrad:eager:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_bwd:triton-vs-eager:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:eager:some_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_router_bwd:eager:softmax_then_topk]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_wgrad:aten-grouped-vs-eager:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_router_bwd:triton-vs-eager:topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_fwd:aten-grouped:all_rows_one_expert]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rope_fwd:eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[embed_bwd_accum:triton-vs-eager:fresh]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rope_bwd:eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[layernorm_bwd:triton-vs-eager:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_aux_lb_grad:triton:single_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_dispatch_fwd:aten:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_router_bwd:triton:softmax_then_topk]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_apply:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gated_rmsnorm_fwd:eager:saturated_gate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_seq_aux_grad:eager:ragged_with_len1_seq]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_pack_bits:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_sparse_attn_fwd:eager:self_only_early_tiles_dead]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_bwd:triton:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_aux_lb_grad:triton-vs-eager:single_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rmsnorm_fwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_apply:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[dsa_index_bwd:triton-vs-eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[muon_step:aten:expert_batched]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:eager:boundary_targets]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:eager:all_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rope_fwd:triton:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_every_registered_op_is_audited` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_packed_bwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rmsnorm_fwd:triton-vs-eager:zero_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_fwd_out:triton-vs-eager:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_fwd:aten-grouped:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_wgrad:aten-grouped:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_fwd:triton:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_index_bwd:eager:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[ce_loss_fwd_bwd:triton-vs-eager:odd_vocab]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[causal_conv1d_silu_fwd:fla-triton:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_fwd_out:triton:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:eager:boundary_targets]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_rowdot:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_router_bwd_sigmoid:eager:basic]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_sigmoid_noaux:eager:extreme_bias]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_index_scores:triton:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_fwd:eager:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[adamw_step:triton:huge_grad]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_topk_softmax:triton-vs-eager:total_ties]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_index_bwd:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_bwd:triton-vs-eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_scale_rows:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gated_rmsnorm_fwd:eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[muon_step:aten:odd_matrix]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:triton:k_equals_e]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[causal_conv1d_silu_bwd:fla-triton:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_bwd:eager:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rope_fwd:triton:huge_positions]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:eager:odd_vocab]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_router_bwd_sigmoid:eager:basic]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_combine_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[muon_step:aten:zero_grad]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gated_rmsnorm_bwd:fla-fused:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_topk:eager:k_exceeds_short_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_aux_lb_grad:triton:max_imbalance]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_aux_lb_grad:eager:max_imbalance]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_dispatch_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rmsnorm_bwd:triton-vs-eager:single_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_sort:aten:uniform]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[swiglu_fwd_out:triton-vs-eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[dsa_index_scores:triton-vs-eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[embed_bwd_accum:triton-vs-eager:all_same_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gelu_fwd_out:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gelu_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_rowdot:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[adamw_step:eager:first_step]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[rope_fwd:triton-vs-eager:huge_positions]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[causal_conv1d_silu_bwd:fla-triton:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gated_rmsnorm_fwd:fla-fused:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_wgrad:aten-grouped:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_router_bwd:triton:topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[swiglu_fwd_out:triton:single]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_aux_lb_grad:eager:single_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_wgrad:triton:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_grouped_mm_dgrad:aten-grouped-vs-eager:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[layernorm_fwd:triton:constant_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gelu_bwd:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_bwd:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:triton:total_ties]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_probs_sum:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_noweight:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_bwd:triton:odd_n]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_wgrad:eager:empty_experts_accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[adamw_step:eager:first_step]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_combine_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_sort:aten:all_one_expert_rest_empty]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_scale_rows:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_scale_rows:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:triton:all_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_sparse_attn_fwd:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_router_bwd:eager:topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_fwd:eager:constant_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_index_bwd:triton:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_grouped_mm_wgrad:triton:empty_experts_create]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_rowdot:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_router_bwd:triton:softmax_then_topk]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_sparse_attn_bwd:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[flash_fwd:aten:ragged]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:triton:mode_topk_then_softmax]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:triton:extreme_logits]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[swiglu_fwd_out:eager:saturated]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_topk_sigmoid_noaux:eager:basic]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:triton:extreme_logits]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_seq_aux_grad:eager:ragged_with_len1_seq]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[rmsnorm_noweight:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[rope_bwd:triton:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[gated_rmsnorm_fwd:fla-fused:saturated_gate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[ce_loss_fwd_bwd:eager:some_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[embed_bwd_accum:triton:all_same_token]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[causal_conv1d_silu_bwd:eager:packed_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_topk_softmax:triton:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[layernorm_fwd:triton:constant_row]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[dsa_index_bwd:eager:ragged_with_len1]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_aux_lb_grad:triton:max_imbalance]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[embed_bwd_accum:eager:fresh]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[adamw_step:triton-vs-eager:huge_grad]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[embed_bwd_accum:eager:accumulate]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[layernorm_fwd:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[moe_dispatch_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[flash_fwd:aten:unit_segments]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gated_rmsnorm_bwd:eager:odd_rows]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:eager:extreme_logits]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:triton:some_ignored]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_dgrad:aten-grouped:empty_experts]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[gelu_fwd_out:eager:odd_shape]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_finite[dsa_index_bwd:eager:single_seq_odd_len]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_degenerate_cross_impl[moe_aux_lb_grad:triton-vs-eager:max_imbalance]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[causal_conv1d_silu_fwd:eager:packed_with_len1]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:eager:all_ignored]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[swiglu_fwd_out:triton:single]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[rope_bwd:triton:odd_rows]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_aux_lb_grad:triton:single_token]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[ce_loss_fwd_bwd:triton-vs-eager:some_ignored]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[gated_rmsnorm_fwd:fla-fused-vs-eager:saturated_gate]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[dsa_index_bwd:triton:single_seq_odd_len]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[adamw_step:triton:first_step]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[moe_topk_softmax:triton-vs-eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[swiglu_packed_fwd:triton-vs-eager:single]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[moe_dispatch_fwd:aten:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[flash_fwd:aten:ragged]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[moe_grouped_mm_fwd:triton-vs-eager:empty_experts]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[swiglu_bwd:triton:saturated]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_bwd:eager:odd_n]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[causal_conv1d_silu_fwd:eager:packed_with_len1]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[gelu_fwd_out:eager:saturated]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:triton:k_equals_e]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:triton:boundary_targets]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[ce_loss_fwd_bwd:triton-vs-eager:boundary_targets]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[swiglu_packed_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[dsa_index_scores:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[dsa_index_scores:eager:single_seq_odd_len]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[moe_grouped_mm_dgrad:eager:empty_experts]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_dispatch_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[swiglu_packed_fwd:eager:single]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_packed_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_fwd:eager:all_rows_one_expert]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[layernorm_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[rope_fwd:triton:odd_rows]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[rope_fwd:eager:odd_rows]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[dsa_sparse_attn_fwd:eager:self_only_early_tiles_dead]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[flash_bwd:aten:ragged]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_cross_impl[causal_conv1d_silu_fwd:fla-triton-vs-eager:packed_with_len1]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:triton:odd_vocab]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:eager:k_equals_e]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[flash_bwd:aten:unit_segments]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[layernorm_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_packed_bwd:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_topk_softmax:eager:mode_topk_then_softmax]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[gated_rmsnorm_fwd:eager:odd_rows]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[rmsnorm_bwd:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_combine_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[muon_step:aten:odd_matrix]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[layernorm_fwd:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_fwd_out:eager:odd_n]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_bwd:triton:odd_n]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[gated_rmsnorm_bwd:fla-fused:odd_rows]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[rmsnorm_fwd:eager:zero_row]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[rmsnorm_fwd:triton:zero_row]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[layernorm_apply:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[dsa_index_scores:triton:single_seq_odd_len]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[embed_bwd_accum:eager:all_same_token]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[dsa_sparse_attn_fwd:triton:self_only_early_tiles_dead]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_router_bwd:eager:softmax_then_topk]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[rope_fwd:triton:huge_positions]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[rmsnorm_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[embed_bwd_accum:eager:accumulate]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_wgrad:aten-grouped:empty_experts_accumulate]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[muon_step:aten:expert_batched]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[moe_router_bwd:triton:topk_then_softmax]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[adamw_step:eager:huge_grad]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[rmsnorm_noweight:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[swiglu_packed_fwd:triton:single]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[dsa_index_bwd:triton:ragged_with_len1]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_fwd_out:triton:saturated]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[embed_bwd_accum:triton:accumulate]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_packed_fwd:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[causal_conv1d_silu_fwd:eager:seqs_shorter_than_kernel]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[layernorm_bwd:triton:single_row]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[moe_rowdot:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[rmsnorm_apply:triton:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[swiglu_fwd_out:eager:single]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[rmsnorm_apply:eager:odd_shape]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[moe_grouped_mm_dgrad:triton:empty_experts]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_degenerate_finite[adamw_step:triton:first_step]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_write_coverage_poison_invariance[causal_conv1d_silu_fwd:fla-triton:seqs_shorter_than_kernel]` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/examples/test_rl_training.py` — 26.0s total, 6 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_rl_training_parity_ppo[qwen35]` | 4.68s | 0.00s | 0.10s | 4.78s |
| `test_rl_training_parity_ppo[glm52]` | 4.24s | 0.00s | 0.10s | 4.34s |
| `test_rl_training_parity_reinforce` | 4.23s | 0.00s | 0.10s | 4.33s |
| `test_rl_training_parity_ppo[dsv32]` | 4.17s | 0.00s | 0.10s | 4.27s |
| `test_rl_training_parity_ppo[qwen3moe]` | 4.07s | 0.00s | 0.10s | 4.17s |
| `test_rl_training_parity_ppo[llama3]` | 4.05s | 0.00s | 0.10s | 4.15s |

### `tests/dataflow_training/data/test_data_pipeline.py` — 21.6s total, 23 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_checkpoint_resume_tail_matches_uninterrupted_run` | 8.81s | 0.00s | 0.11s | 8.92s |
| `test_shard_doc_mode_tokens_in_range_and_cursor_resume` | 5.03s | 0.00s | 0.10s | 5.13s |
| `test_legacy_doc_configuration_pinned` | 4.95s | 0.00s | 0.11s | 5.06s |
| `test_parquet_source_roundtrip` | 0.10s | 0.00s | 0.12s | 0.22s |
| `test_tiktoken_gpt2_vocab_bounds_and_eot_id` | 0.07s | 0.00s | 0.11s | 0.18s |
| `test_ffd_invariants_and_determinism` | 0.05s | 0.00s | 0.11s | 0.16s |
| `test_cursor_roundtrip_regenerates_next_step` | 0.04s | 0.00s | 0.11s | 0.15s |
| `test_legacy_block_configuration_pinned` | 0.04s | 0.00s | 0.10s | 0.14s |
| `test_threaded_packer_cursor_roundtrip` | 0.02s | 0.00s | 0.11s | 0.13s |
| `test_capture_roundtrip` | 0.01s | 0.01s | 0.11s | 0.13s |
| `test_cursor_to_json_matches_eager_and_is_json_clean` | 0.01s | 0.00s | 0.11s | 0.12s |
| `test_threaded_feed_equals_sync` | 0.01s | 0.00s | 0.11s | 0.12s |
| `test_threaded_feed_error_surfaces` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_jsonl_end_to_end_pack_determinism` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_prepacked_feed_bypass` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_greedy_no_split_defers_and_underfills` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_sequence_validation_rejects_bad_ids` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_jsonl_source_targets_and_masking` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_feed_requeue_leads_and_cursor_carries_content` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_jsonl_cursor_resume_and_epoch_wrap` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_txt_source_delimiter_split` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_synthetic_determinism_and_cursor_resume` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_spec_parser` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow/runtime/test_engine_stress.py` — 20.7s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_measured_costs_replan_still_golden` | 20.08s | 0.00s | 0.10s | 20.18s |
| `test_poison_on_free_changes_nothing` | 0.32s | 0.00s | 0.10s | 0.42s |
| `test_result_invariant_to_runtime_jitter` | 0.04s | 0.00s | 0.10s | 0.14s |

### `tests/dataflow_training/pretrain/test_client_model_step.py` — 18.8s total, 4 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_client_model_step_matches_in_process_olmoe` | 5.83s | 0.00s | 0.13s | 5.96s |
| `test_client_model_step_matches_in_process_llama3` | 5.48s | 0.00s | 0.13s | 5.61s |
| `test_client_model_step_llama3_passes` | 4.98s | 0.00s | 0.13s | 5.11s |
| `test_out_of_process_daemon_boots_and_reaps` | 2.00s | 0.00s | 0.13s | 2.13s |

### `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` — 11.1s total, 2 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_kill_escalates_past_sigterm` | 10.55s | 0.00s | 0.15s | 10.70s |
| `test_kill_terminates_and_verifies` | 0.25s | 0.00s | 0.15s | 0.40s |

### `tests/dataflow_training/training/surfaces/test_replicate_load.py` — 10.1s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_world1_replicate_steps_once_bitwise` | 9.95s | 0.00s | 0.16s | 10.11s |

### `tests/dataflow/service/test_shared_server_self_heal.py` — 4.9s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_self_heal_respawns_after_illegal_access` | 4.85s | 0.00s | 0.10s | 4.95s |

### `tests/dataflow_training/tasks/test_kernels.py` — 4.7s total, 21 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_swiglu_fused[4099-14336]` | 0.85s | 0.00s | 0.16s | 1.01s |
| `test_ce_fused[513-128256]` | 0.38s | 0.00s | 0.16s | 0.54s |
| `test_ce_fused[1024-32003]` | 0.14s | 0.00s | 0.16s | 0.30s |
| `test_adamw_fused[16777219]` | 0.09s | 0.00s | 0.15s | 0.24s |
| `test_ce_fused_past_int32_elements` | 0.04s | 0.00s | 0.16s | 0.20s |
| `test_rmsnorm_fused[2048-4096]` | 0.05s | 0.00s | 0.15s | 0.20s |
| `test_swiglu_fused[2048-1024]` | 0.02s | 0.00s | 0.15s | 0.17s |
| `test_rmsnorm_fused[4099-1024]` | 0.02s | 0.00s | 0.15s | 0.17s |
| `test_swiglu_fused[129-517]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_rmsnorm_fused[129-517]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_ce_fused[128-517]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_adamw_fused[1048576]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_rope_fused[seq2-6-128]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_rope_fused[128-8-64]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_builtin_ops_all_have_eager_fallback` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_fused_set_is_fully_fused_and_deterministic` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_rope_fused[seq1-8-64]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_registry_selection_override_and_gating` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_rope_fused[seq3-8-64]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_adamw_fused[129]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_fused_steady_state_no_torch_allocation` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_sim/engine/test_simulator.py` — 3.8s total, 38 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_single_task_emits_start_then_end` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_snapshot_free_run_keeps_intervals_without_events` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_reference_stream_in_snapshot_includes_future_outputs` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_repeated_transfers_of_same_object_get_unique_task_ids` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_prefetch_emits_transfer_events_and_lands_live_on_compute` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_transfer_queue_serializes_multiple_offloads` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_offload_overwrites_existing_backing_copy` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_same_id_on_backing_and_compute_coexist` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_prefetch_defers_when_source_still_offloading` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_queued_prefetch_consumes_no_compute_bytes_until_transfer_starts` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_active_task_inputs_show_in_reference_stream_at_task_start` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_releases_after_emits_release_event_and_frees_memory` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_compute_stalls_waiting_on_inbound_prefetch` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_release_nonexistent_raises` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_missing_input_raises` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_compute_stalls_waiting_on_fast_memory_capacity_to_free` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_prefetch_already_on_compute_raises` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_prefetch_blocks_at_start_until_compute_release_frees_bytes` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_offload_emits_transfer_events_and_frees_compute` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_offload_overwrite_size_mismatch_raises` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_snapshot_free_run_can_emit_compact_memory_trace` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_offload_deadlocks_when_backing_memory_capacity_too_tight` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_location_propagates_to_memory_entries` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_reference_stream_in_snapshot_tracks_next_use` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_missing_bandwidth_raises_when_used` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_output_collision_raises` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_prefetch_deadlocks_when_fast_memory_capacity_too_tight` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_truly_absent_backing_source_still_raises` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_output_reserved_at_start_visible_at_end` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_queued_offload_consumes_no_backing_bytes_until_transfer_starts` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_per_trigger_runtime_overrides_bandwidth` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_output_visible_only_for_downstream_not_self` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_compute_waits_for_needed_queued_inputs_not_unneeded_tail` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_task_intervals_match_runtime_sum` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_backing_memory_capacity_enforced_on_initial_memory` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_prefetch_defers_when_backing_source_pending_inbound` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_fast_memory_capacity_enforced` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_compute_rejects_backing_only_input` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow/service/test_slice_snapshots.py` — 3.8s total, 12 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_slice_roundtrip_and_compose` | 0.45s | 0.00s | 0.10s | 0.55s |
| `test_restore_runs_in_background_and_parks_writers` | 0.41s | 0.00s | 0.11s | 0.52s |
| `test_remap_extraction_restore` | 0.28s | 0.00s | 0.12s | 0.40s |
| `test_shifted_dst_recompose` | 0.23s | 0.00s | 0.11s | 0.34s |
| `test_corrupt_payload_refused_store_untouched` | 0.22s | 0.00s | 0.11s | 0.33s |
| `test_duplicate_snapshots_full_and_independent` | 0.18s | 0.00s | 0.11s | 0.29s |
| `test_second_daemon_on_live_socket_refuses` | 0.16s | 0.00s | 0.11s | 0.27s |
| `test_snapshot_json_schema_and_hashes` | 0.13s | 0.00s | 0.11s | 0.24s |
| `test_refusal_mid_list_leaves_store_untouched` | 0.13s | 0.00s | 0.11s | 0.24s |
| `test_snapshot_has_no_group_concept` | 0.12s | 0.00s | 0.11s | 0.23s |
| `test_restore_status_lifecycle` | 0.11s | 0.00s | 0.11s | 0.22s |
| `test_slice_validation_refusals` | 0.05s | 0.00s | 0.11s | 0.16s |

### `tests/dataflow_sim/planning/policies/test_auto_policy.py` — 3.7s total, 36 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_auto_policy_L10_works_down_to_cap_500[500]` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_auto_policy_L10_works_down_to_cap_500[600]` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_iterative_refinement_recovers_valid_plan_at_tight_cap` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_auto_policy_L10_works_down_to_cap_500[1000]` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_auto_policy_L10_works_down_to_cap_500[None]` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_auto_policy_L10_works_down_to_cap_500[1500]` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_auto_policy_L10_works_down_to_cap_500[800]` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_activation_offload_fires_eagerly_at_production` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_auto_policy_L5_works_down_to_cap_500[1000]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_initial_placement_must_place_T1_inputs` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_smart_initial_placement_defers_to_leave_room_for_outputs` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_zero_stalls_at_unlimited` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L5_works_down_to_cap_500[800]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_unlimited_emits_no_transfers_without_final_locations` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_next_use_after_returns_first_use_at_or_after` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_compute_uses_collects_input_timestamps` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_works_at_loose_caps[None]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_releases_weight_instead_of_offloading_when_backing_copy_exists` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_at_least_as_fast_as_sliding_window_at_unlimited_cap` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_works_at_loose_caps[500]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_unlimited_honors_final_backing_locations` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_works_at_loose_caps[1000]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_per_task_use_events_collapse_duplicates` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_initial_placement_leaves_slack_for_widest_task` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_writes_back_final_backing_objects` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L5_works_down_to_cap_500[1200]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_initial_placement_raises_when_widest_T1_too_big` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_works_at_loose_caps[600]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_works_at_loose_caps[800]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L5_works_down_to_cap_500[600]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L3_works_at_loose_caps[1200]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L5_works_down_to_cap_500[500]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_roundtrip_enumeration_finds_wide_gap_weights` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_emits_releases_at_tight_cap` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_auto_policy_L5_works_down_to_cap_500[None]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_smart_initial_placement_at_loose_cap_eliminates_forward_stalls` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow/service/test_daemon_relaunch.py` — 3.5s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses` | 3.42s | 0.00s | 0.10s | 3.52s |

### `tests/dataflow_training/training/lowering/test_layout_registry.py` — 3.0s total, 21 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_registry_covers_external_family` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_registry_addresses_every_weight_root[glm52]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_registry_validates_and_digest_pinned[qwen35moe]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_registry_validates_and_digest_pinned[llama3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[qwen3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[qwen3moe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[dsv3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[dsv3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[qwen35moe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[qwen3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[glm52]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[llama3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[gpt2]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[gpt2]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[olmoe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[dsv32]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[qwen35]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[qwen3moe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_validates_and_digest_pinned[dsv32]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[qwen35]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_registry_addresses_every_weight_root[olmoe]` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/modules/test_moe.py` — 3.0s total, 24 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_grouped_mm_vs_dense_loop[False]` | 0.00s | 0.00s | 0.13s | 0.13s |
| `test_grouped_mm_vs_dense_loop[True]` | 0.00s | 0.00s | 0.13s | 0.13s |
| `test_topk_softmax_vs_reference_incl_ties[topk_then_softmax]` | 0.00s | 0.00s | 0.13s | 0.13s |
| `test_grouped_mm_bitwise_repeatable` | 0.00s | 0.00s | 0.13s | 0.13s |
| `test_moe_tail_fwd_bwd_vs_reference[False-0.0-softmax_then_topk]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_moe_tail_fwd_bwd_vs_reference[True-0.001-topk_then_softmax]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_topk_sigmoid_noaux_kernel_vs_reference_and_semantics` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_seq_aux_grad_vs_autograd` | 0.00s | 0.01s | 0.12s | 0.13s |
| `test_moe_tail_fwd_bwd_vs_reference[False-0.01-softmax_then_topk]` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_moe_tail_eager_kernels_match_fused` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_moespec_rejects_invalid_configs` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_router_bwd_vs_autograd[softmax_then_topk]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_topk_softmax_vs_reference_incl_ties[softmax_then_topk]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_bias_update_sign_rule_and_aux_counts_layout` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_router_bwd_sigmoid_vs_autograd` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_moe_mlp_reference_ungated_shared_and_noaux_mode` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_topk_eager_matches_reference_bitwise` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_partial_ownership_sizes_and_sharded_experts_math` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_moe_sort_permutation_offsets_stability` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_moe_tail_recompute_reproduces_ctx_bitwise` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_aux_grad_vs_autograd_and_finite_difference` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_dispatch_and_combine_vs_einsum` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_router_bwd_vs_autograd[topk_then_softmax]` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_swiglu_packed_matches_unpacked_bitwise` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow/service/test_service_skeleton.py` — 3.0s total, 10 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_fast_path_answers_while_dispatcher_held` | 0.60s | 0.06s | 0.11s | 0.77s |
| `test_engine_status_lists_all_client_sessions` | 0.40s | 0.06s | 0.11s | 0.57s |
| `test_shutdown_terminates_daemon` | 0.27s | 0.00s | 0.11s | 0.38s |
| `test_ticket_async_and_blocking_parity` | 0.02s | 0.07s | 0.11s | 0.20s |
| `test_session_status_tracks_calls` | 0.00s | 0.07s | 0.11s | 0.18s |
| `test_subscribe_since_seq_zero_replays_all_events` | 0.00s | 0.07s | 0.11s | 0.18s |
| `test_schema_skew_rejected` | 0.00s | 0.08s | 0.10s | 0.18s |
| `test_unknown_op_error_envelope` | 0.00s | 0.07s | 0.10s | 0.17s |
| `test_handshake_and_health` | 0.00s | 0.06s | 0.11s | 0.17s |
| `test_register_program_requires_resolver_kind` | 0.00s | 0.06s | 0.11s | 0.17s |

### `tests/dataflow_sim/core/test_validate_chain.py` — 2.9s total, 29 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_invalid_chain_rejected[make_invalid_release_mutation_release_then_later_reference-make_invalid_release_mutation_release_then_later_reference]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_output_id_collision-make_invalid_id_resolution_output_id_collision]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_trigger_validity_prefetch_and_offload_same_object_same_task-make_invalid_trigger_validity_prefetch_and_offload_same_object_same_task]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_prefetch_unknown_obj-make_invalid_id_resolution_prefetch_unknown_obj]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_capacity_forced_footprint_exceeds_cap-make_invalid_capacity_forced_footprint_exceeds_cap]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_offload_unknown_obj-make_invalid_id_resolution_offload_unknown_obj]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_trigger_validity_duplicate_prefetch_same_task-make_invalid_trigger_validity_duplicate_prefetch_same_task]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_capacity_input_footprint_exceeds_cap-make_invalid_capacity_input_footprint_exceeds_cap]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_release_mutation_release_and_offload_same_object-make_invalid_release_mutation_release_and_offload_same_object]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_release_not_in_inputs-make_invalid_id_resolution_release_not_in_inputs]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_topological_self_cycle-make_invalid_topological_self_cycle]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_topological_forward_reference-make_invalid_topological_forward_reference]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_validate_can_be_skipped` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_trigger_validity_prefetch_already_on_compute-make_invalid_trigger_validity_prefetch_already_on_compute]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_unknown_input-make_invalid_id_resolution_unknown_input]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_release_mutation_no_backing_copy_with_later_use-make_invalid_release_mutation_no_backing_copy_with_later_use]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_topological_duplicate_input_id-make_invalid_topological_duplicate_input_id]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_release_mutation_mutation_never_offloaded-make_invalid_release_mutation_mutation_never_offloaded]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_release_mutation_dirty_with_later_use-make_invalid_release_mutation_dirty_with_later_use]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_capacity_initial_backing_overflow-make_invalid_capacity_initial_backing_overflow]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_trigger_validity_duplicate_offload_same_task-make_invalid_trigger_validity_duplicate_offload_same_task]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_topological_unproduced_input-make_invalid_topological_unproduced_input]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_topological_empty_chain-make_invalid_topological_empty_chain]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_output_shadows_initial-make_invalid_id_resolution_output_shadows_initial]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_capacity_output_footprint_exceeds_cap-make_invalid_capacity_output_footprint_exceeds_cap]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_trigger_validity_offload_not_on_compute-make_invalid_trigger_validity_offload_not_on_compute]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_id_resolution_mutates_not_in_inputs-make_invalid_id_resolution_mutates_not_in_inputs]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_capacity_initial_compute_overflow-make_invalid_capacity_initial_compute_overflow]` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_chain_rejected[make_invalid_release_mutation_release_not_in_inputs-make_invalid_release_mutation_release_not_in_inputs]` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/pretrain/test_client_fetch_surface.py` — 2.8s total, 2 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_client_fetch_surface_dense` | 1.35s | 0.00s | 0.13s | 1.48s |
| `test_client_fetch_surface_moe_aux` | 1.19s | 0.00s | 0.13s | 1.32s |

### `tests/dataflow/service/test_service_store.py` — 2.8s total, 16 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_real_boot_family_init_byte_identity` | 0.65s | 0.00s | 0.11s | 0.76s |
| `test_real_boot_init_fits_tight_slab` | 0.17s | 0.00s | 0.11s | 0.28s |
| `test_program_content_id_matches_registration` | 0.00s | 0.03s | 0.11s | 0.14s |
| `test_create_object_semantics` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_host_program_registration_guards` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_duplicate_copies_bytes_independently` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_protected_object_survives_wipe_unless_forced` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_unknown_object` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_materialize_zeros_and_tokens` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_overwrite_same_size_ok_mismatch_rejected` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_host_run_fills_in_place` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_put_get_file_forms` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_put_get_roundtrip_bytes` | 0.00s | 0.03s | 0.10s | 0.13s |
| `test_query_backing_usage` | 0.00s | 0.02s | 0.10s | 0.12s |
| `test_allocator_coalesce_and_reuse` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_allocator_capacity_error_detail` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/training/e2e/test_varlen_e2e.py` — 2.7s total, 11 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_model_step_ragged_matches_golden_all_families[qwen35moe]` | 0.32s | 0.00s | 0.15s | 0.47s |
| `test_qwen35_model_step_ragged_matches_golden` | 0.25s | 0.00s | 0.15s | 0.40s |
| `test_model_step_ragged_matches_golden_all_families[glm52]` | 0.13s | 0.00s | 0.15s | 0.28s |
| `test_model_step_ragged_matches_golden_all_families[olmoe]` | 0.07s | 0.00s | 0.16s | 0.23s |
| `test_model_step_ragged_matches_golden_all_families[dsv32]` | 0.08s | 0.00s | 0.15s | 0.23s |
| `test_model_step_ragged_matches_golden_all_families[dsv3]` | 0.06s | 0.00s | 0.15s | 0.21s |
| `test_llama3_model_step_ragged_matches_golden` | 0.04s | 0.00s | 0.15s | 0.19s |
| `test_model_step_ragged_matches_golden_all_families[qwen3moe]` | 0.04s | 0.00s | 0.15s | 0.19s |
| `test_model_step_ragged_matches_golden_all_families[qwen3]` | 0.04s | 0.00s | 0.15s | 0.19s |
| `test_varlen_single_launch_matches_reference_autograd` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_flash_ragged_matches_reference_autograd` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_sim/app/test_server.py` — 2.7s total, 16 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_simulate_large_chain_uses_snapshot_free_response` | 1.03s | 0.00s | 0.10s | 1.13s |
| `test_simulate_recompute_toggle_off_matches_default_and_can_change_makespan` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_simulate_final_model_state_on_backing_is_opt_in` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_simulate_keeps_exact_training_step_count` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_preview_model_training_workload_returns_dataflow_schema` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_simulate_exposes_pressurefit_diagnostics` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_preview_accepts_datatype_policy_and_exports_metadata` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_simulate_uploaded_schema_workload` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_presets_include_sram_accelerator_hardware` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_presets_include_only_public_model_workloads` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_simulate_omits_policy_diagnostics_for_other_policies` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_preview_accepts_uploaded_schema_and_returns_bare_chain` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_simulate_accepts_fractional_fast_memory_budget` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_simulate_accepts_asymmetric_transfer_bandwidths` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_presets_include_gb300_hardware` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_h100_marks_fp4_matmul_as_unsupported` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/training/e2e/test_freeze_plan.py` — 2.6s total, 17 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_model_step_truncated_olmoe` | 0.04s | 0.00s | 0.14s | 0.18s |
| `test_model_step_partial_fields` | 0.03s | 0.00s | 0.15s | 0.18s |
| `test_model_step_pair_freeze` | 0.03s | 0.00s | 0.14s | 0.17s |
| `test_model_step_truncated_prefix` | 0.02s | 0.00s | 0.15s | 0.17s |
| `test_model_step_truncated_ga2` | 0.02s | 0.00s | 0.15s | 0.17s |
| `test_model_step_frozen_head` | 0.02s | 0.00s | 0.14s | 0.16s |
| `test_model_step_passthrough` | 0.02s | 0.00s | 0.14s | 0.16s |
| `test_all_families_truncated_prefix_lowers` | 0.01s | 0.00s | 0.14s | 0.15s |
| `test_initial_values_refill_identity` | 0.01s | 0.00s | 0.14s | 0.15s |
| `test_no_freeze_derives_none` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_bench_default_stream_semantics` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_passthrough_plan` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_train_indexer_unified_into_policy` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_composer_semantics` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_all_frozen_ce_rejected` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_plan_repr_compact` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_truncated_prefix_plan` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_sim/workloads/test_modular_workload_builder.py` — 2.6s total, 24 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_constrained_memory_recompute_planning_selects_useful_variants` | 0.14s | 0.00s | 0.10s | 0.24s |
| `test_varied_family_workloads_run_and_report_kpis` | 0.03s | 0.00s | 0.10s | 0.13s |
| `test_deepseek_v32_modules_emit_expected_subop_chains_blocks_and_indexer_dtypes` | 0.01s | 0.00s | 0.10s | 0.11s |
| `test_gpt_oss_modules_emit_expected_subop_chains_and_block_keys` | 0.00s | 0.01s | 0.10s | 0.11s |
| `test_glm52_indexshare_blocks_skip_shared_indexer_work` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_backward_matmul_policy_separates_activation_and_parameter_gradients` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_qwen_deepseek_and_gpt_oss_reuse_existing_moe_subops_without_router_ops` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_matmul_accumulate_epilogue_bytes` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_op_helper_formulas_are_hand_checkable` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_datatype_policy_changes_program_bytes_and_compute_precision` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_muon_uses_split_real_matrix_shapes_and_expert_counts` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_nemotron_modules_emit_expected_subop_chains_and_dtype_lanes` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_family_presets_are_easy_to_override` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_dtype_policy_defaults_and_low_precision_sizes` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_tied_embeddings_share_one_weight_grad_and_optimizer` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_moe_recompute_variant_rewires_activation_producer_and_block_metadata` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_family_public_preset_values_match_source_configs` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_lm_head_optimizer_defaults_to_adamw_for_non_none_layer_optimizer` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_training_program_uses_model_order_reverse_backward_and_optimizer` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_compute_precision_changes_realized_runtime_under_unlimited_memory` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_invalid_ep_group_size_for_moe_raises_clear_error` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_expert_dispatch_dtype_changes_forward_and_backward_dispatch_lanes` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_module_optimizer_matrices_use_real_parameter_shapes` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_ep_group_size_shards_routed_experts_and_uses_scale_up_movement` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/training/planning/test_planning.py` — 2.5s total, 10 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_backing_capacity_drives_recompute` | 0.37s | 0.00s | 0.16s | 0.53s |
| `test_measured_programs_are_built_in_one_place` | 0.30s | 0.00s | 0.15s | 0.45s |
| `test_recompute_fires_under_starved_interconnect` | 0.27s | 0.00s | 0.15s | 0.42s |
| `test_recompute_never_plans_slower_than_saving_everything` | 0.03s | 0.00s | 0.15s | 0.18s |
| `test_capacity_sweep_monotone_tiny` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_plan_with_recompute_tiny` | 0.01s | 0.00s | 0.14s | 0.15s |
| `test_incomplete_cost_table_is_not_reported_as_infeasible` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_burst_sampled_profiles_are_refused_as_a_production_price` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_pressurefit_plan_tiny` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_level_pins_cover_every_variant_the_search_prices` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/training/surfaces/test_solo_resume_bitwise.py` — 2.3s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_solo_resume_reproduces_tail_bitwise` | 2.19s | 0.00s | 0.15s | 2.34s |

### `tests/dataflow_training/pretrain/test_presets.py` — 2.3s total, 16 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_preset_lowers[l3_350m]` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_preset_lowers[l3_760m]` | 0.01s | 0.00s | 0.14s | 0.15s |
| `test_preset_lowers[l3_1b]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_cfg_dict_round_trip_matches_dims` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_param_counts_monotone_and_1b_is_1b` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_preset_shapes_consistent[l3_350m]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_resolve_preset_bare_unique_names_across_families` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_resolve_preset_unknown_name_points_at_the_table` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_locked_token_budget` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_preset_shapes_consistent[l3_1b]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_resolve_preset_qualified_name_disambiguates` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_resolve_preset_ambiguous_name_lists_qualified_forms` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_preset_shapes_consistent[l3_125m]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_preset_shapes_consistent[l3_760m]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_smoke_preset_lowers_and_plans` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_preset_lowers[l3_125m]` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_sim/planning/policies/test_min_grow.py` — 2.2s total, 22 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_derive_schedule_mutated_backing_init_offloads_at_mutator` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_max_plan_pre_places_backing_init_with_a_minus_1` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_analytic_pre_pass_reaches_static_feasibility` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_min_plan_separates_non_adjacent_uses` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_enumerate_reductions_generates_split_for_gappable_interval` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_score_with_peak_returns_both_values` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_respects_static_cap_passes_with_unlimited` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_derive_schedule_pre_places_backing_init_with_a_minus_1` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_end_to_end_returns_runnable_chain` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_enumerate_reductions_generates_shrink_for_extended_interval` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_derive_schedule_released_after_last_use` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_min_plan_inputs_use_half_open_interval` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_max_plan_produced_starts_at_producer` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_derive_schedule_emits_prefetch_for_non_pre_placed_backing_init` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_end_to_end_unlimited_cap_returns_max_immediately` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_max_plan_releases_after_last_use` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_respects_static_cap_includes_next_outputs_reservation` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_smart_prefetch_returns_int_in_range` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_min_plan_output_merges_with_downstream_input` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_bare_invariant_check` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_infeasible_raises_with_forced_footprint_exceeding_cap` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_max_plan_mutated_grad_exits_at_mutator` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/training/surfaces/test_checkpoint_record.py` — 2.2s total, 4 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_load_checkpoint_targets` | 0.45s | 0.00s | 0.15s | 0.60s |
| `test_conductor_save_resume_round_trip` | 0.45s | 0.00s | 0.15s | 0.60s |
| `test_load_checkpoint_engines_mapping` | 0.44s | 0.00s | 0.15s | 0.59s |
| `test_load_checkpoint_refuses_small_engine` | 0.25s | 0.00s | 0.16s | 0.41s |

### `tests/dataflow_training/pretrain/test_flops.py` — 2.1s total, 15 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_every_family_walks[llama3]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_every_family_walks[dsv3]` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_attention_split_factors` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[qwen35moe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[qwen3]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[glm52]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[qwen35]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_optimizer_bucket_policy_aware` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_varlen_quadratic_scaling` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_hybrid_split_causal_vs_static` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[gpt2]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[olmoe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[dsv32]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_every_family_walks[qwen3moe]` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_gpt2_walker_matches_hand_formula` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/pretrain/test_sharding.py` — 1.9s total, 13 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_serialization_roundtrip_and_required_groups` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_expert_shards_views_and_ep_rejection` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_zero1rs_eligibility_rejections` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_zero1_halves_invariants_and_views` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_tp_serialization_roundtrip` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_tp_axis_validation` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_world4_world8_plans_balance_cover_and_comm_identity` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_equal_shards_honest` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_tp_mlp_shards_plan_and_views` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_row_split_and_cover_validation` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_zero1rs_block_params_sizing` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_muon_rejects_row_splits` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_real_llama3_layouts_shard` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/pretrain/test_parity_smoke.py` — 1.8s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_reference_vs_service_parity_smoke` | 1.70s | 0.00s | 0.15s | 1.85s |

### `tests/dataflow_training/training/e2e/test_lbl_modes.py` — 1.8s total, 8 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_retained_router_delta_is_ga_invariant_per_round_is_not` | 0.24s | 0.00s | 0.15s | 0.39s |
| `test_modes_differ_on_router_at_ga4` | 0.07s | 0.00s | 0.15s | 0.22s |
| `test_recompute_never_double_counts` | 0.07s | 0.00s | 0.15s | 0.22s |
| `test_counts_overall_monotone_across_steps` | 0.06s | 0.00s | 0.15s | 0.21s |
| `test_noaux_bias_ga_invariant_bit_exact` | 0.06s | 0.00s | 0.15s | 0.21s |
| `test_counts_ga_invariant_and_sum_exact` | 0.06s | 0.00s | 0.15s | 0.21s |
| `test_modes_agree_on_router_at_ga1_and_differ_upstream` | 0.03s | 0.00s | 0.15s | 0.18s |
| `test_retained_structure_and_default_stability` | 0.00s | 0.00s | 0.15s | 0.15s |

### `tests/dataflow/service/test_pinned_slab.py` — 1.7s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_slab_costs_what_it_asks_for` | 0.70s | 0.00s | 0.10s | 0.80s |
| `test_slab_frees_what_it_pinned` | 0.47s | 0.00s | 0.10s | 0.57s |
| `test_host_memory_is_pinned_in_exactly_one_place` | 0.25s | 0.00s | 0.10s | 0.35s |

### `tests/dataflow/service/test_peer_protocol.py` — 1.7s total, 17 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_exact_chunk_multiple_boundary_byte_identity` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_reordered_chunks_are_protocol_violation` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_chunked_happy_path_byte_identity` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_dead_sender_frees_reservation_nothing_torn` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_busy_lease_backoff_then_success` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_eager_happy_path` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_capacity_retries_exhaust_to_error` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_duplicate_rts_single_reservation` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_async_commit_defers_ack_and_survives_inactivity` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_collision_never_retries` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_eager_duplicate_rts_recommit_guard` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_collective_queue_fifo` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_lost_done_ack_resend_reacks_without_recommit` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_per_dest_fifo_and_cross_dest_interleave` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_overwrite_matrix` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_negotiating_inactivity_aborts` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_corrupt_chunk_aborts_without_commit` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/tasks/test_optim.py` — 1.7s total, 11 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_muon_recipe_string_model_step_vs_hand_replica` | 0.02s | 0.00s | 0.15s | 0.17s |
| `test_layer_indexed_policy_sizes_and_model_step` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_muon_orthogonalizes_2d_and_falls_back_1d` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_hyper_overrides_and_schedule_model_step` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_mixed_policy_model_step_vs_hand_replica` | 0.01s | 0.00s | 0.15s | 0.16s |
| `test_sgd_step_matches_inline_formula` | 0.00s | 0.01s | 0.15s | 0.16s |
| `test_muon_recipe_classification_and_3d` | 0.01s | 0.00s | 0.14s | 0.15s |
| `test_policy_dispatch_and_validation` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_opt_state_layout_slots_follow_policy` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_sgdm_step_matches_inline_formula` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_lr_schedules_shapes` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/training/e2e/test_dtype_policy_e2e.py` — 1.7s total, 7 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_qwen35_model_step_mixed_policy` | 0.26s | 0.00s | 0.15s | 0.41s |
| `test_qwen35_model_step_depth_dependent` | 0.24s | 0.00s | 0.15s | 0.39s |
| `test_llama_model_step_mixed_policy` | 0.05s | 0.00s | 0.15s | 0.20s |
| `test_llama_model_step_depth_dependent` | 0.05s | 0.00s | 0.15s | 0.20s |
| `test_llama_block_backward_matches_golden_mixed_policy` | 0.02s | 0.00s | 0.16s | 0.18s |
| `test_mixed_policy_layout_shapes` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_depth_dependent_layer_sizes_diverge` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/test_import_boundaries.py` — 1.6s total, 7 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_sim_required_only_under_lowering` | 0.27s | 0.00s | 0.15s | 0.42s |
| `test_workload_uses_engine_public_surfaces_only` | 0.14s | 0.00s | 0.16s | 0.30s |
| `test_engine_never_imports_workload_or_twins` | 0.06s | 0.00s | 0.15s | 0.21s |
| `test_tools_stay_near_package_roots` | 0.04s | 0.00s | 0.15s | 0.19s |
| `test_runtime_never_imports_torch_or_sim` | 0.04s | 0.00s | 0.15s | 0.19s |
| `test_core_is_dependency_free` | 0.03s | 0.00s | 0.15s | 0.18s |
| `test_blocks_never_imports_sim` | 0.01s | 0.00s | 0.15s | 0.16s |

### `tests/dataflow/service/test_engine_determinism.py` — 1.5s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_same_daemon_rerun_bitwise` | 1.41s | 0.00s | 0.10s | 1.51s |

### `tests/dataflow_training/modules/test_dsa.py` — 1.4s total, 10 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_index_scores_vs_hand_loop` | 0.10s | 0.00s | 0.12s | 0.22s |
| `test_mask_form_equals_gather_form_fwd_and_bwd` | 0.08s | 0.00s | 0.12s | 0.20s |
| `test_dsv32_block_fwd_recompute_bwd_match_golden[dense]` | 0.01s | 0.00s | 0.13s | 0.14s |
| `test_dsv32_block_fwd_recompute_bwd_match_golden[moe]` | 0.02s | 0.00s | 0.12s | 0.14s |
| `test_absorbed_op_matches_expanded_reference` | 0.00s | 0.00s | 0.13s | 0.13s |
| `test_dsa_kernels_vs_references_and_autograd` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_indexer_kl_grad_is_softmax_minus_p_and_inputs_detached` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_topk_padding_is_mask_safe_and_tie_rule_consistent` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_dsa_index_bwd_vs_autograd` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_index_scores_ragged_packing_matches_per_sequence` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow_training/models/test_gpt2.py` — 1.4s total, 10 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_model_step_nobias` | 0.04s | 0.00s | 0.13s | 0.17s |
| `test_model_step_tied` | 0.05s | 0.00s | 0.12s | 0.17s |
| `test_model_step_uniform` | 0.04s | 0.00s | 0.12s | 0.16s |
| `test_qkv_bias_grad_sections` | 0.04s | 0.00s | 0.12s | 0.16s |
| `test_model_step_ragged` | 0.04s | 0.00s | 0.12s | 0.16s |
| `test_twin_packed_matches_per_sequence_fp32` | 0.01s | 0.00s | 0.12s | 0.13s |
| `test_layernorm_apply_matches_fwd` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_twin_rejects_overlong_segment` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_layernorm_backward` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_gelu_backward_matches_autograd` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow/service/test_service_snapshot.py` — 1.4s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_checkpoint_roundtrip_bit_continuity` | 0.16s | 0.53s | 0.10s | 0.79s |
| `test_leased_writer_parks_until_release` | 0.30s | 0.00s | 0.11s | 0.41s |
| `test_snapshot_status_unknown_id_rejected` | 0.00s | 0.00s | 0.20s | 0.20s |

### `tests/dataflow_training/models/test_glm52.py` — 1.4s total, 7 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_glm52_grad_accum_two_rounds_matches_reference` | 0.14s | 0.00s | 0.12s | 0.26s |
| `test_glm52_aux_zero_model_step_vs_golden` | 0.13s | 0.00s | 0.12s | 0.25s |
| `test_glm52_frozen_indexer_ablation` | 0.11s | 0.00s | 0.13s | 0.24s |
| `test_glm52_dense_warmup_model_step` | 0.09s | 0.00s | 0.13s | 0.22s |
| `test_glm52_dense_warmup_freeze_and_movement` | 0.03s | 0.00s | 0.12s | 0.15s |
| `test_glm52_full_scale_presets_lower_and_validate` | 0.02s | 0.00s | 0.12s | 0.14s |
| `test_glm52_partial_ownership_lowering_rejected` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow_training/models/test_block_isolation.py` — 1.4s total, 5 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_isolated_block_at_floor[glm52-isolate0-6]` | 0.34s | 0.00s | 0.12s | 0.46s |
| `test_isolated_block_at_floor[qwen35moe-isolate2-4]` | 0.21s | 0.00s | 0.12s | 0.33s |
| `test_isolated_block_at_floor[qwen35moe-isolate3-4]` | 0.11s | 0.00s | 0.12s | 0.23s |
| `test_isolated_block_at_floor[glm52-isolate1-6]` | 0.08s | 0.00s | 0.11s | 0.19s |
| `test_isolated_block_at_floor[dsv3-isolate4-3]` | 0.04s | 0.00s | 0.12s | 0.16s |

### `tests/dataflow/service/test_service_runs.py` — 1.3s total, 5 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_rebind_two_token_slabs` | 0.02s | 0.54s | 0.11s | 0.67s |
| `test_transients_visible_and_reclaimed` | 0.01s | 0.00s | 0.21s | 0.22s |
| `test_cancel_mid_run_leaves_healthy_daemon` | 0.12s | 0.00s | 0.10s | 0.22s |
| `test_weights_adopted_not_refilled` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_poison_isolation_and_next_run_succeeds` | 0.01s | 0.00s | 0.10s | 0.11s |

### `tests/dataflow_training/training/lowering/test_shaped_program.py` — 1.3s total, 9 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_8b_shape_totals` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_grad_accum_mutation_pattern` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_recompute_variant_moves_A_production` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_tied_embeddings_chain_structure` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_tiny_validates` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_interleaved_optimizer_fires_at_final_grad_mutation` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_heterogeneous_kinds_emit_per_kind_keys_and_sizes` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_rewrites_cover_all_saved_contexts` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_interleaved_optimizer_respects_reader_ordering` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/models/test_dsv32.py` — 1.3s total, 7 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_dsv32_frozen_indexer_ablation` | 0.11s | 0.00s | 0.12s | 0.23s |
| `test_dsv32_aux_zero_model_step_vs_golden` | 0.10s | 0.00s | 0.12s | 0.22s |
| `test_dsv32_short_sequences_lt_index_topk_vs_golden` | 0.08s | 0.00s | 0.12s | 0.20s |
| `test_dsv32_ga2_matches_reference` | 0.08s | 0.00s | 0.12s | 0.20s |
| `test_dsv32_dense_warmup_model_step` | 0.07s | 0.00s | 0.11s | 0.18s |
| `test_dsv32_full_scale_presets_lower_and_validate` | 0.03s | 0.00s | 0.11s | 0.14s |
| `test_dsv32_partial_ownership_lowering_rejected` | 0.00s | 0.00s | 0.11s | 0.11s |

### `tests/dataflow/runtime/test_parity_vs_sim.py` — 1.2s total, 9 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_parity_8b_starved_pcie_recompute` | 0.24s | 0.00s | 0.10s | 0.34s |
| `test_parity_grad_accum_8b_scale` | 0.05s | 0.00s | 0.10s | 0.15s |
| `test_parity_8b_16gib` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_buffer_reuse_happens_at_scale` | 0.02s | 0.00s | 0.10s | 0.12s |
| `test_parity_8b_tight_budget` | 0.02s | 0.00s | 0.09s | 0.11s |
| `test_parity_tiny_tighter` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_parity_tiny_grad_accum` | 0.01s | 0.00s | 0.09s | 0.10s |
| `test_parity_tiny` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_parity_tiny_recompute_all` | 0.00s | 0.00s | 0.09s | 0.09s |

### `tests/dataflow_sim/planning/policies/test_pressurefit.py` — 1.2s total, 12 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_pressurefit_diagnostics_describe_selected_candidate` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_preplace_rejects_unknown_mode` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_extends_prefetch_intervals_under_strict_cap` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_prefetches_late_object_instead_of_preplacing` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_preserves_final_backing_mutation_writeback` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_uses_timing_relief_when_static_boundary_is_impossible` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_runs_training_chain_at_moderate_cap` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_evaluates_exactly_four_inbound_schedules` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressurefit_can_release_disposable_mutation_after_final_use` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_pressure_initial_placement_skips_hidden_future_use` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_packed_fifo_clamps_prefetch_fire_to_pressure` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_preplace_task0_limits_initial_fast_to_task0_inputs` | 0.00s | 0.00s | 0.10s | 0.10s |

### `tests/dataflow_training/training/e2e/test_packed_args_e2e.py` — 1.2s total, 7 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_packed_args_with_forced_recompute` | 0.07s | 0.00s | 0.14s | 0.21s |
| `test_packed_args_match_golden` | 0.06s | 0.00s | 0.15s | 0.21s |
| `test_no_args_is_legacy` | 0.05s | 0.00s | 0.14s | 0.19s |
| `test_packed_mode_materializes_positions_once` | 0.02s | 0.00s | 0.15s | 0.17s |
| `test_workload_segments_derives_max_seqlen_and_mirrors` | 0.00s | 0.00s | 0.15s | 0.15s |
| `test_resolve_segments_materializes_cuda_positions_and_caches` | 0.00s | 0.00s | 0.14s | 0.14s |
| `test_packed_mode_has_no_lowering_surface` | 0.00s | 0.00s | 0.14s | 0.14s |

### `tests/dataflow_training/training/surfaces/test_source_policy_drills.py` — 1.2s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_simple_policy_round_trip_world2` | 0.36s | 0.00s | 0.15s | 0.51s |
| `test_dedup_policy_covers_with_disjoint_slices` | 0.25s | 0.00s | 0.16s | 0.41s |
| `test_replication_drift_refuses_before_record` | 0.10s | 0.00s | 0.16s | 0.26s |

### `tests/dataflow/runtime/test_engine_semantics.py` — 1.2s total, 13 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_ledger_inversion_without_valve_deadlocks` | 0.00s | 0.00s | 0.10s | 0.10s |
| `test_release_of_non_live_object_errors` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_deferred_prefetch_waits_for_offload` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_stale_final_location_detected` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_initial_memory_over_capacity_errors` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_annotate_rename_rewrites_nvtx_only` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_cached_chain_index_matches_derived_across_repeat_runs` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_capacity_deadlock_raises` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_ledger_inversion_parity_baseline` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_re_prefetch_gets_distinct_interval_name` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_ledger_inversion_eviction_valve` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_blocked_queue_head_starts_when_bytes_free` | 0.00s | 0.00s | 0.09s | 0.09s |
| `test_mutation_offload_overwrites_backing_in_place` | 0.00s | 0.00s | 0.09s | 0.09s |

### `tests/dataflow_training/models/test_qwen35.py` — 1.2s total, 6 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_qwen35_tied_model_step_vs_golden` | 0.25s | 0.00s | 0.13s | 0.38s |
| `test_fla_chunk_bwd_matches_reference_autograd` | 0.10s | 0.00s | 0.13s | 0.23s |
| `test_reference_recurrence_matches_fla_naive` | 0.04s | 0.00s | 0.12s | 0.16s |
| `test_fla_chunk_fwd_matches_reference` | 0.02s | 0.00s | 0.13s | 0.15s |
| `test_conv_and_l2norm_helpers_match_references` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_qwen35_stage_context_completeness` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow_training/models/test_llama3.py` — 1.1s total, 9 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_model_step_muon_policy_golden_parity` | 0.03s | 0.00s | 0.13s | 0.16s |
| `test_block_backward_vs_autograd` | 0.01s | 0.00s | 0.14s | 0.15s |
| `test_rmsnorm_backward` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_ce_loss_fused` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_flash_wrapper_matches_autograd` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_embed_roundtrip` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_rope_backward_is_transpose` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_swiglu_backward` | 0.00s | 0.00s | 0.12s | 0.12s |
| `test_adamw_step_matches_manual` | 0.00s | 0.00s | 0.12s | 0.12s |

### `tests/dataflow_training/data/test_packing.py` — 1.1s total, 10 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_token_conservation_multiset_identity` | 0.01s | 0.00s | 0.11s | 0.12s |
| `test_pack_batch_deterministic_for_same_input` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_lpt_spread_bounded_by_largest_item` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_overflow_split_preserves_content` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_fixed_n_rounds_pads_whole_round` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_sum_len_sq_statistic` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_boundaries_and_tail_padding` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_overflow_error_policy` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_exact_fill_when_divisible` | 0.00s | 0.00s | 0.11s | 0.11s |
| `test_s_max_enforced` | 0.00s | 0.00s | 0.11s | 0.11s |

### `tests/test_program_hashes.py` — 1.0s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_lowered_program_hashes_stable` | 0.06s | 0.00s | 0.97s | 1.03s |


## Methodology and caveats

- Source: `--durations=0` phase report; pytest omits phases under 5ms (`--durations-min` default), so per-file sums are floors and very cheap tests may be absent entirely.
- Session/module-scoped fixture cost lands in the setup phase of the first test that triggers it — a file's first test can look artificially heavy.
- Serial single-run measurement under the serial-battery rule: concurrent GPU work on the box invalidates comparisons (contention reds are not regressions).
- The deselected count is the opt-in lanes (fleet); their cost is not in this document.
- MEASURE ON A WARM PROFILE CACHE. `impl_fingerprint()` hashes `run/profiling.py`, so any edit to that file invalidates every cached profile and the next run re-measures every signature from cold. Measured immediately after such an edit this suite came in at 28:33 against a true 14:16 — the first run after touching the profiling module times the cold path, not the suite.
- Files that profile under `tmp_path` (test_profiling_memory, test_engine_stress) are cold BY DESIGN and pay real device work every run; a warm cache does not help them.

