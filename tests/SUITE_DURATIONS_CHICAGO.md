# Test-suite duration reference (chicago (RTX 5090))

Measured at commit `886c041` on chicago (RTX 5090), single serial run of the canonical suite invocation (`python -m pytest -q --durations=0`, `dataflow` conda env, box otherwise idle). Use it as the expectation baseline for how long suite tasks take at this commit; re-measure after structural suite changes.

Stack: torch 2.13.0+cu130 / triton 3.7.1.

## Summary

- **Wall time: 13:24** (804s) — 1334 passed, 1 skipped.
- Time accounted to individual tests below: 793s (99% of wall; the rest is collection, session setup, and the many sub-5ms phases pytest omits).
- 118 test files with measurable time; distribution by per-file total: 4 files over 60s, 6 in 10-60s, 53 in 1-10s, 55 under 1s.
- Concentration: the top 10 files hold 645s (81% of accounted time) — they are the levers if the suite ever needs to get faster.

### Top 10 files

| # | file | tests | total | share of accounted |
|---|---|---|---|---|
| 1 | `tests/dataflow_training/pretrain/test_engine_parity_families.py` | 22 | 242.3s | 30.6% |
| 2 | `tests/dataflow_training/models/test_model_families.py` | 110 | 119.8s | 15.1% |
| 3 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` | 3 | 74.5s | 9.4% |
| 4 | `tests/dataflow_training/models/test_engine_vs_reference.py` | 20 | 61.9s | 7.8% |
| 5 | `tests/dataflow_training/tasks/test_kernel_audit.py` | 371 | 55.1s | 7.0% |
| 6 | `tests/examples/test_rl_training.py` | 6 | 27.1s | 3.4% |
| 7 | `tests/dataflow_training/data/test_data_pipeline.py` | 22 | 22.6s | 2.8% |
| 8 | `tests/dataflow_training/pretrain/test_client_model_step.py` | 4 | 19.6s | 2.5% |
| 9 | `tests/dataflow_training/training/surfaces/test_replicate_load.py` | 1 | 11.4s | 1.4% |
| 10 | `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` | 2 | 11.1s | 1.4% |

### Top 10 individual tests

| # | test | total |
|---|---|---|
| 1 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 32.5s |
| 2 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[qwen35_smoke_preset]` | 31.6s |
| 3 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_resume_reproduces_tail_bitwise` | 25.6s |
| 4 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_remapped_resume_bitwise` | 24.6s |
| 5 | `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py::test_world2_moe_persistent_set_round_trips` | 24.2s |
| 6 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_qwen35moe_engine_vs_reference` | 22.8s |
| 7 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[glm52_smoke_preset]` | 13.4s |
| 8 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_execute_padding_equivalence` | 13.3s |
| 9 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_poisoned_tail_is_dead_bytes` | 13.0s |
| 10 | `tests/dataflow_training/pretrain/test_engine_parity_families.py::test_underfull_engine_vs_reference[dsv32_smoke_preset]` | 12.8s |

## Per-file breakdown

Sorted by total attributed time (call + setup + teardown of every test in the file).

| file | tests | total | slowest test | its total |
|---|---|---|---|---|
| `tests/dataflow_training/pretrain/test_engine_parity_families.py` | 22 | 242.3s | `test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 32.5s |
| `tests/dataflow_training/models/test_model_families.py` | 110 | 119.8s | `test_poison_on_free_changes_nothing[glm52]` | 5.7s |
| `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` | 3 | 74.5s | `test_world2_resume_reproduces_tail_bitwise` | 25.6s |
| `tests/dataflow_training/models/test_engine_vs_reference.py` | 20 | 61.9s | `test_engine_matches_reference_uniform[qwen35moe]` | 9.7s |
| `tests/dataflow_training/tasks/test_kernel_audit.py` | 371 | 55.1s | `test_write_coverage_poison_invariance[ce_loss_fwd_bwd:triton:all_ignored]` | 0.2s |
| `tests/examples/test_rl_training.py` | 6 | 27.1s | `test_rl_training_parity_ppo[qwen35]` | 4.9s |
| `tests/dataflow_training/data/test_data_pipeline.py` | 22 | 22.6s | `test_checkpoint_resume_tail_matches_uninterrupted_run` | 9.3s |
| `tests/dataflow_training/pretrain/test_client_model_step.py` | 4 | 19.6s | `test_client_model_step_matches_in_process_olmoe` | 6.2s |
| `tests/dataflow_training/training/surfaces/test_replicate_load.py` | 1 | 11.4s | `test_world1_replicate_steps_once_bitwise` | 11.4s |
| `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` | 2 | 11.1s | `test_kill_escalates_past_sigterm` | 10.7s |
| `tests/dataflow/runtime/test_profiling_memory.py` | 1 | 7.3s | `test_tables_leave_no_reserved_memory` | 7.3s |
| `tests/dataflow/service/test_shared_server_self_heal.py` | 1 | 5.2s | `test_self_heal_respawns_after_illegal_access` | 5.2s |
| `tests/dataflow_training/tasks/test_kernels.py` | 21 | 4.8s | `test_swiglu_fused[4099-14336]` | 1.0s |
| `tests/dataflow_sim/engine/test_simulator.py` | 38 | 4.0s | `test_repeated_transfers_of_same_object_get_unique_task_ids` | 0.1s |
| `tests/dataflow_sim/planning/policies/test_auto_policy.py` | 36 | 3.9s | `test_auto_policy_L10_works_down_to_cap_500[500]` | 0.1s |
| `tests/dataflow/service/test_slice_snapshots.py` | 12 | 3.8s | `test_slice_roundtrip_and_compose` | 0.6s |
| `tests/dataflow/service/test_daemon_relaunch.py` | 1 | 3.7s | `test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses` | 3.7s |
| `tests/dataflow_training/modules/test_moe.py` | 24 | 3.1s | `test_moe_tail_fwd_bwd_vs_reference[True-0.001-topk_then_softmax]` | 0.1s |
| `tests/dataflow_training/training/lowering/test_layout_registry.py` | 21 | 3.1s | `test_registry_covers_external_family` | 0.2s |
| `tests/dataflow/service/test_service_skeleton.py` | 10 | 3.1s | `test_fast_path_answers_while_dispatcher_held` | 0.8s |
| `tests/dataflow_sim/core/test_validate_chain.py` | 29 | 3.0s | `test_invalid_chain_rejected[make_invalid_release_mutation_dirty_with_later_use-make_invalid_release_mutation_dirty_with_later_use]` | 0.1s |
| `tests/dataflow_training/pretrain/test_client_fetch_surface.py` | 2 | 3.0s | `test_client_fetch_surface_dense` | 1.6s |
| `tests/dataflow/service/test_service_store.py` | 16 | 3.0s | `test_real_boot_family_init_byte_identity` | 0.8s |
| `tests/dataflow_training/training/e2e/test_freeze_plan.py` | 17 | 2.8s | `test_model_step_truncated_olmoe` | 0.2s |
| `tests/dataflow_sim/app/test_server.py` | 16 | 2.8s | `test_simulate_large_chain_uses_snapshot_free_response` | 1.1s |
| `tests/dataflow_training/training/e2e/test_varlen_e2e.py` | 11 | 2.8s | `test_model_step_ragged_matches_golden_all_families[qwen35moe]` | 0.5s |
| `tests/dataflow_sim/workloads/test_modular_workload_builder.py` | 24 | 2.7s | `test_constrained_memory_recompute_planning_selects_useful_variants` | 0.2s |
| `tests/dataflow_training/training/surfaces/test_checkpoint_record.py` | 4 | 2.5s | `test_load_checkpoint_targets` | 0.8s |
| `tests/dataflow_training/training/surfaces/test_solo_resume_bitwise.py` | 1 | 2.5s | `test_solo_resume_reproduces_tail_bitwise` | 2.5s |
| `tests/dataflow_training/training/planning/test_planning.py` | 9 | 2.4s | `test_backing_capacity_drives_recompute` | 0.5s |
| `tests/dataflow_training/pretrain/test_presets.py` | 16 | 2.3s | `test_preset_lowers[l3_760m]` | 0.2s |
| `tests/dataflow_sim/planning/policies/test_min_grow.py` | 22 | 2.3s | `test_derive_schedule_pre_places_backing_init_with_a_minus_1` | 0.1s |
| `tests/dataflow_training/pretrain/test_flops.py` | 15 | 2.1s | `test_every_family_walks[llama3]` | 0.1s |
| `tests/dataflow/runtime/test_engine_stress.py` | 3 | 2.1s | `test_measured_costs_replan_still_golden` | 1.5s |
| `tests/dataflow_sim/workloads/test_dataflow_schema.py` | 18 | 2.0s | `test_metrics_preview_and_summary_metadata` | 0.1s |
| `tests/dataflow_training/pretrain/test_sharding.py` | 13 | 1.9s | `test_real_llama3_layouts_shard` | 0.1s |
| `tests/dataflow_training/pretrain/test_parity_smoke.py` | 1 | 1.8s | `test_reference_vs_service_parity_smoke` | 1.8s |
| `tests/dataflow_training/training/e2e/test_lbl_modes.py` | 8 | 1.8s | `test_retained_router_delta_is_ga_invariant_per_round_is_not` | 0.4s |
| `tests/dataflow/service/test_pinned_slab.py` | 3 | 1.8s | `test_slab_costs_what_it_asks_for` | 0.8s |
| `tests/dataflow/service/test_peer_protocol.py` | 17 | 1.7s | `test_eager_happy_path` | 0.1s |
| `tests/dataflow_training/tasks/test_optim.py` | 11 | 1.7s | `test_mixed_policy_model_step_vs_hand_replica` | 0.2s |
| `tests/test_import_boundaries.py` | 7 | 1.7s | `test_sim_required_only_under_lowering` | 0.4s |
| `tests/dataflow_training/training/e2e/test_dtype_policy_e2e.py` | 7 | 1.7s | `test_qwen35_model_step_mixed_policy` | 0.4s |
| `tests/dataflow/service/test_engine_determinism.py` | 1 | 1.5s | `test_same_daemon_rerun_bitwise` | 1.5s |
| `tests/dataflow_training/models/test_gpt2.py` | 10 | 1.5s | `test_qkv_bias_grad_sections` | 0.2s |
| `tests/dataflow_training/modules/test_dsa.py` | 10 | 1.5s | `test_index_scores_vs_hand_loop` | 0.2s |
| `tests/dataflow_training/models/test_glm52.py` | 7 | 1.5s | `test_glm52_dense_warmup_model_step` | 0.3s |
| `tests/dataflow/service/test_service_snapshot.py` | 3 | 1.4s | `test_checkpoint_roundtrip_bit_continuity` | 0.8s |
| `tests/dataflow_training/models/test_block_isolation.py` | 5 | 1.4s | `test_isolated_block_at_floor[glm52-isolate0-6]` | 0.5s |
| `tests/dataflow/service/test_service_runs.py` | 5 | 1.4s | `test_rebind_two_token_slabs` | 0.7s |
| `tests/dataflow_training/training/lowering/test_shaped_program.py` | 9 | 1.4s | `test_tied_embeddings_chain_structure` | 0.2s |
| `tests/dataflow_training/models/test_dsv32.py` | 7 | 1.3s | `test_dsv32_dense_warmup_model_step` | 0.2s |
| `tests/dataflow/runtime/test_parity_vs_sim.py` | 9 | 1.3s | `test_parity_8b_starved_pcie_recompute` | 0.3s |
| `tests/dataflow_training/training/surfaces/test_source_policy_drills.py` | 3 | 1.2s | `test_simple_policy_round_trip_world2` | 0.6s |
| `tests/dataflow_training/training/e2e/test_packed_args_e2e.py` | 7 | 1.2s | `test_packed_args_with_forced_recompute` | 0.2s |
| `tests/dataflow_sim/planning/policies/test_pressurefit.py` | 12 | 1.2s | `test_pressurefit_runs_training_chain_at_moderate_cap` | 0.1s |
| `tests/dataflow_training/data/test_packing.py` | 10 | 1.2s | `test_token_conservation_multiset_identity` | 0.1s |
| `tests/dataflow_training/models/test_llama3.py` | 9 | 1.2s | `test_model_step_muon_policy_golden_parity` | 0.2s |
| `tests/dataflow/runtime/test_engine_semantics.py` | 12 | 1.2s | `test_stale_final_location_detected` | 0.1s |
| `tests/dataflow_training/models/test_qwen35.py` | 6 | 1.1s | `test_qwen35_tied_model_step_vs_golden` | 0.4s |
| `tests/dataflow/core/test_ir_validate.py` | 11 | 1.1s | `test_tensor_size_mismatch_rejected` | 0.1s |
| `tests/test_program_hashes.py` | 1 | 1.0s | `test_lowered_program_hashes_stable` | 1.0s |
| `tests/dataflow/runtime/test_run_contract.py` | 10 | 1.0s | `test_task_raise_no_crash_on_cuda` | 0.1s |
| `tests/dataflow/runtime/test_cuda_backend.py` | 5 | 1.0s | `test_mini_program_execution_matches_plan` | 0.4s |
| `tests/dataflow_training/modules/test_mla.py` | 7 | 0.9s | `test_dsv3_block_fwd_recompute_bwd_accum_match_autograd_golden[moe]` | 0.1s |
| `tests/dataflow_training/training/e2e/test_ga_invariance.py` | 4 | 0.9s | `test_sgd_rounds_are_memory_optimization` | 0.3s |
| `tests/dataflow_training/training/lowering/test_responsibility.py` | 6 | 0.9s | `test_world1_full_coverage` | 0.1s |
| `tests/dataflow_training/tasks/test_varlen_attention.py` | 6 | 0.9s | `test_no_hidden_syncs` | 0.1s |
| `tests/dataflow/checkpoint/test_record_layer.py` | 9 | 0.9s | `test_slice_reference_bounds` | 0.1s |
| `tests/dataflow_sim/planning/test_recompute.py` | 8 | 0.9s | `test_zero_runtime_recompute_placeholders_are_schedule_neutral` | 0.1s |
| `tests/dataflow/service/test_nccl_binding.py` | 2 | 0.8s | `test_binding_world1_roundtrip` | 0.7s |
| `tests/dataflow/runtime/test_placement.py` | 7 | 0.8s | `test_parity_with_placement_8b` | 0.2s |
| `tests/dataflow_training/pretrain/test_reference_muon.py` | 5 | 0.8s | `test_tiny_muon_reference_trains` | 0.2s |
| `tests/dataflow_training/pretrain/test_sharding_lowering.py` | 5 | 0.8s | `test_programs_json_serializable_and_plain_unchanged` | 0.2s |
| `tests/dataflow_training/models/test_qwen3moe.py` | 5 | 0.8s | `test_qwen3moe_grad_accum_two_rounds_matches_reference` | 0.2s |
| `tests/dataflow/runtime/test_vmm.py` | 7 | 0.7s | `test_e2e_mini_vmm_matches_static` | 0.1s |
| `tests/dataflow_training/tasks/test_dtype_policy.py` | 5 | 0.7s | `test_mixed_roles_carry_independently` | 0.1s |
| `tests/dataflow_training/pretrain/test_topology.py` | 3 | 0.7s | `test_daemonize_detach_and_group_kill` | 0.4s |
| `tests/dataflow_training/models/test_glm52_lowering.py` | 5 | 0.6s | `test_full_scale_presets_lower` | 0.1s |
| `tests/dataflow/service/test_service_packed_args.py` | 1 | 0.6s | `test_daemon_packed_args_bit_equal` | 0.6s |
| `tests/dataflow_training/models/test_dsv3.py` | 4 | 0.6s | `test_dsv3_aux_zero_model_step_vs_golden` | 0.2s |
| `tests/dataflow_training/models/test_olmoe.py` | 4 | 0.6s | `test_olmoe_aux_zero_model_step_vs_golden` | 0.2s |
| `tests/dataflow_training/training/lowering/test_parallelism_scheme.py` | 4 | 0.6s | `test_data_parallel_axis_views` | 0.1s |
| `tests/dataflow_training/training/surfaces/test_plugins.py` | 4 | 0.6s | `test_validate_family_reports_broken_surface` | 0.1s |
| `tests/dataflow_training/pretrain/test_schedule.py` | 4 | 0.6s | `test_matches_engine_lrschedule_exactly` | 0.1s |
| `tests/dataflow/core/test_sim_convert.py` | 6 | 0.6s | `test_annotated_chain_validates` | 0.1s |
| `tests/test_docstring_index.py` | 2 | 0.6s | `test_index_matches_test_functions` | 0.3s |
| `tests/dataflow_training/tasks/test_ignore_index_ce.py` | 4 | 0.6s | `test_no_ignore_rows_matches_torch_ce_and_rerun_is_bitwise[triton]` | 0.1s |
| `tests/dataflow_training/models/test_qwen3.py` | 4 | 0.5s | `test_qwen3_block_backward` | 0.1s |
| `tests/dataflow_sim/core/test_reference_stream.py` | 5 | 0.5s | `test_next_ref_finds_first_appearance` | 0.1s |
| `tests/dataflow/checkpoint/test_record_targets.py` | 5 | 0.5s | `test_all_targets_resolve_each_byte_once` | 0.1s |
| `tests/dataflow/core/test_json_roundtrip.py` | 5 | 0.5s | `test_recompute_variant_roundtrips` | 0.1s |
| `tests/dataflow_training/training/lowering/test_round_prologue.py` | 3 | 0.5s | `test_round_prologue_publishes_round_index_via_run_values_and_object` | 0.2s |
| `tests/dataflow_training/training/lowering/test_group_annotation.py` | 3 | 0.4s | `test_zero1rs_annotation_matches_builder` | 0.1s |
| `tests/dataflow/checkpoint/test_persistent_targets.py` | 4 | 0.4s | `test_marker_default_and_emit_when_true` | 0.1s |
| `tests/dataflow/service/test_service_events.py` | 2 | 0.4s | `test_event_coverage_and_reattach` | 0.2s |
| `tests/dataflow/service/test_peer_groups.py` | 2 | 0.4s | `test_group_lifecycle_and_error_fanout` | 0.2s |
| `tests/dataflow/service/test_error_codes.py` | 1 | 0.4s | `test_every_raised_error_code_is_registered` | 0.4s |
| `tests/dataflow_training/training/e2e/test_batch_ga.py` | 2 | 0.4s | `test_batch_ga_model_step_matches_reference` | 0.2s |
| `tests/dataflow_training/data/test_shard_corpus.py` | 3 | 0.3s | `test_header_parse` | 0.1s |
| `tests/dataflow_training/training/surfaces/test_webapp_upload.py` | 2 | 0.3s | `test_simulate_schema_upload` | 0.2s |
| `tests/dataflow/runtime/test_reserve_inversion.py` | 2 | 0.3s | `test_caller_priority_prevents_poke_starvation` | 0.2s |
| `tests/dataflow/runtime/test_backing_free.py` | 3 | 0.3s | `test_backing_freed_after_last_use` | 0.1s |
| `tests/test_external_family.py` | 2 | 0.3s | `test_external_family_composes_with_service_path` | 0.1s |
| `tests/dataflow_training/training/lowering/test_persist_marker.py` | 2 | 0.3s | `test_moe_aux_rides_the_marker` | 0.1s |
| `tests/dataflow_training/tasks/test_staged_blocks.py` | 2 | 0.3s | `test_stage_context_completeness` | 0.1s |
| `tests/dataflow_training/pretrain/test_tp_layouts.py` | 2 | 0.3s | `test_per_rank_layout_shapes_and_sizes` | 0.1s |
| `tests/dataflow/service/test_registration.py` | 2 | 0.2s | `test_register_all_resolves_every_family` | 0.1s |
| `tests/dataflow/service/test_hostbw.py` | 2 | 0.2s | `test_probe_reports_positive_lanes` | 0.1s |
| `tests/dataflow_training/test_client_only.py` | 1 | 0.2s | `test_workload_tests_are_client_only` | 0.2s |
| `tests/dataflow/service/test_active_pools.py` | 2 | 0.2s | `test_active_pools_reports_live_pools_scoped_to_a_daemon` | 0.1s |
| `tests/dataflow/runtime/test_view_lifetime.py` | 2 | 0.2s | `test_free_evicts_cache_no_stale_view` | 0.1s |
| `tests/dataflow_training/training/lowering/test_lowering_stability.py` | 1 | 0.2s | `test_lowered_programs_bit_identical` | 0.2s |
| `tests/dataflow/test_workload_blind.py` | 1 | 0.1s | `test_engine_tests_are_workload_blind` | 0.1s |
| `tests/dataflow_training/models/test_qwen35moe.py` | 1 | 0.1s | `test_qwen35moe_stage_context_completeness` | 0.1s |
| `tests/dataflow/service/test_store_allocator_concurrency.py` | 1 | 0.1s | `test_two_writer_allocator_invariants` | 0.1s |
| `tests/dataflow/runtime/test_trace_dict.py` | 1 | 0.1s | `test_trace_to_dict_covers_every_task_interval` | 0.1s |
| `tests/dataflow/checkpoint/test_record_boundary.py` | 1 | 0.1s | `test_checkpoint_package_is_workload_blind` | 0.1s |

## Individual tests at or above 1.0s

Grouped by file, slowest first within each. Tests under 1.0s are covered by the per-file totals above.

### `tests/dataflow_training/pretrain/test_engine_parity_families.py` — 242.3s total, 22 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_underfull_engine_vs_reference[qwen35moe_smoke_preset]` | 32.33s | 0.01s | 0.15s | 32.49s |
| `test_underfull_engine_vs_reference[qwen35_smoke_preset]` | 31.48s | 0.00s | 0.15s | 31.63s |
| `test_qwen35moe_engine_vs_reference` | 22.62s | 0.00s | 0.14s | 22.76s |
| `test_underfull_engine_vs_reference[glm52_smoke_preset]` | 13.21s | 0.00s | 0.16s | 13.37s |
| `test_underfull_execute_padding_equivalence` | 13.18s | 0.00s | 0.15s | 13.33s |
| `test_underfull_poisoned_tail_is_dead_bytes` | 12.82s | 0.00s | 0.15s | 12.97s |
| `test_underfull_engine_vs_reference[dsv32_smoke_preset]` | 12.69s | 0.00s | 0.15s | 12.84s |
| `test_underfull_engine_vs_reference[dsv3_smoke_preset]` | 12.60s | 0.00s | 0.15s | 12.75s |
| `test_underfull_engine_vs_reference[olmoe_smoke_preset]` | 12.43s | 0.00s | 0.15s | 12.58s |
| `test_underfull_engine_vs_reference[qwen3moe_smoke_preset]` | 12.33s | 0.00s | 0.15s | 12.48s |
| `test_underfull_engine_vs_reference[smoke_preset]` | 12.04s | 0.00s | 0.14s | 12.18s |
| `test_underfull_engine_vs_reference[qwen3_smoke_preset]` | 11.93s | 0.00s | 0.14s | 12.07s |
| `test_underfull_engine_vs_reference[gpt2_smoke_preset]` | 11.88s | 0.00s | 0.14s | 12.02s |
| `test_gpt2_docaware_engine_vs_reference` | 7.13s | 0.00s | 0.13s | 7.26s |
| `test_glm52_engine_vs_reference` | 3.20s | 0.00s | 0.14s | 3.34s |
| `test_dsv32_engine_vs_reference` | 2.70s | 0.00s | 0.14s | 2.84s |
| `test_olmoe_engine_vs_reference_lbl_on` | 2.61s | 0.00s | 0.14s | 2.75s |
| `test_olmoe_engine_vs_reference` | 2.58s | 0.00s | 0.14s | 2.72s |
| `test_qwen3moe_engine_vs_reference` | 2.54s | 0.00s | 0.14s | 2.68s |
| `test_dsv3_engine_vs_reference` | 2.53s | 0.00s | 0.14s | 2.67s |
| `test_qwen3_engine_vs_reference` | 2.18s | 0.00s | 0.14s | 2.32s |
| `test_gpt2_engine_vs_reference` | 2.10s | 0.00s | 0.14s | 2.24s |

### `tests/dataflow_training/models/test_model_families.py` — 119.8s total, 110 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_poison_on_free_changes_nothing[glm52]` | 5.53s | 0.00s | 0.13s | 5.66s |
| `test_poison_on_free_changes_nothing[qwen35moe]` | 5.53s | 0.00s | 0.13s | 5.66s |
| `test_poison_on_free_changes_nothing[dsv32]` | 5.50s | 0.00s | 0.13s | 5.63s |
| `test_poison_on_free_changes_nothing[qwen3moe]` | 5.48s | 0.00s | 0.13s | 5.61s |
| `test_poison_on_free_changes_nothing[llama3]` | 5.46s | 0.00s | 0.13s | 5.59s |
| `test_poison_on_free_changes_nothing[olmoe]` | 5.43s | 0.00s | 0.13s | 5.56s |
| `test_poison_on_free_changes_nothing[qwen3]` | 5.38s | 0.01s | 0.13s | 5.52s |
| `test_poison_on_free_changes_nothing[gpt2]` | 5.32s | 0.00s | 0.13s | 5.45s |
| `test_poison_on_free_changes_nothing[dsv3]` | 5.25s | 0.00s | 0.13s | 5.38s |
| `test_poison_on_free_changes_nothing[qwen35]` | 5.17s | 0.08s | 0.13s | 5.38s |
| `test_golden_model_step[dsv3]` | 2.86s | 2.06s | 0.13s | 5.05s |
| `test_golden_model_step[qwen35moe]` | 1.29s | 0.00s | 0.13s | 1.42s |
| `test_golden_model_step[qwen35]` | 1.27s | 0.00s | 0.13s | 1.40s |
| `test_reference_twin_build_is_stateless[qwen35moe]` | 1.14s | 0.00s | 0.13s | 1.27s |
| `test_grad_accum_two_rounds[qwen35moe]` | 1.09s | 0.00s | 0.13s | 1.22s |
| `test_reference_twin_build_is_stateless[qwen35]` | 1.06s | 0.00s | 0.13s | 1.19s |
| `test_grad_accum_two_rounds[qwen35]` | 1.02s | 0.00s | 0.13s | 1.15s |

### `tests/dataflow_training/training/surfaces/test_world2_resume_bitwise.py` — 74.5s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_world2_resume_reproduces_tail_bitwise` | 25.45s | 0.00s | 0.15s | 25.60s |
| `test_world2_remapped_resume_bitwise` | 24.49s | 0.00s | 0.16s | 24.65s |
| `test_world2_moe_persistent_set_round_trips` | 24.07s | 0.00s | 0.16s | 24.23s |

### `tests/dataflow_training/models/test_engine_vs_reference.py` — 61.9s total, 20 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_engine_matches_reference_uniform[qwen35moe]` | 9.58s | 0.00s | 0.13s | 9.71s |
| `test_engine_matches_reference_uniform[qwen35]` | 9.53s | 0.00s | 0.13s | 9.66s |
| `test_engine_matches_reference_ragged[qwen35moe]` | 5.69s | 0.00s | 0.14s | 5.83s |
| `test_engine_matches_reference_ragged[qwen35]` | 5.35s | 0.00s | 0.13s | 5.48s |
| `test_engine_matches_reference_ragged[glm52]` | 2.15s | 0.00s | 0.13s | 2.28s |
| `test_engine_matches_reference_uniform[glm52]` | 2.07s | 0.00s | 0.13s | 2.20s |
| `test_engine_matches_reference_ragged[dsv32]` | 1.93s | 0.00s | 0.14s | 2.07s |
| `test_engine_matches_reference_ragged[olmoe]` | 1.94s | 0.00s | 0.13s | 2.07s |
| `test_engine_matches_reference_uniform[olmoe]` | 1.91s | 0.00s | 0.13s | 2.04s |
| `test_engine_matches_reference_uniform[dsv32]` | 1.91s | 0.00s | 0.12s | 2.03s |
| `test_engine_matches_reference_ragged[qwen3moe]` | 1.86s | 0.00s | 0.13s | 1.99s |
| `test_engine_matches_reference_ragged[dsv3]` | 1.85s | 0.00s | 0.14s | 1.99s |
| `test_engine_matches_reference_uniform[qwen3moe]` | 1.83s | 0.00s | 0.13s | 1.96s |
| `test_engine_matches_reference_uniform[dsv3]` | 1.80s | 0.00s | 0.13s | 1.93s |
| `test_engine_matches_reference_uniform[gpt2]` | 1.67s | 0.00s | 0.13s | 1.80s |
| `test_engine_matches_reference_uniform[qwen3]` | 1.66s | 0.00s | 0.13s | 1.79s |
| `test_engine_matches_reference_uniform[llama3]` | 1.64s | 0.00s | 0.13s | 1.77s |
| `test_engine_matches_reference_ragged[qwen3]` | 1.64s | 0.00s | 0.13s | 1.77s |
| `test_engine_matches_reference_ragged[llama3]` | 1.61s | 0.00s | 0.14s | 1.75s |
| `test_engine_matches_reference_ragged[gpt2]` | 1.61s | 0.00s | 0.13s | 1.74s |

### `tests/examples/test_rl_training.py` — 27.1s total, 6 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_rl_training_parity_ppo[qwen35]` | 4.82s | 0.00s | 0.10s | 4.92s |
| `test_rl_training_parity_ppo[glm52]` | 4.43s | 0.00s | 0.10s | 4.53s |
| `test_rl_training_parity_reinforce` | 4.43s | 0.00s | 0.10s | 4.53s |
| `test_rl_training_parity_ppo[dsv32]` | 4.39s | 0.00s | 0.11s | 4.50s |
| `test_rl_training_parity_ppo[qwen3moe]` | 4.23s | 0.00s | 0.10s | 4.33s |
| `test_rl_training_parity_ppo[llama3]` | 4.17s | 0.00s | 0.10s | 4.27s |

### `tests/dataflow_training/data/test_data_pipeline.py` — 22.6s total, 22 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_checkpoint_resume_tail_matches_uninterrupted_run` | 9.19s | 0.00s | 0.11s | 9.30s |
| `test_shard_doc_mode_tokens_in_range_and_cursor_resume` | 5.24s | 0.00s | 0.11s | 5.35s |
| `test_legacy_doc_configuration_pinned` | 5.22s | 0.00s | 0.11s | 5.33s |

### `tests/dataflow_training/pretrain/test_client_model_step.py` — 19.6s total, 4 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_client_model_step_matches_in_process_olmoe` | 6.04s | 0.00s | 0.14s | 6.18s |
| `test_client_model_step_matches_in_process_llama3` | 5.80s | 0.00s | 0.14s | 5.94s |
| `test_client_model_step_llama3_passes` | 5.17s | 0.00s | 0.13s | 5.30s |
| `test_out_of_process_daemon_boots_and_reaps` | 2.07s | 0.00s | 0.13s | 2.20s |

### `tests/dataflow_training/training/surfaces/test_replicate_load.py` — 11.4s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_world1_replicate_steps_once_bitwise` | 11.21s | 0.00s | 0.15s | 11.36s |

### `tests/dataflow_training/training/surfaces/test_daemonize_kill.py` — 11.1s total, 2 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_kill_escalates_past_sigterm` | 10.55s | 0.00s | 0.15s | 10.70s |

### `tests/dataflow/runtime/test_profiling_memory.py` — 7.3s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_tables_leave_no_reserved_memory` | 7.17s | 0.00s | 0.11s | 7.28s |

### `tests/dataflow/service/test_shared_server_self_heal.py` — 5.2s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_self_heal_respawns_after_illegal_access` | 5.09s | 0.00s | 0.10s | 5.19s |

### `tests/dataflow_training/tasks/test_kernels.py` — 4.8s total, 21 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_swiglu_fused[4099-14336]` | 0.86s | 0.00s | 0.16s | 1.02s |

### `tests/dataflow/service/test_daemon_relaunch.py` — 3.7s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_relaunched_daemon_same_program_reruns_clean_and_reproduces_losses` | 3.55s | 0.00s | 0.11s | 3.66s |

### `tests/dataflow_training/pretrain/test_client_fetch_surface.py` — 3.0s total, 2 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_client_fetch_surface_dense` | 1.47s | 0.00s | 0.13s | 1.60s |
| `test_client_fetch_surface_moe_aux` | 1.25s | 0.00s | 0.13s | 1.38s |

### `tests/dataflow_sim/app/test_server.py` — 2.8s total, 16 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_simulate_large_chain_uses_snapshot_free_response` | 1.04s | 0.00s | 0.10s | 1.14s |

### `tests/dataflow_training/training/surfaces/test_solo_resume_bitwise.py` — 2.5s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_solo_resume_reproduces_tail_bitwise` | 2.30s | 0.00s | 0.16s | 2.46s |

### `tests/dataflow/runtime/test_engine_stress.py` — 2.1s total, 3 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_measured_costs_replan_still_golden` | 1.37s | 0.00s | 0.11s | 1.48s |

### `tests/dataflow_training/pretrain/test_parity_smoke.py` — 1.8s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_reference_vs_service_parity_smoke` | 1.70s | 0.00s | 0.15s | 1.85s |

### `tests/dataflow/service/test_engine_determinism.py` — 1.5s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_same_daemon_rerun_bitwise` | 1.42s | 0.00s | 0.11s | 1.53s |

### `tests/test_program_hashes.py` — 1.0s total, 1 tests

| test | call | setup | teardown | total |
|---|---|---|---|---|
| `test_lowered_program_hashes_stable` | 0.07s | 0.00s | 0.97s | 1.04s |

## Cross-box expectations

Banked full-suite wall times for the same suite (gate runs, not per-test measurements): chicago (RTX 5090) ~12.5-13.5 min; tubingen (RTX 3090) ~15-18 min; della H100 node ~17-20 min (slower CPU and GPFS dominate the non-GPU phases). Per-test ratios are NOT uniform across boxes — GPU-heavy tests track the device, data/IO-heavy tests track the filesystem.

## Methodology and caveats

- Source: `--durations=0` phase report; pytest omits phases under 5ms (`--durations-min` default), so per-file sums are floors and very cheap tests may be absent entirely.
- Session/module-scoped fixture cost lands in the setup phase of the first test that triggers it — a file's first test can look artificially heavy.
- Serial single-run measurement under the serial-battery rule: concurrent GPU work on the box invalidates comparisons (contention reds are not regressions).
- The deselected count is the opt-in lanes (fleet); their cost is not in this document.
