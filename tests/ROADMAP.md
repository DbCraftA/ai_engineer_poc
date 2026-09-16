# Tests Roadmap

| ID | Concept | Test | Code src cible | Statut |
| --- | --- | --- | --- | --- |
| 1.1 | environment, CPU/GPU, disponibilité CUDA | tests/01_tensors/test_environment.py::test_pytorch_is_available | infrastructure uniquement | TODO |
| 1.1 | environment, CPU/GPU, disponibilité CUDA | tests/01_tensors/test_environment.py::test_cuda_tests_are_skipped_when_cuda_is_unavailable | tests/conftest.py | TODO |
| 1.2 | tensor, shape, dimension, numel | tests/01_tensors/test_tensor_shape_and_storage.py::test_tensor_shape_represents_logical_dimensions | src/inference_lab/tensors/inspection.py | TODO |
| 1.2 | tensor, shape, dimension, numel | tests/01_tensors/test_tensor_shape_and_storage.py::test_numel_is_product_of_dimensions | src/inference_lab/tensors/inspection.py | TODO |
| 1.3 | storage, partage mémoire | tests/01_tensors/test_tensor_shape_and_storage.py::test_view_shares_storage_with_source_tensor | src/inference_lab/tensors/inspection.py | TODO |
| 1.3 | storage, partage mémoire | tests/01_tensors/test_tensor_shape_and_storage.py::test_clone_owns_independent_storage | src/inference_lab/tensors/inspection.py | TODO |
| 1.4 | stride | tests/01_tensors/test_stride_and_layout.py::test_contiguous_tensor_has_expected_strides | src/inference_lab/tensors/layout.py | TODO |
| 1.4 | stride | tests/01_tensors/test_stride_and_layout.py::test_stride_maps_indices_to_storage_offsets | src/inference_lab/tensors/layout.py | TODO |
| 1.5 | transpose | tests/01_tensors/test_stride_and_layout.py::test_transpose_changes_strides_without_reordering_storage | src/inference_lab/tensors/layout.py | TODO |
| 1.6 | contiguous, materialization | tests/01_tensors/test_stride_and_layout.py::test_contiguous_materializes_transposed_layout | src/inference_lab/tensors/layout.py | TODO |
| 1.7 | view, reshape | tests/01_tensors/test_view_and_reshape.py::test_view_does_not_copy_when_layout_allows_it | src/inference_lab/tensors/layout.py | TODO |
| 1.7 | view, reshape | tests/01_tensors/test_view_and_reshape.py::test_reshape_may_materialize_when_required | src/inference_lab/tensors/layout.py | TODO |
| 1.8 | FP32, FP16, BF16 | tests/01_tensors/test_dtypes.py::test_dtype_controls_bytes_per_element | src/inference_lab/tensors/dtypes.py | TODO |
| 1.8 | FP32, FP16, BF16 | tests/01_tensors/test_dtypes.py::test_reduced_precision_changes_numerical_accuracy | src/inference_lab/tensors/dtypes.py | TODO |
| 1.8 | plage dynamique, overflow | tests/01_tensors/test_dtypes.py::test_fp16_overflows_where_bf16_keeps_dynamic_range | src/inference_lab/tensors/dtypes.py | TODO |
| 1.8 | plage dynamique, flush-to-zero | tests/01_tensors/test_dtypes.py::test_fp16_flushes_tiny_values_to_zero_while_absolute_error_hides_it | src/inference_lab/tensors/dtypes.py | TODO |
| 1.9 | mémoire tensor | tests/01_tensors/test_dtypes.py::test_tensor_memory_equals_numel_times_element_size | src/inference_lab/tensors/memory.py | TODO |
| 1.10 | matmul, M/N/K | tests/01_tensors/test_matmul.py::test_matrix_multiplication_produces_expected_shape | src/inference_lab/calculators/flops.py | TODO |
| 1.10 | matmul, M/N/K | tests/01_tensors/test_matmul.py::test_matmul_flops_can_be_estimated_from_mnk | src/inference_lab/calculators/flops.py | TODO |
| 1.11 | GEMM / GEMV | tests/01_tensors/test_matmul.py::test_gemv_is_matmul_with_single_output_row_or_vector_workload | documentation/calculators | TODO |
| 2.1 | projection linéaire Q/K/V | tests/02_attention/test_qkv_projection.py::test_qkv_projections_produce_expected_shapes | src/inference_lab/nn/attention/qkv.py | TODO |
| 2.2 | score QKᵀ | tests/02_attention/test_attention_scores.py::test_attention_scores_compare_every_query_with_every_key | src/inference_lab/nn/attention/naive.py | TODO |
| 2.3 | scaling | tests/02_attention/test_attention_scores.py::test_attention_scores_are_scaled_by_inverse_sqrt_head_dimension | src/inference_lab/nn/attention/naive.py | TODO |
| 2.4 | causal mask | tests/02_attention/test_causal_mask.py::test_causal_mask_prevents_future_tokens_from_contributing | src/inference_lab/nn/attention/mask.py | TODO |
| 2.5 | softmax | tests/02_attention/test_attention_softmax.py::test_attention_probabilities_sum_to_one | src/inference_lab/nn/attention/naive.py | TODO |
| 2.5 | softmax | tests/02_attention/test_attention_softmax.py::test_masked_positions_receive_zero_probability | src/inference_lab/nn/attention/naive.py | TODO |
| 2.6 | weighted V | tests/02_attention/test_attention_output.py::test_attention_output_is_weighted_sum_of_values | src/inference_lab/nn/attention/naive.py | TODO |
| 2.7 | MHA | tests/02_attention/test_multi_head_attention.py::test_hidden_dimension_is_split_across_attention_heads | src/inference_lab/nn/attention/mha.py | TODO |
| 2.7 | MHA | tests/02_attention/test_multi_head_attention.py::test_attention_heads_are_concatenated_back_to_hidden_dimension | src/inference_lab/nn/attention/mha.py | TODO |
| 2.8 | GQA | tests/02_attention/test_grouped_query_attention.py::test_gqa_uses_fewer_kv_heads_than_query_heads | src/inference_lab/nn/attention/gqa.py | TODO |
| 2.8 | GQA | tests/02_attention/test_grouped_query_attention.py::test_multiple_query_heads_share_key_value_heads | src/inference_lab/nn/attention/gqa.py | TODO |
| 2.9 | MQA | tests/02_attention/test_grouped_query_attention.py::test_mqa_uses_single_key_value_head | src/inference_lab/nn/attention/gqa.py | TODO |
| 2.10 | RoPE | tests/02_attention/test_rope.py::test_rope_preserves_vector_norm | src/inference_lab/nn/positional/rope.py | TODO |
| 2.10 | RoPE | tests/02_attention/test_rope.py::test_rope_changes_representation_according_to_position | src/inference_lab/nn/positional/rope.py | TODO |
| 2.11 | RMSNorm | tests/02_attention/test_rmsnorm.py::test_rmsnorm_preserves_shape | src/inference_lab/nn/normalization/rmsnorm.py | TODO |
| 2.11 | RMSNorm | tests/02_attention/test_rmsnorm.py::test_rmsnorm_matches_reference_formula | src/inference_lab/nn/normalization/rmsnorm.py | TODO |
| 2.12 | SwiGLU | tests/02_attention/test_swiglu.py::test_swiglu_uses_gate_and_up_projections | src/inference_lab/nn/mlp/swiglu.py | TODO |
| 2.12 | SwiGLU | tests/02_attention/test_swiglu.py::test_swiglu_projects_back_to_hidden_dimension | src/inference_lab/nn/mlp/swiglu.py | TODO |
| 2.13 | TransformerBlock | tests/02_attention/test_transformer_block.py::test_transformer_block_preserves_hidden_shape | src/inference_lab/nn/transformer_block.py | TODO |
| 2.13 | TransformerBlock | tests/02_attention/test_transformer_block.py::test_transformer_block_contains_attention_and_mlp_residual_paths | src/inference_lab/nn/transformer_block.py | TODO |
| 3.1 | configuration modèle | tests/03_models/test_minimal_config.py::test_minimal_transformer_config_defines_model_dimensions | src/inference_lab/models/minimal_transformer/config.py | TODO |
| 3.2 | embeddings | tests/03_models/test_embeddings.py::test_embedding_maps_token_ids_to_hidden_vectors | src/inference_lab/nn/embeddings.py | TODO |
| 3.3 | modèle complet | tests/03_models/test_minimal_transformer.py::test_minimal_transformer_stacks_requested_number_of_blocks | src/inference_lab/models/minimal_transformer/model.py | TODO |
| 3.4 | forward | tests/03_models/test_minimal_transformer.py::test_model_forward_outputs_logits_for_each_token_and_vocabulary_entry | src/inference_lab/models/minimal_transformer/model.py | TODO |
| 3.5 | LM Head | tests/03_models/test_minimal_transformer.py::test_lm_head_projects_hidden_dimension_to_vocabulary_size | src/inference_lab/models/minimal_transformer/model.py | TODO |
| 3.6 | greedy sampling | tests/03_models/test_sampling.py::test_greedy_sampling_selects_highest_logit | src/inference_lab/inference/sampling.py | TODO |
| 3.7 | temperature | tests/03_models/test_sampling.py::test_temperature_changes_probability_distribution | src/inference_lab/inference/sampling.py | TODO |
| 3.8 | top-k | tests/03_models/test_sampling.py::test_top_k_excludes_tokens_outside_k_highest_logits | src/inference_lab/inference/sampling.py | TODO |
| 3.9 | top-p | tests/03_models/test_sampling.py::test_top_p_limits_candidates_by_cumulative_probability | src/inference_lab/inference/sampling.py | TODO |
| 3.10 | génération | tests/03_models/test_generation.py::test_generation_appends_one_token_per_iteration | src/inference_lab/inference/generation.py | TODO |
| 3.10 | génération | tests/03_models/test_generation.py::test_generation_stops_on_eos | src/inference_lab/inference/generation.py | TODO |
| 3.11 | training minimal | tests/03_models/test_training_smoke.py::test_tiny_model_can_reduce_loss_on_tiny_dataset | src/inference_lab/training/ | TODO |
| 4.1 | coût decode naïf | tests/04_inference/test_naive_decode.py::test_naive_decode_reprocesses_previous_tokens | instrumentation inference | TODO |
| 4.2 | structure KV cache | tests/04_inference/test_kv_cache.py::test_kv_cache_stores_keys_and_values_for_every_layer | src/inference_lab/cache/kv_cache.py | TODO |
| 4.3 | croissance cache | tests/04_inference/test_kv_cache.py::test_kv_cache_grows_one_position_per_decode_step | src/inference_lab/cache/kv_cache.py | TODO |
| 4.4 | prefill | tests/04_inference/test_prefill.py::test_prefill_processes_complete_prompt | src/inference_lab/inference/prefill.py | TODO |
| 4.4 | prefill | tests/04_inference/test_prefill.py::test_prefill_populates_initial_kv_cache | src/inference_lab/inference/prefill.py | TODO |
| 4.5 | decode | tests/04_inference/test_decode.py::test_decode_processes_only_new_token | src/inference_lab/inference/decode.py | TODO |
| 4.5 | decode | tests/04_inference/test_decode.py::test_decode_reuses_cached_keys_and_values | src/inference_lab/inference/decode.py | TODO |
| 4.6 | correctness KV | tests/04_inference/test_cached_generation.py::test_cached_and_uncached_generation_produce_same_logits | cache + inference | TODO |
| 4.6 | correctness KV | tests/04_inference/test_cached_generation.py::test_cached_and_uncached_greedy_generation_produce_same_tokens | cache + inference | TODO |
| 4.7 | position cache | tests/04_inference/test_cached_generation.py::test_decode_position_advances_with_cache_length | inference | TODO |
| 4.8 | GQA + cache | tests/04_inference/test_gqa_kv_cache.py::test_gqa_kv_cache_stores_only_kv_heads_not_query_heads | cache | TODO |
| 5.1 | mémoire paramètres | tests/05_calculators/test_model_memory.py::test_parameter_memory_is_parameter_count_times_dtype_size | src/inference_lab/calculators/model_memory.py | TODO |
| 5.2 | KV cache memory | tests/05_calculators/test_kv_cache_memory.py::test_kv_cache_memory_matches_layers_heads_tokens_formula | src/inference_lab/calculators/kv_cache_memory.py | TODO |
| 5.3 | impact contexte | tests/05_calculators/test_kv_cache_memory.py::test_kv_cache_memory_scales_linearly_with_context_length | calculator | TODO |
| 5.4 | impact GQA | tests/05_calculators/test_kv_cache_memory.py::test_reducing_kv_heads_reduces_kv_cache_memory_proportionally | calculator | TODO |
| 5.5 | FLOPs linear | tests/05_calculators/test_flops.py::test_linear_layer_flops_are_estimated_from_matrix_dimensions | src/inference_lab/calculators/flops.py | TODO |
| 5.6 | FLOPs modèle | tests/05_calculators/test_flops.py::test_model_decode_flops_can_be_estimated_from_architecture | calculator | TODO |
| 5.7 | bytes déplacés | tests/05_calculators/test_memory_traffic.py::test_weight_streaming_bytes_can_be_estimated_from_parameter_size | src/inference_lab/calculators/memory_traffic.py | TODO |
| 5.8 | arithmetic intensity | tests/05_calculators/test_arithmetic_intensity.py::test_arithmetic_intensity_is_flops_divided_by_bytes | src/inference_lab/calculators/roofline.py | TODO |
| 5.9 | roofline | tests/05_calculators/test_arithmetic_intensity.py::test_roofline_limit_is_minimum_of_compute_and_bandwidth_limits | src/inference_lab/calculators/roofline.py | TODO |
| 5.10 | bottleneck | tests/05_calculators/test_arithmetic_intensity.py::test_low_arithmetic_intensity_is_classified_as_bandwidth_limited | calculator | TODO |
| 6.1 | device CUDA | tests/06_gpu/test_cuda_device.py::test_tensor_can_be_created_on_cuda | helpers GPU | TODO |
| 6.1 | device CUDA | tests/06_gpu/test_cuda_device.py::test_cpu_and_cuda_tensors_report_different_devices | helpers GPU | TODO |
| 6.2 | async CUDA | tests/06_gpu/test_cuda_synchronization.py::test_cuda_timing_requires_synchronization_or_cuda_events | src/inference_lab/benchmarks/timing.py | TODO |
| 6.3 | CUDA Event | tests/06_gpu/test_cuda_timing.py::test_cuda_timer_returns_positive_elapsed_time | src/inference_lab/benchmarks/timing.py | TODO |
| 6.4 | allocation VRAM | tests/06_gpu/test_cuda_memory.py::test_gpu_allocation_increases_allocated_memory | src/inference_lab/profiling/memory.py | TODO |
| 6.4 | allocation VRAM | tests/06_gpu/test_cuda_memory.py::test_peak_memory_can_be_recorded | src/inference_lab/profiling/memory.py | TODO |
| 6.5 | CPU→GPU | tests/06_gpu/test_transfers.py::test_tensor_transfer_preserves_values | backend | TODO |
| 6.6 | GEMM/GEMV | tests/06_gpu/test_gpu_matmul.py::test_gpu_matmul_matches_cpu_reference | benchmark only | TODO |
| 6.7 | dtype GPU | tests/06_gpu/test_gpu_dtypes.py::test_fp16_bf16_and_fp32_matmul_preserve_expected_shapes | benchmark | TODO |
| 6.8 | GPU absence | tests/06_gpu/test_gpu_absence.py::test_gpu_marked_tests_skip_cleanly_without_cuda | tests/conftest.py | TODO |
| 7.1 | benchmark warmup | tests/07_performance/test_benchmark_runner.py::test_benchmark_executes_warmup_before_measurement | src/inference_lab/benchmarks/runner.py | TODO |
| 7.2 | répétitions | tests/07_performance/test_benchmark_runner.py::test_benchmark_collects_multiple_measurements | runner | TODO |
| 7.3 | médiane | tests/07_performance/test_benchmark_runner.py::test_benchmark_reports_median_latency | runner | TODO |
| 7.4 | prefill metrics | tests/07_performance/test_prefill_benchmark.py::test_prefill_benchmark_reports_latency_and_input_tokens_per_second | benchmark | TODO |
| 7.5 | decode metrics | tests/07_performance/test_decode_benchmark.py::test_decode_benchmark_reports_time_per_output_token | benchmark | TODO |
| 7.6 | profiler | tests/07_performance/test_profiler_helpers.py::test_profiler_can_capture_model_operations | src/inference_lab/profiling/pytorch_profiler.py | TODO |
| 7.7 | résultat benchmark | tests/07_performance/test_benchmark_result.py::test_benchmark_result_contains_hardware_and_model_metadata | src/inference_lab/benchmarks/result.py | TODO |
| 8.1 | SDPA | tests/08_optimized/test_sdpa.py::test_naive_attention_matches_pytorch_sdpa_for_supported_case | attention | TODO |
| 8.2 | causal SDPA | tests/08_optimized/test_sdpa.py::test_causal_sdpa_matches_naive_causal_attention | attention | TODO |
| 8.3 | torch.compile | tests/08_optimized/test_torch_compile.py::test_compiled_model_matches_eager_output | compilation helper | TODO |
| 8.4 | fusion correctness | tests/08_optimized/test_fusion_reference.py::test_fused_operation_matches_unfused_reference | src/inference_lab/kernels/pytorch/ | TODO |
| 9.1 | Triton disponibilité | tests/09_triton/test_triton_environment.py::test_triton_tests_skip_when_triton_or_cuda_is_unavailable | infrastructure | TODO |
| 9.2 | vector add | tests/09_triton/test_vector_add.py::test_triton_vector_add_matches_pytorch | src/inference_lab/kernels/triton/vector_add.py | TODO |
| 9.3 | reduction | tests/09_triton/test_reduction.py::test_triton_reduction_matches_pytorch | src/inference_lab/kernels/triton/reduction.py | TODO |
| 9.4 | RMSNorm | tests/09_triton/test_rmsnorm.py::test_triton_rmsnorm_matches_pytorch_reference | src/inference_lab/kernels/triton/rmsnorm.py | TODO |
| 9.5 | SwiGLU | tests/09_triton/test_swiglu.py::test_triton_swiglu_matches_pytorch_reference | src/inference_lab/kernels/triton/swiglu.py | TODO |
| 10.1 | nanochat-like config | tests/10_nanochat/test_config.py::test_nanochat_like_config_defines_complete_architecture | src/inference_lab/models/nanochat_like/config.py | TODO |
| 10.2 | modèle complet | tests/10_nanochat/test_model.py::test_nanochat_like_model_outputs_expected_logits_shape | src/inference_lab/models/nanochat_like/model.py | TODO |
| 10.3 | training | tests/10_nanochat/test_training.py::test_nanochat_like_model_can_overfit_tiny_batch | modèle/training | TODO |
| 10.4 | génération | tests/10_nanochat/test_generation.py::test_nanochat_like_model_can_generate_tokens | generation | TODO |
| 10.5 | cache | tests/10_nanochat/test_kv_cache.py::test_nanochat_like_cached_generation_matches_uncached_generation | cache/inference | TODO |
| 11.1 | custom model baseline | tests/11_custom_llm/test_baseline.py::test_custom_llm_baseline_matches_declared_configuration | src/inference_lab/models/custom_llm/ | TODO |
| 11.2 | MHA/GQA configurable | tests/11_custom_llm/test_attention_configuration.py::test_custom_llm_can_switch_between_mha_and_gqa | custom model | TODO |
| 11.3 | normalisation configurable | tests/11_custom_llm/test_normalization_configuration.py::test_custom_llm_can_select_normalization_implementation | custom model | TODO |
| 11.4 | attention backend | tests/11_custom_llm/test_attention_backend.py::test_custom_llm_can_select_naive_or_sdpa_attention | custom model | TODO |
| 11.5 | comportement identique | tests/11_custom_llm/test_attention_backend.py::test_equivalent_attention_backends_produce_close_outputs | custom model | TODO |
| 12.1 | configuration Qwen | tests/12_qwen/test_config_mapping.py::test_qwen_config_maps_to_internal_model_dimensions | src/inference_lab/models/qwen/config.py | TODO |
| 12.2 | RMSNorm Qwen | tests/12_qwen/test_rmsnorm_compatibility.py::test_our_rmsnorm_matches_qwen_reference | Qwen components | TODO |
| 12.3 | RoPE Qwen | tests/12_qwen/test_rope_compatibility.py::test_our_rope_matches_qwen_reference | Qwen components | TODO |
| 12.4 | GQA Qwen | tests/12_qwen/test_attention_compatibility.py::test_our_gqa_matches_qwen_reference_on_small_input | Qwen attention | TODO |
| 12.5 | MLP Qwen | tests/12_qwen/test_mlp_compatibility.py::test_our_mlp_matches_qwen_reference_on_small_input | Qwen MLP | TODO |
| 12.6 | poids | tests/12_qwen/test_weight_loading.py::test_qwen_weights_can_be_loaded_into_internal_modules | src/inference_lab/models/qwen/weight_loader.py | TODO |
| 12.7 | modèle | tests/12_qwen/test_model_compatibility.py::test_internal_qwen_model_matches_reference_logits_on_small_input | modèle Qwen | TODO |
| 13.1 | TTFT | tests/13_benchmark/test_ttft.py::test_ttft_metric_measures_time_until_first_token | metrics | TODO |
| 13.2 | TPOT | tests/13_benchmark/test_tpot.py::test_tpot_metric_measures_decode_time_per_output_token | metrics | TODO |
| 13.3 | throughput | tests/13_benchmark/test_throughput.py::test_throughput_reports_tokens_per_second | metrics | TODO |
| 13.4 | mémoire | tests/13_benchmark/test_memory_metrics.py::test_benchmark_reports_peak_device_memory | metrics | TODO |
| 13.5 | metadata | tests/13_benchmark/test_result_metadata.py::test_results_record_model_dtype_batch_context_and_hardware | benchmark schema | TODO |
| 13.6 | prédiction vs mesure | tests/13_benchmark/test_theoretical_predictions.py::test_theoretical_model_produces_comparable_prediction_schema_to_measurements | calculators/benchmark | TODO |
| 14.1 | static batching | tests/14_engine/test_batching.py::test_static_batch_groups_requests_before_execution | src/inference_lab/engine/batching.py | TODO |
| 14.2 | continuous batching | tests/14_engine/test_scheduler.py::test_finished_sequences_can_leave_batch_while_others_continue | scheduler | TODO |
| 14.3 | admission | tests/14_engine/test_scheduler.py::test_scheduler_rejects_or_delays_requests_when_capacity_is_exhausted | scheduler | TODO |
| 14.4 | KV blocks | tests/14_engine/test_paged_cache.py::test_block_allocator_assigns_fixed_size_kv_blocks | memory manager | TODO |
| 14.5 | block table | tests/14_engine/test_paged_cache.py::test_logical_kv_blocks_map_to_physical_blocks | paged cache | TODO |
| 14.6 | fragmentation | tests/14_engine/test_allocator.py::test_block_allocator_reuses_released_blocks | allocator | TODO |
| 14.7 | prefix caching | tests/14_engine/test_prefix_cache.py::test_identical_prefix_can_reuse_cached_kv_blocks | prefix cache | TODO |
| 14.8 | chunked prefill | tests/14_engine/test_chunked_prefill.py::test_long_prefill_can_be_split_into_multiple_chunks | scheduler | TODO |
| 15.1 | weight quantization | tests/15_advanced/test_quantization.py::test_quantized_weights_require_less_storage_than_fp16_weights | future quantization | TODO |
| 15.2 | dequantization | tests/15_advanced/test_quantization.py::test_dequantized_output_remains_close_to_reference | future quantization | TODO |
| 15.3 | KV quantization | tests/15_advanced/test_kv_quantization.py::test_quantized_kv_cache_reduces_cache_memory | future KV quantization | TODO |
| 15.4 | speculative decoding | tests/15_advanced/test_speculative_decoding.py::test_verified_speculative_tokens_match_target_model_distribution | future speculative decoding | TODO |
| 15.5 | draft acceptance | tests/15_advanced/test_speculative_decoding.py::test_acceptance_rate_is_computed_from_verified_draft_tokens | future speculative decoding | TODO |
| 16.1 | column parallel | tests/16_distributed/test_tensor_parallel.py::test_column_parallel_linear_reconstructs_reference_output | future distributed | TODO |
| 16.2 | row parallel | tests/16_distributed/test_tensor_parallel.py::test_row_parallel_linear_reconstructs_reference_output | future distributed | TODO |
| 16.3 | sharded heads | tests/16_distributed/test_attention_parallel.py::test_attention_heads_can_be_partitioned_across_devices | future distributed | TODO |
| 16.4 | AllGather | tests/16_distributed/test_collectives.py::test_all_gather_reconstructs_all_shards | future distributed | TODO |
| 16.5 | AllReduce | tests/16_distributed/test_collectives.py::test_all_reduce_matches_single_device_reference_sum | future distributed | TODO |
| 16.6 | distributed model | tests/16_distributed/test_distributed_model.py::test_tensor_parallel_model_matches_single_device_reference | future distributed | TODO |
| 17.1 | backend contract | tests/17_backends/test_backend_contract.py::test_backend_exposes_required_tensor_and_execution_capabilities | future backends | TODO |
| 17.2 | CPU backend | tests/17_backends/test_cpu_backend.py::test_cpu_backend_executes_reference_operation | future backends | TODO |
| 17.3 | CUDA backend | tests/17_backends/test_cuda_backend.py::test_cuda_backend_matches_cpu_reference | future backends | TODO |
| 17.4 | Spyre backend | tests/17_backends/test_spyre_backend.py::test_spyre_backend_matches_reference_for_supported_operation | future backends | TODO |
| 17.5 | capability discovery | tests/17_backends/test_capability_discovery.py::test_backend_reports_supported_dtypes_and_operations | future backends | TODO |
| 17.6 | cross-hardware | tests/17_backends/test_cross_hardware.py::test_supported_backends_produce_numerically_compatible_results | future backends | TODO |
