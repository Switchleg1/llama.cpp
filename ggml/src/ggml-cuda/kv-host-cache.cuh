#pragma once

#include "common.cuh"

// VRAM page cache for KV tensors in the CUDA_KV_Host buffer, see ggml_backend_kv_host_cache.
bool ggml_backend_cuda_kv_host_cache(ggml_tensor * tensor, ggml_tensor * pool, ggml_tensor * meta, uint32_t page_rows);

// Gathers rows of a cached tensor through its pages. False: not cached here, use the plain path.
bool ggml_cuda_kv_host_cache_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// After rows of a cached tensor were written: drop their pages.
void ggml_cuda_kv_host_cache_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * set_rows);

// Around one graph: reset caches that host-side writes made stale, and those that other ops in the graph wrote.
void ggml_cuda_kv_host_cache_begin_graph(const ggml_cgraph * cgraph, cudaStream_t stream);
void ggml_cuda_kv_host_cache_end_graph(const ggml_cgraph * cgraph, cudaStream_t stream);

// Host-side writes into a CUDA_KV_Host buffer.
void ggml_cuda_kv_host_cache_mark_dirty(const void * base, size_t size);
