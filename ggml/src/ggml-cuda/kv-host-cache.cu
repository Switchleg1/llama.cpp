// VRAM page cache in front of a KV tensor in mapped host memory (Strata's KV streaming).
// A page holds page_rows consecutive rows (cells). The host copy stays authoritative: writes go there and drop the
// page, so a page never holds stale rows. A gather resolves pages on the device: hits are stamped, misses claim a
// page by clock (second chance, never one this call uses) and are copied in from the host copy. Rows that find no
// page are read from the host copy directly.

#include "kv-host-cache.cuh"
#include "convert.cuh"
#include "../ggml-backend-moe.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace {

struct kv_host_cache {
    const ggml_tensor * tensor = nullptr;
    const char * base = nullptr;
    size_t bytes = 0;
    char * pool = nullptr;
    int32_t * meta = nullptr;
    size_t row_bytes = 0;
    int64_t n_rows = 0;
    int32_t page_rows = 0;
    int32_t n_pages = 0;
    int32_t n_blocks = 0;
    std::atomic<bool> dirty{false};
    unsigned long long * verify = nullptr;   // debug: rows served from pages, rows that differed from the host copy
};

std::mutex & kv_host_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<kv_host_cache>> & kv_host_caches() {
    static std::vector<std::unique_ptr<kv_host_cache>> caches;
    return caches;
}

std::atomic<int> & kv_host_cache_count() {
    static std::atomic<int> count{0};
    return count;
}

kv_host_cache * kv_host_cache_find(const ggml_tensor * t) {
    if (t == nullptr || kv_host_cache_count().load(std::memory_order_relaxed) == 0) {
        return nullptr;
    }
    const ggml_tensor * root = t->view_src != nullptr ? t->view_src : t;
    std::lock_guard<std::mutex> lock(kv_host_cache_mutex());
    for (auto & cache : kv_host_caches()) {
        if (cache->tensor == root) {
            return cache.get();
        }
    }
    return nullptr;
}

// meta layout (int32): page_of_block[n_blocks], block_of_page, stamp, ref, miss_block, miss_page [n_pages each],
// then hand, epoch, n_miss
struct kv_meta {
    int32_t * page_of_block;
    int32_t * block_of_page;
    int32_t * stamp;
    int32_t * ref;
    int32_t * miss_block;
    int32_t * miss_page;
    int32_t * control;
};

__host__ __device__ kv_meta kv_meta_of(int32_t * meta, int n_blocks, int n_pages) {
    kv_meta m;
    m.page_of_block = meta;
    m.block_of_page = meta + n_blocks;
    m.stamp         = m.block_of_page + n_pages;
    m.ref           = m.stamp + n_pages;
    m.miss_block    = m.ref + n_pages;
    m.miss_page     = m.miss_block + n_pages;
    m.control       = m.miss_page + n_pages;
    return m;
}

size_t kv_meta_size(int n_blocks, int n_pages) {
    return (size_t) n_blocks + 6 * (size_t) n_pages + 3;
}

__global__ void kv_cache_reset(int32_t * meta, int n_blocks, int n_pages) {
    const kv_meta m = kv_meta_of(meta, n_blocks, n_pages);
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks; i += stride) {
        m.page_of_block[i] = -1;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_pages; i += stride) {
        m.block_of_page[i] = -1;
        m.stamp[i] = 0;
        m.ref[i] = 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        m.control[0] = 0;
        m.control[2] = 0;
    }
}

// One block. out[i] is the pool row of ids[i], or -1 when no page could be given.
__global__ void kv_cache_resolve(const int32_t * ids, int n_ids, int64_t row0, int page_rows, int32_t * meta,
        int n_blocks, int n_pages, int32_t * out) {
    const kv_meta m = kv_meta_of(meta, n_blocks, n_pages);
    __shared__ int epoch;
    __shared__ int n_miss;
    if (threadIdx.x == 0) {
        epoch = m.control[1] + 1;
        m.control[1] = epoch;
        n_miss = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int32_t block = (int32_t) ((row0 + ids[i]) / page_rows);
        const int32_t page = m.page_of_block[block];
        if (page >= 0) {
            m.stamp[page] = epoch;
            m.ref[page] = 1;
        } else if (page == -1 && atomicCAS(&m.page_of_block[block], -1, -2) == -1) {
            const int k = atomicAdd(&n_miss, 1);
            if (k < n_pages) {
                m.miss_block[k] = block;
            } else {
                m.page_of_block[block] = -1;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int n = min(n_miss, n_pages);
        int hand = m.control[0];
        int placed = 0;
        for (int k = 0; k < n; ++k) {
            const int32_t block = m.miss_block[k];
            int page = -1;
            for (int tries = 0; tries < 2 * n_pages; ++tries) {
                const int candidate = hand;
                hand = hand + 1 == n_pages ? 0 : hand + 1;
                if (m.stamp[candidate] == epoch) {
                    continue;
                }
                if (m.ref[candidate]) {
                    m.ref[candidate] = 0;
                    continue;
                }
                page = candidate;
                break;
            }
            if (page < 0) {
                m.page_of_block[block] = -1;   // read from the host copy below
                continue;
            }
            const int32_t old = m.block_of_page[page];
            if (old >= 0) {
                m.page_of_block[old] = -1;
            }
            m.block_of_page[page] = block;
            m.page_of_block[block] = page;
            m.stamp[page] = epoch;
            m.ref[page] = 1;
            m.miss_block[placed] = block;
            m.miss_page[placed] = page;
            ++placed;
        }
        m.control[0] = hand;
        m.control[2] = placed;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int64_t row = row0 + ids[i];
        const int32_t page = m.page_of_block[row / page_rows];
        out[i] = page >= 0 ? page * page_rows + (int32_t) (row % page_rows) : -1;
    }
}

// Copies the pages that resolve just placed from the host copy.
__global__ void kv_cache_fill(const char * base, char * pool, const int32_t * meta_c, int n_blocks, int n_pages,
        int page_rows, size_t row_bytes) {
    const kv_meta m = kv_meta_of(const_cast<int32_t *>(meta_c), n_blocks, n_pages);
    const int n = m.control[2];
    const size_t page_words = (size_t) page_rows * row_bytes / sizeof(uint4);
    const size_t total = (size_t) n * page_words;
    for (size_t w = (size_t) blockIdx.x * blockDim.x + threadIdx.x; w < total; w += (size_t) gridDim.x * blockDim.x) {
        const size_t k = w / page_words, word = w % page_words;
        const uint4 * src = reinterpret_cast<const uint4 *>(base + (size_t) m.miss_block[k] * page_rows * row_bytes);
        uint4 * dst = reinterpret_cast<uint4 *>(pool + (size_t) m.miss_page[k] * page_rows * row_bytes);
        dst[word] = src[word];
    }
}

template <typename src_t, typename dst_t>
__global__ void kv_cache_gather(const char * base, const char * pool, const int32_t * ids, const int32_t * slots,
        int64_t row0, int n_ids, int64_t row_elems, size_t row_bytes, dst_t * dst, unsigned long long * verify) {
    const int64_t total = (int64_t) n_ids * row_elems;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / row_elems, e = i % row_elems;
        const int32_t slot = slots[r];
        const char * host = base + (size_t) (row0 + ids[r]) * row_bytes;
        const char * row = slot >= 0 ? pool + (size_t) slot * row_bytes : host;
        const src_t value = reinterpret_cast<const src_t *>(row)[e];
        dst[i] = ggml_cuda_cast<dst_t>(value);
        if (verify != nullptr && slot >= 0 && e == 0) {
            atomicAdd(&verify[0], 1ull);
            bool same = true;
            for (int64_t k = 0; k < (int64_t) (row_bytes / sizeof(uint32_t)); ++k) {
                same = same && reinterpret_cast<const uint32_t *>(row)[k] == reinterpret_cast<const uint32_t *>(host)[k];
            }
            if (!same) {
                atomicAdd(&verify[1], 1ull);
            }
        }
    }
}

__global__ void kv_cache_drop(const void * idxs, bool idx64, int64_t n, int64_t row0, int page_rows, int32_t * meta,
        int n_blocks, int n_pages) {
    const kv_meta m = kv_meta_of(meta, n_blocks, n_pages);
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t idx = idx64 ? static_cast<const int64_t *>(idxs)[i] : static_cast<const int32_t *>(idxs)[i];
        const int32_t block = (int32_t) ((row0 + idx) / page_rows);
        const int32_t page = m.page_of_block[block];
        if (page >= 0) {
            m.page_of_block[block] = -1;
            m.block_of_page[page] = -1;
        }
    }
}

void kv_cache_reset_async(const kv_host_cache & cache, cudaStream_t stream) {
    kv_cache_reset<<<64, 256, 0, stream>>>(cache.meta, cache.n_blocks, cache.n_pages);
    CUDA_CHECK(cudaGetLastError());
}

bool kv_row_view(const kv_host_cache & cache, const ggml_tensor * t, int64_t & row0) {
    const size_t offs = t->view_src != nullptr ? t->view_offs : 0;
    if (t->nb[1] != cache.row_bytes || offs % cache.row_bytes != 0 || (size_t) t->ne[0] * ggml_type_size(t->type) != cache.row_bytes) {
        return false;
    }
    row0 = (int64_t) (offs / cache.row_bytes);
    return true;
}

} // namespace

bool ggml_backend_cuda_kv_host_cache(ggml_tensor * tensor, ggml_tensor * pool, ggml_tensor * meta, uint32_t page_rows) {
    if (tensor == nullptr) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(kv_host_cache_mutex());
        auto & caches = kv_host_caches();
        for (auto it = caches.begin(); it != caches.end(); ++it) {
            if ((*it)->tensor == tensor) {
                if ((*it)->verify != nullptr) {
                    unsigned long long counts[2] = {};
                    CUDA_CHECK(cudaMemcpy(counts, (*it)->verify, sizeof(counts), cudaMemcpyDeviceToHost));
                    GGML_LOG_WARN("kv host cache %s: %llu rows served from pages, %llu differed from the host copy\n",
                        tensor->name, counts[0], counts[1]);
                    CUDA_CHECK(cudaFree((*it)->verify));
                }
                caches.erase(it);
                kv_host_cache_count().fetch_sub(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    if (pool == nullptr) {
        return true;
    }
    if (meta == nullptr || page_rows == 0 || tensor->data == nullptr || pool->data == nullptr || meta->data == nullptr ||
            pool->type != tensor->type || tensor->nb[0] != ggml_type_size(tensor->type) || ggml_is_quantized(tensor->type) ||
            pool->nb[1] != tensor->nb[1] || meta->type != GGML_TYPE_I32 || !ggml_is_contiguous(tensor) ||
            tensor->nb[1] % sizeof(uint4) != 0) {
        return false;
    }
    auto cache = std::make_unique<kv_host_cache>();
    cache->tensor = tensor;
    cache->base = static_cast<const char *>(tensor->data);
    cache->bytes = ggml_nbytes(tensor);
    cache->pool = static_cast<char *>(pool->data);
    cache->meta = static_cast<int32_t *>(meta->data);
    cache->row_bytes = tensor->nb[1];
    cache->n_rows = (int64_t) (cache->bytes / cache->row_bytes);
    cache->page_rows = (int32_t) page_rows;
    cache->n_pages = (int32_t) (ggml_nrows(pool) / page_rows);
    cache->n_blocks = (int32_t) (cache->n_rows / page_rows);
    if (cache->n_rows % page_rows != 0 || cache->n_pages <= 0 ||
            (size_t) ggml_nelements(meta) < kv_meta_size(cache->n_blocks, cache->n_pages)) {
        return false;
    }
    if (ggml_moe_debug_getenv("LLAMA_KV_CACHE_VERIFY") != nullptr) {
        CUDA_CHECK(cudaMalloc(&cache->verify, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(cache->verify, 0, 2 * sizeof(unsigned long long)));
    }
    kv_cache_reset_async(*cache, cudaStreamPerThread);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    std::lock_guard<std::mutex> lock(kv_host_cache_mutex());
    kv_host_caches().push_back(std::move(cache));
    kv_host_cache_count().fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool ggml_cuda_kv_host_cache_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    kv_host_cache * cache = kv_host_cache_find(src0);
    int64_t row0 = 0;
    if (cache == nullptr || !kv_row_view(*cache, src0, row0) || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->type != GGML_TYPE_I32 || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            dst->ne[0] != src0->ne[0] || ggml_nrows(dst) != ggml_nelements(src1)) {
        return false;
    }
    const int n_ids = (int) ggml_nelements(src1);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int32_t> slots(ctx.pool(), n_ids);
    const int32_t * ids = static_cast<const int32_t *>(src1->data);
    kv_cache_resolve<<<1, 1024, 0, stream>>>(ids, n_ids, row0, cache->page_rows, cache->meta, cache->n_blocks,
        cache->n_pages, slots.get());
    kv_cache_fill<<<128, 256, 0, stream>>>(cache->base, cache->pool, cache->meta, cache->n_blocks, cache->n_pages,
        cache->page_rows, cache->row_bytes);
    const int64_t row_elems = src0->ne[0];
    const int64_t total = (int64_t) n_ids * row_elems;
    const int blocks = (int) std::min<int64_t>((total + 255) / 256, 4096);
#define KV_GATHER(S, D) kv_cache_gather<S, D><<<blocks, 256, 0, stream>>>(cache->base, cache->pool, ids, slots.get(), \
        row0, n_ids, row_elems, cache->row_bytes, static_cast<D *>(dst->data), cache->verify)
    if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        KV_GATHER(half, float);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        KV_GATHER(half, half);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        KV_GATHER(nv_bfloat16, float);
    } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        KV_GATHER(float, float);
    } else {
        GGML_ABORT("kv host cache: unsupported gather %s -> %s", ggml_type_name(src0->type), ggml_type_name(dst->type));
    }
#undef KV_GATHER
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_kv_host_cache_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * set_rows) {
    kv_host_cache * cache = kv_host_cache_find(set_rows);
    if (cache == nullptr) {
        return;
    }
    const ggml_tensor * idx = set_rows->src[1];
    int64_t row0 = 0;
    if (!kv_row_view(*cache, set_rows, row0) || !ggml_is_contiguous(idx) ||
            (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64)) {
        cache->dirty.store(true, std::memory_order_relaxed);   // unknown layout: drop everything at the next graph
        return;
    }
    const int64_t n = ggml_nelements(idx);
    kv_cache_drop<<<(int) std::min<int64_t>((n + 255) / 256, 1024), 256, 0, ctx.stream()>>>(idx->data,
        idx->type == GGML_TYPE_I64, n, row0, cache->page_rows, cache->meta, cache->n_blocks, cache->n_pages);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_kv_host_cache_begin_graph(const ggml_cgraph * cgraph, cudaStream_t stream) {
    GGML_UNUSED(cgraph);
    if (kv_host_cache_count().load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(kv_host_cache_mutex());
    for (auto & cache : kv_host_caches()) {
        if (cache->dirty.exchange(false, std::memory_order_relaxed)) {
            kv_cache_reset_async(*cache, stream);
        }
    }
}

void ggml_cuda_kv_host_cache_end_graph(const ggml_cgraph * cgraph, cudaStream_t stream) {
    if (kv_host_cache_count().load(std::memory_order_relaxed) == 0 || cgraph == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(kv_host_cache_mutex());
    for (auto & cache : kv_host_caches()) {
        // ops other than set_rows that write the tensor (a K shift, cell moves) leave pages stale
        bool written = false;
        for (int i = 0; i < cgraph->n_nodes && !written; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            const ggml_tensor * root = node->view_src != nullptr ? node->view_src : node;
            written = root == cache->tensor && node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW &&
                node->op != GGML_OP_RESHAPE && node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE &&
                node->op != GGML_OP_SET_ROWS;
        }
        if (written) {
            kv_cache_reset_async(*cache, stream);
        }
    }
}

void ggml_cuda_kv_host_cache_mark_dirty(const void * base, size_t size) {
    if (kv_host_cache_count().load(std::memory_order_relaxed) == 0) {
        return;
    }
    const char * lo = static_cast<const char *>(base);
    std::lock_guard<std::mutex> lock(kv_host_cache_mutex());
    for (auto & cache : kv_host_caches()) {
        if (cache->base < lo + size && lo < cache->base + cache->bytes) {
            cache->dirty.store(true, std::memory_order_relaxed);
        }
    }
}
