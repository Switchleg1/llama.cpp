#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

// Raw expert bundles. The thread-safe reader must outlive calls, or be retained through reader_owner.
class llama_moe_expert_store {
    struct impl;
public:
    struct expert {
        uint32_t group;
        uint32_t id;
    };
    struct group {
        uint32_t n_experts;
        std::vector<size_t> bank_bytes;
    };
    using reader = bool (*)(void *, expert, uint32_t bank, void * dst, size_t bytes);

    class lease {
    public:
        lease() = default;
        ~lease();
        lease(lease && other) noexcept;
        lease & operator=(lease && other) noexcept;
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        explicit operator bool() const { return state != nullptr; }
        const void * data(uint32_t bank) const;
        // Borrowed, read-only single-expert view. Keep the lease until all consumers finish.
        bool tensor_view(uint32_t bank, const ggml_tensor & prototype, ggml_tensor & view) const;
        bool resident() const { return state != nullptr && staged == UINT32_MAX; }
    private:
        friend class llama_moe_expert_store;
        void release();
        std::shared_ptr<impl> state;
        uint32_t index = UINT32_MAX;
        uint32_t staged = UINT32_MAX;
    };

    class exchange {
    public:
        exchange() = default;
        ~exchange();
        exchange(exchange && other) noexcept;
        exchange & operator=(exchange && other) noexcept;
        exchange(const exchange &) = delete;
        exchange & operator=(const exchange &) = delete;
        explicit operator bool() const { return state != nullptr; }
        const void * incoming(uint32_t bank) const;
        void * outgoing(uint32_t bank) const;
        // Caller must finish all transfers before commit or cancel. Cancellation preserves old GPU ownership.
        bool commit();
        void cancel();
    private:
        friend class llama_moe_expert_store;
        std::shared_ptr<impl> state;
        uint32_t in = UINT32_MAX;
        uint32_t out = UINT32_MAX;
        uint32_t staged = UINT32_MAX;
    };

    // Initialize once. host_buft (e.g. pinned CUDA host memory) is tried first, then pageable CPU memory.
    // GPU experts are excluded; unranked experts stay file-backed when rank is given.
    bool init(const std::vector<group> & groups, const std::vector<expert> & gpu,
              const std::vector<expert> & rank, size_t resident_budget, size_t staging_budget,
              uint32_t max_staged, reader read, void * user, std::shared_ptr<void> reader_owner = {},
              ggml_backend_buffer_type_t host_buft = nullptr);
    // File-backed experts are copied into bounded staging. Exhaustion and read failures return an empty lease.
    lease acquire(expert key);
    exchange begin_exchange(expert in, expert out);
    // File-backed experts read at least min_hits times since they last moved, most read first.
    std::vector<std::pair<expert, uint32_t>> file_hot(uint32_t min_hits) const;
    // Reads a file-backed expert into the RAM position of a resident one, which becomes file-backed.
    // Fails without changes when either is in use or the incoming bundle is larger. Callers must stop GPU
    // reads of the outgoing position first.
    bool promote(expert in, expert out);
    size_t resident_bytes() const;
    size_t staging_bytes() const;
    // True when both buffers came from host_buft rather than the pageable CPU fallback.
    bool pinned() const;
    // Bundles read from the file tier since init.
    uint64_t file_reads() const;
    // Resident bank bytes without a lease, or null. Valid until the store is destroyed; callers must not use
    // this while exchanges can run, since commit moves bundles.
    const void * resident_data(expert key, uint32_t bank) const;
private:
    std::shared_ptr<impl> state;
};
