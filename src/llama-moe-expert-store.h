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

    // Per-expert location: LOC_FILE, loc_gpu(slot) for an expert that only GPU slot `slot` holds, or loc_ram():
    // a byte offset in the resident buffer, with (slot + 1) above LOC_OFFSET_BITS when a GPU slot also holds the
    // expert. A device that exchanges experts writes these values in place.
    static constexpr int64_t LOC_FILE = -1;
    static constexpr int LOC_OFFSET_BITS = 40;
    static constexpr int64_t loc_gpu(uint32_t slot) { return -2 - int64_t(slot); }
    static constexpr int64_t loc_ram(size_t offset, int64_t slot = -1) {
        return int64_t(offset) | ((slot + 1) << LOC_OFFSET_BITS);
    }
    static constexpr bool gpu_only(int64_t loc) { return loc <= -2; }
    static constexpr int64_t loc_offset(int64_t loc) { return loc >= 0 ? loc & ((int64_t(1) << LOC_OFFSET_BITS) - 1) : -1; }
    static constexpr int64_t loc_slot(int64_t loc) { return loc <= -2 ? -2 - loc : loc >= 0 ? (loc >> LOC_OFFSET_BITS) - 1 : -1; }

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
        size_t offset = SIZE_MAX;
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
    // The k-th GPU expert of a group owns GPU slot k. The last gpu_copies of each group's GPU experts also keep a
    // RAM copy while the budget allows; the others are excluded. When host_buft is given but its memory is
    // unavailable, GPU experts are placed like the rest, since pageable memory cannot take device exchanges.
    // Unranked experts stay file-backed when rank is given.
    bool init(const std::vector<group> & groups, const std::vector<expert> & gpu,
              const std::vector<expert> & rank, size_t resident_budget, size_t staging_budget,
              uint32_t max_staged, reader read, void * user, std::shared_ptr<void> reader_owner = {},
              ggml_backend_buffer_type_t host_buft = nullptr, uint32_t gpu_copies = 0);
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
    int64_t location(expert key) const;
    // Debug check: rereads every resident expert and compares its bytes. Counts experts per tier and mismatches.
    struct tiers { size_t ram = 0, gpu = 0, gpu_copies = 0, file = 0, mismatched = 0, gpu_moved = 0; };
    tiers verify() const;
    // Device-writable view for exchanges: the group's locations (pinned host memory), the resident buffer and the
    // offset of each bank in a bundle. Null when the store is not pinned or holds no GPU experts.
    int64_t * locations(uint32_t group) const;
    const void * resident_base() const;
    size_t bank_offset(uint32_t group, uint32_t bank) const;
private:
    std::shared_ptr<impl> state;
};
