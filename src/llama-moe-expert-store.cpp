// Complement planning and exchange ownership adapted from Strata's expert_source.cpp.
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "llama-moe-expert-store.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <atomic>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

struct llama_moe_expert_store::impl {
    struct layout {
        uint32_t first;
        uint32_t count;
        size_t bytes;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
    };
    struct entry {
        expert key;
        uint32_t readers = 0;
        uint32_t file_hits = 0;   // file-tier reads since the expert last moved
        bool exchanging = false;
        bool ranked = false;
    };
    std::vector<layout> layouts;
    std::vector<entry> entries;
    ggml_backend_buffer_ptr resident;
    ggml_backend_buffer_ptr staging;
    ggml_backend_buffer_ptr location_buffer;
    std::vector<int64_t> location_fallback;
    volatile int64_t * locations = nullptr;   // per entry, see LOC_FILE; a device may write it during exchanges
    bool exchangeable = false;

    int64_t loc(uint32_t index) const { return locations[index]; }
    void set_loc(uint32_t index, int64_t value) { locations[index] = value; }
    bool resident_at(uint32_t index) const { return loc(index) >= 0; }
    int64_t offset(uint32_t index) const { return loc_offset(loc(index)); }
    std::vector<bool> busy;
    size_t resident_size = 0;
    size_t staging_size = 0;
    size_t stride = 0;
    reader read;
    void * user;
    std::shared_ptr<void> reader_owner;
    std::atomic<uint64_t> file_reads{0};
    std::mutex mutex;

    uint32_t index(expert key) const {
        if (key.group >= layouts.size() || key.id >= layouts[key.group].count) {
            return UINT32_MAX;
        }
        return layouts[key.group].first + key.id;
    }
    uint32_t stage() {
        for (uint32_t i = 0; i < busy.size(); ++i) {
            if (!busy[i]) {
                busy[i] = true;
                return i;
            }
        }
        return UINT32_MAX;
    }
    bool pinned = true;
    ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t buft, size_t size) {
        ggml_backend_buffer_t buffer = buft ? ggml_backend_buft_alloc_buffer(buft, size) : nullptr;
        if (!buffer) {
            pinned = false;
            buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        }
        return buffer;
    }
    uint8_t * resident_data() const { return static_cast<uint8_t *>(ggml_backend_buffer_get_base(resident.get())); }
    uint8_t * staging_data() const { return static_cast<uint8_t *>(ggml_backend_buffer_get_base(staging.get())); }
    bool fetch(uint32_t index, uint8_t * dst) const {
        const auto & key = entries[index].key;
        const auto & layout = layouts[key.group];
        try {
            for (uint32_t bank = 0; bank < layout.sizes.size(); ++bank) {
                if (!read(user, key, bank, dst + layout.offsets[bank], layout.sizes[bank])) {
                    return false;
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }
};

bool llama_moe_expert_store::init(const std::vector<group> & groups, const std::vector<expert> & gpu,
        const std::vector<expert> & rank, size_t resident_budget, size_t staging_budget,
        uint32_t max_staged, reader read, void * user, std::shared_ptr<void> reader_owner,
        ggml_backend_buffer_type_t host_buft, uint32_t gpu_copies) {
    if (!read || groups.empty() || groups.size() > UINT32_MAX || state) {
        return false;
    }
    try {
        auto next = std::make_shared<impl>();
        next->read = read;
        next->user = user;
        next->reader_owner = std::move(reader_owner);
        next->pinned = host_buft != nullptr;
        for (uint32_t i = 0; i < groups.size(); ++i) {
            const auto & group = groups[i];
            if (!group.n_experts || group.bank_bytes.empty() || group.bank_bytes.size() > UINT32_MAX ||
                    group.n_experts > UINT32_MAX - next->entries.size()) {
                return false;
            }
            impl::layout layout = {uint32_t(next->entries.size()), group.n_experts, 0, {}, group.bank_bytes};
            for (size_t bytes : group.bank_bytes) {
                // Every bank begins at a cache-line boundary, including tiny scale banks.
                constexpr size_t alignment = 64;
                if (!bytes || layout.bytes > SIZE_MAX - (alignment - 1)) {
                    return false;
                }
                const size_t offset = (layout.bytes + alignment - 1) / alignment * alignment;
                if (bytes > SIZE_MAX - offset) {
                    return false;
                }
                layout.offsets.push_back(offset);
                layout.bytes = offset + bytes;
            }
            if (layout.bytes > SIZE_MAX - 63) {
                return false;
            }
            layout.bytes = (layout.bytes + 63) / 64 * 64;
            next->stride = std::max(next->stride, layout.bytes);
            next->layouts.push_back(std::move(layout));
            for (uint32_t id = 0; id < group.n_experts; ++id) {
                next->entries.push_back({{i, id}});
            }
        }
        if (host_buft) {
            next->location_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, next->entries.size() * sizeof(int64_t)));
        }
        if (next->location_buffer) {
            next->locations = static_cast<int64_t *>(ggml_backend_buffer_get_base(next->location_buffer.get()));
        } else {
            next->location_fallback.resize(next->entries.size());
            next->locations = next->location_fallback.data();
        }
        for (uint32_t i = 0; i < next->entries.size(); ++i) {
            next->set_loc(i, LOC_FILE);
        }
        std::vector<uint32_t> gpu_slots(next->layouts.size(), 0);
        for (expert key : gpu) {
            const uint32_t index = next->index(key);
            if (index == UINT32_MAX || next->loc(index) != LOC_FILE) {
                return false;
            }
            next->set_loc(index, loc_gpu(gpu_slots[key.group]++));
        }
        // GPU experts that keep a RAM copy, by entry: their slot, or -1.
        std::vector<int64_t> copy_slot(next->entries.size(), -1);
        for (uint32_t i = 0; i < next->entries.size(); ++i) {
            const int64_t slot = loc_slot(next->loc(i));
            const uint32_t n_gpu = gpu_slots[next->entries[i].key.group];
            if (slot >= 0 && uint64_t(slot) + gpu_copies >= n_gpu) {
                copy_slot[i] = slot;
            }
        }
        auto admit = [&](uint32_t index) {
            const size_t bytes = next->layouts[next->entries[index].key.group].bytes;
            if ((next->loc(index) == LOC_FILE || copy_slot[index] >= 0) && bytes <= resident_budget - next->resident_size) {
                next->set_loc(index, loc_ram(next->resident_size, copy_slot[index]));
                next->resident_size += bytes;
            }
        };
        if (rank.empty()) {
            for (uint32_t i = 0; i < next->entries.size(); ++i) {
                admit(i);
            }
        } else {
            for (expert key : rank) {
                const uint32_t index = next->index(key);
                if (index == UINT32_MAX || next->entries[index].ranked) {
                    return false;
                }
                next->entries[index].ranked = true;
                admit(index);
            }
        }
        const size_t slots = std::min<size_t>(max_staged, staging_budget / next->stride);
        next->staging_size = slots * next->stride;
        next->busy.resize(slots, false);
        if (next->resident_size) {
            next->resident.reset(next->allocate(host_buft, next->resident_size));
            if (!next->resident) {
                return false;
            }
        }
        if (next->staging_size) {
            next->staging.reset(next->allocate(host_buft, next->staging_size));
            if (!next->staging) {
                return false;
            }
        }
        // Pageable memory cannot take device exchanges: GPU experts then lose their slots.
        next->exchangeable = !gpu.empty() && host_buft && next->pinned && next->location_buffer;
        if (host_buft && !next->exchangeable) {
            for (uint32_t i = 0; i < next->entries.size(); ++i) {
                next->set_loc(i, gpu_only(next->loc(i)) ? LOC_FILE :
                    next->resident_at(i) ? loc_ram(size_t(next->offset(i))) : next->loc(i));
            }
        }
        for (uint32_t i = 0; i < next->entries.size(); ++i) {
            if (next->resident_at(i) && !next->fetch(i, next->resident_data() + next->offset(i))) {
                return false;
            }
        }
        state = std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
}

llama_moe_expert_store::lease llama_moe_expert_store::acquire(expert key) {
    lease result;
    if (!state) {
        return result;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        const uint32_t index = state->index(key);
        if (index == UINT32_MAX) {
            return result;
        }
        auto & entry = state->entries[index];
        const int64_t location = state->loc(index);
        if (gpu_only(location) || entry.exchanging || entry.readers == UINT32_MAX) {
            return result;
        }
        uint32_t staged = UINT32_MAX;
        if (location == LOC_FILE) {
            staged = state->stage();
            if (staged == UINT32_MAX) {
                return result;
            }
            entry.file_hits += entry.file_hits < UINT32_MAX;
        }
        ++entry.readers;
        result.state = state;
        result.index = index;
        result.staged = staged;
        result.offset = location >= 0 ? size_t(loc_offset(location)) : SIZE_MAX;
    }
    if (result.staged != UINT32_MAX) {
        result.state->file_reads.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.staged != UINT32_MAX && !result.state->fetch(result.index,
            result.state->staging_data() + size_t(result.staged) * result.state->stride)) {
        result.release();
    }
    return result;
}

void llama_moe_expert_store::lease::release() {
    if (state) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->entries[index].readers;
            if (staged != UINT32_MAX) {
                state->busy[staged] = false;
            }
        }
        state.reset();
    }
}
llama_moe_expert_store::lease::~lease() { release(); }
llama_moe_expert_store::lease::lease(lease && other) noexcept { *this = std::move(other); }
llama_moe_expert_store::lease & llama_moe_expert_store::lease::operator=(lease && other) noexcept {
    if (this != &other) {
        release();
        state = std::move(other.state);
        index = other.index;
        staged = other.staged;
        offset = other.offset;
    }
    return *this;
}
const void * llama_moe_expert_store::lease::data(uint32_t bank) const {
    if (!state) {
        return nullptr;
    }
    const auto & entry = state->entries[index];
    const auto & layout = state->layouts[entry.key.group];
    if (bank >= layout.offsets.size()) {
        return nullptr;
    }
    const uint8_t * base = staged == UINT32_MAX ? state->resident_data() + offset :
        state->staging_data() + size_t(staged) * state->stride;
    return base + layout.offsets[bank];
}

bool llama_moe_expert_store::lease::tensor_view(uint32_t bank, const ggml_tensor & prototype, ggml_tensor & view) const {
    if (!state || &prototype == &view || prototype.op != GGML_OP_NONE || prototype.view_src != nullptr ||
            prototype.ne[2] <= 0 || prototype.ne[3] != 1 || !ggml_is_contiguous(&prototype)) {
        return false;
    }
    const auto & layout = state->layouts[state->entries[index].key.group];
    if (bank >= layout.sizes.size() || layout.sizes[bank] != prototype.nb[2]) {
        return false;
    }
    view = prototype;
    view.data = const_cast<void *>(data(bank));
    view.buffer = staged == UINT32_MAX ? state->resident.get() : state->staging.get();
    view.extra = nullptr;
    view.ne[2] = 1;
    view.nb[3] = view.nb[2];
    return true;
}

std::vector<std::pair<llama_moe_expert_store::expert, uint32_t>> llama_moe_expert_store::file_hot(uint32_t min_hits) const {
    std::vector<std::pair<expert, uint32_t>> result;
    if (!state) {
        return result;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (uint32_t i = 0; i < state->entries.size(); ++i) {
        const auto & entry = state->entries[i];
        if (state->loc(i) == LOC_FILE && entry.file_hits >= min_hits) {
            result.push_back({entry.key, entry.file_hits});
        }
    }
    std::stable_sort(result.begin(), result.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
    return result;
}

bool llama_moe_expert_store::promote(expert in, expert out) {
    if (!state) {
        return false;
    }
    uint32_t i = UINT32_MAX, o = UINT32_MAX, staged = UINT32_MAX;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        i = state->index(in);
        o = state->index(out);
        if (i == UINT32_MAX || o == UINT32_MAX || i == o) {
            return false;
        }
        auto & incoming = state->entries[i];
        auto & outgoing = state->entries[o];
        if (state->loc(i) != LOC_FILE || incoming.readers || incoming.exchanging ||
                !state->resident_at(o) || outgoing.readers || outgoing.exchanging ||
                state->layouts[in.group].bytes > state->layouts[out.group].bytes) {
            return false;
        }
        staged = state->stage();
        if (staged == UINT32_MAX) {
            return false;
        }
        incoming.exchanging = outgoing.exchanging = true;
    }
    // Read into staging first: a failed read must not damage the outgoing resident bytes.
    uint8_t * stage = state->staging_data() + size_t(staged) * state->stride;
    const bool fetched = state->fetch(i, stage);
    std::lock_guard<std::mutex> lock(state->mutex);
    auto & incoming = state->entries[i];
    auto & outgoing = state->entries[o];
    if (fetched) {
        // An outgoing expert that a GPU slot also holds stays there, now without a RAM copy.
        const int64_t slot = loc_slot(state->loc(o));
        memcpy(state->resident_data() + state->offset(o), stage, state->layouts[in.group].bytes);
        state->set_loc(i, loc_ram(size_t(state->offset(o))));
        state->set_loc(o, slot >= 0 ? loc_gpu(uint32_t(slot)) : LOC_FILE);
        incoming.file_hits = outgoing.file_hits = 0;
    }
    incoming.exchanging = outgoing.exchanging = false;
    state->busy[staged] = false;
    return fetched;
}

llama_moe_expert_store::exchange llama_moe_expert_store::begin_exchange(expert in, expert out) {
    exchange result;
    if (!state || in.group != out.group) {
        return result;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const uint32_t i = state->index(in), o = state->index(out);
    if (i == UINT32_MAX || o == UINT32_MAX || i == o) {
        return result;
    }
    auto & incoming = state->entries[i];
    auto & outgoing = state->entries[o];
    if (!state->resident_at(i) || loc_slot(state->loc(i)) >= 0 || incoming.readers || incoming.exchanging ||
            !gpu_only(state->loc(o)) || outgoing.readers || outgoing.exchanging) {
        return result;
    }
    const uint32_t staged = state->stage();
    if (staged == UINT32_MAX) {
        return result;
    }
    incoming.exchanging = outgoing.exchanging = true;
    result.state = state;
    result.in = i;
    result.out = o;
    result.staged = staged;
    return result;
}
const void * llama_moe_expert_store::exchange::incoming(uint32_t bank) const {
    if (!state) {
        return nullptr;
    }
    const auto & entry = state->entries[in];
    const auto & layout = state->layouts[entry.key.group];
    return bank < layout.offsets.size() ? state->resident_data() + state->offset(in) + layout.offsets[bank] : nullptr;
}
void * llama_moe_expert_store::exchange::outgoing(uint32_t bank) const {
    if (!state) {
        return nullptr;
    }
    const auto & layout = state->layouts[state->entries[out].key.group];
    return bank < layout.offsets.size() ? state->staging_data() + size_t(staged) * state->stride + layout.offsets[bank] : nullptr;
}
bool llama_moe_expert_store::exchange::commit() {
    if (!state) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        auto & incoming = state->entries[in];
        auto & outgoing = state->entries[out];
        const size_t bytes = state->layouts[incoming.key.group].bytes;
        const int64_t position = state->offset(in);
        std::memcpy(state->resident_data() + position, state->staging_data() + size_t(staged) * state->stride, bytes);
        state->set_loc(in, state->loc(out));
        state->set_loc(out, loc_ram(size_t(position)));
        incoming.exchanging = outgoing.exchanging = false;
        state->busy[staged] = false;
    }
    state.reset();
    return true;
}
void llama_moe_expert_store::exchange::cancel() {
    if (state) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->entries[in].exchanging = state->entries[out].exchanging = false;
            state->busy[staged] = false;
        }
        state.reset();
    }
}
llama_moe_expert_store::exchange::~exchange() { cancel(); }
llama_moe_expert_store::exchange::exchange(exchange && other) noexcept { *this = std::move(other); }
llama_moe_expert_store::exchange & llama_moe_expert_store::exchange::operator=(exchange && other) noexcept {
    if (this != &other) {
        cancel();
        state = std::move(other.state);
        in = other.in;
        out = other.out;
        staged = other.staged;
    }
    return *this;
}
size_t llama_moe_expert_store::resident_bytes() const { return state ? state->resident_size : 0; }
const void * llama_moe_expert_store::resident_data(expert key, uint32_t bank) const {
    if (!state) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const uint32_t index = state->index(key);
    if (index == UINT32_MAX) {
        return nullptr;
    }
    const auto & layout = state->layouts[key.group];
    const int64_t location = state->offset(index);
    if (location < 0 || bank >= layout.offsets.size()) {
        return nullptr;
    }
    return state->resident_data() + location + layout.offsets[bank];
}

int64_t llama_moe_expert_store::location(expert key) const {
    if (!state) {
        return LOC_FILE;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const uint32_t index = state->index(key);
    return index == UINT32_MAX ? LOC_FILE : state->loc(index);
}

llama_moe_expert_store::tiers llama_moe_expert_store::verify() const {
    tiers result;
    if (!state) {
        return result;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    std::vector<uint8_t> expected(state->stride);
    for (uint32_t i = 0; i < state->entries.size(); ++i) {
        const int64_t location = state->loc(i);
        const int64_t slot = loc_slot(location);
        if (slot >= 0) {
            ++result.gpu;
            // with even ranking, expert k of a group starts in slot k
            result.gpu_moved += uint64_t(slot) != state->entries[i].key.id;
        }
        if (location == LOC_FILE) {
            ++result.file;
        } else if (location >= 0) {
            ++(slot >= 0 ? result.gpu_copies : result.ram);
            const auto & layout = state->layouts[state->entries[i].key.group];
            const uint8_t * held = state->resident_data() + loc_offset(location);
            bool same = state->fetch(i, expected.data());
            for (size_t bank = 0; same && bank < layout.sizes.size(); ++bank) {
                same = std::memcmp(held + layout.offsets[bank], expected.data() + layout.offsets[bank], layout.sizes[bank]) == 0;
            }
            result.mismatched += !same;
        }
    }
    return result;
}

int64_t * llama_moe_expert_store::locations(uint32_t group) const {
    if (!state || !state->exchangeable || group >= state->layouts.size()) {
        return nullptr;
    }
    return const_cast<int64_t *>(state->locations) + state->layouts[group].first;
}

const void * llama_moe_expert_store::resident_base() const {
    return state && state->resident ? state->resident_data() : nullptr;
}

size_t llama_moe_expert_store::bank_offset(uint32_t group, uint32_t bank) const {
    if (!state || group >= state->layouts.size() || bank >= state->layouts[group].offsets.size()) {
        return SIZE_MAX;
    }
    return state->layouts[group].offsets[bank];
}

uint64_t llama_moe_expert_store::file_reads() const {
    return state ? state->file_reads.load(std::memory_order_relaxed) : 0;
}

bool llama_moe_expert_store::pinned() const {
    return state && state->pinned && (state->resident || state->staging);
}

size_t llama_moe_expert_store::staging_bytes() const { return state ? state->staging_size : 0; }
