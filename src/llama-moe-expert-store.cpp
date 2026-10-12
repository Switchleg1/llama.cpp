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
        size_t offset = SIZE_MAX;
        uint32_t readers = 0;
        uint32_t file_hits = 0;   // file-tier reads since the expert last moved
        bool gpu = false;
        bool exchanging = false;
        bool ranked = false;
    };
    std::vector<layout> layouts;
    std::vector<entry> entries;
    ggml_backend_buffer_ptr resident;
    ggml_backend_buffer_ptr staging;
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
        ggml_backend_buffer_type_t host_buft) {
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
        for (expert key : gpu) {
            const uint32_t index = next->index(key);
            if (index == UINT32_MAX || next->entries[index].gpu) {
                return false;
            }
            next->entries[index].gpu = true;
        }
        auto admit = [&](uint32_t index) {
            auto & entry = next->entries[index];
            const size_t bytes = next->layouts[entry.key.group].bytes;
            if (!entry.gpu && bytes <= resident_budget - next->resident_size) {
                entry.offset = next->resident_size;
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
        for (uint32_t i = 0; i < next->entries.size(); ++i) {
            const auto & entry = next->entries[i];
            if (entry.offset != SIZE_MAX && !next->fetch(i, next->resident_data() + entry.offset)) {
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
        if (entry.gpu || entry.exchanging || entry.readers == UINT32_MAX) {
            return result;
        }
        uint32_t staged = UINT32_MAX;
        if (entry.offset == SIZE_MAX) {
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
    const uint8_t * base = staged == UINT32_MAX ? state->resident_data() + entry.offset :
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
    for (const auto & entry : state->entries) {
        if (!entry.gpu && entry.offset == SIZE_MAX && entry.file_hits >= min_hits) {
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
        if (incoming.gpu || incoming.offset != SIZE_MAX || incoming.readers || incoming.exchanging ||
                outgoing.gpu || outgoing.offset == SIZE_MAX || outgoing.readers || outgoing.exchanging ||
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
        memcpy(state->resident_data() + outgoing.offset, stage, state->layouts[in.group].bytes);
        incoming.offset = outgoing.offset;
        outgoing.offset = SIZE_MAX;
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
    if (incoming.gpu || incoming.offset == SIZE_MAX || incoming.readers || incoming.exchanging ||
            !outgoing.gpu || outgoing.readers || outgoing.exchanging) {
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
    return bank < layout.offsets.size() ? state->resident_data() + entry.offset + layout.offsets[bank] : nullptr;
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
        std::memcpy(state->resident_data() + incoming.offset, state->staging_data() + size_t(staged) * state->stride, bytes);
        outgoing.offset = incoming.offset;
        incoming.offset = SIZE_MAX;
        incoming.gpu = true;
        outgoing.gpu = false;
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
    const auto & entry = state->entries[index];
    const auto & layout = state->layouts[key.group];
    if (entry.gpu || entry.offset == SIZE_MAX || bank >= layout.offsets.size()) {
        return nullptr;
    }
    return state->resident_data() + entry.offset + layout.offsets[bank];
}

uint64_t llama_moe_expert_store::file_reads() const {
    return state ? state->file_reads.load(std::memory_order_relaxed) : 0;
}

bool llama_moe_expert_store::pinned() const {
    return state && state->pinned && (state->resident || state->staging);
}

size_t llama_moe_expert_store::staging_bytes() const { return state ? state->staging_size : 0; }
