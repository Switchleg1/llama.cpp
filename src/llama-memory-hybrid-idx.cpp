#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
llama_memory_placement_options placement,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type, offload, placement,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        // the indexer reads every key each step, so its cache stays in device memory
        llama_memory_placement_options placement_idx = placement;
        placement_idx.kv_host_mapped = false;

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, placement_idx, "idx_");
    }()) {
    if (!mem_idx) {
        return;
    }

    const auto & hp = model.hparams;
    const uint32_t n_stream = mem_idx->get_n_stream();

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context *>> ctx_list;
    for (const uint32_t il : mem_idx->get_layer_ids()) {
        const uint32_t r = hp.dsv4_compress_ratios[il];
        const ggml_tensor * k = mem_idx->get_k_storage(il);
        if (r == 0 || k == nullptr || k->buffer == nullptr) {
            continue;
        }
        // same device as the indexer keys the rows are pooled from
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(k->buffer);
        auto it = std::find_if(ctx_list.begin(), ctx_list.end(), [&](const auto & e) { return e.first == buft; });
        if (it == ctx_list.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(hp.n_layer())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                throw std::runtime_error("failed to create the QSA pool context");
            }
            ctx_list.emplace_back(buft, ctx);
            it = ctx_list.end() - 1;
        }
        ggml_tensor * t = ggml_new_tensor_3d(it->second, GGML_TYPE_F32, hp.indexer_head_size, mem_idx->get_size()/r + 2, n_stream);
        ggml_format_name(t, "qsa_pool_l%u", il);
        qsa_pools[(int32_t) il]      = t;
        qsa_pool_ratio[(int32_t) il] = r;
        qsa_pool_stale[r].assign(n_stream, 1);
    }

    for (auto & [buft, ctx] : ctx_list) {
        ggml_backend_buffer_t buf;
        if (hp.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error("failed to allocate the QSA pool buffer");
        }
        // finite rows everywhere: a row that no block owns still meets a -inf bias, never a nan
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s QSA pool buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf),
            (hp.no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft) : ggml_backend_buffer_get_size(buf))/1024.0/1024.0);
        qsa_pool_bufs.emplace_back(ggml_context_ptr(ctx), ggml_backend_buffer_ptr(buf));
    }
}

void llama_memory_hybrid_idx::qsa_pool_invalidate() {
    for (auto & [r, stale] : qsa_pool_stale) {
        std::fill(stale.begin(), stale.end(), 1);
    }
}

ggml_tensor * llama_memory_hybrid_idx::get_qsa_pool(int32_t il) const {
    const auto it = qsa_pools.find(il);
    return it == qsa_pools.end() ? nullptr : it->second;
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_reserve(uint32_t n_kv) {
    return std::make_unique<llama_memory_hybrid_idx_context>(
            this,
            get_mem_attn()->init_reserve(n_kv),
            mem_idx == nullptr ? nullptr : mem_idx->init_reserve(n_kv));
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);
    qsa_pool_invalidate();

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
    qsa_pool_invalidate();

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);
    qsa_pool_invalidate();

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
    qsa_pool_invalidate();

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
    qsa_pool_invalidate();

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    for (const auto & [ctx, buf] : qsa_pool_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        mb[buft] += ggml_backend_buffer_get_base(buf.get()) == nullptr ?
            ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft) : ggml_backend_buffer_get_size(buf.get());
    }

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    // restored cells come without pooled rows
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        qsa_pool_invalidate();
    }

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * spare_cells,
        ggml_tensor * spare_val,
        ggml_tensor * blk_sel,
        ggml_tensor * pool_cells,
        ggml_tensor * pool_pos,
        ggml_tensor * pool_rows) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT((spare_cells == nullptr) == (blk_sel == nullptr));
    GGML_ASSERT(get_mem_idx() != nullptr);
    GGML_ASSERT((spare_cells == nullptr) == (spare_val == nullptr) && (spare_cells == nullptr || blk_bias));

    if (pool_cells != nullptr) {
        const qsa_pool_plan plan = get_qsa_pool_plan(*ubatch, (uint32_t) cell_blk->ne[1], ratio, cell_blk->ne[0], blk_bias);
        GGML_ASSERT(plan.on && plan.cap == pool_rows->ne[0] && "qsa: pool plan changed after the graph was built");
        // the graph reads blk_pos only to recompute every block, for checks
        if (blk_pos->buffer != nullptr) {
            const int64_t nb = plan.n_blocks;
            int32_t * dst = (int32_t *) blk_pos->data;
            for (int64_t b = 0; b < nb; ++b) {
                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst[sec*nb + b] = (int32_t) ((plan.pb_lo + b)*ratio);
                }
            }
        }
        set_input_qsa_pool(plan, cell_blk, blk_cells, bias, ubatch, ratio, spare_cells, spare_val, blk_sel,
            pool_cells, pool_pos, pool_rows);
        return;
    }

    // this ubatch does not maintain the pool rows of its streams
    if (auto stale = qsa_pool_stale.find(ratio); stale != qsa_pool_stale.end()) {
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            stale->second[get_mem_idx()->get_seq_stream(ubatch->seq_id[i][0])] = 1;
        }
    }

    // block-level selection leaves cell_blk out of the graph, so it may have no buffer
    GGML_ASSERT(cell_blk->buffer == nullptr || ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    std::vector<int32_t> cell_blk_scratch;
    if (cell_blk->buffer == nullptr) {
        cell_blk_scratch.resize(ggml_nelements(cell_blk));
    }
    int32_t * dst_cell_blk  = cell_blk->buffer != nullptr ? (int32_t *) cell_blk->data : cell_blk_scratch.data();
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        // cells that no full block covers: the spare block's cells
        std::vector<int32_t> spare;
        if (spare_cells != nullptr && have_dead) {
            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j) && blk_of[j] < 0) {
                    spare.push_back((int32_t) j);
                }
            }
        }
        if (blk_sel != nullptr) {
            float * dst_sel = (float *) blk_sel->data + s*n_blocks;
            std::fill(dst_sel, dst_sel + n_blocks, 0.0f);
            if (have_dead) {
                dst_sel[dead_bid] = -INFINITY;
            }
        }
        if (spare_cells != nullptr) {
            const int64_t n_spare = spare_cells->ne[0];
            GGML_ASSERT((int64_t) spare.size() <= n_spare && "qsa: more unpooled cells than the graph holds");
            int32_t * dst_cells = (int32_t *) spare_cells->data + s*n_tps*n_spare;
            float   * dst_val   = (float   *) spare_val->data   + s*n_tps*n_spare;
            for (int64_t ii = 0; ii < n_tps; ++ii) {
                for (int64_t k = 0; k < n_spare; ++k) {
                    const bool used = k < (int64_t) spare.size();
                    // the spare block's bias is always 1e9, see below
                    dst_cells[ii*n_spare + k] = used ? spare[k] : 0;
                    dst_val  [ii*n_spare + k] = used ? 1e9f : -INFINITY;
                }
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // a block wholly after the query has no visible cell; -inf keeps block-level selection from taking it
                    if (bid_idx[b] > q) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

llama_memory_hybrid_idx::qsa_pool_plan llama_memory_hybrid_idx::get_qsa_pool_plan(
        const llama_ubatch & ubatch, uint32_t ns, uint32_t ratio, int64_t n_kv, bool blk_bias) const {
    qsa_pool_plan plan;

    const auto stale = qsa_pool_stale.find(ratio);
    if (stale == qsa_pool_stale.end() || !blk_bias || ns != 1 || ubatch.n_tokens == 0 || !qsa_contiguous(ubatch, ns)) {
        return plan;
    }

    const llama_seq_id seq = ubatch.seq_id[0][0];
    const auto & cells = get_mem_idx()->get_cells(seq);

    const int64_t r  = ratio;
    const int64_t p0 = cells.seq_pos_min(seq);
    const int64_t p1 = cells.seq_pos_max(seq);

    plan.stream   = get_mem_idx()->get_seq_stream(seq);
    plan.n_rows   = get_mem_idx()->get_size()/r + 2;
    plan.pb_lo    = p0/r;
    plan.n_blocks = (n_kv + r - 1)/r + 1;

    // the view and the padding row must stay inside the pool
    if (p1/r - plan.pb_lo >= plan.n_blocks || plan.pb_lo + plan.n_blocks > plan.n_rows - 1) {
        return plan;
    }

    auto complete = [&](int64_t pb) { return pb*r >= p0 && pb*r + r - 1 <= p1; };

    if (stale->second[plan.stream]) {
        for (int64_t pb = plan.pb_lo; pb <= p1/r; ++pb) {
            if (complete(pb)) {
                plan.dirty.push_back(pb);
            }
        }
    } else {
        // a block is pooled by the ubatch that writes its last missing cell
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const int64_t pb = ubatch.pos[i]/r;
            if (complete(pb)) {
                plan.dirty.push_back(pb);
            }
        }
        std::sort(plan.dirty.begin(), plan.dirty.end());
        plan.dirty.erase(std::unique(plan.dirty.begin(), plan.dirty.end()), plan.dirty.end());
    }

    const int64_t n = (int64_t) plan.dirty.size();
    plan.cap = n <= 4 ? 4 : (n + 63)/64*64;
    plan.on  = true;

    return plan;
}

void llama_memory_hybrid_idx::set_input_qsa_pool(
        const qsa_pool_plan & plan,
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        ggml_tensor * spare_cells,
        ggml_tensor * spare_val,
        ggml_tensor * blk_sel,
        ggml_tensor * pool_cells,
        ggml_tensor * pool_pos,
        ggml_tensor * pool_rows) const {
    const int64_t r        = ratio;
    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_blocks = plan.n_blocks;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t cap      = plan.cap;

    GGML_ASSERT(blk_cells->ne[0] == r*n_blocks && bias->ne[0] == n_blocks && bias->ne[1] == n_tokens);
    GGML_ASSERT(pool_cells->ne[0] == r*cap && pool_pos->ne[0] == 4*cap && pool_rows->ne[0] == cap);

    const llama_seq_id seq = ubatch->seq_id[0][0];
    const auto & cells = get_mem_idx()->get_cells(seq);

    const int64_t p0 = cells.seq_pos_min(seq);
    const int64_t p1 = cells.seq_pos_max(seq);

    // one pass maps positions to cells; qsa_contiguous guarantees every position has exactly one
    std::vector<int32_t> cell_at(p1 - p0 + 1, -1);
    for (int64_t j = 0; j < n_kv; ++j) {
        if (!cells.is_empty(j)) {
            cell_at[cells.pos_get(j) - p0] = (int32_t) j;
        }
    }

    auto complete = [&](int64_t pb) { return pb*r >= p0 && pb*r + r - 1 <= p1; };

    // cells no complete block covers share one spare block, which is always visible
    std::vector<int32_t> spare;
    int64_t dead = -1;
    for (int64_t p = p0; p <= p1; ++p) {
        GGML_ASSERT(cell_at[p - p0] >= 0 && "qsa: position without a cell");
        if (!complete(p/r)) {
            spare.push_back(cell_at[p - p0]);
            dead = p/r - plan.pb_lo;
        }
    }
    const bool have_dead = dead >= 0;
    if (!have_dead) {
        // only empty cells point there, and the mask drops them
        dead = n_blocks - 1;
    }

    // inputs the graph does not read have no buffer
    auto host = [](ggml_tensor * t) { return t != nullptr && t->buffer != nullptr ? t->data : nullptr; };

    if (auto * dst = (int32_t *) host(cell_blk)) {
        std::fill(dst, dst + n_kv, (int32_t) dead);
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t pb = plan.pb_lo + b;
            if (complete(pb)) {
                for (int64_t m = 0; m < r; ++m) {
                    dst[cell_at[pb*r + m - p0]] = (int32_t) b;
                }
            }
        }
    }

    if (auto * dst = (int32_t *) host(blk_cells)) {
        std::fill(dst, dst + r*n_blocks, 0);
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t pb = plan.pb_lo + b;
            if (complete(pb)) {
                for (int64_t m = 0; m < r; ++m) {
                    dst[b*r + m] = cell_at[pb*r + m - p0];
                }
            }
        }
    }

    if (auto * dst = (float *) host(blk_sel)) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            dst[b] = complete(plan.pb_lo + b) ? 0.0f : -INFINITY;
        }
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
    float * dst_bias = (float *) bias->data;
    std::vector<float> base(n_blocks);
    for (int64_t b = 0; b < n_blocks; ++b) {
        base[b] = complete(plan.pb_lo + b) ? 0.0f : -INFINITY;
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        const int64_t q          = ubatch->pos[i];
        const int64_t tail_start = (q + 1)/r*r;
        float * cur = dst_bias + i*n_blocks;
        std::copy(base.begin(), base.end(), cur);
        // blocks are in position order: the ones that start after the query are a suffix
        const int64_t b_after = std::clamp<int64_t>(q/r - plan.pb_lo + 1, 0, n_blocks);
        std::fill(cur + b_after, cur + n_blocks, -INFINITY);
        // a complete block that starts at tail_start <= q is always visible; finite, so it never meets a -inf
        const int64_t b_tail = tail_start/r - plan.pb_lo;
        if (tail_start <= q && b_tail >= 0 && b_tail < n_blocks && base[b_tail] == 0.0f) {
            cur[b_tail] = 1e9f;
        }
        if (have_dead) {
            cur[dead] = 1e9f;
        }
    }

    if (spare_cells != nullptr) {
        const int64_t n_spare = spare_cells->ne[0];
        GGML_ASSERT((int64_t) spare.size() <= n_spare && "qsa: more unpooled cells than the graph holds");
        int32_t * dst_cells = (int32_t *) spare_cells->data;
        float   * dst_val   = (float   *) spare_val->data;
        for (int64_t i = 0; i < n_tokens; ++i) {
            for (int64_t k = 0; k < n_spare; ++k) {
                const bool used = k < (int64_t) spare.size();
                dst_cells[i*n_spare + k] = used ? spare[k] : 0;
                dst_val  [i*n_spare + k] = used ? 1e9f : -INFINITY;
            }
        }
    }

    // padding slots pool cell 0 into the last row, which no view reads
    int32_t * dst_cells = (int32_t *) pool_cells->data;
    int32_t * dst_pos   = (int32_t *) pool_pos->data;
    int32_t * dst_rows  = (int32_t *) pool_rows->data;
    for (int64_t k = 0; k < cap; ++k) {
        const bool used = k < (int64_t) plan.dirty.size();
        const int64_t pb = used ? plan.dirty[k] : 0;
        for (int64_t m = 0; m < r; ++m) {
            dst_cells[k*r + m] = used ? cell_at[pb*r + m - p0] : 0;
        }
        for (int64_t sec = 0; sec < 4; ++sec) {
            dst_pos[sec*cap + k] = used ? (int32_t) (pb*r) : 0;
        }
        dst_rows[k] = used ? (int32_t) pb : (int32_t) (plan.n_rows - 1);
    }

    qsa_pool_stale.at(ratio)[plan.stream] = 0;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
      llama_memory_context_ptr   ctx_attn,
      llama_memory_context_ptr   ctx_idx) :
    llama_memory_hybrid_context(mem, std::move(ctx_attn)),
    mem(mem),
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(std::move(ctx_idx)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * spare_cells,
        ggml_tensor * spare_val,
        ggml_tensor * blk_sel,
        ggml_tensor * pool_cells,
        ggml_tensor * pool_pos,
        ggml_tensor * pool_rows) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, spare_cells, spare_val, blk_sel,
        pool_cells, pool_pos, pool_rows);
}

llama_memory_hybrid_idx::qsa_pool_plan llama_memory_hybrid_idx_context::get_qsa_pool_plan(
        const llama_ubatch & ubatch, uint32_t ratio, int64_t n_kv, bool blk_bias) const {
    return mem != nullptr ? mem->get_qsa_pool_plan(ubatch, get_n_stream(), ratio, n_kv, blk_bias)
                          : llama_memory_hybrid_idx::qsa_pool_plan();
}

ggml_tensor * llama_memory_hybrid_idx_context::get_qsa_pool(int32_t il) const {
    return mem != nullptr ? mem->get_qsa_pool(il) : nullptr;
}

bool llama_memory_hybrid_idx_context::qsa_contiguous(const llama_ubatch & ubatch) const {
    return mem != nullptr && mem->qsa_contiguous(ubatch, get_n_stream());
}

bool llama_memory_hybrid_idx::qsa_contiguous(const llama_ubatch & ubatch, uint32_t ns) const {
    if (get_mem_idx() == nullptr || ns == 0 || ubatch.n_tokens % ns != 0) {
        return false;
    }
    const uint32_t n_tps = ubatch.n_tokens/ns;
    for (uint32_t s = 0; s < ns; ++s) {
        const llama_seq_id seq = ubatch.seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq);
        const llama_pos p0 = cells.seq_pos_min(seq);
        const llama_pos p1 = cells.seq_pos_max(seq);
        if (p0 < 0 || cells.get_used() != uint32_t(p1 - p0 + 1)) {
            return false;
        }
        for (llama_seq_id other = 0; other < LLAMA_MAX_SEQ; ++other) {
            if (other != seq && cells.seq_pos_min(other) >= 0) {
                return false;
            }
        }
    }
    return true;
}
