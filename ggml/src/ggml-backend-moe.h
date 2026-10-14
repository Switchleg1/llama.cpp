#pragma once

#include <stdlib.h>

#include "ggml-backend.h"

// Private fork protocol shared by llama and backend implementations.
// Resolve these procedures through the backend registry; this header is not installed.
#ifdef __cplusplus
extern "C" {
#endif

#define GGML_BACKEND_MOE_CACHE_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_buffer_type"
#define GGML_BACKEND_MOE_CACHE_BOUNDED_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_bounded_buffer_type"
#define GGML_BACKEND_MOE_STAGING_SIZE_V1_PROC_NAME "ggml_backend_moe_staging_size_v1"
#define GGML_BACKEND_MOE_DEVICE_SIZE_V1_PROC_NAME "ggml_backend_moe_device_size_v1"
#define GGML_BACKEND_MOE_DEVICE_SIZE_V2_PROC_NAME             "ggml_backend_moe_device_size_v2"
#define GGML_BACKEND_MOE_CACHE_FREE_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_free_buffer_type"
#define GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME "ggml_backend_moe_cache_configure_sources"
#define GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME "ggml_backend_moe_cache_is_buffer_type"
#define GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME "ggml_backend_moe_cache_buffer_from_host_ptr"
#define GGML_BACKEND_MOE_CACHE_WRITABLE_LOAD_DATA_PROC_NAME "ggml_backend_moe_cache_writable_load_data"
#define GGML_BACKEND_MOE_CACHE_SET_DEBUG_PROC_NAME "ggml_backend_moe_cache_set_debug"
#define GGML_BACKEND_MOE_EARLY_ROUTER_SET_ENABLED_PROC_NAME "ggml_backend_moe_early_router_set_enabled"
#define GGML_BACKEND_MOE_EARLY_ROUTER_SET_MAX_ROWS_PROC_NAME "ggml_backend_moe_early_router_set_max_rows"
#define GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME "ggml_backend_moe_cache_log_and_reset_stats"

typedef ggml_backend_buffer_type_t (*ggml_backend_moe_cache_buffer_type_t)(void);
typedef ggml_backend_buffer_type_t (*ggml_backend_moe_cache_bounded_buffer_type_t)(size_t bytes);
typedef void (*ggml_backend_moe_cache_free_buffer_type_t)(ggml_backend_buffer_type_t buft);
typedef bool (*ggml_backend_moe_cache_is_buffer_type_t)(ggml_backend_buffer_type_t buft);
typedef ggml_backend_buffer_t (*ggml_backend_moe_cache_buffer_from_host_ptr_t)(ggml_backend_buffer_type_t buft, void * ptr, size_t size);
typedef void * (*ggml_backend_moe_cache_writable_load_data_t)(ggml_backend_buffer_t buffer, void * data, size_t size);
// Experimental logging remains process-wide, including retired backend counters.
typedef void (*ggml_backend_moe_cache_set_debug_t)(bool enabled);
typedef void (*ggml_backend_moe_early_router_set_enabled_t)(bool enabled);
typedef void (*ggml_backend_moe_early_router_set_max_rows_t)(ggml_backend_t backend, uint32_t max_rows);
typedef void (*ggml_backend_moe_cache_log_and_reset_stats_t)(void);

#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_V1_PROC_NAME "ggml_backend_moe_candidate_replace_v1"
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC 0x4d4f4531u
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION 1u
#define GGML_BACKEND_MOE_CANDIDATE_REPLACE_V2_PROC_NAME "ggml_backend_moe_candidate_replace_v2"
#define GGML_BACKEND_REQUIRED_GROUPED_EXECUTION_SUPPORTED_PROC_NAME "ggml_backend_required_grouped_execution_supported"
// Diagnostic: recomputes the resident routes of one region and compares them with the executed output.
#define GGML_BACKEND_MOE_HYBRID_SHADOW_PROC_NAME "ggml_backend_moe_hybrid_shadow"

struct ggml_backend_moe_hybrid_shadow_result {
    int32_t n_routes;
    int32_t n_hot;
    double  nmse;
};

typedef bool (*ggml_backend_moe_hybrid_shadow_t)(
        ggml_backend_t backend,
        const struct ggml_tensor * ids,
        const struct ggml_tensor * input,
        const struct ggml_tensor * down_output,
        const struct ggml_tensor * gate,
        const struct ggml_tensor * up,
        const struct ggml_tensor * down,
        bool gelu,
        struct ggml_backend_moe_hybrid_shadow_result * result);

// Lets the owner compute cold routes with its own CPU path. Activation rows are activation_stride floats apart;
// rows[i] names the row of route i. Results are n_embd floats per route, written result_stride floats apart. Return false to let the backend fall back. Clear with a null handler before
// the user data is destroyed; clearing waits for a call in progress.
#define GGML_BACKEND_MOE_HYBRID_SET_COLD_HANDLER_PROC_NAME "ggml_backend_moe_hybrid_set_cold_handler"

typedef bool (*ggml_backend_moe_hybrid_cold_handler_t)(
        void * user, const struct ggml_tensor * down, const float * activation, size_t activation_stride,
        const void * activation_q8_1, size_t q8_1_row_bytes, const int32_t * rows, const int32_t * experts, uint32_t count,
        float * results, size_t result_stride);
typedef void (*ggml_backend_moe_hybrid_set_cold_handler_t)(ggml_backend_moe_hybrid_cold_handler_t handler, void * user);

// Debug and legacy environment switches of the MoE split take effect only when LLAMA_MOE_DEBUG is set, so a
// normal run is configured by its parameters alone.
static inline const char * ggml_moe_debug_getenv(const char * name) {
    const char * debug = getenv("LLAMA_MOE_DEBUG");
    return debug != NULL && debug[0] != '\0' && debug[0] != '0' ? getenv(name) : NULL;
}

// Expert split configuration: when split is set, experts that miss the GPU cache run on the owner's CPU
// handler (cold routes) while the GPU computes the cached ones. promote_budget bounds in-call promotions per
// layer call, promote_misses is the number of cold calls before an expert may be promoted. Process-wide; call
// before the first graph that uses the expert cache. Without a call, debug environment variables apply.
#define GGML_BACKEND_MOE_HYBRID_CONFIGURE_PROC_NAME "ggml_backend_moe_hybrid_configure"
struct ggml_backend_moe_hybrid_config {
    bool     split;
    uint32_t promote_budget;
    uint32_t promote_misses;
};
typedef bool (*ggml_backend_moe_hybrid_configure_t)(const struct ggml_backend_moe_hybrid_config * config);

// Strict mode: a refused or failed cold packet is not recomputed from the backend's own bank pointers.
// The cumulative failure count covers strict refusals and device wait timeouts. Unplanned packets carry no
// routes for the current token and are not failures.
// Read it after the stream is synchronized; an increase means the last graph output is invalid.
#define GGML_BACKEND_MOE_HYBRID_SET_COLD_STRICT_PROC_NAME "ggml_backend_moe_hybrid_set_cold_strict"
#define GGML_BACKEND_MOE_HYBRID_COLD_FAILURES_PROC_NAME "ggml_backend_moe_hybrid_cold_failures"
typedef void (*ggml_backend_moe_hybrid_set_cold_strict_t)(bool strict);
typedef uint64_t (*ggml_backend_moe_hybrid_cold_failures_t)(void);

// Pinned per-expert sources for a routed bank tensor; grouped resources created afterwards fill from them.
// A null entry marks an expert the owner does not hold: it is never filled and stays with the CPU.
// The owner keeps the memory valid until it unregisters (null hosts) and the backend has released resources.
// Planned-call count per expert since the grouped resource holding this bank was created. Synchronizes.
#define GGML_BACKEND_MOE_EXPERT_USAGE_PROC_NAME "ggml_backend_moe_expert_usage"
typedef bool (*ggml_backend_moe_expert_usage_t)(
        ggml_backend_t backend, const struct ggml_tensor * tensor, uint32_t * counts, uint32_t n);

// Repoints one registered expert (null: stop filling it) in the registry and in existing grouped resources.
// The caller guarantees no grouped work reading this bank is in flight.
#define GGML_BACKEND_MOE_UPDATE_EXPERT_SOURCE_PROC_NAME "ggml_backend_moe_update_expert_source"
typedef bool (*ggml_backend_moe_update_expert_source_t)(const struct ggml_tensor * tensor, uint32_t expert, const void * host);

// Strata-style adaptive residency: call between graphs on the backend's thread; every N calls it swaps the most
// used non-resident experts into the slots of the least used residents. Seeds come from per-bank priors.
#define GGML_BACKEND_MOE_ADAPT_RESIDENCY_PROC_NAME "ggml_backend_moe_adapt_residency"
#define GGML_BACKEND_MOE_SET_EXPERT_PRIOR_PROC_NAME "ggml_backend_moe_set_expert_prior"
typedef void (*ggml_backend_moe_adapt_residency_t)(ggml_backend_t backend);
typedef void (*ggml_backend_moe_set_expert_prior_t)(const struct ggml_tensor * tensor, const float * scores, uint32_t n);

// Pinned, device-mapped host memory that the accelerator claims: its kernels read and write tensors in it over the
// bus, and the scheduler does not copy them. For KV caches whose readers touch only a few rows per step.
#define GGML_BACKEND_KV_HOST_BUFFER_TYPE_PROC_NAME "ggml_backend_kv_host_buffer_type"
typedef ggml_backend_buffer_type_t (*ggml_backend_kv_host_buffer_type_t)(void);

// VRAM page cache for a tensor in the KV host buffer: pool (device memory, same type and row size, a multiple of
// page_rows rows) holds pages of page_rows rows, meta is an int32 device tensor of at least
// ggml_backend_kv_host_cache_meta_size() elements. Gathers of the tensor then read through the pages and writes
// drop them. Null pool unregisters.
#define GGML_BACKEND_KV_HOST_CACHE_PROC_NAME "ggml_backend_kv_host_cache"
typedef bool (*ggml_backend_kv_host_cache_t)(struct ggml_tensor * tensor, struct ggml_tensor * pool,
        struct ggml_tensor * meta, uint32_t page_rows);
static inline size_t ggml_backend_kv_host_cache_meta_size(size_t n_blocks, size_t n_pages) {
    return n_blocks + 6 * n_pages + 3;
}

#define GGML_BACKEND_MOE_SET_EXPERT_SOURCES_PROC_NAME "ggml_backend_moe_set_expert_sources"
typedef bool (*ggml_backend_moe_set_expert_sources_t)(
        const struct ggml_tensor * tensor, const void * const * hosts, uint32_t n_experts, size_t stride);

// Exchange residency for a bank registered above: RAM need not hold the experts that GPU slots hold. locations
// (pinned host memory, one value per expert) is -1 for a file-backed expert, -2 - slot for an expert that only GPU
// slot `slot` holds, or else a byte offset into resident_base in the low 40 bits, with slot + 1 above them when a
// GPU slot also holds the expert. Resources seed the slots from the tensor data. A fill that evicts an expert
// without a RAM copy swaps its bytes into the incoming expert's RAM position; either way locations are updated in
// place. bank_offset is this bank's offset inside an expert bundle. Register after the sources; null unregisters.
#define GGML_BACKEND_MOE_SET_EXPERT_EXCHANGE_PROC_NAME "ggml_backend_moe_set_expert_exchange"
typedef bool (*ggml_backend_moe_set_expert_exchange_t)(
        const struct ggml_tensor * tensor, int64_t * locations, const void * resident_base, size_t bank_offset);

#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC 0x4d4f4532u
#define GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION 2u

enum ggml_backend_moe_candidate_snapshot_flag {
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_FLAG_NONE = 0,
};

enum ggml_backend_moe_candidate_group_flag {
    GGML_BACKEND_MOE_CANDIDATE_GROUP_FLAG_NONE = 0,
};

enum ggml_backend_moe_candidate_layout {
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID       = 0,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE      = 1,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP = 2,
    GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED       = 3,
};

enum ggml_backend_moe_candidate_bank_role {
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID             = 0,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT         = 1,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT           = 2,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT      = 3,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT         = 4,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE          = 5,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE            = 6,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_SCALE       = 7,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE          = 8,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BLOCK_SCALE    = 9,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BLOCK_SCALE      = 10,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BLOCK_SCALE = 11,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BLOCK_SCALE    = 12,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS           = 13,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS             = 14,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS        = 15,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS           = 16,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_INPUT_SCALE    = 17,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_INPUT_SCALE      = 18,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_INPUT_SCALE    = 19,
    GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_COUNT,
};

enum ggml_backend_moe_candidate_replace_result {
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED         = 0,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED         = 1,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ARGUMENT = 2,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_INVALID_ABI      = 3,
    GGML_BACKEND_MOE_CANDIDATE_REPLACE_ERROR            = 4,
};

enum {
    GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS = 512,
    GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS  = 16,
    GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2 = 16384,
};

enum ggml_backend_moe_staging_family_v1 {
    GGML_BACKEND_MOE_STAGING_FAMILY_V1_GROUPED     = 1u << 0,
    GGML_BACKEND_MOE_STAGING_FAMILY_V1_LEGACY      = 1u << 1,
    GGML_BACKEND_MOE_STAGING_FAMILY_V1_HOST_STAGED = 1u << 2,
};

enum ggml_backend_moe_staging_flag_v1 {
    GGML_BACKEND_MOE_STAGING_FLAG_V1_PREDICTION_CONTROL = 1u << 0,
};

struct ggml_backend_moe_staging_query_v1 {
    uint32_t struct_size;
    uint32_t n_slots;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t n_banks;
    uint32_t staged_bank_mask;
    uint32_t family_mask;
    uint32_t flags;
    uint64_t bank_expert_strides[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
};

struct ggml_backend_moe_staging_size_v1 {
    uint32_t struct_size;
    uint32_t reserved32;
    uint64_t grouped_min_bytes;
    uint64_t legacy_min_bytes;
    uint64_t host_staged_min_bytes;
    uint64_t optional_growth_max_bytes;
    uint64_t prepack_tile_bytes;
};

typedef bool (*ggml_backend_moe_staging_size_v1_t)(
    const struct ggml_backend_moe_staging_query_v1 * query,
    struct ggml_backend_moe_staging_size_v1 * result);

enum ggml_backend_moe_device_size_flag_v1 {
    GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG = 1u << 0,
};

struct ggml_backend_moe_device_size_query_v1 {
    uint32_t struct_size;
    uint32_t n_slots;
    uint32_t n_experts;
    uint32_t n_banks;
    uint32_t n_slot_auxiliaries;
    uint32_t flags;
    uint32_t early_width;
    uint32_t early_experts;
    uint32_t early_top_k;
    uint32_t early_hc_rank;
    uint64_t slot_auxiliary_values;
    uint64_t original_shadow_bytes;
    uint64_t prefill_copy_bytes;
    uint64_t bank_expert_strides[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int64_t bank_ne0[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int32_t bank_types[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
};

struct ggml_backend_moe_device_size_v1 {
    uint32_t struct_size;
    uint32_t reserved32;
    uint64_t group_fixed_bytes;
    uint64_t group_per_slot_bytes;
    uint64_t context_fixed_bytes;
};

typedef bool (*ggml_backend_moe_device_size_v1_t)(
    const struct ggml_backend_moe_device_size_query_v1 * query,
    struct ggml_backend_moe_device_size_v1 * result);

struct ggml_backend_moe_device_size_query_v2 {
    uint32_t struct_size;
    uint32_t n_slots;
    uint32_t n_experts;
    uint32_t n_banks;
    uint32_t n_slot_auxiliaries;
    uint32_t flags;
    uint32_t early_width;
    uint32_t early_experts;
    uint32_t early_top_k;
    uint32_t early_hc_rank;
    uint64_t slot_auxiliary_values;
    uint64_t original_shadow_bytes;
    uint64_t prefill_copy_bytes;
    uint64_t bank_expert_strides[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int64_t  bank_ne0[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    int32_t  bank_types[GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS];
    // Native graph-cloned router scratch is not available during model fit accounting.
    uint32_t early_route_capacity;
    uint32_t early_row_capacity;
    uint32_t early_groups;
    uint64_t early_expert_bytes;
};

struct ggml_backend_moe_device_size_v2 {
    uint32_t struct_size;
    uint32_t reserved32;
    uint64_t group_fixed_bytes;
    uint64_t group_per_slot_bytes;
    uint64_t context_fixed_bytes;
    uint64_t host_fixed_bytes;
};

typedef bool (*ggml_backend_moe_device_size_v2_t)(const struct ggml_backend_moe_device_size_query_v2 * query,
                                                  struct ggml_backend_moe_device_size_v2 *             result);

struct ggml_backend_moe_candidate_bank_v1 {
    const struct ggml_tensor * tensor;
    uint32_t role;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_group_v1 {
    const struct ggml_backend_moe_candidate_bank_v1 * banks;
    uint32_t n_banks;
    uint32_t layout;
    uint32_t flags;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_snapshot_v1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    // Implementations reject unknown flags and nonzero reserved fields.
    uint32_t flags;
    // Physical configuration; not part of a logical group signature.
    uint32_t n_slots;
    uint32_t n_groups;
    const struct ggml_backend_moe_candidate_group_v1 * groups;
    uint64_t reserved[2];
};

// A well-formed call replaces the complete backend-local snapshot.
// Arrays are borrowed for the call. Tensor pointers must outlive the accepted snapshot.
typedef int32_t (*ggml_backend_moe_candidate_replace_v1_t)(ggml_backend_t backend, const struct ggml_backend_moe_candidate_snapshot_v1 * snapshot);

enum ggml_backend_moe_candidate_domain_v2 {
    GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_INVALID  = 0,
    GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY = 1,
    GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK    = 2,
};

enum ggml_backend_moe_candidate_status_v2 {
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INVALID       = 0,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE   = 1,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE  = 2,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS   = 3,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE   = 4,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED  = 5,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED = 6,
    GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE  = 7,
};

enum ggml_backend_moe_candidate_snapshot_flag_v2 {
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_NONE             = 0,
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_TENSOR_OVERRIDES = 1u << 0,
    GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE       = 1u << 1,
};

enum ggml_backend_moe_candidate_group_flag_v2 {
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE             = 0,
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA      = 1u << 0,
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES = 1u << 1,
    GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE       = 1u << 2,
};

enum ggml_backend_moe_candidate_tensor_flag_v2 {
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_NONE             = 0,
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER    = 1u << 0,
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA      = 1u << 1,
    GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES = 1u << 2,
};

struct ggml_backend_moe_candidate_group_v2 {
    uint32_t layout;
    uint32_t domain;
    uint32_t flags;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_tensor_v2 {
    const struct ggml_tensor * tensor;
    uint32_t group_index;
    uint32_t role;
    uint32_t status;
    uint32_t flags;
    uint32_t reserved;
};

struct ggml_backend_moe_candidate_snapshot_v2 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t flags;
    uint32_t n_slots;
    uint32_t n_groups;
    const struct ggml_backend_moe_candidate_group_v2 * groups;
    uint32_t n_tensors;
    uint32_t reserved32;
    const struct ggml_backend_moe_candidate_tensor_v2 * tensors;
    uint64_t reserved[2];
};

typedef int32_t (*ggml_backend_moe_candidate_replace_v2_t)(ggml_backend_t backend, const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot);
typedef bool (*ggml_backend_moe_cache_configure_sources_t)(ggml_backend_buffer_type_t buft, const struct ggml_backend_moe_candidate_snapshot_v2 * snapshot);
typedef bool (*ggml_backend_required_grouped_execution_supported_t)(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
