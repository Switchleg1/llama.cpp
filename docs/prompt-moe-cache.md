# Prompt MoE cache size

`--prompt-moe-expert-cache-size N` sets the target cache size during the prompt workspace phase. The default, `0`, disables this feature: it does not disable the normal cache and adds no phase-triggered cache clearing.

Example:

```bat
--moe-expert-cache-size 40 ^
--prompt-moe-expert-cache-size 32 ^
-ub 768 ^
```

A positive setting automatically enables phase-aware compute workspace sizing. The target uses the prompt size before reserving the larger prompt workspace and returns to the normal size in the generation workspace phase. Draft cache sizing remains controlled by `--spec-draft-moe-expert-cache-size`.

The prompt value must be no larger than the normal target size and no smaller than the model's active expert count. It requires a backend that supports the MoE candidate cache. Unsupported configurations fail at context initialization.

Each actual size change synchronizes GPU work, invalidates captured CUDA graphs containing old cache addresses, and clears the affected target cache. New allocations and expert contents are populated on demand. Repeated chunks in the same workspace phase do not resize again. Equal prompt and generation sizes do not trigger size-change clearing. This does not preserve hot experts across a resize.

The phase boundary follows the existing workspace planner: batches larger than the decode reservation use prompt capacity. Small prompt batches that fit the decode reservation use generation capacity. The configured context, micro-batch size, model weights, and KV contents are unchanged.

Look for `moe-cache-phase-resize: phase=prefill slots=40->32` and the reverse `phase=decode` message. Memory savings and total turn performance need validation with the workload; shared workspace and backend allocator retention can affect observed VRAM use. This is explicit sizing, not automatic free-VRAM management.
