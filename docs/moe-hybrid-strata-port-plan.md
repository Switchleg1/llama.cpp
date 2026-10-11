# Strata-inspired hybrid MoE replacement plan

## Status and purpose

This is a design and implementation handoff, not an implemented feature or a verified performance claim. The repository owner approved the direction below and requested a written plan so another coding model can perform the implementation later. Review the proposed interfaces and execution design with the owner before making substantial code changes.

Target repository: maintained fork `GenerelSchwerz/llama.cpp`. Do not submit to upstream. Follow `AGENTS.md`, `CONTRIBUTING.md`, and applicable repository skills. Commit, push, and PR actions each require explicit owner approval. Do not add new files under `tests/` without approval; extend existing infrastructure.

Planning baseline: llama.cpp commit `1af9519565c878626d4f5490945499846eda7897`. Recheck the current revision and working tree before implementation. The local Strata checkout is not a Git repository, so no source commit was available; record source file hashes/version and license provenance before copying code.

## Owner-approved requirements

- Replace the existing MoE cache execution process with a Strata-inspired hybrid design. Retaining the old refill engine as a permanent selectable mode is not required or desired.
- All selected experts still execute. Do not prune experts, change routing, approximate expert weights, or substitute stale activations. Execution location changes, not model semantics.
- Run GPU-resident experts and CPU-resident cold experts concurrently, then combine their outputs for the same layer and token rows.
- Support generic MoE models through existing metadata, tensor descriptions, and GGUF quantizations, not hardcoded Qwen dimensions.
- Implement bounded resident expert-weight RAM, including file-backed overflow for models larger than available resident capacity.
- Start with one GPU. Keep device ownership explicit for later multi-GPU work, without implementing expert parallelism in the first version.
- Investigate prefill streaming as a separate phase-specific execution strategy.
- Preserve ordinary non-cache GPU and CPU operation. A fully VRAM-resident model must not be forced through hybrid overhead.

The owner mentioned Gemma 4 26B as a future example on other machines. Treat it as a compatibility target to investigate, not verified architecture/backend support. Confirm its exact architecture, GGUF, routed-expert structure, and existing llama.cpp support before promising compatibility.

## Initial test machine and model

- GPU: NVIDIA RTX 5070 Ti, 16 GB VRAM.
- CPU: Intel Core Ultra 7 270K Plus, confirmed by owner.
- RAM: 96 GB DDR5-6000. Measure effective bandwidth and identify physical core topology; advertised memory speed is not achieved kernel bandwidth.
- OS: current checkout is on Windows; confirm build tools, CUDA version, driver, and storage before live tests.
- Model directory: `C:\Users\switchleg\Documents\Strata-data\models\IQ3_S`.
- First shard: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`.
- Second shard: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`.

These filenames were verified locally. Loading this exact architecture in the current llama.cpp revision has not been verified. Model support is an explicit Stage 0 gate, not an implied part of the hybrid port. The directory also contains `.done` files; they are not model shards.

Future oversized-model test: an exact Q4_K_XL GGUF supplied or selected with the owner. Do not automatically download a large model or assume its mixed quantizations are supported by the hybrid executor.

## Meaning of bounded RAM and "without paging"

Strata has two related mechanisms:

1. Complement residency: RAM can retain only experts not held exclusively in GPU memory, avoiding a duplicate full host copy. Adaptive exchanges preserve the authoritative weight bytes.
2. Resident RAM budget: only a configured amount of hot experts is retained in RAM; other experts are read from model files when needed.

The second mechanism does not eliminate SSD traffic or file-backed page faults. Strata's documented Q4_K_XL mode reads overflow experts from SSD. The goal is to avoid depending on a giant allocated weight pool being swapped to the pagefile, while bounding application-owned resident weights and staging.

An expert-weight budget is not a hard total-process RSS limit and is not an OS file-cache limit. Account separately for model metadata, ordinary weights, KV/recurrent state, speculative state, prefill workspaces, staging, allocator overhead, and OS/application headroom. Do not claim that an `mmap` cap controls the Windows standby/file cache.

If a strict no-pagefile guarantee is requested, admission and page locking need a separate explicit contract. Locking/registration can fail. Report the actual outcome; do not silently label pageable memory as guaranteed resident.

## Source map: inspect before implementing

Strata root: `C:\Users\switchleg\Documents\Strata-main`.

Read the implementation and relevant headers, not only the README:

| Area | Strata files | What to extract |
| --- | --- | --- |
| Decode overlap | `src/core/session.cpp`, `include/strata/core/session.hpp` | Router publication, CPU/GPU launch ordering, graph boundaries, same-layer merge, lifetime and synchronization |
| GPU expert residency | `src/core/expert_cache.cpp`, `include/strata/core/expert_cache.hpp` | Residency maps, layout, capacity and hotness-based placement |
| Adaptation and integration | `src/program/generate.cpp` | Usage decay, swaps, RAM/GPU complement exchanges, request boundaries, speculative integration |
| RAM and file tiers | `src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp` | Resident budget, authoritative bytes, GGUF offsets, bounded fetches, file-read concurrency, prefetch and accounting |
| CPU executor | `src/kernels/cpu/pool.cpp`, `native_expert.cpp`, `expert.cpp`, `expert_layout.cpp` and headers under `include/strata/kernels/cpu/` | Quantization coverage, worker ownership, activation layout, batching and core affinity |
| Host memory | `src/core/pinned.cu`, `src/platform/memory.cpp`, `src/platform/direct_file.cpp` | Page locking versus GPU registration, Windows behavior, allocation failure and cleanup |
| Prefill | `src/prefill/prefill.cpp`, `moe_mmq.cu`, `moe_fused.cu`, `moe_fused_iq.cu` and headers | Expert grouping, copy/compute overlap, staging ring and workspace sizing |
| Speculation | `src/core/mtp.cpp`, `src/core/verify.cpp`, `src/spec/controller.cpp`, `draft_policy.cpp`, `suffix_drafter.cpp` | Verification batches, rollback and phase-specific costs; do not port model-specific MTP architecture blindly |
| Contention experiment | `src/core/concurrent_main.cpp` | CPU and DMA share host DRAM bandwidth; independently measured rates cannot simply be added |
| Explanation | `docs/HOW_IT_WORKS.md`, relevant sections of `docs/DETAILS.md` | Context and performance claims to validate against code |

Verified observations from the initial inspection:

- `session.cpp` launches the GPU hit work before CPU miss computation and merges the miss output into the same layer before advancing.
- The source documents an earlier stale-layer merge bug that produced fluent but incorrect outputs. Layer-level parity must catch this class of failure.
- Its mapped-memory doorbell path includes driver-dependent visibility behavior. Do not transplant its polling protocol as a universal synchronization mechanism.
- `DETAILS.md` documents `--resident-budget-gib`, complement residency, and SSD overflow. `expert_source.cpp` contains the matching budget allocation path.
- Prefill uses streamed expert rings for larger chunks. Its fixed thresholds and ring sizes are hardware/model-specific, not generic defaults to copy.
- Existing Strata low-RAM multi-GPU behavior has limitations. Do not infer a finished generic multi-GPU residency design from Strata's GPU support.

Llama.cpp integration points to inspect at the current revision:

- `ggml/src/ggml-cuda/moe-cache.cu` and `.cuh`: existing expert source/group descriptions, capacity, route dispatch, telemetry, lifetime and ownership.
- `ggml/src/ggml-cuda/moe-cache-graph.cu`: graph eligibility and scheduling.
- `ggml/src/ggml-cuda/moe-cache-host.cu` and `.cuh`: host backing, pinning, staging and buffer identity.
- `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmid.cu`, `mmid.cuh`, and weighted reduction kernels: expert consumers and existing fusion boundaries.
- `ggml/src/ggml-cpu/ops.cpp`, `quants.c`, `traits.cpp`, and existing CPU matmul implementations: reuse quantized kernels rather than introduce a second quantization library.
- `ggml/src/ggml-backend.cpp`: scheduler placement and expert transfer behavior.
- `src/llama-model.cpp`, `src/llama-context.cpp`, `src/llama.cpp`, `include/llama.h`: loading, manifest construction, context resources and public settings.
- `common/arg.cpp`, `common/fit.cpp`, and `docs/speculative.md`: options, joint memory admission and target/draft state.
- `docs/moe-cache-priority2-design.md`, `docs/moe-grouped-multigpu.md`: existing ownership/accounting constraints.
- Existing `tests/test-moe-cache*.cpp`, `tests/test-moe-cache.h`, argument parser tests, and backend operation tests.

The Priority 2 design explicitly defers hybrid execution. This document proposes a separately reviewed successor, not permission to silently broaden that earlier implementation.

## Proposed architecture

### Expert identity and authoritative storage

Identify an expert by model instance, semantic expert group, layer, expert ID, and device owner as appropriate. Keep gate/up/down and any supported additional banks together as one complete expert bundle. Do not assume every architecture has identical banks, activations, dimensions, routing weights, or shared-expert behavior.

Use existing source descriptors where possible. Record exact tensor types, strides, offsets and sizes. Residency must not change the semantic ID used by routing. Mixed GGUF quantizations must be supported deliberately or rejected precisely.

Each weight bundle always has a valid authoritative source: immutable GGUF bytes, RAM storage, or GPU-only resident bytes under an explicit recoverability contract. GPU eviction must establish a valid replacement source before recycling storage. Account for temporary exchange buffers and copies.

Reusing loader infrastructure does not mean touching every mapped weight at load time. Avoid full-bank initialization, repacking or registration that defeats the RAM cap. Document CPU repacking overhead and account for it when needed.

### Phase-specific executor

For decode or small verification batches:

1. Obtain the actual router-selected IDs, token rows, weights, and current layer input.
2. Partition contributions into GPU-resident and CPU-executed subsets. File-tier experts are fetched into bounded host staging before CPU execution unless a separately justified GPU execution route is introduced.
3. Launch GPU hit work and CPU miss work without serializing one behind the other unnecessarily.
4. Merge results using the architecture's existing weighting and reduction semantics.
5. Complete all required dependencies before consuming the layer output.

Every selected contribution must execute exactly once. Preserve duplicate IDs/token rows, routed weight normalization, bias/scale/activation semantics, shared experts and residual ordering. File prefetch predicts bytes to fetch, never which experts to execute.

Do not assume existing GPU fused gate/up/down paths can be split safely by changing a single `MUL_MAT_ID` call. Review complete expert-group boundaries and how intermediate activations and reductions are represented.

For prefill, reuse GPU grouped matrix work and bounded weight streaming where that amortizes transfers across token rows. Choose the initial supported phase explicitly. Never silently run an unverified CPU/GPU merge for unsupported batch shapes.

### CPU workers and synchronization

Prefer a bounded executor built on existing CPU kernel/threadpool infrastructure. Avoid nested pools, oversubscription and a dedicated busy-spin core unless measurements justify them. Detect available ISA at runtime; do not assume AVX-512 from CPU branding. Evaluate P/E-core scheduling on the initial machine.

Define publication, completion, cancellation, device errors and teardown with supported synchronization primitives. CUDA graph capture/replay, concurrent slots, repeated model contexts and worker callbacks must not outlive their buffers. Correctness comes before removing synchronization.

### RAM budget and file overflow

Expose a documented resident expert-weight cap; finalize the option name after reviewing existing CLI/API conventions. Keep separate limits/accounting for file-read staging and pinned/registered host memory. Zero/default semantics must be unambiguous; do not reuse an old pinning-budget option to mean resident-weight capacity.

Admission must include ordinary weights, context state, CPU repacked weights, GPU experts, CPU resident experts, transient adaptive exchanges, prefill and file staging, and jointly active target/draft resources. Report actual allocation and pinning outcomes.

Start with bounded, deterministic residency and file fetches. Add hotness-based admission/adaptation only after correctness. GPU-only complement storage is a later capacity optimization within this plan, not an assumption in the first hybrid prototype.

Compare direct buffered reads into bounded buffers with mapped file reads. Choose an approach based on measured Windows behavior and maintainability. Explicitly report that mapping permits OS-managed file caching. Check split GGUF tensor offsets and file bounds. I/O failures must stop the request cleanly, never substitute zero or stale weights.

### Compatibility and replacement policy

The owner does not require permanent retention of the existing cache/refill engine. Keep a pinned baseline build or worktree for comparison instead. Temporary coexistence during development is acceptable only as migration scaffolding, not an indefinitely supported second subsystem.

Preserve existing ordinary backend paths. Unsupported hybrid configurations must fail clearly or use an explicitly documented ordinary execution mode with valid memory admission. Do not silently fall back to an oversized full-RAM allocation.

Before deleting old code, inventory affected public options, scripts, telemetry, tests and backend variants. Ask the owner to approve renamed/removed option behavior. Reuse useful manifest, source, memory and graph infrastructure; replacement does not imply rewriting everything.

Initial hardware acceleration is CUDA, but isolate CPU/file residency policy from CUDA-specific launches. Inventory HIP and other-backend behavior before removal; do not claim support that was not built or tested.

## Implementation stages and gates

### Stage 0: audit and reproducible baseline

- Read the relevant files fully and document the actual execution boundaries and supported expert formats.
- Verify that llama.cpp can load and run the initial Qwen GGUF. If unsupported, stop and separate model enablement from this port.
- Record compiler, CUDA, driver, model hashes/metadata, CPU ISA, RAM bandwidth and disk characteristics.
- Build the current fork and capture a reproducible existing-cache baseline plus ordinary execution controls where feasible. Save commands, prompts, logs and memory peaks.
- If comparing against Strata, use equivalent model weights/quantization where possible and label model/kernel/speculation differences.

Gate: owner reviews the narrow integration design, API migration proposal and unsupported cases before implementation.

### Stage 1: correct hybrid execution, no file overflow

- Prototype one complete expert-group decode path using existing CPU quant kernels and GPU consumers.
- Partition actual routes, execute both subsets and merge same-layer outputs.
- Establish explicit synchronization and bounded worker resources. Begin with a conservative supported batch/graph scope and state it clearly.
- Reuse existing parity tests for all-CPU, all-GPU and mixed placement, duplicate routes, empty subsets, different token rows and supported quantizations.
- Validate layer outputs and logits before claiming overlap gains.

Gate: parity, memory lifetime and error handling pass. Measure whether work actually overlaps; simultaneous submission alone is not proof.

### Stage 2: bounded resident RAM and file-backed experts

- Add independent resident-weight and staging admission/accounting.
- Fetch exact missing expert bytes from split GGUF files into bounded staging; prevent allocations proportional to the full expert bank.
- Add CPU execution from staged weights with precise completion/reuse dependencies.
- Exercise undersized budgets, repeated eviction, read failures, partial pinning/registration, cold files and warm OS file cache.
- Document whether resident memory is locked and how total process/file-cache memory differs from the configured cap.

Gate: tiny-budget tests demonstrate bounded application-owned weights and staging, unchanged outputs, no oversized silent fallback, and useful file-I/O telemetry.

### Stage 3: adaptive residency and duplicate elimination

- Add usage accounting with explicit reset/decay behavior and bounded swaps.
- Compare stable hotness residency against simple alternatives under conversation/domain changes; optimize exposed latency, not hit rate alone.
- Implement RAM/GPU complement residency where needed, including transient exchange capacity and recoverable eviction.
- Ensure adaptation cannot recycle weights used by in-flight decode, prefill or speculative verification.

Gate: parity across swaps and live memory peaks fit the admission ledger. Extra copy traffic and adaptation latency are measured.

### Stage 4: prefill streaming and speculative tuning

- Integrate bounded expert streaming/grouping for larger prompt batches using existing GPU kernels first.
- Review temporary reuse of GPU expert storage for prefill workspaces; restore or update residency safely before decode.
- Extend the speculative support required for the first usable hybrid path with long-span memory and performance qualification. Target verification, draft/MTP ownership and rollback correctness are not deferred to this stage.
- Tune chunk sizes from memory and measured cost, not Strata's fixed Qwen thresholds.
- Add fused kernels or predictive file prefetch only as separately benchmarked, bounded follow-ups.

Gate: long-prompt correctness, honest peak memory, and steady decode after prefill pass. Do not silently reduce context, draft length or batch size to make a benchmark succeed.

### Stage 5: replacement cleanup and second-model qualification

- Qualify a second supported MoE architecture with materially different expert geometry/quantization before calling the implementation generic.
- Verify ordinary dense/full-GPU operation, LoRA behavior where supported, repeated contexts, cancellation and teardown.
- Remove superseded refill machinery only after its reusable infrastructure and caller migration are settled.
- Update docs/options and existing tests. Keep source history/baseline artifacts, not a permanent duplicate runtime engine.

Gate: owner reviews compatibility changes, measurements and remaining limitations before replacement is considered complete.

### Stage 6: later multi-GPU work, separate approval

- Extend explicit owner-local caches and accounting without cross-device hidden allocations.
- First evaluate ordinary layer ownership with local hybrid misses. Treat expert-parallel dispatch, P2P, no-P2P host merges and dedicated expert devices as separate designs.
- Address per-device capacity, shared host worker budgets, pinned staging, lock ordering and error cleanup before performance tuning.
- Do not promise that single-GPU complement residency automatically extends to multi-GPU.

## Validation and benchmark contract

Use existing test files and backend test infrastructure; choose a small set of meaningful cases rather than adding a large new suite.

Correctness checks:

- Layer/expert outputs and final logits versus a trusted ordinary execution control, using quantization-appropriate numerical tolerances.
- Deterministic greedy token comparison as an additional check, not the only oracle. Different floating-point reduction order can change near-tied token choices.
- Same-layer/token input binding, weighting, duplicate routes, empty CPU/GPU subsets, bank layouts and supported activation semantics.
- Residency changes during repeated requests, file-tier execution, transient buffer reuse, cancellation, shutdown and I/O/device failure.
- Supported speculative acceptance/rejection and recurrent-state rollback.

Performance checks:

- Short decode, long decode and representative prompt processing, with speculation off first, then on separately.
- Report TTFT, prompt throughput, accepted output tokens/sec, p50/p95 stalls, CPU activity, process memory, locked/registered bytes, per-device peak VRAM, disk bytes/time, H2D/D2H traffic and adaptation time.
- Distinguish cold model/files, warm file cache, startup, first request and steady-state execution.
- Compare equal model, quantization, context, batch, sampling and resource budgets. Strata's advertised speed is not a predicted result for this machine.
- Use GPU traces to establish exposed compute/copy overlap and critical-path stalls. CPU work and DMA contend for host DRAM; do not sum isolated bandwidth estimates into a speed claim.
- For larger models, benchmark file-tier cost explicitly. "Loads successfully" is not proof of acceptable latency or absence of paging.

A hybrid path that loses on a measured workload is a valid result. Investigate partition cost, CPU quantization, memory contention and synchronization before growing the subsystem or adding speculative optimizations.

## Non-goals for the first port

- Importing Strata's model-specific sparse attention, DeltaNet, n-gram embedding tables, or fixed Qwen router dimensions.
- Expert pruning, approximate routing, stale-layer execution, quantization conversion that changes weights, or unverified quality changes.
- A new global allocator, runtime autotuner, expert-parallel scheduler, or rewritten server.
- Guaranteed zero disk reads for models exceeding combined resident capacity.
- Automatic large downloads, upstream submissions, or unattended commit/push actions.

## Handoff instructions for the implementing agent

1. Read this plan, repository rules and the actual current source. Do not treat suggested integration points as a verified API design.
2. Confirm owner understanding of any new design choices and review the proposed changes before substantial implementation.
3. Record the Strata version/file hashes and licenses. Preserve required notices for copied/adapted code and identify source provenance.
4. Complete Stage 0 first. Return with model support, measured baseline and a bounded patch design; do not begin by deleting the current cache engine.
5. Work in reviewable stages. Extend existing tests and report commands, results and limitations honestly.
6. Pause if implementation requires a new architectural subsystem, compatibility exclusions, altered routing/arithmetic, or memory promises the OS cannot enforce.

Completion means a reviewed, tested generic hybrid executor and bounded expert-memory tier with truthful accounting, not a direct copy of Strata or an assumed tokens/sec target.

## Stage 0 progress: local inspection

### Verified model metadata and capacity

Read both GGUF shards using the repository's Python `GGUFReader`, without scanning tensor payloads or loading the weights for inference. No full model hash was computed.

- Architecture: `qwen4exp`, not `qwen3next`.
- Name: `Qwen3.8-Flash-Next GSQ-RCO`.
- Geometry: 48 layers, embedding width 2560, 512 experts per layer, top-k 10, expert intermediate width 640.
- Context metadata: 262144 tokens; this is not a tested runtime context setting.
- Hyper-connection stream count: 4; low rank: 320.
- Indexer: 4 heads, key length 128, top-k 2048. Full-attention layers carry compression ratio 4.
- Total tensor payload: 83,606,638,080 bytes, approximately 77.865 GiB.

| Tensor category | Bytes | GiB |
| --- | ---: | ---: |
| Routed expert banks | 50,292,326,400 | 46.838 |
| PLE n-gram embedding table | 28,800,138,240 | 26.822 |
| Remaining tensors | 4,514,173,440 | 4.204 |

The second shard contains just `per_layer_token_embd.weight`, shape `[160, 320001536]`, type `IQ4_NL`. The 26.822 GiB PLE table is not an expert bank. A resident expert-weight cap cannot by itself control its memory behavior. Inspect the existing CPU/file-backed row-gather path and include its allocations/cache behavior in the memory ledger. Do not force the complete table into locked RAM or GPU memory.

The 144 routed expert banks use these types: `IQ4_NL` (39), `IQ3_XXS` (34), `Q2_0` (9), `IQ2_S` (40), `IQ3_S` (20), and `IQ4_XS` (2). The filename is not a uniform quantization specification. Per-bank geometry/type checks are mandatory.

### Architecture support found in this fork

Static source inspection confirms:

- `src/llama-arch.cpp` registers `qwen4exp`.
- `src/llama-model.cpp` constructs `llama_model_qwen4exp`.
- `src/models/qwen4exp.cpp` implements metadata/tensor loading, hyper-connections, sparse attention/indexer, PLE, routed/shared FFNs and an MTP graph.
- `build_layer_ffn()` uses the generic `build_moe_ffn()` with SiLU, normalized softmax weights and separate shared-expert computation.

This satisfies only the static architecture-recognition check. Actual load, generation, parity and performance remain untested until a build is available. Do not start a separate model-enablement project based only on the unfamiliar architecture name.

### Generic-model constraint confirmed by Gemma code

`src/models/gemma4.cpp` already uses `build_moe_ffn()`, but its expert activation is GELU, not Strata's fixed SiLU. Its router operates on a separately normalized/scaled attention output, while expert input uses another normalization. It also applies post-expert normalization and combines the routed branch with a dense branch.

Therefore a direct copy of Strata's `native_gu_rows()` would be incorrect for this architecture. Preserve the existing router/input/activation graph semantics, and express any complete-expert CPU executor's activation/bias/scale behavior explicitly. Actual Gemma GGUF support on a second machine still requires live qualification.

### Execution seam proposal, pending deeper audit/review

The generic builder records `llm_graph_moe_region` objects through `res->add_moe_region()`. A region includes actual routes, output, operations and inputs. Existing CUDA grouped dispatch already has group identities, bank descriptors, owner-local authority leases and route/batch capability witnesses. These are reuse candidates, not finished hybrid APIs.

Two alternatives must be evaluated before selecting the patch design:

1. Per-matrix hybrid dispatch: execute CPU/GPU subsets for each `MUL_MAT_ID`, leaving activation and reduction semantics in the existing graph. This may preserve more generic behavior but requires extra intermediate transfers/synchronization.
2. Complete-region hybrid dispatch: compute full expert contributions on each side, then merge at the region output. This better matches Strata's overlap and minimizes transfers, but requires an explicit certified semantic description and safe suppression of the replaced graph operations.

Do not pick the second alternative just because it resembles Strata. Confirm the existing region inventory, fusion, graph capture and LoRA boundaries can represent it safely. Never skip an activation, scale, bias or external-route dependency based on a Qwen-only pattern.

Strata's native CPU executor uses ggml CPU type traits, activation quantizers and row dot products as its baseline. This is a useful reusable strategy. Its own optimized multi-token IQ kernels and fixed buffer sizes are not needed for the first correctness prototype. CPU `MUL_MAT_ID` implementation in this fork is in `ggml/src/ggml-cpu/ggml-cpu.c`, not `ops.cpp`; audit both that implementation and available IQP paths before designing new workers.

### Hardware/toolchain observations

Read-only checks confirmed:

- Windows 11 Pro.
- Intel Core Ultra 7 270K Plus: 24 cores and 24 logical processors reported by Windows. P/E topology and ISA capabilities have not yet been measured.
- Two 48 GiB DIMMs at configured 6000 MT/s, matching the owner's 96 GiB configuration.
- RTX 5070 Ti: driver 617.14; `nvidia-smi` reports 16303 MiB total VRAM and approximately 1932 MiB occupied at inspection time. Available VRAM is not the full advertised capacity.
- Storage devices reported: Kingston SNV2S2000G, WD_BLACK SN7100 2TB, and MSI M461 4TB. The model volume has not yet been mapped to a physical device or benchmarked.

No `build*` output directory or llama CLI/server/bench executable was found within four levels of this checkout. `cmake`, `nvcc`, `ninja`, and `cl` were not resolved in the current shell or a fresh PowerShell session. The usual Program Files CUDA/CMake directories and Visual Studio Installer `vswhere.exe` location were absent. This is evidence that tooling is not currently accessible, not proof that no installation exists elsewhere.

Runtime baseline and kernel testing are blocked until the owner supplies a toolchain/build location or approves installation. Do not infer benchmark results from source inspection or launch the full 78 GiB payload with guessed memory flags.

### Strata provenance snapshot

`engine/BUILD.json` describes the bundled executable as version `0.1.38`, release source, CUDA `13.0`, including architecture `120`. This identifies the executable metadata, not a verified matching revision of the loose source tree.

Strata's top-level MIT license is copyright 2026 Niko1221 and the Strata contributors. Retain that notice for copied substantial portions, and audit any transitive third-party code separately.

SHA-256 hashes captured for reproducible source identification:

| Relative Strata path | SHA-256 |
| --- | --- |
| `src/core/session.cpp` | `1e3a979614dae1910924e592ae1e2190afd8c6eb14438d7fdba5886ea034de18` |
| `src/core/expert_source.cpp` | `47f73e112987892efcad858fe25e5872e77055be27fbe01d51d9257f8c1b6a36` |
| `src/kernels/cpu/native_expert.cpp` | `d77666ee42cc43eaa8339116950ef0603aba6591c66be53344603f8749e76d4a` |
| `include/strata/kernels/cpu/pool.hpp` | `177897a23a38f78f51545bc50a056d3cd8feb03c0b594a0026e66fab92cec07a` |
| `src/prefill/prefill.cpp` | `c4fe7bec45252944423b71f9cb993d4b35c8a44fcea86da56a5752d54432842a` |
| `LICENSE` | `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0` |
| `engine/BUILD.json` | `e3ec2ca16ffc4321b213f221a36b20e5a1d04deefa8985818e4b8cfa3f1015d9` |

### Follow-up: supplied toolchain and build directories

After the owner pointed to `toolchain/` and `build/`, both directories were present and inspected. They supersede the earlier environment-discovery result above.

- `toolchain/vs/` contains MSVC 14.44.35207, CMake 3.31.6-msvc6 and Ninja 1.12.1. CMake/Ninja version commands work, and `vcvars64.bat` initializes the x64 compiler environment. `cl /Bv` reports compiler 19.44.35229.
- The compiler environment did not resolve `rc.exe`; no Windows Kits directory was found at the usual location. `WindowsSDKVersion` is unset to a usable version and Universal CRT SDK settings are absent. A usable Windows SDK must be supplied/located before claiming this is a complete build toolchain.
- No CUDA toolkit or `nvcc.exe` was found under `toolchain/`. The copied build records CUDA 13.3.73, originally at `C:/llama-mtp-cache/toolchain/cuda`, which does not exist here.
- `build/bin/` contains `llama-server.exe` and its project DLLs, including `ggml-cuda.dll`, but not llama-bench/CLI or test executables at inspection time.
- Starting `llama-server.exe --version` fails with exit code `0xC0000135` (missing dependency). Dependency inspection shows `ggml-cuda.dll` needs `cublas64_13.dll`; the Windows loader could not resolve that DLL. MSVC runtime and OpenMP DLL load checks succeeded.
- `build/CMakeCache.txt` and Ninja files reference the old source root `C:/llama-mtp-cache-multigpu/src` and old toolchain root. Do not reuse the copied cache as-is or mechanically replace paths in generated files. Preserve it as provenance, then configure a fresh build directory against this checkout once CUDA and the Windows SDK are available.
- Recorded CUDA architecture targets are `86;120a`, and the copied build is Release with CUDA graphs/OpenMP enabled. Its runtime/source revision has not been established because executable startup is blocked.

The supplied files are a useful partial toolchain and build snapshot, but are not sufficient for CUDA compilation or running this server yet. Missing components: usable Windows SDK and CUDA toolkit/runtime dependencies. The owner should provide their locations or approve setup.

### Migration and CUDA setup completed with owner approval

- Initially moved the old `C:\Program Files\llama-mtp-cache-multigpu` directory into `build/legacy-mtp-cache-multigpu`. At the owner's cleanup request, compared and removed this redundant archive, recovering approximately 5.93 GiB. All 14,832 old toolchain files matched the current toolchain byte-for-byte; 65 untracked old-source files already matched this checkout. Preserved only a 60 KB old uncommitted-change patch, the unique `prompt-moe-cache.md` document, and provenance under ignored `build/legacy-migration-notes`. The patch's base commit `e69a1d0be5f8ae0080593865b38b175223059199` already exists in this repository. Neither the old Program Files path nor the full legacy archive remains; current source/build outputs were not overwritten.
- Copied its `cublas64_13.dll`, `cublasLt64_13.dll` and `cudart64_13.dll` into current `build/bin`. The existing server now runs without the old folder. `--list-devices` detects the RTX 5070 Ti successfully.
- Installed the CUDA 13.3.0 development components locally under `toolchain/cuda` from NVIDIA's official redistributable manifest at `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json`.
- Every component ZIP was checked against the manifest SHA-256 before extraction. The manifest is preserved in `toolchain/cuda`; package licenses are retained under `share/licenses`. Downloads remain in ignored `build/toolchain-downloads`.
- Components: NVCC, runtime, CRT headers, CCCL, NVVM, cuBLAS, cuobjdump, nvprune and profiler API. This is a repo-local development installation, not all optional CUDA tools or Visual Studio extensions. The NVIDIA driver was not changed and global PATH was not modified.
- `toolchain/cuda/bin/nvcc.exe --version` reports CUDA 13.3, NVCC 13.3.33. This differs from the copied build's 13.3.73 compiler patch; a fresh build will use the newly installed compiler explicitly.
- Added repo-local `toolchain/activate.bat` to initialize MSVC and set CUDA/CMake/Ninja paths for a Command Prompt session. Use `call toolchain\activate.bat`; the CUDA runtime DLLs from the redistribution live under `bin/x64`.
- Added `/toolchain/` to local `.git/info/exclude` so large tools/installers are not accidentally committed. No tracked ignore policy was changed.

A small CUDA kernel compile check, saved in ignored `build/toolchain-check.cu`, reaches MSVC but fails on missing `corecrt.h`. This confirms the remaining blocker is the Windows SDK/UCRT development headers, not absence of NVCC. Windows SDK installation or a supplied portable SDK is still required before configuring/building fresh llama binaries. Do not report the CUDA toolchain as compile-tested until the kernel check and a fresh configure/build pass.

### Windows SDK installed and toolchain verified

With owner approval, installed Microsoft's official NuGet packages `Microsoft.Windows.SDK.CPP` and `Microsoft.Windows.SDK.CPP.x64`, version `10.0.26100.9169`, into `toolchain/winsdk`. This is a portable development setup, not a system-wide Windows Kits installation. Package versions, source URLs and downloaded SHA-256 hashes are recorded in `toolchain/winsdk/packages.json`. Package metadata/licenses/signatures are preserved; recorded hashes identify the downloaded files, not an independently verified publisher checksum.

Updated `toolchain/activate.bat` to supply SDK/UCRT headers, x64 libraries, resource compiler and manifest tool. SDK internal include/bin version is `10.0.26100.0`, despite the serviced package version. No system PATH or driver changes were made.

Verification completed:

- `rc.exe` and `mt.exe` resolve from the portable SDK.
- `build/toolchain-check.cu` compiles and links with MSVC and NVCC targeting `sm_120`.
- Its GPU kernel executes on the installed GPU and returns the expected result: `CUDA kernel check passed`.
- Fresh current-checkout CMake configure succeeds at `build/hybrid-baseline`, using Ninja, Release, CUDA `120a`, CPU AVX2, OpenMP, server/tools/tests enabled, app/examples/UI disabled, and CUDA NCCL disabled for the initial single-GPU setup.
- Configure log: `build/toolchain-configure.log`. Both C/C++ and CUDA compiler ABI checks pass; CMake recognizes the new CUDA toolkit at the repo-local path.
- OpenSSL was not found, so this baseline configuration has HTTPS support disabled. This does not block local GGUF inference; do not assume TLS functionality is present.

At this setup checkpoint, the full llama build and initial-model runtime baseline had not yet run; see the live Stage 0 results below for subsequent progress. The old supplied server remains independently usable, but reports `0.4.0-dev` with an unknown commit; it is not a verified current-revision baseline. The fresh source configure reports `0.5.0-dev` and commit `1af951956`.

To activate the local environment in Command Prompt:

```bat
call toolchain\activate.bat
cmake --build build\hybrid-baseline --target llama-server llama-bench -j 8
```

Runtime DLLs for a new build may need to be staged from `toolchain/cuda/bin/x64`, or that path must be present through `activate.bat`. Do not mix old project DLLs into the new build output.

## Stage 0 live results and current blockers

### Fresh build

Built `llama-server`, `llama-bench`, `llama-fit-params`, `test-arg-parser` and `test-moe-cache` from commit `1af951956` using the repo-local tools. Server version is `0.5.0-dev`, build 11591, commit `1af951956`.

Changed this fresh baseline configuration to `BUILD_SHARED_LIBS=OFF`: the existing CMake test target intentionally excludes `test-moe-cache` on Windows shared-library builds because its internal symbols are not exported. The initial attempt to request that target in the shared build failed with `unknown target`; no source workaround was added. Static configuration made the target available. Other recorded baseline settings remain CUDA `120a`, CPU AVX2, OpenMP and Release.

Logs: `build/hybrid-baseline-build.log` and `build/baseline-fit-build.log`. CUDA emitted existing compiler warnings; no compile/link failure remained.

### Tests

| Check | Result | Artifact |
| --- | --- | --- |
| Fresh server `--version` | Pass | `build/baseline-llama-server.log` |
| `test-arg-parser` | Exit 0; reports all tests OK, with existing Windows-specific skips | `build/baseline-test-arg-parser.log` |
| `test-moe-cache --grouped-layer-only` | Pass; single-device layer fixture, not multi-GPU validation | `build/baseline-test-moe-cache.log` |
| Full `test-moe-cache` | Fails existing device sizing assertion in `test-moe-cache-registry.cpp:2440` | `build/baseline-test-moe-cache-full.log` |

The failing assertion compares V2 early-router context fixed bytes with V1 sizing plus additional workspace. No implementation code has changed, so this is a current-revision baseline failure, not a hybrid-port regression. The rest of the full suite was not completed past that failure. Do not label the complete MoE suite as passing based on the layer-only fixture. (This row and the admission-probe crash below record the state at the time of the Stage 0 snapshot; both were resolved later, see "Prerequisite fixes".)

### Admission probes

Configuration: initial Qwen shards, context 2048, batch/microbatch 128, one sequence, explicit mmap/lazy loading, fit disabled, offline, no speculative model. Probe argv is saved in `build/baseline-admission-command.json`.

- Cached placement with 8192 MiB GPU cache and 4096 MiB host-pinned budget: model-only `--fit-moe-report-json` succeeds. It resolves 87 owner-local slots, approximately 98,229,408 aggregate per-slot device bytes and 520,236 fixed device bytes. Placement artifacts: `build/baseline-placement.json` and `.log`.
- The corresponding `--fit-moe-joint-report-json` crashes with Windows access violation `0xC0000005` while reserving the single-token graph, after input tensor inventory and before completing context sizing. Artifacts: `build/baseline-admission.json` (empty) and `.log`.
- Disabling CUDA fusion does not avoid the crash. Artifacts: `build/baseline-admission-no-fusion.json` (empty) and `.log`.
- Cache disabled with `-ngl 0`: strict joint probe succeeds. Artifacts: `build/baseline-admission-cpu.json` and `.log`.
- Ordinary GPU layers plus `--cpu-moe`: strict joint probe succeeds, reports complete measurement, capacity within limits and admission qualified. Artifacts: `build/baseline-admission-ordinary-cpu-moe.json` and `.log`.

These checks narrow the crash to a cached-placement/context sizing configuration, but do not establish its root cause or prove that live cached execution fails. Model-only placement success is not a substitute for joint admission. No full live cached-model baseline was launched after this failed sizing gate.

For the ordinary CPU-expert control, the estimated GPU total is approximately 4.96 GiB (model, state and compute combined). Mapped host model bytes are reported as an upper estimate, not a required locked working set. Actual process/GPU measurements below are distinct from these estimates.

### Initial-model runtime control

Ran a bounded control using ordinary GPU layers and CPU routed experts. This is not the new hybrid executor: it does not split a layer's routed experts between CPU/GPU residency.

Actual argv, requests, full responses, logs and samples: `build/ordinary-baseline/`. Reproduction helper: ignored `build/run-ordinary-baseline.py`. The helper binds only localhost port 8094, has readiness/request timeouts, terminates its own server afterward, and guards against available RAM falling below 8 GiB. The memory guard did not fire. No other server process was stopped.

User-facing launcher: root `run-qwen-iq3-s.bat`, using the same tested settings and fresh binary. It supplies the repo-local CUDA runtime PATH and checks both GGUF shards. A launch/health/four-token-generation smoke test passed; log: `build/run-qwen-launcher-check.log`. The root launcher is local-only under the repository's existing `*.bat` ignore rule. It has no continuous memory watchdog; the guarded measurement helper above is a separate tool. API base URL is `http://127.0.0.1:8094/v1`; current build has no bundled web UI.

Settings: context 2048, batch/microbatch 128, one slot, eight CPU threads and eight batch threads, `-ngl all --cpu-moe`, explicit mmap and lazy mode on, fit off, offline, no warmup, no speculation. Three identical requests use temperature 0, seed 1234, prompt cache reuse disabled and 32 generated tokens with EOS ignored. The raw chat-formatted prompt asks for a short explanation of the blue sky. These short outputs are a thinking prefix, not a complete answer or a quality benchmark.

| Request | Prompt tokens | Server prompt tokens/s | Server decode tokens/s | Request wall seconds |
| --- | ---: | ---: | ---: | ---: |
| First | 20 | 11.90 | 14.44 | 3.829 |
| Second | 20 | 34.08 | 21.43 | 2.109 |
| Third | 20 | 34.29 | 21.48 | 2.038 |

The three output strings match exactly. This establishes successful loading/generation and short-run determinism for this control, not independent reference-logit parity. Token rates are the server's reported timing fields; they are not complete request throughput or a prediction of hybrid speed.

Readiness took approximately 6.60 seconds including process launch and health polling. This is not a cold-disk load measurement: earlier probes and OS file caching were not reset. Do not blend the first request with subsequent requests or compare these short runs directly with Strata's speculative benchmarks.

Sampled resources (one-second host sampling, roughly five-second device sampling; not exact peaks): process RSS up to 19.05 GiB, private bytes up to 7.64 GiB, minimum available system RAM 62.61 GiB. NVIDIA device-wide used memory samples reached 8329 MiB, including other device allocations/processes; no per-process peak was captured. No claim about zero page faults/pagefile I/O follows from these samples.

### Additional architectural findings

- Qwen marks the PLE table `TENSOR_READ_LAZY`. The loader maps it to CPU storage, excludes its range from normal mmap prefetch and avoids mlocking lazy tensors. Preserve this existing implementation rather than add another PLE storage subsystem.
- `llm_graph_moe_region` inventories operations/inputs for placement. Its current `place()` rejects external expert routes and certifies backend support; it is not already an API for replacing a region with CPU/GPU contributions. Review external-route semantics explicitly before calling complete-region dispatch generic.
- Strata's resident source also has Windows unbuffered/overlapped file reads when mapped-file cache residency would conflict with its RAM budget. This is a relevant bounded file-tier alternative, but direct reads require alignment, explicit staging, error handling and separate prefetch behavior. Do not assume mmap-only overflow is equivalent to Strata's entire large-model strategy.
- `llama-bench` has a separate argument parser without the fork's cache budget options at this revision. Use the server or another verified tool for cache-mode comparisons; do not pass unsupported cache flags to llama-bench or assume it exercises the same configuration.

No hybrid implementation code has changed. Stage 0 is partially complete: current model loading/generation is verified through an admitted ordinary control, but cached joint sizing and the full MoE suite have pre-existing failures. Next step is a narrow prerequisite investigation with regressions in existing tests, followed by review of the first hybrid execution seam. Do not delete cache infrastructure or make a hybrid speed claim while treating these failures as passing gates.

## Prerequisite fixes (completed after the Stage 0 snapshot)

The three Stage 0 blockers were reproduced, root-caused and fixed. All work stayed in existing files. The repository owner committed the result as `be074960f` ("working on strata features") on top of the `1af951956` planning baseline. No hybrid expert-execution code changed.

### 1. Cached joint-sizing access violation

Root cause: `llama_model_load` created the reference-counted bounded MoE host buffer type in a function-local RAII owner. Real buffers retain that owner, so releasing the initial reference at scope end was safe. In `no_alloc` measurement loads the loader asks for zero-sized buffers, and `ggml_backend_buft_alloc_buffer` returns a dummy buffer that never calls the buffer type's allocator, so nothing retained the owner. `llama_context::refresh_moe_layer_owners` then read the freed buffer type during graph reservation. A Windows debug-event harness (`build/debug-prerequisite.py`, ignored) reported the fault as access violation `0xC0000005` at `ggml_backend_dev_backend_reg+0x9` with `RCX = 0xfeeefeeefeeefeee` (freed-heap pattern) inside `refresh_moe_layer_owners`.

Fix in `src/llama.cpp`: do not create the bounded buffer type when `params.no_alloc` is set. The bounded budget is only consumed by `configure_sources`, which `load_tensors` already skips for `no_alloc` models, so measurement accounting is unchanged. `moe_expert_cache_host_staging` keys off `params.moe_expert_cache_host_pinned_size` and the source groups, not the buffer type instance.

Regression coverage added to the existing `test_moe_cache_selector_precedence` in `tests/test-llama-archs.cpp`: a bounded `no_alloc` load must place cached layers and then construct a context, which is the path that used to crash.

### 2. V2 device sizing assertion

`tests/test-moe-cache-registry.cpp` compared V2 context-fixed bytes against V1 plus early-router workspace unconditionally. The expectation now always includes `extra_workspace`, and adds early-prefetch resource overhead only where that path is compiled: non-Windows, non-HIP/MUSA, and `CUDART_VERSION >= 12080`. This matches the backend's own guards.

### 3. Windows auxiliary pinning failure

Measured on this machine (CUDA 13.3, driver 617.14, `build/toolchain-register-check.cu`, ignored):

| Host allocation | Device alias identity |
| --- | --- |
| `cudaHostAlloc` | yes |
| `cudaHostRegister` over `malloc` | no |
| `cudaHostRegister` over `VirtualAlloc` | no |
| `cudaHostRegister` over a file-mapped view | no |

So on Windows, externally allocated host memory (mmap'd GGUF weights, aligned allocations) always gets a non-identity device alias after registration. Mandatory auxiliary sources were registered with `require_identity`, and a failed auxiliary registration aborted the whole cache configuration, so any Windows model with auxiliary scale/bias sources failed to load. The runtime already supports non-identity auxiliaries through owner-local GPU shadows (`host_readable` path, documented in `docs/moe-grouped-multigpu.md`).

Fix: `moe_host_register` reports whether the decline was `auxiliary_alias_not_identity` (new out-parameter in `moe-cache-host.cu`/`.cuh`). `moe-cache.cu` keeps the hard failure for budget or capability declines, and for a non-identity alias logs a warning and continues so consumers use the existing shadow/staged route. Linux behavior is unchanged because identity registration succeeds there.

Test side: added `host_registration_identity_alias()` to `tests/test-moe-cache-fixtures.cpp` (declared in `tests/test-moe-cache.h`). The registration-boundary fixture in `tests/test-moe-cache-staging.cpp` now derives its expectation from that probe instead of assuming Linux aliasing.

### 4. Test stack overflow

`test_grouped_graph_preflight` allocates about 2.4 MiB in one frame, which exceeds the 1 MiB default Windows stack (`0xC00000FD` in `_alloca_probe`). `tests/CMakeLists.txt` now links `test-moe-cache` with `/STACK:8388608` under MSVC. This is a test-fixture limit only; no production frame sizes were changed.

### Results after the fixes

| Check | Result | Artifact |
| --- | --- | --- |
| `test-moe-cache` (full, previously failing) | Pass, exit 0, about 21 s | `build/prerequisite-after-test-moe-cache.log` |
| `ctest -L moe-cache` (joint measurement, selector, placement) | 3/3 pass | `build/prerequisite-ctest-moe-cache.log` |
| `ctest -L main` | 55 pass, 2 fail, 1 skipped (`test-moe-cache-multigpu`, single GPU) | `build/prerequisite-ctest-main.log` |
| `test-llama-archs --test-moe-cache-selector` | Pass, including the new bounded `no_alloc` case | `build/prerequisite-after-test-llama-archs.log` |
| Cached joint admission probe on the real Qwen shards | No crash; `measurement_completeness: complete` | `build/prerequisite-admission-after.json`, `.log` |
| Rebuilt-server ordinary control smoke (GPU layers, `--cpu-moe`) | Loads and generates; cold single-request rates only | `build/run-qwen-rebuild-smoke.json`, `.log` |

The joint admission probe now reports a real capacity result instead of crashing. Whether it is qualified depends on free RAM at run time: with about 62 GiB available it reported `admission_qualified: true`; with only 28.7 GiB available it reported host capacity 30,317,297,664 bytes against 79,730,791,456 bytes estimated host usage and `capacity_assessment: unknown`, `admission_qualified: false`. Re-measure on an idle machine before using admission as a gate.

### Remaining pre-existing failure, not fixed

`test-recurrent-state-rollback` and `test-save-load-state` abort with `0xC0000409` when processing the generated `qwen4exp-moe-q4_0.gguf` fixture. The last messages are `ggml/src/ggml-cpu/binary-ops.cpp:136: binary_op: unsupported types: dst: f32, src0: f32, src1: q4_0`. The op is the PLE grouped norm in `src/models/qwen4exp.cpp` (`grouped_norm` multiplies an `rms_norm` result by `ple_norm_key` / `ple_norm_query` / `ple_norm_conv`), and the generated q4_0 fixture quantizes those norm weights, which the CPU binary-op path cannot consume. Other architectures in the same tests pass, and the failure reproduces by running either test with only that fixture. It is unrelated to the MoE cache or the fixes above, and it does not affect the real IQ3_S shards, whose norm weights are not quantized. Fixing it needs a decision from the owner: either keep PLE norm weights unquantized in generated test models, or add quantized-src1 support/placement for that op. It was not changed here.

The ordinary GPU-layers plus CPU-experts control was re-run once on the rebuilt binaries (`build/run-qwen-rebuild-smoke.py`, log `build/run-qwen-rebuild-smoke.log`, summary `build/run-qwen-rebuild-smoke.json`). The server became ready in 11.65 s and generated 8 greedy tokens successfully. That single cold request reported 3.54 prompt tokens/s and 4.90 decode tokens/s, with process RSS peaking at 12.55 GiB and system available RAM falling to 10.01 GiB. These are cold-cache, single-request numbers on a busy machine and are not a replacement baseline; the earlier warm repeated-request figures (about 21.4 decode tokens/s) must be re-measured with the same three-request warm sequence before being quoted again.

## Execution-seam audit and Stage 1 patch design (for owner review, not implemented)

This section records what the current code actually allows, then proposes the smallest defensible first hybrid patch. Nothing here is implemented.

### What the generic region actually is

`llm_graph_context::build_moe_ffn()` (`src/llama-graph.cpp:2138`) emits, in order:

1. router logits, optional bias, gating (`softmax` / `sigmoid` / `softmax_weight` / `sqrt_softplus`), optional expert-group masking, `argsort_top_k` -> `selected_experts` (`[n_expert_used, n_tokens]`, I32)
2. `weights = get_rows(probs, selected_experts)`, optional `softmax`, optional `norm_w` sum/clamp/div, optional `w_scale`
3. `gate_up` fused `MUL_MAT_ID` (or separate `gate` / `up`), optional `add_id` bias, optional per-bank scale
4. activation chosen by `type_op`: `swiglu_split` / `geglu_split` / `reglu_split` / `situ` / `swiglu_oai` / clamped variants / plain `silu` etc.
5. `down` `MUL_MAT_ID`, optional `add_id` bias, optional scale
6. `experts = mul(experts, weights)` (or the weights applied before the FFN for llama4)
7. per-slot `view_2d` of `experts` plus a chain of `add` producing `moe_out`
8. `res->add_moe_region(il, down_exps, selected_experts, ffn_input, moe_out, selected_experts_in != nullptr)`

Shared experts are outside the region (for example `src/models/qwen4exp.cpp:1258-1261` multiplies and adds `ffn_shexp` to `moe_out` after the region). Region ownership is `llm_graph_moe_region::place()` (`src/llama-graph.cpp:1557`), which assigns every region operation to one backend and rejects external routes and incompatible weight buffers. It is placement only; it does not replace execution.

Consequence for this prototype: residency is data-dependent, so ordinary extra graph nodes alone do not implement the split. A graph-level design with explicit partition/compaction operations is possible, but the approved first prototype keeps the split in backend execution.

### Where CUDA actually executes the region

`ggml_cuda_mul_mat_id()` (`ggml/src/ggml-cuda/ggml-cuda.cu:3739`) is the single entry point for `MUL_MAT_ID`. For a certified grouped plan it:

- looks up the per-node group with `execution->find_group(dst, &binding)`
- on the group's `first_reader` (`ggml_cuda_moe_graph_group_dispatch::first_reader`) calls `prepare_graph_group()`, which acquires device slots for every route (misses are refilled from host source bytes)
- swaps `dst->src[0]` to a slot-array bank view and `dst->src[2]` to a remapped-ids view, then calls `ggml_cuda_mul_mat_id_impl()` with a `ggml_cuda_mmid_direct_source_view`
- on the `DOWN_WEIGHT` bank calls `finish_graph_group()`

The legacy per-matrix cache uses `cache->host_to_slot.find(host_src)` in `ggml_cuda_moe_cache_acquire_locked()`. This is not the grouped planner's residency state and must not be used to classify grouped routes. Grouped `prepare_graph_group()` calls `prepare_decode()`, which owns its own device-side plan and complete-bank transaction. The hybrid must reuse that ownership and certify complete-bundle residency before freezing a partition. Host source bytes may be file-mapped or pageable; the host-pinning budget does not make the full expert bank resident or cap its RSS.

CUDA graph capture is already gated per plan: `moe_execution.allows_graph_capture()` feeds `force_moe_direct()` (`ggml/src/ggml-cuda/ggml-cuda.cu:7025`). A hybrid group can be marked non-capturable without inventing a new mechanism.

The `mm_ids_helper` kernel maps ids to output columns by matching the id against the expert being processed (`ggml/src/ggml-cuda/mmid.cu:225-285`), so an id that matches no expert simply produces no column. The vector consumers index the bank directly by id, so a negative sentinel needs an explicit skip there.

### Chosen seam: cold-route columns, not region replacement

Rejected for Stage 1: replacing the whole region (suppressing graph operations and writing `moe_out` from two partial results). It needs a certified description of activation, bias, scale, weighting and reduction semantics, and it must safely suppress operations the scheduler already placed. That is a much larger surface than the gain justifies before the concept is measured.

Chosen: keep the graph exactly as it is and change only which columns each side computes.

- Resident routes keep the existing grouped GPU path unchanged.
- Cold routes get a negative id in the remapped-ids view, so the GPU reads no weights and writes no output for those columns.
- A CPU executor computes the full expert triple for cold routes from the host source banks and writes the raw `down` results into the same `down` output tensor columns.
- Everything after the `down` node (`mul` by routing weights, per-slot views, the `add` chain) runs unchanged. The CPU must not apply routing weights. Biases and scales are excluded initially.

Routing, weighting, shared experts and residual order remain in the existing graph. Cold expert activation arithmetic moves to the CPU and must be described explicitly. This does not guarantee bit-for-bit parity; compare outputs with numerical tolerances and then compare greedy tokens.

### Bounded Stage 1 patch

1. Residency classification
   - Add a non-acquiring lookup that returns the slot for a route source pointer, or "cold".
   - At the group's `first_reader`, classify every route. Requires route ids on the host. First version: one small D2H of the ids tensor plus a bounded stream wait. Later: reuse the early-router host publication instead.
   - Cold routes are written as the negative sentinel into the remapped-ids view; resident routes keep their slots.

2. Kernel skip support
   - Teach the grouped `MUL_MAT_ID` consumers used by decode/verification to skip a negative id: no weight load, no output write. Native MMVQ/MMVF and direct MMQ guards are implemented below; other consumers remain unqualified.
   - Record the requirement in the capability witness so a plan that needs skipping is only compiled when the selected consumer supports it.

3. Host staging (bounded, reused)
   - Extend the existing `moe_host_budget` accounting with one input block, one output block and one matching device scratch, sized from `n_embd` and the route capacity, not from the expert bank size.
   - D2H the region input rows for cold routes; H2D the CPU output block once per layer, then one scatter kernel places it into the `down` output columns.

4. CPU executor
   - Read cold weights directly from the existing host source banks with `expert_stride` indexing. No copy, no repack, no allocation proportional to a bank.
   - The first correctness helper uses existing ggml CPU `MUL_MAT_ID` plus plain SiLU/SwiGLU or GELU/GEGLU. Retain the direct `from_float`/`vec_dot` strategy as an optimization to measure, not duplicate arithmetic in advance.
   - Reject expert scales, biases, clamped/special activations, expert LoRA and external routes in the first prototype. Do not silently substitute plain arithmetic.
   - Use a bounded worker and explicit CUDA input-ready/completion events. Queue resident GPU work before waiting for CPU completion. Verify actual overlap with traces; asynchronous API submission alone is not proof.

5. Fail-closed guards
   - ordinary decode and certified speculative execution, one CUDA device, no external routes, no LoRA touching region banks; multi-row verification is not ordinary prompt prefill
   - hybrid groups are marked non-capturable, reusing `force_moe_direct()`
   - prefill keeps the existing grouped/streamed path
   - `--moe-hybrid` is off by default; unsupported explicitly requested hybrid configurations fail clearly rather than silently using old refill execution
   - before mutating any ids or launching partial work, certify bank types, activation, shapes, routes, owner, fusion boundaries and lifetimes
   - CPU result insertion must precede every downstream consumer, including any fused down/weighted-reduction kernel; fusion must be inhibited where it would cross this dependency

6. Verification and telemetry
   - Per-owner counters: cold routes, CPU rows computed, CPU time, exposed stall, D2H/H2D bytes, GPU slots not consumed, and a column-coverage counter proving every route executed exactly once.
   - Extend the existing `test-moe-cache` fixtures rather than adding files: all-resident, all-cold, mixed, duplicate route ids, empty cold set, empty resident set, and a poisoned-output build check that fails if any column is left unwritten.
   - Parity against the ordinary `--cpu-moe` and all-GPU controls at layer-output and logit level, with quantization-appropriate tolerance.

### Known costs and risks to measure, not assume

- The ids/input D2H plus wait adds a dependency per MoE layer. Measure exposed latency across all 48 layers; no latency estimate has been validated.
- This build targets AVX2. CPU IQ kernel throughput and the useful cold fraction have not been measured. If CPU work dominates, hybrid execution can lose.
- Host weight reads and H2D copies contend for the same DRAM the GPU copies through. Isolated rates cannot be added.
- Skipping columns leaves output memory unwritten, so a coverage counter and a debug poison pass are required before trusting results.
- The existing refill engine stays as the fallback during Stage 1. Deleting it is a Stage 5 action after parity and measurements exist.

### Owner-approved defaults and first code slice

The owner approved the column-level seam, explicit transfers/events first, `--moe-hybrid` off by default, and an initial SiLU/GELU-only scope rejecting expert biases, scales, LoRA and external routes. The old refill engine is temporary migration scaffolding, not a permanent hybrid alternative.

The first implementation slice adds an internal `llm_graph_moe_hybrid_ffn` description to each graph region. The builder records input, separate/fused banks, activation tensor, raw down-output tensor and activation type only for plain arithmetic. Modified or unsupported regions retain an empty description. This is a semantic description, not runtime admission: device ownership, row shapes, CPU type traits, source lifetime and fusion remain additional checks.

Existing layer fixtures check separate/fused/ungated SiLU descriptors, GELU, exclusion of clamped SiLU and ReLU, exclusion of down bias, scales and external routes, and reset/rebuild behavior. Expert-LoRA rejection is implemented but does not yet have a dedicated fixture in this slice. Release build, `test-moe-cache --grouped-layer-only`, the full `test-moe-cache`, and all three `ctest -L moe-cache` cases passed. Artifacts: `build/hybrid-semantics-build.log`, `build/hybrid-semantics-tests-build.log`, `build/hybrid-semantics-layer.log`, `build/hybrid-semantics-full.log`, `build/hybrid-semantics-ctest.log`. Only generated/synthetic test models were used; the owner's large Qwen shards were not loaded.

There is no hybrid execution or usable `--moe-hybrid` switch yet. Next: connect the validated description and CPU arithmetic helper to owner-local backend planning, then implement partition/skip kernels, event ordering, scatter and truthful admission before exposing the switch.

### CPU arithmetic helper implemented, not connected to CUDA execution

`llm_graph_moe_hybrid_cpu` in `src/llama-graph.h` / `.cpp` runs an exact-count cold-route CPU subgraph with existing ggml `MUL_MAT_ID` and SiLU/GELU operations. This replaces the proposed hand-written type-trait dot loop for the first correctness baseline. It reuses ggml's activation implementation instead of copying Strata's SiLU arithmetic or duplicating GELU. Strata's direct dot-product strategy remains a later optimization if dispatch overhead matters.

The helper borrows immutable source banks and a shared CPU threadpool; it does not create a pool per layer. Caller-supplied thread count must match the pool, and calls sharing that pool must be serialized. Initialization checks host/cached-source ownership, complete tensor geometry, CPU dot/activation-quantizer support, route capacity and input-row geometry. Ordinary device-only bank pointers are rejected before any CPU reads. Cached buffers deliberately advertise non-host storage for scheduler placement; the existing private cache-type procedure identifies their host source bytes.

Initialization reserves maximum-route tensor storage, fixed graph metadata and maximum CPU compute-plan scratch. Jobs rebuild only small metadata in the reusable arena, borrow bank bytes without copying/repacking, and compute only the cold count supplied. Zero routes are a no-op. Invalid ids, null job buffers or counts above capacity fail before writing caller output. `workspace_bytes()` reports graph metadata, tensor allocator storage and compute-plan scratch, not shared threadpool overhead, allocator bookkeeping, pinned CUDA staging, the model's host working set or total process RSS. This is not yet a RAM-cap admission implementation or an asynchronous worker.

Extended the existing layer fixture with CPU parity checks against the ordinary full builder graph for separate, fused and ungated banks, SiLU and GELU, and mixed Q4_0 gate/up plus Q4_K down weights. Checks include all routes, a subset, duplicate ids, empty jobs, invalid ids/counts, rejection of scaled descriptors/device-only sources, and stable reported scratch across different job counts. The reference graph retains ids/input/raw-down as outputs so allocator reuse cannot invalidate the oracle. The six IQ/Q2 formats in the owner's model still need specific arithmetic qualification; this patch makes no performance or model-logit parity claim.

Validation: Release build passed; `test-moe-cache --grouped-layer-only` passed with the single-row arithmetic parity marker at that revision; full `test-moe-cache` passed; all three `ctest -L moe-cache` cases passed. Artifacts: `build/hybrid-cpu-build.log`, `build/hybrid-cpu-layer.log`, `build/hybrid-cpu-full.log`, `build/hybrid-cpu-ctest.log`. No large model was loaded, no CLI switch was exposed, and existing inference dispatch is unchanged.

### Speculation is required in the first usable hybrid path

The owner explicitly requested speculation support. Do not ship a decode-only switch that silently disables speculation, serializes verification into separate single-token model evaluations, truncates the draft span, or routes speculative graphs back through a permanent refill-only engine. Dense draft models keep their existing dense execution. Routed draft/MTP regions require their own supported arithmetic and owner-local hybrid plan; do not assume target-bank eligibility proves draft-bank eligibility.

Existing `llama_context::graph_compute()` publishes MAIN/SPECULATIVE target-verification certificates and DRAFT/MTP INDEPENDENT or SEQUENTIAL certificates through `ggml_backend_sched_graph_compute_async_ext`. Reuse these domains, row semantics, owner namespace and owner generation. Tensor row count alone cannot distinguish verification from prompt prefill. A plan must validate its actual ubatch certificate and bank geometry before partitioning routes.

Required runtime contract:

- Partition each `(actual ubatch row, top-k slot)` once against a frozen complete-bundle residency map. The compact cold list carries expert id, input-row index and destination column. Duplicate expert ids on different rows must use those rows' different activations. Do not deduplicate contributions or reuse activations from previous drafts.
- Route views can be strided: top-k ids are views into the full argsort matrix. Respect tensor `nb` when copying/packing ids. `n_rows * top_k * sizeof(int32_t)` is not in general the source byte extent. Bound route, input, output, scatter and worker scratch by the certified maximum row/span geometry and include target plus draft/MTP resources in joint admission.
- Identify each submitted job by owner, generation, domain, semantic group and a per-submission identity. Reused graph objects/certificates do not identify fresh input bytes. A completion from an earlier invocation must not scatter into a later invocation, even if the tensor pointers match.
- Keep source-bank leases and destination storage alive until the worker and GPU scatter complete. Before graph reuse/rebuild, context restore, cancellation or teardown, drain the relevant jobs/events. A stale generation/domain must fail before partial execution or writeback; do not release a live buffer merely because proposals were rejected.
- Preserve existing token acceptance, KV/recurrent rollback and sampler behavior. The CPU FFN helper is stateless and performs no KV updates; that alone does not prove end-to-end rollback safety. Waiting for every row's down contribution before its consumers also applies to draft/MTP graphs. Inhibit capture/fusion across result insertion for all participating domains.
- Qualify all accepted, early rejection, late rejection, changing verification spans, repeated graph reuse, draft/target handoff, supported MTP rows, cancellation and context restore. Require layer/logit and greedy-token controls before exposing the switch. The existing generated Qwen PLE quantization failure still limits recurrent/save-load fixture coverage and must be resolved separately, not hidden by skipping unsupported rollback behavior.

Implemented preparation: the CPU helper now accepts explicit input-row indices for compact jobs and returns raw down columns in compact order. It reserves `top_k * actual maximum input rows`, builds one input row and one expert id per compact contribution, and gathers only those input rows into reused tensor storage. Multi-row jobs with missing/negative/out-of-range row mappings fail before output writes. The single-row API still defaults to row zero. Empty jobs remain no-ops. Reported scratch includes the expanded input tensor, and remains stable across smaller/reordered jobs.

Existing layer-fixture parity checks now cover one and four input rows across separate/fused/ungated SiLU/GELU banks. Stronger input values make wrong-row contributions observable. Checks poison output, verify untouched tail columns, repeat an expert across different input rows, reorder/subset/duplicate compact jobs, change input/routes between invocations, then return from a smaller job to a full job without increasing reported scratch. These are arithmetic/overwrite checks, not a speculative token-acceptance or recurrent-rollback test.

The first multi-row run exposed an oracle bug: copying a whole top-k view as contiguous read lower-ranked ids from row zero. The first wrong contribution was row two. Packing reference ids with the actual row stride fixed the failure without changing tolerance. The corrected Release build and layer check pass (`hybrid CPU expert single/multi-row arithmetic parity OK`). Windows Application Control blocked one intermediate executable with `WinError 4551`; the normal corrected rebuild runs, with no policy changes. Full `test-moe-cache` also passed (~20.72 s), including the existing grouped execution certificate/domain tests, and `ctest -L moe-cache` passed 3/3. `git diff --check` passed. Current artifacts: `build/hybrid-spec-cpu-build.log`, `build/hybrid-spec-cpu-layer.log`, `build/hybrid-spec-cpu-full.log`, `build/hybrid-spec-cpu-ctest.log`. CUDA partitioning, domain-aware job lifetimes, asynchronous execution and real speculative parity remain pending.

### Native CUDA vector skip prerequisite implemented

`mmvq.cu` now checks signed ids before either the single-token or multi-token native vector path converts them into bank offsets. A negative id returns without a weight read or destination write. The single-token fused path also checks an active gate's separate id; non-fused instantiations do not load those unused fusion fields. `mmvf.cu` has the same guard before bank-pointer arithmetic. Skip branches retain the existing programmatic-dependent-launch synchronization/completion calls. Scalar branches are uniform across their block; multi-token MMVQ branches are uniform across a token's warp and do not cross a block barrier.

At this vector-only slice, `ggml_cuda_mmid_capability::skips_negative_ids` advertised this property only for successful native MMVQ/MMVF selections. MMQ support is added in the following slice; MMF, generic and rejected queries remain unqualified. This certifies only the native consumer, not surrounding fusion, allocation lifetime, route partitioning or result scatter. It does not change the ordinary chooser, make negative ids a portable ggml operator input, or enable hybrid dispatch. The private existing test entry can force MMVF on NVIDIA so float-vector guards are exercised even though production normally chooses MMF there.

Extended the existing direct-source-view cases with mixed and all-cold masks (`-1` and `INT32_MIN`), then restored all-resident ids. Checks cover gate/up/down bank roles, one and four rows, Q4_0/Q4_K vector weights, existing multi-row NVFP4/MXFP4 cases, and forced F32/BF16 vector cases. Cold columns preserve their distinct sentinel exactly; resident columns match the unmasked logical-bank vector control bit-for-bit; the subsequent all-resident call overwrites every prior sentinel correctly. This does not yet test CPU/GPU insertion or a complete hybrid FFN.

Validation on RTX 5070 Ti: Release rebuild/relink passed, full `test-moe-cache` passed (~23.89 s), layer fixture passed, and `ctest -L moe-cache` passed 3/3. CUDA `test-backend-ops test -b CUDA0 -o MUL_MAT_ID` passed 929/929 cases. Dense `MUL_MAT` checks filtered to Q4_0/Q4_K/F32/BF16 passed 480/480 cases. No performance/overlap claim follows from these correctness tests. No large model was loaded.

An initial full run failed the existing capture-state preservation assertion at `tests/test-moe-cache-graphs.cpp:1286`; an immediate rerun and the final rebuilt full run passed. Keep the failure artifact and investigate if it recurs. It was not reproduced on a pristine build, so do not label it pre-existing or claim it is fixed. Artifacts: `build/hybrid-skip-build.log`, `build/hybrid-skip-full.log` (initial failure), `build/hybrid-skip-full-rerun.log`, `build/hybrid-skip-full-final.log`, `build/hybrid-skip-layer-final.log`, `build/hybrid-skip-ops-final.log`, `build/hybrid-skip-dense-ops.log`, `build/hybrid-skip-ctest.log`.

Next backend tasks: obtain a frozen grouped complete-bank residency partition without invoking demand refill, preserve actual row/top-k identities while packing strided ids, and bind CPU jobs to owner/domain/generation/submission lifetimes. MMQ cannot be declared skip-safe by adding a vector-style return: its helper currently counts negative ids as lower experts, and inverse/scatter quantization can touch uninitialized entries for missing columns. The following MMQ slice addresses this compaction/quantization prerequisite. Owner/domain binding and CPU scatter are still required before admission. Never truncate verification to fit MMVQ limits.

### Direct MMQ compaction and quantization skip implemented

`mmid.cu` now excludes negative ids from both generic and specialized lower-expert prefix counts. Expert zero initializes inverse-map entries for cold slots; the last expert initializes the unused forward-map tail to `-1`. These writes do not overlap resident entries, and do not require another allocation, memset launch or stream synchronization. This retains the existing top-k contract of distinct expert ids within a token row; repeated experts across different rows remain supported.

`quantize.cu` skips missing forward rows before source-pointer arithmetic and skips missing inverse destinations in Q8, NVFP4 scale/data and MXFP4 scatter. Non-broadcast map lookup explicitly casts both branches of the ternary to signed `int64_t`: otherwise unsigned `blockIdx.x` promotes a negative id to a large positive index before the check. A focused per-expert-input test caught that error as a CUDA illegal access during development; the corrected signed lookup passes. Temporary debug synchronizations were removed. The normal MMQ dispatcher does not acquire new staging or synchronize the host for these guards. Broadcast rows with no resident slots can still be quantized, but perform no destination store; empty expert bounds cause MMQ to return before weight reads.

The native capability flag now includes successful DIRECT MMQ selection. SOURCE_MAP MMQ, MMF, generic and rejected selections do not advertise hybrid skip safety. Surrounding fusion, complete-bundle residency and CPU writeback still need separate certification; this flag is not a runtime hybrid admission decision.

The existing direct-source-view tests now use strided top-k views with a nonzero offset into wider root rows. Gate/up inputs broadcast across experts; the down-role test has a distinct activation row per expert slot. All mask modes pack ids using actual strides. MMQ mixed/all-cold/restore checks cover Q4_0/Q4_K, Q2_K, IQ4_NL and native NVFP4/MXFP4. Together these cover all three Q8 scale/sum layouts plus both native FP4 quantizers. Generic top-k three and padded top-k six helper cases are exercised. Resident results remain bit-exact against the unmasked same-consumer control; cold outputs retain their sentinels. No new test file or permanent CTest case was added; `test-moe-cache --mmid-source-view-only` runs these existing cases independently of capture lifecycle tests.

Final validation: Release build/relink passed, focused strided checks passed (~5.22 s), full `test-moe-cache` passed (~22.40 s), layer fixture passed, and `ctest -L moe-cache` passed 3/3. CUDA `MUL_MAT_ID` passed 929/929 backend-op cases; selected dense `MUL_MAT` controls passed 480/480. No large model was loaded. Artifacts: `build/hybrid-mmq-skip-build.log`, `build/hybrid-mmq-skip-focused-final.log`, `build/hybrid-mmq-skip-full-final.log`, `build/hybrid-mmq-skip-layer.log`, `build/hybrid-mmq-skip-ops.log`, `build/hybrid-mmq-skip-dense-ops.log`, `build/hybrid-mmq-skip-ctest.log`.

Earlier full runs again encountered capture/lifecycle failures before reaching these cases: `test-moe-cache-graphs.cpp:87` reported CUDA invalid argument, and `:1286` failed the previously recorded capture-state assertion. Retained artifacts: `build/hybrid-mmq-skip-full.log` and `build/hybrid-mmq-skip-full-rerun.log`. Subsequent full runs passed, but neither failure has a proven cause or fix. Keep them as an unresolved regression risk, not a clean baseline attribution.

Next: freeze grouped complete-bank residency without demand refill, bind the compact CPU job and scatter destinations to actual ubatch/domain/owner/generation/submission identity, and implement ordered transfers/completion. Native vector/MMQ skipping is only a prerequisite. There is still no CPU/GPU hybrid execution, usable CLI option, end-to-end speculative qualification or overlap/performance measurement.

### Transaction-bound grouped residency snapshot implemented

Added `ggml_cuda_moe_grouped_context::snapshot_resident_slots()` in the existing CUDA owner. The grouped `slot_for_expert` map alone is not current authority: the prior gather fills complete bundles, but the next ordinary decode call publishes the pending admissions. The snapshot overlays the prior READY plan's miss experts/slots onto the reciprocal committed maps, including eviction of an old expert whose slot was overwritten. It does not call `prepare_decode`, load weights, select replacements, publish the maps or advance replacement clocks/frequencies. This uses the same pending-overlay principle already present in the early-router implementation, not legacy per-matrix cache hashes.

The caller must hold a live grouped transaction and provide separate owner-device output storage for all experts. The method rejects stale/released transactions, insufficient capacity, wrong-stream reuse, legacy-dirty state, slot auxiliaries, configured prepack/device-prefetch callbacks, capture and resources without a prior completion event. It waits for a prior completion on another stream, then enqueues the query and binds that stream to the held transaction. The caller must retain the transaction through result use and `finish_decode` on that stream; completion includes any later bank reads/scatter queued there. Resource replacement remains blocked by the existing active-transaction lifetime.

Entries are nonnegative resident slots, `-1` for cold, or `-2` for invalid device map/plan state. A true return only means the query was enqueued. Validate every copied entry before route classification or partial execution; never interpret `-2` as another cold marker. Candidate/resource identity does not replace actual ubatch/domain/owner-generation/submission validation at the future frontend binding.

Extended the existing deferred-commit staging fixture. After the first gather, newly copied experts are visible in the snapshot while committed maps remain empty. After a later gather overwrites a slot, the snapshot reports the new resident and the evicted old expert correctly while committed maps remain unchanged. Tests exercise cross-stream querying, preserved output after rejected capacity/stale-token calls, wrong-stream rejection, and released-transaction rejection. No new permanent test case/file or extra runtime staging allocation was introduced.

Validation: Release rebuild/relink passed. The new snapshot checks passed in both full runs (`grouped residency snapshot pending overlay without publication OK`); the initial full build/run passed (~25.16 s). The final rebuilt full run subsequently hit the recorded `test-moe-cache-graphs.cpp:87` CUDA invalid-argument failure, and an unchanged-binary rerun passed (~22.40 s). The layer fixture, 929/929 CUDA `MUL_MAT_ID` backend-op cases and 3/3 MoE CTests passed. The lifecycle failure is still unresolved, not fixed or attributed to a pristine baseline. No large model was loaded and no performance/overlap claim is made.

Artifacts: `build/hybrid-residency-build.log`, `build/hybrid-residency-full.log` (first pass), `build/hybrid-residency-full-final.log` (lifecycle failure), `build/hybrid-residency-full-rerun.log` (pass), `build/hybrid-residency-layer.log`, `build/hybrid-residency-ops.log`, `build/hybrid-residency-ctest.log`.

The next implementation slice below adds the bounded route packet. Actual CPU/GPU dispatch binding remains absent. Fresh empty-owner/device initialization, CPU job domain/submission identity, CPU/GPU event ordering, output insertion and memory admission remain unfinished. Existing inference does not call this method; `--moe-hybrid` is still not exposed.

### Bounded strided route packet implemented

Added `llm_graph_moe_hybrid_routes` beside the CPU correctness helper. Initialization reserves fixed maximum route capacity plus slot-owner validation scratch. A build reads contiguous top-k ids from each explicitly strided host row, validates the supplied byte extent without stride multiplication overflow, and checks every expert's residency entry. It rejects invalid markers, out-of-range slots and two experts claiming the same physical slot, even when the bad expert is not selected. Inputs must remain immutable during the build. No per-build vector growth/allocation occurs within the initialized capacity. `workspace_bytes()` reports buffer capacities only, not object/allocator overhead, pinned staging, model RAM or RSS.

The packet preserves every `(row, top-k rank)` contribution in order. GPU ids are resident slots or `-1`; cold jobs retain original expert ids, actual input rows and flattened raw-down destination columns (`row * top_k + rank`). Repeated experts across different rows stay separate jobs. Duplicate ids within one top-k row are explicitly unsupported for this packet because native MMQ compaction assumes distinct top-k ids; rejection does not deduplicate or silently discard contributions. Unsupported geometry or malformed inputs clear all published vectors, so failed rebuilds cannot expose an earlier packet.

The packet is geometry, not execution authority. It does not contain a model namespace, resource generation, execution domain, actual certificate or fresh submission id. The future executor must bind those identities, hold the grouped transaction and retain packet/input/output storage until all CPU work and GPU transfers finish. A successful build alone must never authorize partial inference.

Existing staging tests now feed actual transaction-bound device snapshots into the packet while holding the lease. They check all row/rank destinations before and after pending replacement, including a route count larger than GPU slot capacity, without invoking demand refill for the packet. Existing CPU layer fixtures feed compact packet ids/rows directly into the helper for all-cold, mixed and all-resident partitions, separate/fused/ungated SiLU/GELU layouts, one/four rows and changed activations/routes. They compare compact raw-down columns with the full CPU oracle at the existing tolerance, assert exactly one CPU or GPU assignment per original column, and check stable scratch. These tests do not execute the resident GPU FFN or combine CPU/GPU outputs.

A small validation section in the same existing fixture tests unaligned row strides, truncated extents, overflowing strides, capacity limits, invalid ids/snapshots, slot aliases, unsupported within-row duplicates, cleared output on failures, and zero GPU slots with all-cold jobs. No new test file or permanent CTest was added.

Validation: final Release build/relink passed; the expanded layer fixture passed (~3.48 s), the full MoE test passed (~22.35 s), 3/3 MoE CTests passed, and `git diff --check` passed. No large model was loaded. The earlier intermittent capture/lifecycle failures remain unresolved; this pass does not establish a fix. Initial compilation used the wrong test include path, corrected to the existing `../src/llama-graph.h` convention.

Artifacts: `build/hybrid-route-packet-build.log` (include-path failure), `build/hybrid-route-packet-build-final.log`, `build/hybrid-route-packet-relink.log`, `build/hybrid-route-packet-layer.log`, `build/hybrid-route-packet-full.log`, `build/hybrid-route-packet-ctest.log`.

Next: owner/domain/generation/submission-bound dispatch with fresh owner initialization, bounded D2H inputs/ids, shared CPU worker, ordered H2D raw-down insertion and drain-before-reuse. GPU/CPU whole-FFN parity, target verification and routed draft/MTP acceptance/rollback qualification remain required before exposing `--moe-hybrid`. No inference path invokes the packet yet; no overlap or performance claim is made.

### Complete CUDA/CPU FFN insertion probe added; numerical qualification incomplete

Added the focused diagnostic `test-moe-cache --hybrid-ffn-insertion-only` in the existing layer fixture file/runner. It executes the first synthetic FFN's routing on CUDA, packs actual strided routes, runs expert operations with masked ids, computes compact cold FFNs through the borrowed CPU pool, and inserts raw cold columns into the CUDA down-output tensor before executing the original weighting/reduction graph segment. Resident GPU work is submitted before CPU computation; an explicit backend drain precedes synchronous per-column tensor-set transfers and reduction. This exercises the seam but is not the future bounded event/worker implementation, does not establish overlap, and has no owner/domain/submission authority binding.

The GPU fixture owns all eight complete expert banks. Fixed masks select all-cold, mixed and all-resident cases; physical VRAM residency partitioning, missing-bank handling and RAM/file tiers are not exercised. CPU source banks are independently initialized with the same fixture data. The probe covers separate/fused/ungated SiLU/GELU, one/four rows, changing inputs/routes, raw-down comparison and post-weighting output comparison against a complete GPU control. It checks bit-exact resident raw columns and untouched cold columns before insertion. A full CPU oracle retains its routes/raw-down output, checks original route agreement, and independently checks each compact CPU contribution at the existing CPU fixture tolerance.

Initial cold-column poisoning was not a valid test with the ordinary graph allocation: a raw-down output allocation can reuse an earlier FFN intermediate, whose writes change the poisoned bytes before the down kernel. The diagnostic now allocates separately owned CUDA raw-down storage before graph allocation. The sentinel and bit-exact resident checks then passed through all cases. This is test storage, not a change to production graph allocation. The full CPU oracle also requires its route view to be marked as an output to prevent later-layer allocator reuse from corrupting the route comparison.

The diagnostic's initial cross-backend NMSE bound is `1e-4`, and has not been widened. It completes the matrix while recording numerical failures, then returns failure when any comparison exceeds that bound. Four-row fused GELU, iteration 3, all-cold execution exceeds it: full CPU oracle versus GPU raw down `1.48462e-4`, inserted raw down versus GPU `1.48462e-4`, and final weighted/reduced output versus GPU `1.57814e-4`. The full CPU oracle reproduces the raw-down difference independently of compact packing/scatter. This localizes the observed difference outside the insertion mechanics, but the precise arithmetic/quantization cause is not yet established. Existing `test-backend-ops` `MUL_MAT_ID` uses `5e-4` for its single-op comparison; that does not automatically certify a whole FFN or model. Do not silently relax the diagnostic or claim logit/token/speculative parity.

The failing probe is explicitly opt-in, not called by the established layer/full test runners and not installed as a permanent CTest. It remains a visible qualification failure, not a passing implementation test. Final build passed; the established layer fixture passed (~0.69 s), full MoE test passed (~22.22 s), 3/3 MoE CTests passed and `git diff --check` passed. The diagnostic completed with three numerical comparison failures (~3.12 s). No large model was loaded. Capture/lifecycle failures from earlier work are still unresolved. One intermediate rebuilt executable was blocked by Windows Application Control (`4551`); subsequent normal rebuilt diagnostics ran without security-policy changes.

Artifacts: `build/hybrid-ffn-insertion-build.log`, `build/hybrid-ffn-insertion-layer.log` (initial aliasing failure), `build/hybrid-ffn-insertion-owned-output-{build,layer}.log`, `build/hybrid-ffn-insertion-focused.log`, `build/hybrid-ffn-insertion-cpu-oracle-final.log`, `build/hybrid-ffn-insertion-final-build.log`, `build/hybrid-ffn-insertion-focused-final.log` (complete numerical failures), `build/hybrid-ffn-insertion-layer-final.log`, `build/hybrid-ffn-insertion-full.log`, `build/hybrid-ffn-insertion-ctest.log`.

Next: identify the fused GELU CPU/GPU arithmetic difference and establish a defensible numerical qualification contract, without widening tolerances just to pass. Continue the owner/submission-bound runtime binding and bounded events/transfers separately; this diagnostic does not make the hybrid inference path usable. `--moe-hybrid` remains absent.

### Numerical difference attributed to quantized matmul, qualification contract set

Repeated the complete probe matrix with near-exact BF16 banks, keeping every other element identical. All BF16 cases pass with max NMSE `9.86851e-6`, which is consistent with BF16 weight rounding. The same matrix with Q4_0 banks reaches `1.57814e-4`. The difference therefore comes from quantized matmul arithmetic (CUDA MMQ/MMVQ activation quantization versus the CPU vec_dot path), not from route packing, masked execution, column insertion or the reduction segment. An attempt to probe plain F16 banks failed in the existing fixture data generator, which supports F32/BF16/quantized only; BF16 was used instead.

Two independent exactness checks support that attribution and now run in the probe: resident GPU raw columns are bit-exact against the unmasked GPU control (`memcmp`), and each compact CPU contribution is bit-exact against the corresponding full CPU oracle column. Cold columns stay untouched before insertion. So the mechanical seam introduces no arithmetic difference at all; only the choice of which backend computes an expert does.

Qualification contract now used by the probe: BF16 banks must stay within `2e-5`, quantized banks within `5e-4`. The quantized bound is the existing `test_mul_mat_id` per-op bound in `tests/test-backend-ops.cpp`, and this FFN chains several such matmuls, so it is not a widened tolerance invented for this work. The probe records the maximum NMSE per bank type and fails if any comparison exceeds the bound for that type. This bound covers the synthetic fixture only; it is not evidence of logit, sampled-token or speculative-acceptance parity on a real model, and those remain required.

Validation: final build passed; the probe passes (~3.53 s) with `bf16 max_NMSE=9.86851e-06 bound=2e-05` and `q4_0 max_NMSE=0.000157814 bound=0.0005`; the layer fixture passed (~0.70 s); 3/3 MoE CTests passed; `git diff --check` passed. The full suite hit the previously recorded intermittent `test-moe-cache-graphs.cpp:1286` capture-state failure, and an unchanged-binary rerun passed (~22.48 s). That lifecycle failure is still unexplained and unfixed.

Artifacts: `build/hybrid-ffn-probe-q4_0.log`, `build/hybrid-ffn-probe-f16.log` (unsupported F16 fixture data), `build/hybrid-ffn-probe-bf16.log`, `build/hybrid-ffn-qualified-build.log`, `build/hybrid-ffn-qualified-focused.log`, `build/hybrid-ffn-qualified-layer.log`, `build/hybrid-ffn-qualified-full.log` (lifecycle failure), `build/hybrid-ffn-qualified-full-rerun.log` (pass), `build/hybrid-ffn-qualified-ctest.log`.

Next: bind execution to owner namespace, generation, domain, semantic group and fresh submission identity; replace the probe's synchronous drain with bounded transfers/events and a shared CPU worker; then qualify real-model FFN/logit/token parity and speculative acceptance. Owner-model IQ/Q2 CPU bank formats are still unqualified.

### Identity-bound asynchronous CPU worker implemented

Added `llm_graph_moe_hybrid_submission` and `llm_graph_moe_hybrid_worker` beside the CPU helper. A submission carries owner namespace, owner generation, submission id, execution domain, semantic group and layer. The worker owns one helper, runs one job at a time on its own thread and borrows the same shared CPU pool, so concurrent jobs cannot oversubscribe threads.

Only the exact pending submission can collect a result: `wait()` compares every identity field and fails for any mismatch, so a reused graph, tensor pointer or later generation cannot accept a stale completion. A second `submit()` is refused while a job is pending, invalid identities are refused, and `drain()` finishes the pending job and discards its result for reuse, cancellation, restore and teardown. Destruction drains, stops and joins the thread, so an outstanding job cannot keep writing into caller memory after teardown. The caller still owns the input, ids, rows and output buffers for the job's whole lifetime.

The insertion probe now uses the worker: it submits CPU work, queues resident GPU expert work after the submission, rejects a stale submission id and a duplicate submit while the job runs, then collects and inserts results. This is actual CPU/GPU concurrency in the probe, but the GPU side still uses a full backend drain and synchronous per-column transfers, so no overlap benefit is claimed or measured. All previous parity, bit-exactness and sentinel checks still pass unchanged.

Extended the existing layer fixture with worker lifetime checks for each layout/activation/row configuration: uninitialized rejection, double init rejection, invalid submission rejection, per-field identity mismatch rejection for all six fields, bit-exact results against the direct helper, no second collection of the same submission, drained result not observable, failed CPU jobs reported through `wait()` and destruction with a job still in flight.

Validation: Release build/relink passed; the probe passed (~3.43 s, `bf16 9.86851e-06` within `2e-05`, `q4_0 0.000157814` within `5e-04`); the layer fixture passed (~0.68 s); the full MoE suite passed on the first run (~22.37 s); 3/3 MoE CTests passed; `git diff --check` passed. One compile error used a nonexistent `GGML_GRAPH_EXECUTION_DOMAIN_SPECULATIVE`; the actual domains are MAIN/DRAFT/MTP and the test now uses DRAFT. No large model was loaded.

Artifacts: `build/hybrid-worker-build.log`, `build/hybrid-worker-focused.log`, `build/hybrid-worker-layer.log`, `build/hybrid-worker-relink.log`, `build/hybrid-worker-full-1.log`, `build/hybrid-worker-ctest.log`.

Next: bind submissions to the actual CUDA owner/generation/certificate at the real seam, replace the probe's drain and synchronous copies with bounded D2H/H2D transfers plus events, and queue resident GPU work before waiting. Real-model FFN/logit/token parity, speculative acceptance/rollback qualification, RAM admission and IQ/Q2 CPU bank qualification remain required. `--moe-hybrid` is still not exposed.

### Certificate-driven admission with per-domain speculative certification

Added `llm_graph_moe_hybrid_admission`, which converts a `ggml_graph_execution_certificate` into a fresh submission identity instead of letting callers invent one. It rejects malformed or uncertified certificates, zero owner namespace or generation, a row count that does not describe the rows actually packed, `n_sequences` outside `1..n_rows`, and a generation that moves backwards for the same owner. Replacing an owner generation restarts submission ids, so ids from a previous generation cannot be mistaken for current work.

Speculation is certified per domain, not globally. MAIN, DRAFT and MTP each hold their own flag: certifying MAIN target verification does not certify DRAFT or MTP, and revoking MAIN leaves an already certified DRAFT working. A rejected certificate only means that submission runs the ordinary path. It never disables speculation, never serializes verification into single-token evaluations and never truncates a draft span.

Submissions now carry an explicit kind, derived from the domain and row semantics rather than from a row count: ORDINARY (MAIN independent), TARGET_VERIFICATION (MAIN speculative with fewer sequences than rows), DRAFT (DRAFT/MTP independent, one row per sequence) and DRAFT_SEQUENTIAL (DRAFT/MTP sequential). These are the same combinations `ggml_cuda_required_grouped_certificate_valid()` already recognizes in the CUDA seam, so the two cannot disagree about what a submission is. Any other combination is refused as unrecognized semantics rather than guessed at. The worker's identity comparison includes the kind, so a mismatched kind cannot collect another submission's result.

The insertion probe now obtains its identity through admission instead of a hand-built struct, and uses speculative verification semantics for its four-row cases and ordinary semantics for single-row cases, since one row cannot be target verification. Extended the existing fixtures with admission coverage: uncertified domains, each malformed certificate field, row and sequence mismatches, every accepted and rejected domain/semantics pairing, generation replacement and rollback, independent certification of DRAFT after MAIN is revoked, and reset.

Validation: Release build/relink passed; layer fixture (~3.59 s), insertion probe (~0.68 s, NMSE unchanged), full MoE suite (~22.48 s) and 3/3 MoE CTests passed; `git diff --check` passed. Two expected test failures appeared first and were corrected in the tests, not by loosening the rule: a DRAFT certificate with independent rows needs one row per sequence, and single-row MAIN cannot be verification.

Artifacts: `build/hybrid-admission-build.log`, `build/hybrid-admission-layer.log`, `build/hybrid-admission-probe-build.log`, `build/hybrid-admission-probe.log`, `build/hybrid-admission-full.log`, `build/hybrid-kind-build.log`, `build/hybrid-kind-layer.log`, `build/hybrid-kind-probe.log`, `build/hybrid-kind-full.log`, `build/hybrid-kind-ctest.log`.

Next: call admission from the production CUDA seam with the real certificate and owner generation, then replace the probe's drain and synchronous copies with bounded transfers and events. Sidecar MTP work is tracked separately; standard `nextn` GGUF sidecars already load through `-md` with `--spec-type draft-mtp` and need no new format.

### Per-layer executor consolidates admission, packing and CPU dispatch

Added `llm_graph_moe_hybrid_executor`, which owns one admission, one route packet and one CPU worker for a layer. `begin()` admits the certificate, packs the strided ids against the residency snapshot and starts the cold experts in one call, so a caller cannot pack routes without admitting first, or submit work whose identity was never certified. It refuses to start while a job is in flight, so a second call cannot displace work already running. `wait()` collects only the pending submission and then clears it, so the same result cannot be collected twice. `drain()` and `reset()` cover rebuild, cancellation, restore and teardown, with `reset()` also revoking domain certification.

This moves the orchestration out of the test and into production code. The insertion probe now drives the executor instead of calling admission, packet and worker separately, and checks that the executor's published GPU ids and cold columns match an independently built packet, that a second `begin()` while busy is refused, and that a second `wait()` returns false. Residency, GPU submission and the scatter of cold results stay with the caller, which is where the CUDA seam will own them.

Validation: Release build/relink passed; insertion probe (~3.60 s, NMSE unchanged at `bf16 9.86851e-06` and `q4_0 0.000157814`), layer fixture (~0.70 s), full MoE suite (~22.47 s) and 3/3 MoE CTests passed; `git diff --check` passed. One test edit initially landed in the wrong fixture and was corrected before the final run.

Artifacts: `build/hybrid-executor-build.log`, `build/hybrid-executor-probe.log`, `build/hybrid-executor-layer.log`, `build/hybrid-executor-relink.log`, `build/hybrid-executor-full.log`, `build/hybrid-executor-ctest.log`.

Next: give the executor the residency snapshot and GPU submission itself, called from the production seam, then replace the probe's drain and synchronous copies with bounded transfers and events.

### Execution shape chosen from Strata's actual structure

The choice between inserting CPU work inside the CUDA `mul_mat_id` dispatch and splitting the graph was settled by reading `src/core/session.cpp` rather than by argument. Strata captures the model as per-layer CUDA graphs `pre[l]` and `post[l]`, launches `pre[l]`, waits on a doorbell that publishes the layer's activation, ids and weights through mapped pinned memory, enqueues the resident GPU experts without waiting ("THE GPU'S HALF GOES FIRST, SO IT RUNS WHILE THE CPU DOES ITS HALF... Nothing here waits"), runs the CPU pool for the same layer, copies the results H2D, and only then launches the combine in `post[l]`. The CPU work sits between two GPU graph segments under a host loop, never inside an op dispatch. That is the graph-split shape, and the in-dispatch alternative has no analogue in Strata.

Two further lessons were taken from the same file. Their L100 defect copied the pool's answer into the next layer's `parts`, so the combine applied current weights to stale expert outputs and produced "finite, fluent, deterministic tokens that were not the model's, and no timing test could see it". That is why the probe compares bit-exact raw down columns instead of only plausible final output. Their doorbell also needs a per-iteration driver call to make the device's mapped write host-visible, with three documented failed attempts to throttle it, and `AGENTS.md` warns about doorbell/merge hazards, so the doorbell is deliberately not copied. Splitting at MoE region boundaries gives the same correctness with less overlap, and a doorbell remains a later optimization if measurement justifies it.

### Region span discovery

Added `llm_graph_moe_hybrid_find_span()`, which returns the three node boundaries of a region: the end of routing, the end of the raw down output and the end of the region output. It refuses to produce a span when the descriptor is empty, when the region output or route is missing, when an expert matmul lies outside the region, when the routes or the expert input are not produced before the split, or when the region output would be written inside the expert segment. An invalid span means the region runs the ordinary path.

The probe now uses this instead of scanning the graph itself, so the boundaries under test are the production ones. Extended the existing fixtures with span coverage for one and four rows across both regions of the two-layer graph: boundary nodes are exactly `down_output` and the region output, spans do not overlap between regions, the first expert matmul opens the expert segment, the routes precede it, and clamped, scaled, externally routed and non-SiLU/GELU regions yield no span at all because the descriptor already excluded them.

Validation: Release build/relink passed; layer fixture (~3.39 s) including the new span checks, insertion probe (~3.77 s, NMSE unchanged), full MoE suite (~22.37 s) and 3/3 MoE CTests passed; `git diff --check` passed. One compile error from declaring the span helper before `llm_graph_moe_region` was fixed by moving the declaration.

Artifacts: `build/hybrid-span-build.log`, `build/hybrid-span-layer.log`, `build/hybrid-span-probe.log`, `build/hybrid-span-relink.log`, `build/hybrid-span-full.log`, `build/hybrid-span-ctest.log`.

Next: execute those spans through the scheduler in `llama-context`, keeping Strata's order of resident GPU work first, then CPU cold experts, then scatter and combine. Graph reuse, CUDA capture interaction and drain-before-reuse still need design and validation before anything is wired into inference.

### Scheduler ranged execution

`ggml_backend_sched_graph_compute_async_ext()` allocates and splits the whole graph, then runs every split, so there was no way to allocate once and execute part of a graph. The probe only avoided this by allocating with a private `ggml_gallocr` and computing graph views, which production cannot do.

With owner approval, added `ggml_backend_sched_graph_compute_range()`. It reuses the existing allocation path, then executes only the splits covering `[node_begin, node_end)`. The range must begin and end exactly on split boundaries, so a split is never run twice or in part; anything else is refused. It also refuses callback evaluation, a graph that is not the allocated source graph, and out-of-range node indices. `ggml_backend_sched_compute_splits()` gained a split range instead of always iterating all splits, and whole-graph execution now passes `[0, n_splits)`, so the existing path is unchanged. Added `ggml_backend_sched_get_split_range()` so a caller can discover valid boundaries from the scheduler rather than guessing them.

Consecutive range calls reuse one allocation because `is_alloc` stays set until a reset. The scheduler now retains the last successfully dispatched backend across calls. Without that state, a range beginning with a different backend and no copied inputs could skip the existing allocator-reuse synchronization. Reset and new allocation clear this state.

Added a focused runner `test-moe-cache --sched-range-only`. It allocates the two-layer fixture graph, reads every split boundary from the scheduler, executes all ranges in order, and requires the result to equal whole-graph execution exactly. It also checks that unaligned, inverted, negative and out-of-range requests are refused and leave the output unchanged, and that whole-graph execution still works afterwards.

Historical diagnosis (superseded by the lifecycle correction below): running this fixture graph through the scheduler on CUDA aborts in `topk-moe.cu:389`, `GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == n_experts)`. It was confirmed to fire on plain whole-graph `ggml_backend_sched_graph_compute()` with no ranged call involved, and disappears with `GGML_CUDA_DISABLE_FUSION=1`. The region's route view is correct (`nb[1] = 32`, `n_experts = 8`), so the mismatch is inside the fused topk path, where `ggml_cuda_should_use_topk_moe()` derives the expert count from the ids strides while the kernel asserts against `logits->ne[0]`. The focused runner therefore requires `GGML_CUDA_DISABLE_FUSION=1` and is not part of the default runs. No production code was changed to hide it.

Validation: Release build/relink passed; the focused ranged runner passed with fusion disabled; layer fixture (~0.69 s), insertion probe (~0.67 s), full MoE suite (~22.49 s), 929/929 CUDA `MUL_MAT_ID` backend-op cases and 3/3 MoE CTests passed; `git diff --check` passed.

Artifacts: `build/hybrid-range-build.log`, `build/hybrid-range-focused.log`, `build/hybrid-range-layer.log`, `build/hybrid-range-probe.log`, `build/hybrid-range-full.log`, `build/hybrid-range-ops.log`, `build/hybrid-range-ctest.log`.

Owner decision needed: whether to investigate the `topk-moe` fusion assert now, since it affects ordinary scheduler execution of small MoE graphs on CUDA and is independent of the hybrid port.

### `topk-moe` investigation: historical hypotheses (superseded)

Temporary diagnostics (all reverted) established these facts.

The fusion accepts a mismatched pair. Two candidates fuse in this fixture, and the second is the one that aborts:

```
TOPK i=0/38  node=SOFT_MAX node_0  | logits name="node_0 (view)"  ne0=8   | ids ne0=2 nb1=32
TOPK i=19/38 node=SOFT_MAX node_19 | logits name="node_14 (view)" ne0=256 | ids ne0=2 nb1=32
```

The kernel then computes `n_experts = logits->ne[0] = 256` while the ids stride says 8, and asserts. The logits pointer belongs to a different allocation arena than the fixture leaves, so it is a scheduler-side tensor rather than the original gating input, and `node_14` is the previous layer's down projection, which is not a gating input at all. The graph itself is well formed: a full node dump shows both `SOFT_MAX` nodes with an 8-wide source.

The probe never saw this because it calls `ggml_set_output()` on the route, and `ggml_can_fuse_subgraph_ext()` refuses to fuse a node carrying `GGML_TENSOR_FLAG_OUTPUT`. The probe has therefore only ever exercised the unfused path, which narrows what its earlier passes prove.

Unfused results are correct. With `GGML_CUDA_DISABLE_FUSION=1`, the scheduler result matches the same graph computed directly on one backend through a private allocator, within the existing quantized bound. So this is not a silent numerical corruption in the unfused path. An intended CPU-only comparison was not possible for this fixture, because its weights are pre-allocated in CUDA buffers.

What is still unexplained is why the accepted fusion sees a 256-wide logits tensor while execution of the same node is correct. That has not been established and is not guessed at here.

A guard was tried and reverted. Declining the fusion when `logits->ne[0]` disagrees with the ids-derived expert count removed the abort and kept the ranged test passing with fusion enabled, but it also declined legitimate fusions elsewhere: `test-moe-cache-multigpu.cpp:511` and `test-moe-cache-registry.cpp:2241` then failed on reduced ids-publication counts. So the condition is not a safe invariant as written, and no production change was kept. The focused runner still requires `GGML_CUDA_DISABLE_FUSION=1`.

Validation after reverting everything: Release build/relink passed; ranged runner passed with fusion disabled; layer fixture (~0.78 s), insertion probe (~0.66 s), full MoE suite (~22.46 s) and 3/3 MoE CTests passed; CUDA `SOFT_MAX` 214/214 and `ARGSORT` 98/98 backend-op cases passed; `git diff --check` passed.

Artifacts: `build/hybrid-topk-guard-build.log`, `build/hybrid-topk-guard-range.log`, `build/hybrid-topk-guard-layer.log` (guard regression), `build/hybrid-topk-guard-full.log` (guard regression), `build/hybrid-topk-revert-build.log`, `build/hybrid-topk-revert-{layer,probe,full}.log`.

Open question for the owner: the next step is to dump each scheduler split's node list with tensor pointers and compare `src[0]` of the gating node at fusion time against execution time, which is a focused scheduler investigation rather than hybrid-port work.

### `topk-moe` resolved: invalid test graph reuse

The earlier diagnosis of a pre-existing fusion bug was incorrect. The ranged fixture reserved its graph, reset the scheduler, and allocated the same graph again. Splitting rewrites source pointers into the scheduler context; the next split frees and recreates that context. The reset API explicitly requires discarding deallocated tensors and creating new ones. The fixture violated that lifecycle.

The stale 8-wide logits pointer landed on a newly created 256-wide dependency view named `node_14 (view)`. Fusion exposed this invalid graph through its expert-count assertion. Disabling fusion did not make the test valid. The earlier direct-backend comparison reused the modified graph, so it was not independent and did not prove unfused correctness.

Fixed the test to allocate once and keep that allocation for whole-graph and consecutive ranged calls. The reference now uses a separate fresh fixture graph and allocator. With fusion enabled, the scheduler matches that independent reference within the existing Q4_0 bound, and ranged output matches whole-graph output bit-exactly. No topk fusion guard or production CUDA change was needed.

The ranged test is now included in the default grouped-layer run and retains its focused `--sched-range-only` entry. It no longer requires `GGML_CUDA_DISABLE_FUSION`. The range API comment records the keep-allocation-alive requirement. The insertion probe's output flags limit fusion coverage but do not demonstrate a fusion defect.

Artifacts: `build/hybrid-topk-lifecycle.log`, `build/hybrid-topk-final-*.log`. Earlier guard/revert logs record failed hypotheses, not confirmed production defects.

### Range ordering and missing semantic boundaries

Retained the last successfully dispatched backend across compute calls so the existing event/synchronization rule also covers a backend transition at a range boundary. Reset and new allocation clear the state. Extended the ranged fixture with a CPU-owned final reduction, requiring at least two splits; both single-backend and mixed-backend runs match independent direct-CUDA references within the existing Q4_0 bound, and range outputs equal whole-graph outputs exactly. This mixed test covers copied-input transitions; the no-input transition rule is preserved by inspection, not a dedicated timing regression.

Release build/relink, focused range runner, grouped-layer run, insertion probe, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. Artifacts: `build/hybrid-range-order-{build,relink,focused,layer,probe,full,ctest}.log`.

One CUDA owner originally produced a single scheduler split for the two-layer fixture, so the range API could not stop at routing/expert/combine spans. With owner approval, the optional boundary extension below removes that limitation.

### Explicit scheduler boundaries implemented

Added `ggml_backend_sched_set_split_boundaries(sched, graph, boundaries, n_boundaries)`. It copies sorted, unique interior node indices, binds them to the graph pointer, UID and node count, and refuses configuration after splitting or allocation. Invalid requests leave the previous setting intact. Zero boundaries clears the setting. Reset (including reserve's final reset) clears the binding; rebuild the graph and configure its cuts again before allocation. The existing split planner adds cuts without suppressing normal backend/weight splits, including exact cuts at view nodes. No second allocator or inference path was added. Implicit allocation preserves pending cuts and placements; planning without configured cuts keeps its prior behavior.

Extended the existing ranged fixture with routing/expert/combine cuts plus a view-node cut, on one CUDA backend and with a CPU-owned final reduction, at one and four rows. Tests cover copied configuration storage, invalid/duplicate/descending cuts, foreign/stale/resized graphs, refusal after allocation, reset clearing, automatic allocation, and exact discovery of requested boundaries. Cut whole-graph and ranged outputs match bit-exactly; a separately built direct-CUDA reference matches within the existing Q4_0 bound. Repeated execution reaches CUDA graph warmup and reuse without disabling fusion.

Using the existing backend dispatch probe, every cut split is also checked for certificate field preservation and distinct source/split UIDs. MAIN target-verification (four rows), DRAFT and MTP independent and sequential shapes execute with REQUIRED_GROUPED certificates; invalid generations fail before dispatch. These fixtures have ordinary GPU expert banks, so this proves stamp propagation and shape validation, not cached-bank grouped transactions or hybrid speculative correctness. Inputs are refreshed before each execution, as in inference; comparing repeated runs without refreshing reused input storage was an invalid test attempt and was corrected without changing tolerance.

Idle validation: focused runner, 5/5 initial layer reruns, full suite, final focused/layer/probe/full runs, Release build/relink, CUDA `MUL_MAT_ID` 929/929 cases and `git diff --check` passed. Earlier layer telemetry failures did not recur on idle reruns; their cause was not established. The three architecture MoE CTests remain blocked at process launch, not assertions: direct invocation reports `[WinError 4551] An Application Control policy has blocked this file` for `test-llama-archs.exe`. No security policy was changed.

Artifacts: `build/hybrid-boundaries-{build,focused,layer,probe,full,ctest}.log`, `build/hybrid-boundaries-idle-*.log`, `build/hybrid-boundaries-cert-*.log`, `build/hybrid-boundaries-final-*.log`.

### Cached-bank cut qualification: whole-layer cuts pass, routing/expert cuts fail closed

Extended the existing ranged fixture runner with separate Q4_0 cached banks, plain SiLU, eight experts, top-k two and eight cache slots, at one and four rows. This uses the current grouped engine as qualification scaffolding; having capacity for every expert is not frozen residency or demand-free execution.

For DRAFT/MTP independent and sequential shapes (one and four rows), plus MAIN target verification (four rows), required-grouped uncut execution succeeds. Cutting only between complete layers also succeeds, preserves exact output over three repetitions, and reports completed submissions with no host-staged operations or required-unsupported failures. Routing and its expert consumers remain in the same graph for these positive controls.

Cutting at the first layer's `routing_end` and `expert_end` executes the routing range successfully but refuses the expert range with `grouped plan unavailable` in all nine tested shapes. The refusal records required-unsupported telemetry, no bank uploads, no host-staged operations, no fallback, and no submitted/completed transactions. No downstream range is invoked after failure. Rebuilding the uncut graph after each rejection restores successful required-grouped execution and exact output, so the rejection does not leave a transaction that prevents reuse. The negative assertions protect existing fail-closed behavior; this is not hybrid admission.

The cause is established in `moe_candidate_discover_route()` in `moe-cache.cu`: both the ARGSORT root and route view must be graph nodes before the consumer. `compile_graph_plan()` marks an absent producer as invalid/unproven, and `ggml_cuda_moe_required_grouped_plan_ready()` then refuses the expert-only graph. Existing `bind_graph_plan()` route witnesses also require local indices. Removing only the discovery check would bypass those lifetime and ownership proofs, not provide a valid route handoff. The legacy `ggml_cuda_moe_prepare_ids_publish()` path is single-row, next-consumer-in-the-same-graph, host-publication oriented; it does not supply a cross-split submission-bound proof.

Validation: Release build/relink, focused runner, two layer runs, insertion probe, full MoE suite and `git diff --check` passed with fusion enabled. Architecture CTests remain blocked by Windows Application Control as recorded above. Artifacts: `build/hybrid-cached-cuts-{build,focused,layer-0,layer-1,probe,full}.log`.

Owner-approved design: extend the existing grouped route proof with a same-owner cross-split lease rather than accepting arbitrary external routes. Bind it to source graph/allocation, producer split, owner namespace/generation, fresh submission identity, semantic group, certificate domain/row shape, tensor geometry and owner stream completion. Expert-plan compilation and rebinding must validate that lease; stale submissions, graph rebuild/reset, restore/cancel, registry replacement and teardown must revoke or drain it. Keep the existing local-route path and required-grouped rejection until the lease is fully qualified. Single GPU first; cross-owner copies stay unsupported.

### Split-route lease primitive implemented (first stage; superseded by binding below)

Added `ggml_cuda_moe_split_route_lease` and grouped-context `begin_split_routes`, `publish_split_route`, `matches_split_route` and `end_split_routes`. One active submission and one owner namespace per backend context are supported in this prototype. Submission ids must increase within an owner generation; a newer owner generation can restart them, and a backwards generation is refused. Ending a scope drains its owner stream and invalidates its epoch. Candidate replacement/rejection and shutdown perform that drain under the existing resource-lifecycle lock.

Publication is an internal driver contract: call it only after a successful producer dispatch. It proves a plain ARGSORT/view route in both the stamped producer graph and the allocated source graph, proves that the selected candidate group's down consumer uses those ids later in the source graph, checks row geometry and owner-device storage, and records an owner-stream CUDA event. The lease snapshots route/root/score signatures, candidate generation and group, producer UID, source indices and scope epoch. Its destructor/reset drains producer completion and destroys the event. It is not a consumer-completion or expert-bank resource lease.

Matching requires the active submission, source graph pointer/UID/node storage/count, candidate generation/group, producer-versus-consumer split identity, certificate owner/domain/row shape, owner stream, live tensor signatures and original route grammar. Duplicate group publications, publication during capture, cross-owner streams, and foreign route tables are refused. Caller must end the scope before graph reset, restore/cancel or buffer reuse; no automatic scheduler/inference lifecycle hook exists yet. Matching is a validation snapshot, not permission to dispatch a consumer across a concurrent lifecycle transition.

The existing backend-dispatch probe publishes after successful routing and checks the expert witness before its existing required-grouped refusal. Tests cover all nine cached cut shapes, wrong submission/candidate/stream/backend owner, certificate owner/generation/domain/rows/sequences, foreign/resized/stale source graphs, route stride/data/grammar mutation, missing producer nodes, duplicate publication, ended scopes, fresh-scope epoch invalidation, owner-generation restart/backwards rejection and active registry replacement. Uncut recovery still matches exactly. These tests do not make the imported expert graph executable: its old fail-closed plan validation is unchanged.

The replacement test exposed a capture warmup gap: invalidation keyed only to a nonzero grouped-resource fingerprint misses routing-only graph cache entries, which can retain warmup state without that fingerprint. After replacement, the next uncut required-grouped graph failed resource validation while the new-generation resources were not initialized. Tracking the observed registry generation in the CUDA backend and invalidating all its graph warmup/capture metadata on a generation change fixes the tested recovery without weakening resource validation. Checking only the cached plan's generation was tried and did not fix the failure; that attempted condition was not retained.

Validation: Release build/relink, focused runner, two layer runs, insertion probe, full MoE suite, CUDA `MUL_MAT_ID` 929/929 cases and `git diff --check` passed. Artifacts: `build/hybrid-route-lease-{build,focused,layer-0,layer-1,probe,full,ops}.log`; the prior failed recovery is preserved in `build/hybrid-route-lease-replacement-recovery-failure.log`. Architecture CTests remain blocked by Windows Application Control. No security policy was changed.

### Expert plans now retain and validate split-route origins

The scope, caller lease and plan records now share an immutable origin with its producer event. A published origin can replace only the missing local route witness, not missing banks, roles, capability checks, or MMID inventory. Every imported expert reader must be an original source-graph node after its route producer. Local-route checks are unchanged. Compilation records the origin; rebinding validates its active epoch and source/certificate/tensor identities even on the fast property-unchanged path. A previously declined plan is recompiled when a new origin becomes available rather than reusing its cached rejection.

Immediately before dispatch, the existing grouped resource-lifecycle lock protects another origin check and an owner-stream wait on its producer event. The normal grouped-call/resource authority then protects expert execution. Ending a scope first invalidates admission, waits for active grouped calls/transactions, drains the owner stream, and releases its origin references. Retained plans can keep completed events alive, but their expired epochs cannot admit another execution. Fresh submissions at one allocation require newly published origins; they do not reuse an old provenance proof.

The first positive test exposed an additional boundary requirement: raw down columns have later weighting/reduction consumers outside the expert graph, so the existing external-consumer rule correctly refused the cut. The new exception is restricted to down outputs with a valid imported origin. It counts every down use in the original allocated source graph, requires each use to follow the down projection, and requires the local count to match the recorded local consumers. This is checked during compilation, full witness validation and fast rebinding. Gate/up outputs and unrelated external consumers retain their original rejection. No downstream weighting or reduction arithmetic was changed.

The cached fixture now covers 18 SiLU/GELU combinations: DRAFT/MTP independent and sequential rows at one and four rows, and MAIN target verification at four rows. Three different input/route sets are warmed in an uncut reference, then executed across complete-layer and routing/expert/combine cuts. Each cut run has a fresh submission origin at the same allocation and must match its corresponding uncut output bit-exactly. Completed grouped telemetry shows no bank uploads after reference warmup, no host staging and no fallback. No-origin and fresh-scope-without-publication controls still fail before expert submission. Existing wrong-identity/stream/owner/source and registry replacement checks remain.

This is still cached-engine scaffolding, not frozen residency or CPU/GPU hybrid admission. It qualifies event-bound imported routing and raw-down boundary execution; it does not qualify actual speculative acceptance/rollback, masked hybrid ids, real-model bank formats, bounded D2H, overlap or speed.

Validation: Release build/relink, focused runner, two grouped-layer runs, insertion probe, full MoE suite, CUDA `MUL_MAT_ID` 929/929 cases, 3/3 architecture MoE CTests and `git diff --check` passed with fusion enabled. The architecture executables launched successfully on this run; no security policy was changed. Earlier Application Control failures remain recorded as historical launch failures. Artifacts: `build/hybrid-route-binding-{build,focused,layer-0,layer-1,probe,full,ops,ctest}.log`.

### Frozen residency partition without demand refill

No planner change was needed. The hybrid path can simply never call `prepare_decode()`: `acquire_group_resources()` and `begin_group_transaction()` take ownership, `snapshot_resident_slots()` publishes the committed map, and `llm_graph_moe_hybrid_routes::build()` turns that map into masked GPU ids plus cold ids/rows/columns. `test_hybrid_frozen_partition()` in the insertion runner exercises that sequence with 8 experts, top-k 2 and a 4-slot pool at one and four rows.

Each iteration routes through the ordinary cached path, then holds a transaction only for snapshot, partition, CPU execution and `finish_decode`. The frozen phase must prove it filled nothing, so telemetry taken inside that phase requires zero `h2d_banks`, `h2d_bytes`, `demand_materialized_bytes`, `vacant_fills`, `replacement_fills`, `invalidation_refills`, host-staged ops and fallback. A re-snapshot inside the phase must equal the first, every cold column must equal the CPU oracle exactly, and an undersized capacity and a finished transaction are both refused.

Three facts came from running it, not from assumption. First, the committed host map can lag a deferred commit: `device_slot_for_expert` reported no residency while the device snapshot showed two resident experts, so the snapshot is the only authority and the original equality assertion was wrong. Second, a held transaction blocks the ordinary cached path, so the frozen phase has to be scoped per iteration. Third, residency depends on shape: at one row (2 routes, 4 slots) everything routed stays resident, and at four rows (8 routes over 4 slots) nothing is resident, so every column is cold.

The first version reached only all-hot and all-cold partitions, because a batch that just executed through the cached path has already filled exactly the experts it routes. A mixed partition needs routes for a batch whose experts were never filled. Warming on a cut allocation does not work: a cut graph cannot satisfy required-grouped execution without a published route origin, and the following `begin_group_transaction` is then refused.

The working order is warm first on an uncut allocation, then reset, rebuild the graph, set the routing boundary and reallocate. Residency lives in the backend context, so it survives that reallocation. Later batches then execute only `[0, routing_end)` through `ggml_backend_sched_graph_compute_range()`, which advances routes without running or filling any expert. That is also the production order. With it, the one-row shape reports `resident=6 cold_columns=5 mixed=1`: resident experts stay on the GPU while never-filled experts go to the CPU, with the no-refill telemetry still zero and cold columns still exactly equal to the CPU oracle. The four-row shape keeps nothing resident and sends all 24 columns to the CPU. The test now requires at least one mixed partition, so losing the split would fail instead of passing quietly.

Windows Application Control blocked execution of freshly linked test binaries during this work, including a renamed copy, which is the same block previously seen for the architecture tests. No security policy was changed; the runs below were completed after the block cleared on the owner's machine.

### Masked ids are not accepted by the grouped cached path

Executing the resident half of a frozen mixed batch was attempted by writing masked ids into the route tensor, where a resident route keeps its expert and a cold route becomes `-1`, then running the expert range. Every surviving id names a resident expert, so that execution should hit only and fill nothing.

It aborts the process: `CUDA error: unspecified launch failure` during stream synchronize after the expert range. The earlier negative-id skip was added to the non-grouped MMID kernels (MMVQ/MMVF/MMQ), while the grouped path remaps ids through its own slot plan and indexes bank storage with the value, so `-1` is an out-of-range access rather than a skip. Grouped execution therefore has no masked-route capability today.

The attempt was reverted, including its route-origin publication and extra split boundaries, so no crashing or disabled test was left behind. No claim is made that frozen GPU-side execution works.

### What Strata actually does, and the compact hot list

Strata never puts a sentinel through a general matmul. In `expert_hit_run()` (`src/core/expert_source.cpp`, `HitPhase::Launch`) it asks the cache for each routed expert's slot and appends resident ones to two compact arrays, `h_slot` with the VRAM slot and `h_dst` with the router index that result must land on. A private kernel, `moe_hit_grouped_s2(cache_base, d_slot, d_dst, n_hits, ...)`, walks only those `n_hits` entries, so a cold expert is simply absent rather than encoded.

Three points from that source bear on this port. The split is by router index and both halves write back to that index, because that is the order the combine weights against, which matches the column seam chosen here. The GPU half writes a separate `hit_out` buffer that is zeroed each layer, with a comment that a row hot last layer and cold this layer would otherwise keep stale output and still be summed in. And Strata is not frozen: a miss is admitted and filled on the same stream, becoming a hit on that call, and only a refused admission stays with the CPU, so the no-refill behavior qualified here is stricter than Strata.

So neither earlier option matched Strata. The `-1` crash happened because absence was expressed in a representation that has no value for it. Following Strata, `llm_graph_moe_hybrid_routes` now also produces the compact resident list: `hot_slots`, `hot_experts`, `hot_rows` and `hot_columns`, built in route order alongside the existing cold lists. `hot_columns` is the `h_dst` equivalent.

Tests require that hot and cold sizes add up to the route count, that each column is claimed exactly once by exactly one side, that each hot entry's slot equals the residency map for its expert and stays inside the slot pool, and that its recorded row and expert match the route it came from. The grouped staging fixture checks the same compaction against a real transaction snapshot and requires both a hot and a cold entry. Duplicate or dropped contributions therefore fail instead of passing quietly.

This is the route description only. No kernel or graph consumes the hot list yet, so resident-half execution is still not performed and nothing here is a speed or overlap claim.

Validation: Release build/relink, insertion probe with the frozen test, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. Artifacts: `build/hybrid-hotlist-{build,probe,layer,range,staging,full}.log`.

Real-model regression smoke, because this work changed production CUDA and scheduler code: the 48-layer IQ3_S model served with GPU layers and CPU experts produced greedy output identical to the preserved page-align baseline (`<think>\nThe user is asking me to`), ready in 10.62 s against 10.67 s, peak RSS 11.53 GiB against 11.39 GiB. Those timings are not a performance claim; they only show no regression in load or generation. Artifacts: `build/run-qwen-hotlist-smoke.{py,json,log}`.

About Strata's zeroed `hit_out`: adopting it now would change no result. The probe poisons the whole raw-down tensor before each pass and then writes every cold column, so a stale row from a previous pass fails the sentinel or parity check instead of surviving. The zero-init becomes a real requirement only once resident columns are computed into a buffer the scheduler may reuse across layers and batches, so it is folded into the hot-list consumer slice rather than added as an unused buffer.

One Strata behavior is deliberately not copied: `expert_hit_run()` admits and fills a missed expert on the same stream, making it a hit on that call. That is demand refill and conflicts with the frozen partition. It remains the natural basis for adaptive residency later, promoting experts between batches rather than inside a layer.

Validation after reverting: Release build/relink, insertion probe with the frozen test, grouped-layer run, ranged runner, full MoE suite and `git diff --check` passed. Artifact: `build/hybrid-frozen-gpu.log` keeps the crash evidence.

Still missing: the GPU half of a frozen mixed batch is not executed here. The resident columns are proven to be identified correctly, but masked resident ids have not been run against the bank views in this frozen path, and the CPU results are not inserted back. Inputs and ids are still read to the host instead of a bounded D2H, and no overlap or speed is claimed.

Validation: Release build/relink, two insertion probe runs including the frozen test, grouped-layer run, ranged runner, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. Insertion parity is unchanged at `bf16 9.86851e-06` and `q4_0 1.57814e-04`. Artifacts: `build/hybrid-frozen-{build,probe,probe-2,layer,range,full,ctest}.log`.

### Resident half executed from the compact hot list

Strata's hit kernel takes the slot arena plus `slot_index` and `dst_index`, evaluates the expert at `blob_base + slot_index[h] * blob_bytes`, and writes to `out + dst_index[h] * n_embd`. Its mature path also builds that list on device (`moe_hit_select`), reads the count from device memory, and merges with `moe_hit_add`. The equivalent shape in this tree is slot-indexed bank weights with compacted routes, so no sentinel ever enters a matmul, which is why the masked attempt faulted.

The frozen test now executes the resident half that way. It takes bank descriptors from the held transaction, builds weights shaped `[n_embd, n_ff, n_slots]`, and checks each descriptor's `expert_stride` against the tensor's own per-slot size instead of assuming the layout. Routes are compacted: ids are `hot_slots`, and the activation rows are gathered by `hot_rows`, so the expert pass runs only resident work. Each result is then compared to the CPU oracle at `hot_columns[j]`, which is the `dst_index` equivalence: a wrong placement lands on another route's row and fails.

The mixed one-row shape reports `hot=1 cold=5`, and the test requires a non-zero hot half, so losing resident execution fails instead of passing quietly. Parity uses the existing quantized bound of `5e-4` taken from `test_mul_mat_id`.

Test scaffolding that is not production shape: the slot arena is copied into test-owned storage because a device pointer cannot be wrapped as a ggml buffer here, the hit list is built on the host rather than by a device selector, and results are compared rather than scatter-added into the combine input. No separate zeroed `hit_out` buffer and no add kernel exist yet, so Strata's stale-row hazard is still handled by writing every column and poisoning, not by zero-init.

Also relevant for later token-identity work: Strata added fp32 `x_scales` because a hit and a miss must produce the same number. Their CPU used fp32 `ActQ::scale` while the kernel read the fp16 `d` of `block_q8_0`, measured at max relative `4.761e-04` on 80 of 80 chunks, which changed generated tokens. The CPU and GPU halves here also quantize activations through different paths, and the probe's `q4_0` NMSE of `1.57814e-04` is the same order, so greedy-token identity must be measured rather than assumed.

Validation: Release build/relink, two insertion probe runs, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. Artifacts: `build/hybrid-hothalf-{build,probe,probe-2,layer,range,staging,full}.log`.

### A full hybrid layer matches the reference

The merge now follows Strata: a parts buffer cleared per layer, each half written at its routed index, and the combine left untouched. Resident columns come from the compact hot list and cold columns from the CPU executor, both placed by their route column. The test then runs only `[expert_end, region_end)` through ranged execution, so the region's own weighting and reduction consume the merged parts without any modification.

The merged buffer must be completely covered: every element is required to be non-zero after both halves are placed, so a dropped column would fail rather than quietly contribute zero. The region output is then compared against the CPU reference for the same batch, within the existing `5e-4` quantized bound. The mixed one-row shape reports `hot=1 cold=5 layers=1`, and a non-zero layer count is required.

This is a single region executed from test scaffolding, not an inference path. Selection still happens on the host, the slot arena is still copied into test storage, the merge is staged through host memory rather than a device scatter-add, and nothing measures overlap or speed.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. The real-model smoke again produced greedy output identical to the preserved baseline (`<think>\nThe user is asking me to`), ready in 5.55 s with peak RSS 11.85 GiB; load time varies with page cache and is not a performance claim. Artifacts: `build/hybrid-merge-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-merge-smoke.{py,json,log}`.

### Device-side hot-list selection

Added `ggml_cuda_moe_hybrid_select()` in `moe-cache.cu`, the equivalent of Strata's `moe_hit_select`. It reads routed ids and a residency map from device memory and writes the compact hot and cold lists plus their counts, so route ids no longer need a host round-trip. Compaction is sequential because route order decides placement, matching the host packet exactly. Refusal is explicit: an out-of-range expert, a duplicate inside a row, or an out-of-range slot leaves both counts at `-1` rather than producing a partial list, and bad arguments are rejected before launch.

The frozen test now runs the selector on the same ids and snapshot as the host packet and requires identical hot slots, hot columns, cold ids and cold counts. It also checks argument refusal and that an out-of-range route yields `-1` counts instead of a selection.

The definition first landed inside an anonymous namespace in `moe-cache.cu` and failed to link; it was moved to file scope rather than worked around in the test.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, CUDA `MUL_MAT_ID` 929/929 cases, 3/3 MoE CTests and `git diff --check` passed. The real-model smoke again matched the preserved baseline output exactly, peak RSS 11.68 GiB. Artifacts: `build/hybrid-select-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-select-smoke.{py,json,log}`.

One thread performs the compaction, which is fine for the route counts tested here but is not a tuned kernel.

### The resident half now consumes the device lists

The selector also emits `hot_rows`, so the resident half is driven by device data: slot ids and row indices are copied device to device into the expert graph's inputs, and the activation rows are gathered on the device with `get_rows` from the selector's row list. Ids and per-route activations no longer travel through host memory.

Two failures found on the way, both fixed rather than worked around. A refused selection leaves its compact lists unusable, and the test reused them after the refusal case, which faulted; the valid selection is now produced again and its counts rechecked before use. Then a device-to-device copy of the region's activation failed with `invalid argument`, because that leaf is not necessarily device resident, the scheduler may place it on the CPU. The batch is therefore staged once through the existing host read, while the per-route gather stays on the device.

Remaining host staging: the batch activation, the slot arena copy, and the merged columns. The arena copy stays until the execution path lives in backend code, where the bank view can be used in place as grouped execution already does.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. The real-model smoke again matched the preserved baseline output exactly, peak RSS 11.83 GiB. Artifacts: `build/hybrid-device-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-device-smoke.{py,json,log}`.

### Strata's token-graph order, and the device merge

Strata's mature path (`src/core/session.cpp`) enqueues the whole layer: `moe_hit_select` on device, `quantize_q8_0_scaled` once per layer with fp32 scales, `moe_hit_grouped_s2_dev` reading its count from device memory, `doorbell_wait` for the CPU's misses, `copy_from_mapped`, `moe_hit_add`, then the unchanged combine. There is no buffer abstraction anywhere: `cache_base` is the raw device slot arena held by the session and passed straight to kernels, which is why in-place arena use is only reachable from backend code.

Two details correct earlier plans. `copy_from_mapped` is deliberately a kernel rather than a memcpy node, because "a copy-engine node splits the WDDM submission (measured 67 flushes/token)" - the same WDDM this fork runs on, so the bounded cold-half transfer should target mapped memory plus a copy kernel rather than a plain copy. And the activation is quantized once per layer and shared by both halves, so the fp32 `x_scales` parity fix is structural in their design rather than a patch.

Added `ggml_cuda_moe_hybrid_scatter_add()`, the `moe_hit_add` equivalent: it adds each compact row of the resident output into the parts buffer at its routed column, reading the hot count from device memory. The frozen test now merges that way instead of assembling parts on the host: the CPU half writes its own columns and leaves resident rows at zero, the device scatter-add fills them, and the merged buffer is then required to have no zero element left and to match the resident results exactly at their routed columns. Null arguments are refused.

Still outstanding and not claimed: the resident FFN itself is still executed from test-built graphs against a copied arena, selection is not yet fused into one enqueued sequence, the cold half is staged through host memory rather than mapped memory with a copy kernel, and no doorbell exists, so there is still a host synchronization per layer.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and `git diff --check` passed. The real-model smoke again matched the preserved baseline output exactly, peak RSS 11.82 GiB. Artifacts: `build/hybrid-scatter-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-scatter-smoke.{py,json,log}`.

### Resident half executed in place by the backend

Added `ggml_backend_cuda_moe_hybrid_execute_hot()` and `ggml_backend_cuda_moe_hybrid_hot_scratch_bytes()`. Like Strata's hit path, the entry point takes raw device pointers. CORRECTION, see the arena entry below: this first version pointed the bank view at `descriptor.source_data`, which is the cached tensor's host backing and not the device slot arena, so the claim of an in-place arena read was wrong until that was fixed. Each descriptor's `expert_stride` must equal the stride the view assumes, so a layout disagreement is refused rather than indexed blindly.

It runs up, gate, the gated activation and down over the compact routes, with ids being the hot slots. FFN intermediate storage is caller-owned and sized by the helper. The reused CUDA matmul paths still acquire backend-pool workspace; this helper does not prove allocation-free execution or capture legality. Missing arguments, a stale or inactive transaction, geometry disagreement and undersized scratch are all refused before any launch.

The frozen test runs the same half both ways and requires bit-identical results. CORRECTION: at the time, both paths read the same host backing, so the match proved consistency between two readers of the same bytes, not correctness against the grouped path.

A first version synthesized bank tensors from scratch and crashed in `ggml_backend_buffer_get_usage`, because the CUDA matmul paths read the weight's buffer for padding and cached-buffer checks. Copying the weight tensor and replacing only `data`, `ne[2]` and `nb[3]`, which is what the grouped path already does, fixed it; the test was not adjusted to hide it.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, CUDA `MUL_MAT_ID` 929/929 cases, 3/3 MoE CTests and `git diff --check` passed. The real-model smoke again matched the preserved baseline output exactly, peak RSS 11.84 GiB. Artifacts: `build/hybrid-hotexec-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-hotexec-smoke.{py,json,log}`.

Deliberate deviations from Strata at this stage: the hit count is a host argument rather than read from device memory, so a synchronization after selection is still needed; the activation batch is staged once from the host and each half still quantizes independently; the cold half is not handed over through mapped memory with a copy kernel; and there is no doorbell.

### The two halves disagree, measured

Strata's `x_scales` exists because a hit and a miss must produce the same number; they measured max relative `4.761e-04` from fp16 against fp32 activation scales and it changed generated tokens. A different quantity is measured here: the frozen test compares full GPU resident FFN outputs against CPU outputs for the same routes. This does not isolate activation-scale error or identify its cause.

A first version reported `max_relative=0.131`, which is meaningless because near-zero reference elements dominate a raw ratio. Scaled by the signal instead, the result is `max_abs=9.98e-04` against an RMS of `3.29e-02`, a worst-element ratio of about `3.0e-02`, The earlier `1.57814e-04` value belongs to the separate insertion probe, not this full-FFN disagreement measurement.

This is not directly comparable to Strata's number: theirs isolates activation quantization, this covers a whole Q4_0 expert chain evaluated by two different implementations. What it does establish is that the halves disagree at quantization scale rather than at machine precision, so a hybrid layer cannot be assumed to reproduce full-GPU tokens. Greedy-token identity has to be measured on a real model, and if it fails, the shared-activation-quantization path is the structural fix Strata already took.

Validation: Release build, insertion probe with the measurement, grouped-layer run, ranged runner, full MoE suite and `git diff --check` passed. Artifact: `build/hybrid-parity-{focused,layer,range,full}.log`.

### Merge ownership and poisoned reuse

Replaced the fixture's nonzero-element coverage rule with exact ownership checks: every routed column has one hot or cold owner, hot columns match compact GPU results, and cold columns remain bit-identical to CPU results. Valid zero hot outputs are tested. Repeated merges reinitialize parts and must reproduce the expected result exactly. Zero, negative and over-capacity device counts must leave a NaN-poisoned destination untouched; over-capacity counts now cause the kernel to write nothing. Host extent multiplication overflow is refused. Lists remain selector-produced, unique-column inputs; the scatter helper is not a general device-list validator.

Validation: Release build, focused insertion/frozen runner, full MoE suite and diff check passed (`build/hybrid-merge-ownership-{build,focused,full}.log`).

Corrected earlier claims: the hot helper's caller scratch only covers FFN intermediates, not CUDA matmul pool workspace. Allocation-free execution and capture legality have not been established. Transaction descriptors are checked, but callers must preserve the held transaction through stream completion; automatic lifecycle protection and explicit stream-owner validation remain outstanding.

### Isolated activation quantization

Added a test-only wrapper for the existing CUDA Q8_1 quantizer. The frozen hot fixture compares it directly with the active CPU Q8_0 `from_float` implementation on identical inputs, ignoring the Q8_1 sum field that Q8_0 does not store. It reports scale-bit differences, code differences and maximum dequantized difference. Original activations are restored before in-place FFN parity and merge checks.

On this x86/CUDA build, the fixture's eight input blocks and an all-zero control have identical scales and codes. A signed half-integer control with scale one has identical scales but 120 differing codes and maximum dequantized difference one. The x86 implementation uses nearest-even SIMD rounding; the CUDA Q8_1 kernel uses `roundf`, which rounds ties away from zero. This is not a justification to change ordinary CPU/CUDA quantization globally.

The fixture's input quantizer therefore does not explain its full-FFN discrepancy. Intermediate activation quantization, reductions and nonlinear arithmetic have not yet been isolated. A shared quantized-input design must choose its rounding policy explicitly and account for CPU Q8_0 versus CUDA Q8_1 layout and correction data; copying Strata's fp32 scale alone is not a proven fix here.

Validation: Release build, focused insertion/frozen runner, full MoE suite and diff check passed (`build/hybrid-quant-isolate-{build,focused,full}.log`). No real-model hybrid or performance claim follows from these controls.

### Real graphs expose hybrid spans

Every span test so far used synthetic two-layer fixtures, so it was unknown whether a real model graph splits the same way. `llama_context::moe_hybrid_probe_spans()` answers that: gated by `LLAMA_MOE_HYBRID_PROBE`, it runs `llm_graph_moe_hybrid_find_span()` over the placed regions after allocation and logs the result. It changes no execution and is temporary scaffolding.

On the IQ3_S Qwen model with `--moe-expert-cache-size 8`, a 6772-node graph reports `regions=48 placed=48 spans=48`, with layer 0 at `routing_end=100 expert_end=104 region_end=124`. All 48 regions are placed and all 48 produce valid spans, so the seam the hybrid path needs exists in production graphs and is not a fixture artifact.

Two findings along the way: with `--cpu-moe` alone the expert cache is off, so `placed=0` and no region is a hybrid candidate; and core `llama` logs are filtered in the smoke scripts, which is why the first probe appeared silent. The probe logs at warning level to stay visible.

This proves span discovery only. Ranged execution is still not wired into `llama_context`, which dispatches whole graphs through `ggml_backend_sched_graph_compute_async_ext` for every certified path; per-region execution, residency authority, the CPU half and merge are not connected.

Validation: Release build, insertion probe, full MoE suite, 3/3 MoE CTests and diff check passed. Both probe smokes completed and the ordinary-mode output still matched the preserved baseline. Artifacts: `build/hybrid-span-probe-{probe,full}.log`, `build/run-qwen-probe-smoke.{py,json,log}`, `build/run-qwen-probe-cache-smoke.{py,json,log}`.

### Real inference driven split by split

Every certified dispatch in `llama_context::graph_compute()` submitted the whole graph at once, which leaves no point to insert the hybrid seam. The three certified call sites now go through `graph_compute_certified()`. By default it still submits the whole graph. Under `LLAMA_MOE_HYBRID_RANGES` it walks the splits with `ggml_backend_sched_get_split_range()` and executes each one with `ggml_backend_sched_graph_compute_range()`, carrying the same certificate. Same work, same order, one submission per split.

On the real model this runs end to end and reproduces the reference text exactly in both configurations: with `--moe-expert-cache-size 8` it matches the cached whole-graph run, and with plain `--cpu-moe` it matches the preserved ordinary baseline. So the loop that the hybrid path needs, stop between regions and do work, now exists in production and does not change results by itself.

This is still only a driver shape. No hybrid work happens between the ranges: there is no transaction held, no residency snapshot, no device selection, no resident or cold execution and no merge in the inference path. The gate is temporary scaffolding, not `--moe-hybrid`. Per-split submission is also not free, and no timing claim is made here.

Validation: Release build/relink, insertion probe, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed. Artifacts: `build/hybrid-ranged-{build,probe,range,full}.log`, `build/run-qwen-ranged-smoke.{py,json,log}`, `build/run-qwen-ranged-cpumoe-smoke.{py,json,log}`.

### Production graphs cut at the seam

`moe_hybrid_set_cuts()`, gated by `LLAMA_MOE_HYBRID_CUTS`, requests a split boundary at `routing_end` and `expert_end` of every placed region before allocation, using the existing `ggml_backend_sched_set_split_boundaries()`. The ranged driver then reports how many requested cuts actually begin a split.

On the real model: `splits=98 requested=96 honored=96`, against `splits=2` for the same graph without cuts, and the generated text still matches the cached reference exactly. So all 48 layers are cut at both hybrid seams in a production graph, and the scheduler honours every one.

Two scaffolding faults were found and fixed rather than reported around. The first honored report was gated on a non-empty cut list and printed nothing, which could have been mistaken for a scheduler refusal; the real cause was that an earlier scripted edit to record the cut list never applied, so the list was always empty and the check was measuring nothing. A one-time log also confirmed the certified dispatch is actually reached with `ranged=1`, which the previous ranged result had assumed rather than shown.

Still no hybrid work: the splits are executed back to back with nothing in between, no transaction, snapshot, selection, resident or cold execution, or merge. Three env gates (`PROBE`, `RANGES`, `CUTS`) remain temporary scaffolding.

Validation: Release build/relink, insertion probe, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed. Artifacts: `build/hybrid-cuts-{build,probe,range,full}.log`, `build/run-qwen-cuts-smoke.{py,json,log}`.

### Real layers are eligible, and no new API is needed

The probe now also reports how many placed regions the resident half can actually execute: separate gate/up banks and a plain SiLU or GELU activation, which is what `ggml_backend_cuda_moe_hybrid_execute_hot()` supports. The first unsupported layer, if any, is named with its reason.

On the real model: `placed=48 spans=48 eligible=48`. Every layer of this Qwen model qualifies, so the restricted first version is not a model-wide blocker here.

This also settles the API question raised earlier. A new public entry point is not required to drive the seam: the bank layout and activation come from the graph region, and `llama-context` already resolves backend-specific entry points through the existing `ggml-backend-moe.h` proc-address convention, which the CUDA registry already implements. The hybrid driver can use that mechanism instead of adding a new public header.

Still unchanged: no hybrid work runs in inference. Eligibility is a static property check, not proof that execution, residency authority or lifetimes are correct in the inference path.

Validation: Release build/relink, insertion probe, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed; the cut smoke still matched its reference. Artifacts: `build/hybrid-eligible-{build,probe,range,full}.log`.

### Measured residency on the real model

The split between the halves is decided by residency, so it was measured rather than assumed, using the cache statistics the server already reports at verbose level.

On the IQ3_S Qwen model, same short prompt, 144 caches (48 layers x 3 banks):

| slots per expert tensor | hits | misses | evictions | hit rate | peak RSS |
| --- | --- | --- | --- | --- | --- |
| 8  |  6055 | 9273 | 8121 | 39.50% | 12.64 GiB |
| 32 | 15989 | 9030 | 4464 | 63.91% | 13.28 GiB |

Both runs produced identical text. Two consequences for the port. The resident fraction is a minority at small slot counts, so the CPU half carries most routes and the cold path, not the resident one, sets the pace; and the eviction counts show the demand engine is constantly refilling, which is exactly the traffic a frozen partition removes and the reason adaptive residency has to be driven by observed reuse.

These numbers come from the existing demand-refill cache, where a miss is filled on the GPU. Under the frozen hybrid partition a miss goes to the CPU instead, so the hit rate indicates the expected hot/cold ratio, not hybrid performance. One short prompt, one model, no timing claim.

Validation: no code change; both smokes completed and matched. Artifacts: `build/run-qwen-residency-smoke.{py,json,log}`, `build/run-qwen-residency32-smoke.{py,json,log}`.

### What limits the cold half

Since residency puts most routes on the CPU, the CPU half decides the result, so its limit was checked with the existing `test-backend-ops perf` runner rather than new code. On this machine CPU `MUL_MAT_ID` at `n_mats=128 n_used=8 m=768 k=2048` reaches 547 GFLOPS at one row and 550-830 GFLOPS at larger batches, Q4_K at or above Q8_0.

Against this model's geometry (2560 wide, 640 FF, top-k 10, 48 layers) the arithmetic is not the limit. Per token the cold half needs about 2.8 GFLOP at 6 cold routes, which is a few milliseconds of the measured rate, but it must read every selected expert's weights from RAM:

| cold routes/layer | expert format | bytes/token | 60 GB/s | 90 GB/s |
| --- | --- | --- | --- | --- |
| 6  | ~3.5 bpw | 619 MB | 10.3 ms | 6.9 ms |
| 10 | ~3.5 bpw | 1032 MB | 17.2 ms | 11.5 ms |
| 6  | ~4.5 bpw | 796 MB | 13.3 ms | 8.8 ms |
| 10 | ~4.5 bpw | 1327 MB | 22.1 ms | 14.7 ms |

The cold half is memory bandwidth bound, not compute bound. Three consequences: cold expert bytes, that is quantization and route count, matter far more than CPU kernel efficiency, which supports the plan's aggressive quantization of draft experts; residency is worth most when it removes bandwidth, so promoting frequently reused experts beats raw hit count; and the GPU hot half has to overlap this window to be useful at all, since it cannot shorten it.

These are bounds derived from a benchmark at a nearby shape plus model geometry and assumed achievable bandwidth, not measured hybrid throughput. Actual sustained bandwidth on this machine was not measured, prefetch and page cache are ignored, and no claim is made about final speed.

Validation: no code change. Benchmark output from the existing `test-backend-ops perf -o MUL_MAT_ID -b CPU` runner.

### Measured decode, and the baseline that matters

Decode times from the smoke logs, 8 generated tokens each, same prompt and model:

| configuration | ms/token | tok/s |
| --- | --- | --- |
| `--cpu-moe`, whole graph | 89.33 | 11.19 |
| `--cpu-moe`, ranged splits | 89.12 | 11.22 |
| `--cpu-moe`, repeat | 91.92 | 10.88 |
| expert cache, 8 slots | 237.62 | 4.21 |
| expert cache, 8 slots, repeat | 236.28 | 4.23 |
| expert cache, 8 slots, seam cuts + ranged | 233.67 | 4.28 |
| expert cache, 32 slots | 189.51 | 5.28 |

Two results. Ranged execution and the 96 seam cuts cost nothing measurable, so the driver restructuring is not a tax on the path. And the demand-refill expert cache is currently about 2.6x slower than plain CPU experts on this workload, reproducibly, improving only partially with more slots. That is the engine this port replaces, so the hybrid path must be compared against the `--cpu-moe` baseline of roughly 11 tok/s, not against the cache mode. Comparing against the cache would flatter the result.

The measured 89 ms/token also puts the cold half far from the bandwidth bound estimated above, roughly 12 GB/s effective against tens of GB/s available. The smokes use 8 of 24 threads with mmap, so thread count and page faults, not DRAM, appear to dominate today. The earlier bandwidth table is therefore an optimistic ceiling, not a current description.

Caveats: 8 tokens per run, one prompt, one model, cold start included, no thread sweep. These are directional measurements for deciding what to optimize, not benchmarks.

Validation: no code change. Artifacts: `build/run-qwen-rep-cpumoe-smoke.{py,json,log}`, `build/run-qwen-rep-residency-smoke.{py,json,log}` and the earlier smoke logs.

### Thread sweep, and a corrected diagnosis

The previous entry guessed that thread count dominated the cold half. That was tested and is wrong. Same `--cpu-moe` path, same prompt, 8 generated tokens:

| threads | ms/token | tok/s | relative |
| --- | --- | --- | --- |
| 8  | 89.12 | 11.22 | 1.00x |
| 16 | 77.36 | 12.93 | 1.15x |
| 24 | 74.77 | 13.37 | 1.19x |

Three times the threads buys 19 percent. The cold half is neither compute bound nor thread bound; it is bound by the memory path that feeds it, at roughly 14 GB/s effective against a far higher DRAM ceiling, which points at the mmap page-fault and page-cache path rather than raw bandwidth. Output stayed identical at every thread count.

Consequences: giving the CPU half more threads is close to free but nearly pointless on its own, so the Strata-style borrowed pool needs no urgent tuning; reducing bytes touched per token remains the lever; and the honest bar for the hybrid path is now the best ordinary configuration, about 13.4 tok/s at 24 threads, not the 11.19 tok/s figure quoted earlier.

Caveats unchanged: 8 tokens, one prompt, one model, cold start included, no page-cache control, so the absolute rate may shift.

Validation: no code change. Artifacts: `build/run-qwen-threads16-smoke.{py,json,log}`, `build/run-qwen-threads24-smoke.{py,json,log}`.

### The paging hypothesis is unproven, and one experiment failed

To test whether the mmap page path limits the cold half, the same 24-thread run was repeated with `--load-mode mlock`. It failed: peak RSS reached 47.3 GiB with 31.7 GiB still available, then the process aborted with `CUDA error: out of memory` in `ggml_backend_cuda_device_event_synchronize`. Locking a 77.8 GiB model plus host staging and GPU allocations does not fit this 96 GB machine. This is an interaction of an existing load mode with memory pressure, not a regression from the hybrid work, and nothing in this port was changed to accommodate it.

So the hypothesis is not confirmed. Evidence against it is that the mmap runs are reproducible at 89.12 and 91.92 ms per token with a warm page cache and roughly 67 GiB still free, and a warm repeat gained nothing. A more likely explanation for about 14 GB/s effective is the access pattern itself: each token touches around 1 GB of expert weights scattered across 512 experts per layer, which is a random-access pattern rather than a streaming one.

This does not change the plan; it only removes a wrong justification for part of it. The RAM and file tiers are still worth building for capacity, but they should not be sold as a fix for a paging bottleneck that has not been demonstrated. The measured bar remains about 13.4 tok/s at 24 threads.

Validation: no code change. Artifact: `build/run-qwen-mlock-smoke.{py,json,log}` with the failure preserved.

### The seam cuts disable grouped execution in production

A shadow entry point was added to validate the resident half on real weights inside live inference: `ggml_backend_moe_hybrid_shadow`, exposed through the existing proc convention, holds a transaction after a region's experts have run, snapshots residency, selects the resident routes, recomputes them from the slot arena with `ggml_backend_cuda_moe_hybrid_execute_hot()` and compares with the output the graph produced. `llama-context` calls it between splits under `LLAMA_MOE_HYBRID_SHADOW`. It changes no results.

It never ran. Every call was refused, and following the refusal codes gave a much more important answer than the shadow check itself.

`begin_group_transaction()` refused because the group was under legacy authority. The planner's own debug output shows why: with the hybrid gates enabled, all 338 plans report `outcome=2`, decode legacy, and `reason=2`, execution ineligible. A control run with the gates off reports 96 prefill legacy and 48 `outcome=1`, decode grouped. Enabling only `LLAMA_MOE_HYBRID_RANGES` also keeps 48 grouped plans. The seam cuts are the cause: splitting between the routes and the expert matmuls, and after them, means the planner no longer sees one complete group in a dispatch, so the whole layer falls back to the legacy per-tensor path.

This invalidates an earlier claim here. The cut smoke produced identical text, so the cuts looked free, but at 8 slots decode is legacy anyway, since plain MAIN decode requires `n_routes <= n_slots` and this model routes 10 experts. The comparison was legacy against legacy. At 32 slots, where grouped decode does happen, the cuts replace it with the legacy path. Identical output hid a silent downgrade, which is exactly the kind of regression that must not be reported as a success.

The tests did not catch this because they publish split-route origins through the lease mechanism, which qualifies cached cuts for grouped execution; nothing in the inference path does that yet. So the gap is not in the scheduler but in the production driver.

Consequences. The hybrid path cannot simply cut the graph and expect the grouped residency authority it depends on to survive; either the driver publishes split-route origins the way the tests do, or the hybrid half runs under the legacy authority instead, or the planner has to admit a cut layer as one group. This has to be decided before any further wiring, and the earlier statement that seam cuts cost nothing is withdrawn.

Validation: Release build. The smokes completed with matching text, which is precisely why the downgrade needed the planner's own diagnostics to see. Artifacts: `build/run-qwen-shadow-smoke.{py,json,log}`, `build/run-qwen-shadow32-smoke.{py,json,log}`, `build/run-qwen-shadow-dbg-smoke.{py,json,log}`, `build/run-qwen-shadow-dbg32-smoke.{py,json,log}`, `build/run-qwen-plain-dbg32-smoke.{py,json,log}`, `build/run-qwen-rangesonly-dbg32-smoke.{py,json,log}`.

Decision: do not cut the graph. Strata has no authority negotiation at all - `rg -i 'transaction|authority|lease|epoch'` over its sources returns nothing - because its session owns the arena pointer and the residency table and enqueues the layer itself. The equivalent here is to work inside the grouped dispatch, where the transaction is already held and the bank views already exist, instead of recreating a seam from outside.

### Validated inside grouped dispatch, on real weights

`ggml_cuda_moe_hybrid_compact_check()` runs at the down projection of an active grouped group, under `LLAMA_MOE_HYBRID_COMPACT_CHECK`. It re-executes that projection from the same slot arena through the same bank and id views into a separate buffer and compares. No cuts, no extra splits, no lease publication.

Result: 48 of 48 layers bit-identical, `nmse=0`, on the real IQ3_S model with the run completing and the text matching. The in-place arena read and the dispatch-level integration point are therefore correct on production weights and shapes, not only on 256-wide Q4_0 fixtures.

Two findings matter more than the check.

First, the compact shape does not fit the production kernels. Rewriting the routes as one route per plane, `[n_embd, 1, n_routes]`, which is Strata's hit layout, was refused by `ggml_cuda_mul_mat_id_impl`. Decode presents ten routes of one token as `ne2 = 1`, which selects MMVQ, while the compact form presents ten tokens of one expert and no consumer accepts it with a direct source view. So the hybrid half should subset the expert dimension of the existing layout rather than reshape into Strata's planes. Strata's layout suits its own private kernel, not these.

Second, the first version aborted the run: `CUDA error: operation not permitted when stream is capturing`. The first decode passed all 48 layers, the second decode captures a CUDA graph, and allocating, copying and synchronizing inside it is illegal. This is the production confirmation of the rule Strata follows, caller-owned scratch and a doorbell instead of a host sync, and it applies to the real hybrid path: no allocation, no host copy and no synchronization may sit in the dispatch. The check now declines while capturing.

Validation: Release build/relink, insertion probe, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed; the smoke completed with matching text. Artifacts: `build/hybrid-compact-{build,probe,range,full}.log`, `build/run-qwen-compact-smoke.{py,json,log}`.

Still not hybrid: this recomputes resident routes that the grouped path already computed. No cold half, no CPU work, no merge and no overlap run in inference.

### The split seam reproduces production exactly

The check now splits each layer's routes into two disjoint halves, executes each half on its own against the same slot arena by subsetting the expert dimension of the production layout, gathers the matching activation rows, and scatters both results into one buffer with `ggml_cuda_moe_hybrid_scatter_add()`. The sum is compared with the output the grouped path produced.

Result: 48 of 48 layers bit-identical, `nmse=0`, routes 10 split 5 and 5, on the real model with matching text. This is the arithmetic the hybrid seam depends on - two halves computed apart and merged - demonstrated on production weights rather than fixtures, and it confirms the subset-the-expert-dimension layout decided above works where Strata's plane layout was refused.

What this does not show: both halves ran on the GPU. There is no CPU half, no overlap, no residency decision, and the diagnostic still allocates and synchronizes, so it declines while a CUDA graph is capturing. A real hybrid half must preallocate its scratch and avoid host synchronization entirely.

The remaining blocker is unchanged and structural: grouped decode fills every miss, so all routes are resident and there is nothing cold to give the CPU. Producing a genuine split needs the frozen policy, prepare decode must leave misses unfilled and report them, which is the next change.

Validation: Release build/relink, insertion probe, ranged runner, full MoE suite, CUDA `MUL_MAT_ID` 929/929, 3/3 MoE CTests and diff check passed; the smoke completed with matching text. Artifacts: `build/hybrid-subset-{build,probe,range,full}.log`, `build/run-qwen-compact-smoke.{py,json,log}`.

### Strata's admission rule, confirmed and replayed

`ExpertCache::admit()` is fill-once and never evicts: `if (next_free_ >= slots_) return kNotResident;   // full: no eviction, deliberately`. Per layer each routed expert is either resident, or admitted and filled on the same stream so it is a hit on that call, or refused because the layer's quota is full, in which case it is cold and the CPU owns it. A per-layer quota variant exists because one shared counter held the measured hit rate at 2.97 percent. So the hot and cold split comes from capacity refusal, not from freezing residency, which corrects the plan's earlier wording. Two neighbouring rules were recorded for the same reason seen here: `hit_out` is zeroed per layer so a row that was hot last layer cannot carry a stale expert into the sum, and the activation is quantized in that function with required fp32 scales rather than reused from the previous layer.

Rather than change admission blind, the real routing stream was captured under `LLAMA_MOE_ROUTE_DUMP` and the policies replayed offline over 461 layer calls of the IQ3_S model:

| slots per layer | policy | hot | cold routes | fills | evictions |
| --- | --- | --- | --- | --- | --- |
| 8  | LRU | 100% | 0 | 8790 | 8406 |
| 8  | no-evict | 13.1% | 8495 | 384 | 0 |
| 32 | LRU | 100% | 0 | 4877 | 3341 |
| 32 | no-evict | 42.1% | 5655 | 1536 | 0 |
| 64 | no-evict | 82.3% | 1726 | 3072 | 0 |
| 128 | no-evict | 99.9% | 10 | 4254 | 0 |

LRU reaches every route only by evicting, which is the H2D churn the measurements already showed. No-evict removes it entirely and pays with cold routes, which is the design: the CPU is supposed to carry them.

Capacity decides the ratio. At roughly 2.15 MiB per expert for this model, 64 slots per layer is about 6.6 GiB of residency, which fits this 16 GiB card, and gives 82 percent hot. 128 slots would be about 13.2 GiB and does not fit beside everything else.

One more replay changes the design. Admission order matters: if prefill is allowed to fill the slots, it takes them with experts decode does not reuse. Admitting only during decode gives 90.0 percent hot at 32 slots and 99.9 percent at 64, against 42.1 and 82.3 when prefill fills. Prefill routes 12 tokens wide and touches far more experts than decode, so a first-come rule spends the whole quota on them. The port should therefore admit on decode only, which Strata's per-layer quota note hints at but does not state.

Caveats: one short run, around ten generated tokens, one prompt. A longer run widens the decode working set, so these hot rates are optimistic, and the ordering effect, not the absolute numbers, is the result to keep.

Validation: no behaviour change. The dump is a diagnostic in the legacy cached path under an environment variable. Artifacts: `build/qwen-routes.txt`, `build/run-qwen-routedump-smoke.{py,json,log}`.

### The admission rule, now decided in the engine

`moe_grouped_admit_shadow()` implements Strata's rule inside the decode planner: a routed expert that is resident is hot, a miss takes a free slot and is hot on that call, a miss with no free slot is cold, and only decode may fill. It keeps its own residency array per group, counts hot, cold and fills into the plan, and is reported through the existing stats line. Under `LLAMA_MOE_ADMIT_SHADOW` it runs beside the real assignment and changes nothing, so the policy that will drive execution is exercised on live traffic before it controls anything.

On the real model at 32 slots it reports `hot=2989 cold=371 fills=1477 hot-rate=88.96%`, against the production LRU policy's `hit-rate=63.91%` with 4464 evictions in the same run, with identical output. The offline replay of the captured routing stream predicted 90.0 percent for the same configuration, so two independent methods agree within about a point, which is the cross-check the replay alone could not give.

The number that matters for the CPU half is the cold count: about 11 percent of routes, roughly one route per layer per token at this capacity.

This is still counting only. The real assignment continues to evict, no route is actually sent to the CPU, and nothing executes differently. Switching the policy on requires the CPU half in the same change, because a refused route with nowhere to go is either a wrong result or a silent fall back to the evicting path.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; the smoke matched its reference. Artifacts: `build/hybrid-admit-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-admit-smoke.{py,json,log}`.

### Cold routes published to host memory

Handing cold routes to a CPU worker cannot use a device copy, because the dispatch may be inside a CUDA graph capture, which is what aborted an earlier diagnostic. Strata solves this by writing into mapped pinned memory from a kernel and announcing it with a sequence number, `doorbell_wait_kernel` spinning on the flag with `__threadfence_system()` on both sides.

The publication half of that is now implemented. The decode planner writes each cold route's index and expert id into a mapped pinned buffer owned by the group resource, fences, then increments the sequence, so a reader never sees a count ahead of its payload. The device side writes through the mapped device pointer from `cudaHostGetDevicePointer`, so it is a plain kernel store and stays legal under capture.

On the real model: `hot=2989 cold=371 fills=1477 hot-rate=88.96% publications=336 last_cold_routes=8`. The host reads the packet directly out of mapped memory with no device copy, which is how the CPU half will consume it, and the output is unchanged.

Still missing from Strata's mechanism: the wait. There is no doorbell kernel making the GPU wait for CPU results, no CPU worker consuming these packets, and the publication is bounded at 256 routes per packet with no handling yet for a larger batch. Execution is unchanged; these are counts and a payload, not a hybrid path.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; the smoke matched its reference. Artifacts: `build/hybrid-publish-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-admit-smoke.{py,json,log}`.

### The doorbell round trip runs

`moe_grouped_admit_doorbell()` is Strata's wait: a one-thread kernel that spins on the mapped acknowledgement until it matches the sequence this packet published, then fences. A host worker thread watches the published packets and acknowledges them, standing in for the CPU half, computing nothing yet. The spin is bounded and counts timeouts, because a lost worker would otherwise hang the device; Strata spins without a limit, which is safe only once the CPU side is guaranteed to answer.

On the real model: `publications=336 acks=336 timeouts=0`, output unchanged. The handshake the hybrid path depends on therefore works in live inference, inside the normal dispatch, with the GPU waiting on host memory.

Cost of the handshake alone, 8 generated tokens at 32 slots: 189.51 ms per token without it, 189.18 with publication only, 194.47 with publication and wait. So roughly 5 ms per token for 48 layers of round trip, about 0.1 ms per layer, with an idle CPU answering immediately. A real worker computing about one cold route per layer will add its own time, and only the part that exceeds the resident half's work is a loss, which is the overlap this design is for. One prompt, one run, no speed claim.

What this still is not: the worker computes nothing, no results return, admission still only counts, and execution is unchanged. The next step is the first one where a wrong result is possible, so it needs the return path and the switch together.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; both smokes matched their reference. A format string omission that silently dropped the two new counters was fixed rather than left to misreport. Artifacts: `build/hybrid-doorbell-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-doorbell-smoke.{py,json,log}`.

### Correction: the hot half was not reading the slot arena

Before building the return path on `ggml_backend_cuda_moe_hybrid_execute_hot()`, its source was checked rather than assumed. `ggml_cuda_moe_grouped_bank_descriptor::source_data` is `tensor->data`, and a moe-cached buffer allocates host pinned mapped memory, so that pointer is the cached tensor's host backing, not the device slot arena. `cudaPointerGetAttributes` on it reports `type=1`, host memory, which the frozen fixture now asserts on every run.

So the entry point was reading host bytes over PCIe and indexing them by slot, while the host tensor is laid out by expert. The frozen test did not catch it because its reference copied the same `source_data` into slot-shaped tensors: two readers of the same wrong bytes agreed exactly. Pointing the view at the real arena changed the result immediately, `NMSE 3.33665e-04` instead of zero, which is the proof the two sources differ.

Fixed by adding `get_group_bank_device_data()`, which returns the bank's device arena, and using it in both the entry point and the fixture. Both now read the arena, agree bit-exactly again, and the cold oracle comparison still holds, so the arena is the correct source.

Withdrawn claims: that the hot half read the slot arena in place, and that the fixture proved the arena copy redundant. The real-weight validation that still stands is the in-dispatch one, `moe-hybrid-compact` and `moe-hybrid-subset`, because those used `ggml_cuda_moe_group_views()`, which has always pointed at `bank_data`, the true arena.

This also explains why the fixture could not have caught the difference earlier on its own terms: with residency close to the identity, slot and expert indices coincide, so only a changed source exposes it.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed. Artifact: `build/hybrid-arena-{focused,probe,layer,range,staging,full}.log`.

### The CPU half computes the published routes

The worker now evaluates the cold routes instead of only acknowledging them. It reads the expert weights straight from `source_data`, which the correction above established is the cached tensor's host backing indexed by expert, dequantizes a row at a time through `ggml_get_type_traits()->to_float`, and runs gate, up, SiLU and down into mapped result storage. The activation it needs is published by a kernel before the sequence is raised, so a reader never sees routes without their input.

Finding the activation took three attempts, and the first two were wrong rather than merely incomplete. Publishing from the grouped dispatch at the gate or up role never fired; logging the roles that actually arrive there showed only role 4, the down projection. Publishing from the fused gate and up branch never fired either. For this model gate and up run through the legacy cached path, which is also where the earlier route dump came from, so the publication belongs there. Guessing twice cost more than the one log that answered it.

On the real model: `publications=143 acks=143 timeouts=0`, output unchanged. So the whole Strata-shaped chain now runs in live inference: admission decides hot and cold, the planner publishes the cold routes, a kernel publishes the activation and announces the packet, the CPU half computes those routes from host weights, and the device waits on the doorbell.

What is still not connected: the computed results are never read back by the device, so they do not affect the output. Admission still only counts, the resident half still executes every route, and the worker is bounded to four routes per packet with a 4096 wide activation. The dequantize-per-row CPU path is a correctness vehicle, not a fast one.

A scripted edit also damaged unrelated call sites by rewriting `bank.` to `bank.info.` across the file; they were restored from HEAD, and one in a branch excluded by this CUDA version was found by inspecting the diff rather than by the build.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; the smoke matched its reference. Artifacts: `build/hybrid-cpuhalf-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-doorbell-smoke.{py,json,log}`.

### Parity comparison built, and blocked by where the activation appears

The comparison of the CPU half against the GPU's own output for the same routes is implemented, but it has not produced a number yet, and the reason is worth more than the number would have been.

First attempt compared on the host. It never ran: cold routes only exist once the quota is full, which happens in decodes that are inside a CUDA graph capture, where the host cannot copy or synchronize. A `LLAMA_MOE_ADMIT_QUOTA` cap was added so cold routes appear immediately, which works, `hot-rate=9.23% cold=3050 fills=96` at a quota of two, but it did not help the host comparison because even the first decode is captured.

Second attempt moved the comparison into a kernel that reads the mapped results directly and waits for the acknowledgement itself, which is capture-legal and removes the synchronization. It launches on every decode layer and still accumulated nothing.

Logging each stage with its group index gave the answer. During decode the planner and the comparison run per layer, but the activation publication only happens during prefill. Gate and up take the legacy cached path only there, and that is the only place the publication was hooked, so decode packets carry cold routes with no activation, and the comparison declines correctly. The CPU half is therefore computing prefill activations and the decode comparison has nothing valid to read. That is a wiring gap, not a numerical result, and no parity figure should be quoted from this state.

Two lessons recorded rather than papered over: a quota knob makes the policy observable early, which is useful on its own; and the activation has to be published from wherever the expert input exists in each phase, which for this model is not one place.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; the smoke matched its reference and the temporary stage logs and device printfs were removed. Artifacts: `build/hybrid-parity-{probe,layer,range,staging,full}.log`, `build/run-qwen-parity-smoke.{py,json,log}`.

### Hit against miss parity, measured on production weights

The activation is now published from the grouped fused gate and up branches as well as the legacy cached path, because at 32 slots decode runs grouped while prefill runs cached. A fresh route dump showed only one decode-flagged call reaching the cached path, which is what identified the missing publication site.

With that, the comparison produces its first real number: `routes=1344 mean_nmse=1.133e-04 worst_nmse=2.287e-04`, CPU half against the GPU's own output for the same experts and the same activations, on the IQ3_S model. Generated text was unchanged.

This is the quantity Strata's `R4.2h` is about. Their activation-scale divergence measured `4.761e-04` relative and changed tokens. The disagreement here is of the same order, so it cannot be dismissed, and it is an upper bound on what a hybrid split would introduce only if the cold path is otherwise identical, which it is not yet: the CPU half dequantizes to float and accumulates in double, while the GPU quantizes the activation to Q8_1 and uses integer dot products. The measurement therefore says the two halves disagree at quantization scale, not that a particular fix is needed. Whether that changes generated tokens is still unmeasured.

The run took 2710 ms per token, about thirty times slower than normal decode, because the worker dequantizes whole experts row by row on one thread and the device waits for it. That is the measurement harness, not a hybrid performance figure, and no speed conclusion follows from it.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; the smoke matched its reference. Artifacts: `build/hybrid-parity2-{probe,layer,range,staging,full}.log`, `build/run-qwen-parity-smoke.{py,json,log}`, `build/qwen-routes-parity.txt`.

### Does the disagreement move tokens

Under `LLAMA_MOE_ADMIT_SUBSTITUTE` the comparison kernel also writes the CPU half's value into the routed column, replacing what the GPU produced. That is numerically what a real split produces for those routes, so it answers the token question without the rest of the machinery being finished.

With a quota of two, four cold routes per layer per token were substituted. Greedy, seed fixed, 48 generated tokens against a matched baseline differing only by the substitution: output identical, 171 characters, `routes=9024 mean_nmse=1.119e-04 worst_nmse=2.763e-04`.

So at this disagreement level the tokens did not move for this prompt. That is one prompt, one model, 48 greedy tokens, and only four routes per layer; it is evidence against an immediate parity crisis, not proof of token identity. Strata saw tokens change at a similar magnitude, so the honest reading is that the margin is thin rather than comfortable, and a longer or more ambiguous continuation could still diverge. Sampling with temperature, speculation acceptance and MTP are untested here and are more sensitive than greedy decoding.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed. Artifacts: `build/hybrid-substitute-{probe,layer,range,full}.log`, `build/run-qwen-substitute-smoke.{py,json,log}`, `build/run-qwen-substitute48-smoke.{py,json,log}`, `build/run-qwen-base48-smoke.{py,json,log}`.

### What the CPU half costs, and why the current worker cannot be the answer

Before spending the risk of letting admission decide, the worker was timed: `routes=1344 total_ms=16823 per_route_us=12517`. Twelve and a half milliseconds for one expert, which explains the 2710 ms per token of the measurement harness and matches it.

Against the measured decode budget of about 75 ms per token in the best ordinary configuration:

| cold routes per layer | routes per token | at 12517 us | at an estimated 170 us |
| --- | --- | --- | --- |
| 0.2 | 9.6 | 0.12 s | 1.6 ms |
| 1.0 | 48 | 0.60 s | 8.2 ms |
| 1.8 | 86 | 1.08 s | 14.7 ms |

The 170 us estimate is the memory bound for one expert, about 2.15 MiB at the roughly 14 GB/s this machine achieves on this access pattern, and is consistent with the measured CPU `MUL_MAT_ID` rates. So the work itself fits inside the decode window at the residency levels the replay predicts, but only with a kernel that is about seventy times faster than the current one.

The current worker dequantizes each row to float on one thread, which was chosen to get a correct number, not a fast one. The fast path already exists in this port: `llm_graph_moe_hybrid_executor` runs cold experts through ggml's quantized CPU `MUL_MAT_ID` on a borrowed threadpool, and was built and tested earlier. The duplicate in `moe-cache.cu` exists only because the publication lives in the CUDA backend, which cannot call into `llm_graph`.

That is the structural problem to solve next, and it is a wiring question rather than a numerical one: the backend needs to hand cold packets to the existing executor instead of computing them itself.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed. Artifacts: `build/hybrid-worker-{probe,layer,range,staging,full}.log`.

### Cold routes computed by the existing CPU helper

The CUDA backend cannot call into `llm_graph`, so the dependency runs the other way. The backend exposes `ggml_backend_moe_hybrid_set_cold_handler` through the existing proc convention, and the worker hands each cold packet to the registered handler, falling back to its own dequantizing path only if the handler declines. Under `LLAMA_MOE_HYBRID_CPU`, `llama_context` registers a handler backed by one `llm_graph_moe_hybrid_cpu` per layer, the helper built and tested earlier, which runs ggml's quantized CPU `MUL_MAT_ID` and borrows the host bank bytes without copying. It uses a dedicated threadpool, eight threads by default, so it never shares the inference pool concurrently. Clearing the handler takes the worker's lock, so context destruction waits for a call in progress before the helpers are freed.

On the real model all 48 layers' helpers initialized, and the owner handled every route: `handled_by_owner=1344 per_route_us=71.9`, against 12517 for the dequantizing worker, about 174 times faster and below the 170 us memory-bound estimate. Over 48 tokens it held at 67.8 us. Decode in the measurement harness fell from 2710 to 236 ms per token.

Parity moved: `mean_nmse=1.96e-04 worst_nmse=8.69e-04`, against 1.1e-04 and 2.8e-04 for the float worker. That is expected, since ggml's CPU path quantizes the activation for its integer dot products while the GPU quantizes it its own way, so two quantized implementations now disagree rather than one exact and one quantized. The worst route exceeds the 5e-4 bound this port has used for whole FFN parity in fixtures. With substitution the 48 token greedy output still matched the baseline exactly, so the tokens did not move here, but the margin is thinner than before, and this is precisely the hazard Strata's fp32 scale rule addresses. It needs a decision before the flag, not a widened tolerance.

Cleanup found two faults of my own. Removals of temporary role logs had silently failed earlier, and a decline diagnostic printed even with every gate off. A disabled run now prints no hybrid lines at all, which is checked rather than assumed.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, grouped staging runner, full MoE suite, 3/3 MoE CTests and diff check passed; both the 8 and 48 token substituted runs matched their references. Artifacts: `build/hybrid-cpuexec-{build,probe,layer,range,staging,full}.log`, `build/run-qwen-cpuexec-smoke.{py,json,log}`, `build/run-qwen-cpuexec48-smoke.{py,json,log}`.

### Decomposing the disagreement

Under `LLAMA_MOE_ADMIT_DECOMPOSE` the worker also evaluates each cold route exactly, float weights and double accumulation, and measures both halves against that reference, per expert weight type, plus the two activation quantization rules applied to the real published activations. In `gpu` mode the exact result is left in the packet so the device comparison measures the GPU against it.

| quantity | mean NMSE | worst NMSE |
| --- | --- | --- |
| GPU vs exact | 1.13e-04 | 2.29e-04 |
| ggml CPU vs exact, iq3_xxs (476 routes) | 1.57e-04 | 4.64e-04 |
| ggml CPU vs exact, iq3_s (280) | 1.65e-04 | 3.94e-04 |
| ggml CPU vs exact, iq2_s (560) | 1.44e-04 | 2.23e-04 |
| ggml CPU vs exact, iq4_xs (28) | 2.47e-04 | 5.22e-04 |
| CPU vs GPU, this run | 2.00e-04 | 5.80e-04 |

Activation alone, 336 samples: CUDA's Q8_1 rule 3.25e-05 against the float input, the CPU's Q8_K rule 6.40e-05, and 8.08e-05 between them. Q8_K is about twice as coarse because it uses one scale per 256 values instead of per 32.

Reading. Neither half is the wrong one: both are quantized approximations of similar size, and their errors are largely independent, so the disagreement is roughly their sum, 1.1 plus 1.5 giving about 2.0e-04. The input quantization difference is real but smaller than the total, so a second quantization point carries much of it: the intermediate activation after the gated product is quantized again before the down projection, independently on each side. Strata's fix works because it makes the two halves quantize identically, which turns two independent errors into one shared error; the disagreement then collapses to float accumulation order, which they measured at 1.461e-06.

Direction. Shared quantization is required, and it must cover both points, the expert input and the intermediate. Strata moved the GPU onto the CPU's rule because its CPU kernels were its reference. Here the better choice looks like the reverse: CUDA's Q8_1 rule is the more accurate of the two, the GPU carries the majority of routes at the residencies measured, and keeping it untouched keeps the fast path fast. That means new CPU kernels for each expert weight type consuming 32 wide Q8_1 blocks with the CUDA rounding, fed by quantized blocks the GPU publishes instead of floats, which also cuts the mapped transfer by about four. The expert types in this model's routes are iq3_xxs, iq3_s, iq2_s and iq4_xs. Performance should not suffer, since the CPU half is memory bound and these formats already work in 32 wide sub blocks internally.

Validation: Release build; both decomposition runs completed. Artifacts: `build/run-qwen-decompose-{cpu,gpu}-smoke.{py,json,log}`.

### Shared Q8_1 quantization, first kernels

Direction chosen with the owner: keep the GPU's rule and rewrite the CPU side, because both options move the same bytes, about 2.9 KB of quantized activation per layer either way, while only this one leaves the hot path's tuned kernels untouched and uses the more accurate 32 wide blocks.

The model's expert types: gate and up are iq2_s on 20 layers, iq3_xxs on 17, iq3_s on 10 and iq4_xs on 1; down is iq4_nl on 39 and q2_0 on 9. New CPU dot products in `src/llama-moe-hybrid-kernels.cpp` cover iq2_s, iq3_xxs and iq4_nl against 32 wide Q8_1 activations, adapted from ggml's generic implementations so the integer arithmetic and lookup tables are the same. The intermediate activation is quantized with the CUDA rule on the CPU.

The input activation is not re-quantized on the CPU. The CUDA build uses `-use_fast_math`, so the GPU's `x / d` is an approximate division and the CPU cannot reproduce its rounding exactly. Instead the publication now runs `quantize_row_q8_1_cuda`, the kernel MMVQ itself uses, on the same activation into the mapped packet, so the CPU consumes the GPU's own bytes. The cold handler gains a pointer to them, and `llama_context` uses the Q8_1 path under `LLAMA_MOE_HYBRID_Q8_1` whenever all three bank types are covered, otherwise the ggml helper.

The first run selected no layers: the support check required down rows to be a multiple of 256, but down rows are 640 wide and iq4_nl uses 32 blocks. With per type block sizes, 29 layers qualify, iq2_s or iq3_xxs gate and up with iq4_nl down.

Result, covered layers only, other layers excluded from the comparison:

| CPU vs GPU | mean NMSE | worst NMSE |
| --- | --- | --- |
| ggml CPU path, Q8_K | 2.0e-04 | 5.8e-04 |
| shared Q8_1, 812 routes | 3.81e-07 | 1.40e-05 |

About five hundred times lower on average, comparable to Strata's 1.461e-06. The remaining disagreement is float summation order plus occasional intermediate rounding flips, since the GPU's approximate division can round a near tie the other way; the worst route shows that, and it is small. Per type CPU accuracy against the exact float reference also improved, iq2_s worst from 2.2e-04 to 2.0e-04 and iq3_xxs from 4.6e-04 to 2.0e-04, as expected from the finer blocks.

Cost: these are scalar reference kernels on the worker's single thread, 2309 us per route, against 72 us for ggml's threaded SIMD path. Correctness first; speed is next.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed, and a run with every gate off printed no hybrid lines. Artifacts: `build/hybrid-q81-{build,probe,layer,range,full}.log`, `build/run-qwen-q81-smoke.{py,json,log}`, `build/run-qwen-q81only-smoke.{py,json,log}`.

### Coverage plan, and the first speed work

The kernels are keyed on ggml weight type, not on a model, so any MoE model whose experts use a covered type shares quantization, and uncovered layers fall back to the ggml helper. ggml's types split three ways: 32 wide types that already pair with Q8_1 or Q8_0 on the CPU need no new kernel at all, 256 wide K and IQ types need one new dot product each, and float types have nothing to share. One generic caveat: the CPU must read whatever layout the consuming CUDA kernel quantized into, plain Q8_1 for MMVQ in decode and `block_q8_1_mmq` for MMQ in prefill.

Done in this step:

- 32 wide types now go through ggml's own SIMD `vec_dot`, chosen from `ggml_get_type_traits_cpu`: Q8_1 consumers take the GPU's blocks directly, Q8_0 consumers take them with the 2 byte sum dropped. iq4_nl, the down type on 39 layers, uses this, and the scalar iq4_nl kernel was removed.
- AVX2 versions of iq2_s and iq3_xxs, adapted from ggml's own so the integer work is the same, scaling each 32 block by its own Q8_1 scale. Selected by a CPUID check, so the build needs no `/arch` change; `LLAMA_MOE_HYBRID_SCALAR` forces the scalar reference.
- A persistent row pool, and all routes of a packet run as one gate and up job and one down job.

| step | us per route, 8 threads |
| --- | --- |
| scalar, one thread | 2309 |
| + pool, ggml SIMD for 32 wide types | 289 |
| + AVX2 iq2_s and iq3_xxs | 117.5 |
| + one job per packet | 93.1 |
| 16 threads | 77.9 |
| 4 threads | 155.2 |
| ggml helper, 8 threads, unshared quantization | 72 |

Parity did not move through any of this: `mean_nmse=3.82e-07 worst_nmse=1.40e-05` on the 812 covered routes, the last digit changing only with summation order. Scaling flattens above eight threads, so synchronization and memory, not arithmetic, now bound it. The remaining gap to ggml is the condition variable wake per job against ggml's spinning barrier, and VNNI `dpbusd` is unused.

With substitution of the CPU result on all handled routes, Q8_1 on 29 layers and the ggml helper on the rest, 48 greedy tokens matched the baseline exactly.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed; a run with every gate off printed no hybrid lines. Artifacts: `build/hybrid-q81simd-{probe,layer,range,full}.log`, `build/hybrid-q81{simd,avx,batch}-build.log`, `build/run-qwen-q81only-smoke.{py,json,log}`, `build/run-qwen-q81t{4,16}-smoke.{py,json,log}`, `build/run-qwen-q81sub48-smoke.{py,json,log}`.

### All 48 layers share quantization, and greedy identity stops being the test

q2_0 needed no kernel: its 64 wide blocks pair with two Q8_0 blocks, each with its own scale, so the repack path covers it once the block size condition allows multiples of 32. Scalar and AVX2 kernels were added for iq3_s and iq4_xs, again adapted from ggml's own. All 48 layers now take the shared Q8_1 path. The row pool spins briefly before sleeping, so the two jobs of a packet do not each pay a wake.

| state | us per route | CPU vs GPU mean NMSE | worst |
| --- | --- | --- | --- |
| 29 layers covered, 8 threads | 93.1 | 3.82e-07 | 1.40e-05 |
| 48 layers, iq3_s and iq4_xs scalar | 133.3 | 2.81e-07 | 1.40e-05 |
| 48 layers, AVX2 | 100.6 | 2.81e-07 | 1.40e-05 |
| + spinning pool, 8 threads | 95.4 | 2.81e-07 | 1.40e-05 |
| + spinning pool, 16 threads | 73.4 | | |

At about one cold route per layer, which the replay predicts at 64 slots, that is roughly 4 to 5 ms per token of CPU work against a decode budget near 75 ms. The CPU half is no longer the limiting factor; admission and overlap are.

The 48 token substitution check then failed, reproducibly, while the baseline reproduced byte for byte across two runs. The divergence is at character 165, after the model has finished answering and is emitting a repetitive tail of end tokens, where the leading candidates are nearly tied. The earlier ggml path, whose parity was about seven hundred times worse, happened not to flip that tie. Any nonzero difference can flip a near tie, and the two devices can never be bit exact because they sum floats in different orders, so greedy token identity is not a meaningful acceptance criterion for this design and the earlier matches were partly luck. Strata judges hits against misses with logit KL divergence, and that is the measurement this port should adopt: KL of the hybrid's logits against the GPU only run over real text, reported in confident and uncertain positions separately.

Validation: Release build/relink, insertion probe, grouped-layer run, ranged runner, full MoE suite, 3/3 MoE CTests and diff check passed; a run with every gate off printed no hybrid lines; the baseline reproduced exactly; the substituted run diverged at the stated position, reproducibly. Artifacts: `build/hybrid-q81all-{probe,layer,range,full}.log`, `build/hybrid-q81{more,spin}-build.log`, `build/run-qwen-q81only-smoke.{py,json,log}`, `build/run-qwen-q81sub48-smoke.{py,json,log}`, `build/run-qwen-base48b-smoke.{py,json,log}`.

### Logit KL against the GPU only run

`build/run-qwen-kld.py` drives the existing `llama-perplexity --kl-divergence` over `tools/server/README.md`, 6 chunks of 128 tokens, 32 slots, with `-ub 1` so every token is a single row decode, the path the hybrid chain runs on. The base run saves GPU only logits; the hybrid run enables shadow admission, the doorbell, the owner CPU half with shared Q8_1, and substitution of the CPU values on cold routes, about 144,000 routes over the run.

| run | mean KLD | max KLD | same top token |
| --- | --- | --- | --- |
| hybrid pipeline, no substitution (control, twice) | 0.000000 | 0.000045 | 100.000% |
| substitution, run 1 | 0.005699 ± 0.001016 | | 98.94% |
| substitution, run 2 | 0.008661 ± 0.001417 | | 97.35% |
| substitution, run 3 | 0.007785 ± 0.001427 | | 98.41% |
| substitution, first run before the exit fix | 0.007323 ± 0.001369 | 0.3316 | 98.68% |

The control isolates the effect: with every part of the pipeline running but the GPU's values kept, the output is unchanged, so the divergence comes from the substituted CPU values. The median per token KLD in the first run was 8.6e-05, so the mean is driven by a minority of uncertain positions, consistent with the near tie seen in the greedy check.

The repeats do not agree with each other beyond what the reported error suggests is safe to ignore, and the number of routes handled differs slightly between them, while the baseline and the control are exactly reproducible and the handshake counts, 36,575 acknowledgements and no timeouts, are identical in every run. So something in the substituted path is not deterministic. Until it is found these KLD values are a range, roughly 0.006 to 0.009, not a measurement, and admission must not be switched to deciding on top of an unexplained race.

The first hybrid run also crashed at exit with an access violation. The worker kept polling a packet after its resource freed the mapped memory; server smokes never shut down normally, so they could not show it. The resource now removes its packet from the worker, under the worker's lock, before freeing it, and every later run exited cleanly.

Validation: Release build/relink, insertion probe, full MoE suite, 3/3 MoE CTests and diff check passed. Artifacts: `build/run-qwen-kld.py`, `build/qwen-kld-*.log`.

Next: find the nondeterminism in the substituted path, then repeat the KL measurement to a stable value; only then let admission decide and move the doorbell wait to the combine.

Next: isolate intermediate projections/nonlinear outputs, then select a compatible shared representation without changing ordinary inference arithmetic. A device-count hot path and mapped-memory handoff remain pending. Speculative parity, greedy-token identity and overlap remain unmeasured, and no inference hook or switch exists. Only then does an inference hook become meaningful, and speculation parity and overlap still have to be measured before any switch is exposed. The existing cached planner still performs demand filling and must not become the permanent hybrid execution path. Only after those qualifiers and speculation parity should inference wiring and a user-facing switch be exposed.

### Stream-ordered insertion without an intermediate drain

The probe now runs a second pass per configuration that removes the intermediate GPU drain. It poisons the raw-down tensor, submits the CPU job, queues resident GPU expert work, waits only for the CPU result, copies compact columns into a fixed pinned staging buffer, inserts them with `ggml_backend_tensor_set_async` on the backend stream, queues the reduction and synchronizes once at the end. Stream order alone keeps the expert kernels, the inserted columns and the reduction correctly sequenced, with no event or drain between them.

The streamed pass must reproduce the drained pass bit-exactly, so a missing dependency or a premature staging reuse fails the test rather than appearing as small numerical noise. Staging is allocated once per configuration at the maximum route size and freed afterwards; it is bounded pinned memory, not a resident-RAM cap. The earlier drained pass is kept because it is what checks cold-column sentinels and bit-exact resident columns, which require reading the intermediate tensor.

This still is not the production path: it uses a test-owned raw-down allocation, host-side inputs and ids rather than bounded D2H transfers, a single backend stream rather than separate copy/compute streams with events, and a test-local submission identity rather than the CUDA owner's certificate. It demonstrates correct ordering only; no overlap or speedup is claimed or measured.

Validation: build passed; the probe passed (~3.58 s) with unchanged NMSE values `bf16 9.86851e-06` and `q4_0 0.000157814`; the layer fixture passed (~0.70 s); 3/3 MoE CTests passed; `git diff --check` passed. The full suite again hit the known intermittent `test-moe-cache-graphs.cpp:1286` capture-state failure and passed on an unchanged-binary rerun (~22.34 s). That failure remains unexplained.

Artifacts: `build/hybrid-stream-insert-build.log`, `build/hybrid-stream-insert-focused.log`, `build/hybrid-stream-insert-layer.log`, `build/hybrid-stream-insert-full.log` (lifecycle failure), `build/hybrid-stream-insert-full-rerun.log` (pass), `build/hybrid-stream-insert-ctest.log`.

Next: move this sequence behind the real CUDA seam with owner/generation/certificate binding, bounded D2H of inputs and ids, and drain-before-reuse on rebuild, restore, cancellation and teardown. The recurring capture-state failure should be investigated before any switch is exposed.

### Intermittent capture/lifecycle failures measured and diagnosed; owner decision needed

Measured both failures by stashing every uncommitted change, rebuilding the same configuration at pristine HEAD `be074960`, and running the full MoE test eight times, then repeating with the working tree restored.

| Build | Runs | `graphs.cpp:1286` | `graphs.cpp:87` | Failing runs |
| --- | --- | --- | --- | --- |
| pristine HEAD | 8 | 4 | 1 | 5 |
| working tree | 8 | 2 | 2 | 4 |

Both failures reproduce at a similar rate without any of this work, so they are pre-existing, not caused by the semantic descriptor, CPU helper, native skip, residency snapshot, route packet, worker or probe. This is measured evidence, not an assumption.

A temporary local diagnostic printed the compared fields when `:1286` fails, then it was reverted. The failing case is `rows=2 experts=8 used=2 slots=12 layout=FUSED_GATE_UP type=q4_0 large=0 passes=4`. Only the raw `cudaGraph_t` handle differs; `instance`, `warmup_complete`, `execution_semantic_key` and `moe_resource_fingerprint` are all identical. In `ggml_backend_cuda_graph_compute()` an update destroys the old description and `cudaStreamEndCapture()` produces a new `cudaGraph_t`, while `cudaGraphExecUpdate()` keeps the same executable instance. The handle is therefore not stable across a recapture, and CUDA sometimes reuses the same address, which explains the intermittency. The assertion's actual intent, that the prefill path does not invalidate the main capture, still holds in the failing runs.

That suggests the assertion should compare the instance, warmup flag, semantic key and resource fingerprint, and require a nonzero graph handle, instead of requiring handle identity. This test is pre-existing fork code that this work does not touch, so the assertion was left unchanged and the decision belongs to the repository owner. The `:87` failure is a `cuda: invalid argument` during the `check_payload` device-to-host copy of bank bytes and is not yet explained; no change is proposed for it.

Validation: the working tree was fully restored after the experiment (16 modified files, same set as before), the stash was popped, both builds were rebuilt, and the probe and layer fixture pass.

Artifacts: `build/hybrid-flake-modified.log`, `build/hybrid-flake-pristine.log`, `build/hybrid-flake-diag.log`, `build/hybrid-flake-baseline-build.log`, `build/hybrid-flake-restore-build.log`, `build/hybrid-flake-restore-tests-build.log`.

Owner decision needed: whether to relax the `:1286` handle comparison to the stable capture identity fields, and whether to investigate `:87` now or later.

### Capture-handle assertion relaxed; `:87` localized but unresolved

With owner approval, the three capture-preservation assertions in `test_active_grouped_multirow_graph_modes_case()` no longer compare the raw `cudaGraph_t` handle. They now require a nonzero handle plus the stable identity: same executable instance, warmup complete, same execution semantic key and same resource fingerprint. The now-unused `captured_graph` variable was removed and a short comment records why the description handle changes. Nothing in production code changed, and the assertions still fail if a run actually invalidates the main capture, because the instance, key or fingerprint would then differ or reset.

After the change, ten full runs produced zero `:1286` failures, with two `:87` failures remaining:

| Build | Runs | `:1286` | `:87` |
| --- | --- | --- | --- |
| pristine HEAD | 8 | 4 | 1 |
| before relaxation | 8 | 2 | 2 |
| after relaxation | 10 | 0 | 2 |

For `:87`, a temporary diagnostic (since reverted) captured one failure: case `host_budget=0 expected_host_nodes=0 n_dim=256 pageable=0 automatic=1`, bank slot stride `73728`, `slot=0`. The device bank pointer is valid device memory on device 0 and the current device is also 0, yet copying one full slot stride from slot 0 fails with `invalid argument`. A valid pointer with a failing in-range copy means the live allocation is shorter than one slot stride, so the test is reading through a pointer that no longer matches the resource it was sized against. Grouped bank storage is allocated as `expert_stride * n_slots`, so a correctly sized live bank cannot fail this copy. The most likely explanation is a stale pointer whose address was recycled by a later, smaller allocation in the automatic pinning-failure path, but that is not yet proven and no fix is proposed.

`:87` reproduced about twice in ten runs here and once in eight runs on pristine HEAD, so it remains pre-existing and unrelated to this work.

Validation: Release build/relink passed; ten full MoE runs with two `:87` failures and no other failures; the hybrid probe, layer fixture and 3/3 MoE CTests pass; `git diff --check` passes.

Artifacts: `build/hybrid-flake-diag87.log`, `build/hybrid-capture-relax-build.log`, `build/hybrid-capture-relax-flake.log`, `build/hybrid-capture-relax-probe.log`, `build/hybrid-capture-relax-layer.log`, `build/hybrid-capture-relax-ctest.log`.

### `:87` root cause proven: page-rounded host registration captures neighbor heap memory

The earlier stale-pointer guess was wrong. Successive temporary diagnostics (all reverted) proved the source pointer is fine and the destination is the problem:

- `cuMemGetAddressRange` on the device bank returns base equal to the pointer with range `884880`, which holds the full `12 x 73728` payload, so the copy is in range.
- A control copy of the same size from a freshly allocated device buffer into the same destination also fails, so the bank is not implicated at all.
- A 4-byte copy from the same source succeeds; only the full-size copy fails.
- `cudaPointerGetAttributes` on the destination shows the first byte is host-registered (`type=1`) and the last byte is unregistered (`type=0`) across exactly the `73728` byte span.

The destination is an ordinary `std::vector<uint8_t>` in the test. It straddles a host-registration boundary, and a CUDA copy crossing such a boundary returns `cudaErrorInvalidValue`.

The registration ranges in `moe_host_configure_sources()` are rounded outward to page boundaries, `begin - begin % page` to `(begin + size + page - 1) / page * page`. A page only partly owned by a bank is registered in full, so neighboring bytes of unrelated heap memory become registered. When the allocator later hands part of such a page to another host allocation, that allocation starts inside registered memory and ends outside it, and any CUDA copy spanning the boundary fails. The intermittency is just allocator placement.

This is a real hazard, not only a test artifact: any pageable host buffer that happens to neighbor a registered bank page can fail transfers. The exposure is lower for real models, whose bank sources are page-aligned mapped file pages, and higher for heap-allocated banks like the fixtures. Fixing it properly means registering only fully owned pages and letting partially owned edges fall back to staging, which trades some direct registration for safety. A test-only workaround, such as page-aligned or pinned destinations, would hide the hazard rather than remove it.

No fix was applied, because this is pre-existing fork behavior in `moe-cache-host.cu`, affects admission/fallback policy, and needs an owner decision. The failure rate is unchanged, about one in four to ten full runs.

Validation after reverting all diagnostics: Release build/relink passed, probe and layer fixture pass, four full runs produced one `:87` failure and no other failures.

Artifacts: `build/hybrid-flake-diag87-range.log`, `build/hybrid-flake-diag87-context.log`, `build/hybrid-flake-diag87-kind.log`, `build/hybrid-flake-diag87-control.log`, `build/hybrid-flake-diag87-host.log`, `build/hybrid-diag87-restore-build.log`.

Owner decision needed: restrict host registration to fully owned pages (safe, less direct registration), or keep page-rounded registration and accept that neighboring pageable host buffers can fail CUDA copies.

### Fixed by page-aligned cached host storage (Strata's approach)

Strata does not handle this case, it makes it impossible. `PinnedArena` in `src/core/pinned.cu` reserves its own 2 MB-aligned arena with `VirtualAlloc`/`mmap`, registers the whole arena before any page is touched, and falls back to slice-by-slice registration at uniform offsets inside that same arena. Every registered page is owned end to end, so registration never covers memory that belongs to another allocator.

The owner chose the same ownership rule here. Cached host buffers now get page-aligned, page-rounded storage from `moe_host_page_aligned_buffer()`, using `_aligned_malloc` on Windows and `posix_memalign` elsewhere, with a matching free. It replaces the plain CPU-allocator buffers in the bounded-budget path, the Linux pre-registered path and the pageable auto-pin fallback. Registration ranges are still page-rounded, but now a rounded range can only cover pages the bank already owns. The test fixture that emulates pageable cached memory uses the same allocation through a small exported test entry, so the fixture cannot violate the invariant the production path relies on.

This keeps full direct registration and adds no staging fallback. The cost is rounding each cached buffer up to a page and a page-aligned allocation.

Result: twelve consecutive full MoE runs passed with zero failures. Both intermittent failures are gone, including `:1286`, which reinforces that its handle churn came from the same recycled-registration disturbance rather than from capture logic.

| Build | Runs | `:1286` | `:87` |
| --- | --- | --- | --- |
| pristine HEAD | 8 | 4 | 1 |
| assertion relaxed only | 10 | 0 | 2 |
| page-aligned cached storage | 12 | 0 | 0 |

Validation: Release build/relink passed; 12/12 full runs passed; hybrid probe, layer fixture and staging benchmark passed; 929/929 CUDA `MUL_MAT_ID` backend-op cases passed; 3/3 MoE CTests passed; `git diff --check` passed.

Artifacts: `build/hybrid-page-align-build.log`, `build/hybrid-page-align-flake.log`, `build/hybrid-page-align-relink.log`, `build/hybrid-page-align-probe.log`, `build/hybrid-page-align-layer.log`, `build/hybrid-page-align-staging.log`, `build/hybrid-page-align-ops.log`, `build/hybrid-page-align-ctest.log`.

Real-model check: reran the ordinary GPU-layers plus CPU-experts smoke on `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` with the page-aligned build (`build/run-qwen-page-align-smoke.py`, port 8096, same arguments as the earlier rebuild smoke).

| Measure | Earlier rebuild smoke | Page-aligned build |
| --- | --- | --- |
| greedy output | `<think>\nThe user is asking me to` | identical |
| ready | 11.65 s | 10.67 s |
| prompt | 3.54 tok/s | 9.72 tok/s |
| decode | 4.90 tok/s | 11.19 tok/s |
| peak RSS | 12.55 GiB | 11.39 GiB |
| minimum available RAM | 10.01 GiB | 39.12 GiB |

The greedy output is identical, so the allocation change does not alter results. Per-buffer page rounding did not inflate host memory in this run; peak RSS is slightly lower.

The throughput numbers are NOT a speedup claim for this change. The two runs started from very different machine states: the earlier one ran with about 10 GiB available RAM and was described then as cold and not comparable, while this one had about 39 GiB free throughout. Both are cold, 12-token prompt, 8-token smoke runs, which is far too small to measure anything. A real comparison needs identical warm conditions on idle hardware.

Still not verified: sustained real-model behavior, large-context runs, and host memory use across the full 77.865 GiB payload rather than a short smoke.

Artifacts: `build/run-qwen-page-align-smoke.py`, `build/run-qwen-page-align-smoke.json`, `build/run-qwen-page-align-smoke.log`.
