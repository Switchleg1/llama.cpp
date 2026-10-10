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

### Stage 4: prefill and speculative integration

- Integrate bounded expert streaming/grouping for larger prompt batches using existing GPU kernels first.
- Review temporary reuse of GPU expert storage for prefill workspaces; restore or update residency safely before decode.
- Test target/draft memory accounting, supported MTP verification row shapes and recurrent rollback.
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

The failing assertion compares V2 early-router context fixed bytes with V1 sizing plus additional workspace. No implementation code has changed, so this is a current-revision baseline failure, not a hybrid-port regression. The rest of the full suite was not completed past that failure. Do not label the complete MoE suite as passing based on the layer-only fixture.

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
