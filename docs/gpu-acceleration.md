# GPU acceleration paths

Survey of which algorithms in `compression-zoo`'s Rust crates are
candidates for GPU offload, scored by parallelism friendliness vs
expected real-world payoff. The aim is to pick a small number of
targets and ship working CUDA kernels rather than try to GPU-port the
whole pipeline.

## Scoring matrix

| Component               | Crate    | Parallel? | GPU win | Verdict |
|-------------------------|----------|-----------|---------|---------|
| Byte histogram          | (shared) | trivial   | high    | **PoC** |
| Adler-32                | bsc-rs   | tree-ok   | low-med | nice-to-have |
| E8E9 transform          | (shared) | trivial   | medium  | **PoC** |
| Suffix array (SA-IS)    | bsc-rs   | hard      | high    | future  |
| Forward BWT             | bsc-rs   | follows SA| high    | follows SA |
| Inverse BWT (LF walk)   | bsc-rs   | data-dep  | low     | skip    |
| LZP encode              | bsc-rs   | sequential| low     | skip    |
| LZ77 search             | (zpaq)   | window-dep| low-med | skip    |
| QLFC (rank coding)      | bsc-rs   | sequential| low     | skip    |
| Range coder             | bsc-rs   | sequential| zero    | skip    |
| Arith coder (ZPAQ)      | zpaq-rs  | sequential| zero    | skip    |
| ZPAQL VM                | zpaq-rs  | sequential| zero    | skip    |
| Predictor / context-mix | zpaq-rs  | bit-dep   | low     | skip    |
| SHA-1                   | zpaq-rs  | sequential| zero    | skip    |
| Multi-block decode      | zpaq-rs  | per-block | medium  | future  |
| PPMd                    | sevenz-rs| sequential| zero    | skip    |
| LZMA2                   | sevenz-rs| sequential| low     | skip    |

## Why most of the pipeline is sequential

Both BSC and ZPAQ use **arithmetic coders** at their core. Arithmetic
coding maintains a `[low, high]` range that depends on the previous
bit's probability and decoded value. There is no way to decode the
*k*-th bit before bits 0..k−1 are known, so the inner loop is
inherently sequential. The same is true of:

- **ZPAQL VM** — bytecode interpreter with shared state.
- **Predictor / context-mixing** — every bit's prediction reads and
  writes the same component-cm tables.
- **LZP / LZ77 encode** — uses a hash table that's updated per byte.
- **PPMd / LZMA2** — same range-coder argument.

This means the right level for GPU work is *outside* the inner loop:
either preprocessing/postprocessing of the byte stream, or coarse
(per-block) parallelism.

## What's actually GPU-friendly

### Tier 1 — embarrassingly parallel

- **Byte histogram (256 bins).** Used as a building block in:
  - SA-IS bucket counting.
  - BWT bucket layout.
  - First-pass entropy estimation (compress() level selection in
    upstream libzpaq).

  Standard CUDA pattern: per-block shared-memory histogram + atomic
  reduction. ~1-line kernel, dozens of GB/s on RTX-class hardware.
  Used as the PoC kernel in [`gpu-rs`](../rust/gpu-rs/).

- **E8E9 (Intel/AMD jump-relative transform).** Per-byte conditional
  rewrite: when bytes 0xE8/0xE9 (CALL/JMP) are followed by a 4-byte
  relative offset, convert to absolute. Each candidate position is
  independent under one writer; the transform is "scan, then compact"
  in GPU terms. Wins when run as part of a larger preprocessing
  pipeline rather than in isolation (latency vs throughput).

### Tier 2 — parallel but non-trivial

- **Suffix array (SA-IS).** The L/S induced-sort phase is sequential
  by design (each write feeds the next read). However:
  - Bucket counting is trivially parallel.
  - The recursive layer can run on GPU via radix sort (DC3/SkewSA).
  - Ilya Grebnov's [`libcubwt`](https://github.com/IlyaGrebnov/libcubwt)
    already does this end-to-end for BWT.

  Realistic path: *call libcubwt via FFI* rather than re-implementing
  SA-IS on GPU. That gives bsc-rs a 5-10× BWT speedup on >1 MB blocks
  with maybe 200 lines of integration code.

- **Multi-block ZPAQ decompression.** ZPAQ archives are made of
  independent blocks (predictor reset at each block boundary). On
  multi-GB archives the blocks are large enough that one block per
  GPU SM (Streaming Multiprocessor) keeps everything busy. But the
  *inner* decode is still sequential per block, so GPU just gives N×
  speedup where N is block count, not a fundamental win.

### Tier 3 — modest wins from tree reductions

- **Adler-32.** Slice-by-N parallelisation works (split input into
  chunks, compute partial Adler-32s, combine). 4-8× speedup on big
  inputs. Adler-32 is already cheap so the absolute win is small.

- **SHA-1.** Tree-Merkle-style parallelisation is technically
  possible but the per-iteration data dependency makes it painful;
  not worth it for our use case.

## Rejected — cannot meaningfully accelerate

- All inner loops of arithmetic coders (range, ZPAQ, LZMA2, PPMd).
- ZPAQL VM and predictor updates.
- LZP and LZ77 encoders (hash-table dependency chain).
- QLFC adaptive ranks.
- Context-mixing predictors (every bit feeds back into itself).

These are sequential by design. The right GPU strategy for them is
"don't" — focus the GPU on preprocessing or use it for parallel
*independent* compressions (one stream per SM).

## What this session shipped

1. **Crate scaffolding.** `rust/gpu-rs/` joined the workspace. The
   `build.rs` invokes `nvcc` if available and emits a `libgpu_rs_cuda.a`
   static archive that the Rust crate links; on hosts without CUDA
   it falls back to a C stub that returns "unavailable" at runtime.
   `GPU_RS_FORCE_CPU=1` forces the stub path.
2. **PoC kernel: 256-bin byte histogram.** Per-block shared-memory
   histogram + atomic global reduction. Validated against the
   scalar CPU reference (`histogram_u8_cpu`) on 1 MB random data,
   uniform runs, and the empty case (3/3 lib tests pass). Drops in
   trivially as a parallel `bucket_counts` for SA-IS.
3. **Honest benchmark.** `cargo run --release -p gpu-rs --bin
   histogram_bench` on RTX 4080:
   ```
   size=  0.1 MiB  CPU=      16 µs  GPU=    725 µs  speedup=0.02×
   size=  0.2 MiB  CPU=      68 µs  GPU=    771 µs  speedup=0.09×
   size=  1.0 MiB  CPU=     276 µs  GPU=    829 µs  speedup=0.33×
   size=  4.0 MiB  CPU=    1109 µs  GPU=   1715 µs  speedup=0.65×
   size= 16.0 MiB  CPU=    4486 µs  GPU=   3418 µs  speedup=1.31×
   size= 64.0 MiB  CPU=   18060 µs  GPU=  11055 µs  speedup=1.63×
   ```
   Crossover at ~8 MiB. Below that the kernel is strictly slower
   because the H2D + D2H copies dominate. This is the "GPU lesson"
   for compression: standalone kernels on small buffers don't pay,
   but the same kernel run as part of a longer pipeline (where the
   data is *already* on the device) sees the full ~10× memory-
   bandwidth speedup.

Future sessions:
4. libcubwt FFI integration into `bsc-rs` for big-block BWT.
5. CUDA E8E9 forward/inverse for ZPAQ levels 4-7. Trivially
   parallel; same per-block shared-memory pattern as histogram.
6. Per-SM multi-block ZPAQ decode driver — one block per SM,
   independent predictor state, end-to-end on GPU.
7. Pinned host memory + async streams to eliminate the per-call
   allocation cost shown in the benchmark above.

## Why CUDA over wgpu

Both could work, but for our specific situation:

- **wgpu/WGSL pros**: cross-platform (Linux/macOS/Windows + browsers),
  no proprietary toolchain, runs on AMD/Intel/NV.
- **wgpu cons**: ~200 transitive dependencies, slow first build,
  WGSL is more limited (no warp intrinsics, less mature for compute).
- **CUDA pros**: best perf, mature toolchain, full warp-level
  primitives, this machine has nvcc + an RTX 4080.
- **CUDA cons**: NVIDIA-only, requires nvcc at build time.

For a `compression-zoo` that already vendors C/C++ source for the
plugins (which require MinGW for Windows DLLs), assuming a working
C toolchain is fair. We compile `.cu` source via nvcc through a
`build.rs` script and emit a static lib that the Rust crate links.
Everything stays in the workspace.

## How to build / disable

The `gpu-rs` crate auto-detects `nvcc`. If absent, `build.rs` falls
back to a CPU-only stub so the workspace still builds on machines
without CUDA. Set `GPU_RS_FORCE_CPU=1` to force the stub path even
when CUDA is installed (useful for CI / cross-builds).
