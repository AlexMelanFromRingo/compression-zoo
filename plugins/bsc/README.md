# `bsc` — 7-Zip codec plugin for libbsc

[libbsc](https://github.com/IlyaGrebnov/libbsc) by Ilya Grebnov is a
high-performance compressor combining Burrows–Wheeler transform,
Schindler transform and LZP. It typically gives 5–15 % better
compression than LZMA2 ultra at comparable speed, and saturates on a
single thread sooner than 7-Zip's existing methods.

| Property | Value |
|---|---|
| Method name | `bsc` |
| Method ID | `0x4F71200` (proposed; community ID block `0x4F712xx`) |
| Levels | 1–9 (libbsc internal "block sorter intensity") |
| License (wrapper) | Apache-2.0 |
| License (libbsc) | Apache-2.0 (Ilya Grebnov) |

## Status

Not implemented yet. Plan:

1. Add `git submodule` for `https://github.com/IlyaGrebnov/libbsc` into
   `upstream/`.
2. Implement `CBscEncoder` / `CBscDecoder` in `src/BscCoder.cpp`. libbsc
   exposes a clean C API (`bsc_compress` / `bsc_decompress`) which makes
   the wrapper relatively short.
3. Choose a default block size (e.g., 64 MiB); expose it via
   `ICompressSetCoderProperties`.
4. `REGISTER_CODEC_E(Bsc, CBscDecoder, CBscEncoder, 0x4F71200, "bsc")`.

## Build

Same pattern as `plugins/zpaq/Makefile`. Will be added once the codec
source is in place.
