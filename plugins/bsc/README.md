# `bsc` — 7-Zip codec plugin for libbsc

[libbsc](https://github.com/IlyaGrebnov/libbsc) by Ilya Grebnov is a
high-performance compressor combining Burrows–Wheeler transform,
Schindler transform and LZP. It typically gives 5–15 % better
compression than LZMA2 ultra at comparable speed, and saturates on a
single thread sooner than 7-Zip's existing methods.

| Property | Value |
|---|---|
| Method name | `bsc` |
| Method ID | `0x4F71200` (community ID block `0x4F712xx`) |
| Levels | 7-Zip 1–9 → libbsc internal "block sorter intensity" |
| License (wrapper) | Apache-2.0 |
| License (libbsc) | Apache-2.0 (Ilya Grebnov) |

## Status

Built and tested. The DLL ships as a 64-bit Windows codec plugin and
round-trips correctly through 7-Zip's `ICompressCoder` ABI.

  * `tests/test_linux` — 10/10 in-process round-trips driving
    `bsc_compress` / `bsc_decompress` directly. Covers tiny text, 8 KiB
    pseudo-random with periodic structure, 1 MiB sparse, and the
    `bsc_store` fallback for incompressible random input. Linux smoke.

The Windows-side COM round-trip currently relies on a `7z a` / `7z x`
pair against the installed DLL; an in-memory `tests/test_bsc.exe`
mirroring `plugins/zpaq/tests/test_zpaq.cpp` is on the TODO list.

## Build

From `plugins/bsc/`, on Linux/WSL with MinGW-w64:

```bash
make                      # builds build/bsc.dll (~5.9 MiB)
make install              # copies to $CODECS_DIR (default /mnt/l/Programs/7-Zip/Codecs)
```

Then in 7-Zip:

```cmd
7z a archive.7z -m0=bsc -mx5 input.dat
7z x archive.7z
```

## Test

```bash
make -C plugins/bsc        # builds DLL + tests/bsc_cli + tests/test_linux
plugins/bsc/tests/test_linux
plugins/bsc/tests/bsc_cli e in.txt out.bsc      # CLI round-trip
plugins/bsc/tests/bsc_cli d out.bsc back.txt
```

## Level mapping

7-Zip exposes `-mx` levels 0..9. We pass them straight through to
`bsc_compress`'s internal block-sorter level (1..9; libbsc clamps).
Higher levels turn on stronger BWT vs. ST and tighter LZP heuristics.

## Implementation notes

- `src/BscCoder.cpp` reads the entire input into a single block
  (libbsc is a block compressor; streaming would need our own framing
  on top), calls `bsc_compress`, then writes the resulting block out.
- The decoder is symmetric: read one full compressed block, call
  `bsc_decompress`, write the result.
- We compile the C99 SA-IS suffix array library `libsais.c` as plain C
  and the rest of libbsc as C++; the Makefile uses a per-source compile
  rule to flatten paths into `build/` while keeping `..` segments out
  of the output tree.

## Reference

- Upstream libbsc: <https://github.com/IlyaGrebnov/libbsc>
- libsais (the BWT engine): <https://github.com/IlyaGrebnov/libsais>
