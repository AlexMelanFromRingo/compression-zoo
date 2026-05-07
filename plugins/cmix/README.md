# `cmix` — 7-Zip codec plugin for CMIX

[CMIX](https://github.com/byronknoll/cmix) by Byron Knoll is a context-
mixing compressor that holds the top of many lossless benchmarks
(Hutter Prize, [Large Text Compression
Benchmark](http://mattmahoney.net/dc/text.html)). It compresses
enwik9 to about 115 MB versus LZMA2 ultra's ~210 MB, but at a cost
that surprises people:

  - **Time**: hours of single-thread CPU per gigabyte.
  - **Memory**: peak RSS easily exceeds 25 GB on large inputs.
  - **Decompression is symmetric**: as slow as compression.

Use only when ratio absolutely dominates — archival storage, training
corpora, single-shot data shipping over expensive links.

| Property | Value |
|---|---|
| Method name | `CMIX` |
| Method ID | `0x4F71201` (proposed) |
| Levels | none (CMIX has a single mode) |
| License (wrapper) | GPL-3.0 |
| License (CMIX) | GPL-3.0 (Byron Knoll, plus PAQ8 lineage) |

## Licensing note

CMIX is GPL-3.0; this plugin DLL therefore ships under GPL-3.0 too.
That **does not** affect 7-Zip itself: 7-Zip loads the plugin at
runtime, which is widely understood as an arms-length use that does
not impose GPL on the host. Be aware, though, that any code you
**statically link** against `cmix.dll` (e.g., another wrapper) would
need to be GPL-3.0 compatible.

## Status

Not implemented yet. CMIX is the hardest to wrap because:

1. Its source tree contains many submodels (text, image, audio, jpeg,
   x86), each with its own initialisation cost. We need to vendor the
   whole tree (`upstream/cmix/`).
2. The CLI driver in CMIX uses stdin/stdout; the codec interfaces are
   there but not really designed as a library.

Plan:

1. `git submodule add https://github.com/byronknoll/cmix
   plugins/cmix/upstream`.
2. Build `libcmix.a` from the upstream `Makefile` cross-compiled with
   MinGW-w64, with `-DSTANDALONE_LIB` style flags so we don't pull the
   `main()` driver in.
3. Implement `CCmixEncoder` / `CCmixDecoder` in `src/CmixCoder.cpp`,
   buffering each input block (CMIX needs a known total size to model
   well) and calling `cmix::compress` / `cmix::decompress`.
4. Document RAM/time expectations *prominently* in the 7-Zip method
   description string (the plugin name itself can include `(slow,
   needs RAM)`).

## Build

Same pattern as `plugins/zpaq/Makefile`. Will be added once the codec
source is in place.
