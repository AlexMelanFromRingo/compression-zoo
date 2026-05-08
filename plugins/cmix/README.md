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

Implemented and built. The DLL ships as a 7-Zip codec plugin and works
correctly for the typical "compress in one process, extract in another"
workflow.

**Known limitation: same-process round-trip does not work.** CMIX is built
around many `static`-scope mutable variables in `paq8.cpp`, `fxcmv1.cpp`
and other models (e.g. `paq8::y`, `paq8::bpos`, `paq8::blpos`,
`paq8::c4`, `paq8::col`, `paq8::x4`, dozens of function-local `static`s).
These are not reset between `Predictor` instances, so encoding then
decoding inside the same process gets the decoder Predictor a non-zero
starting state and round-trips fail starting from byte 1.

In practice this means:

  * `7z a archive.7z -m0=cmix file` — works (encode-only, fresh process)
  * `7z x archive.7z` later in a new 7-Zip invocation — works
  * Re-using a single 7-Zip process to encode then test the archive — fails

Fixing this for in-process reuse would require a significant CMIX
refactor (resetting every static state across all 30+ source files).
The Linux test harness (`tests/run.sh`) drives encode and decode as
two separate `test_linux` processes for that reason.

## What's vendored

`upstream/` is a snapshot of <https://github.com/byronknoll/cmix> with
two small modifications, marked `// CMIX-MOD:` in the source:

  * `src/coder/encoder.{h,cpp}` — `std::ofstream*` → `std::ostream*`.
  * `src/coder/decoder.{h,cpp}` — `std::ifstream*` → `std::istream*`.

Those let our wrapper feed CMIX from a `std::ostringstream` /
`std::istringstream` instead of demanding a real file. We also omit the
preprocessor (`src/preprocess/`) and the enwik9-specific tools
(`src/enwik9-preproc/`); only the core context-mixing predictor + arith
coder are linked in.

## Build

```bash
make -C plugins/cmix
make -C plugins/cmix install      # copies build/cmix.dll to $CODECS_DIR
```

The DLL is large (~150 MiB) because of the static PAQ8 tables.

## Test

```bash
make -C plugins/cmix
plugins/cmix/tests/run.sh tiny   # 8 bytes
plugins/cmix/tests/run.sh small  # 64 bytes; ~10 s wall clock
```

`run.sh` deliberately spawns separate processes for encode and decode
to side-step the same-process global-state bug.
