# `brotli` — 7-Zip codec plugin for Brotli

Google's Brotli (RFC 7932) wrapped as a 7-Zip codec. At quality 11
Brotli is roughly comparable to xz/LZMA2 on typical text but ~3–5×
faster to decode; at lower qualities it slots in between zstd and xz.

| Property | Value |
|---|---|
| Method name | `Brotli` |
| Method ID | `0x4F71102` (community allocation, originally mcmilk/7-Zip-zstd) |
| Levels | 7-Zip 1–9 → Brotli quality 1, 3, 5, 7, 9, 11 |
| Encode speed | ~5–50 MB/s depending on quality (single thread) |
| Decode speed | ~250–500 MB/s (single thread) |
| License (wrapper) | MIT |
| License (Brotli) | MIT (Google) |

## Status

Built and round-trip-tested on Linux via the Brotli streaming API
(`tests/test_linux` passes 12/12). Windows-side test binary
(`tests/test_brotli.exe`) is built but needs a Windows host to run.

## Build

From `plugins/brotli/`, on Linux/WSL with MinGW-w64:

```bash
make
cp build/brotli.dll "/mnt/<drive>/Programs/7-Zip/Codecs/"
```

Then in a Windows shell:

```cmd
7z a archive.7z -m0=brotli -mx9 input.dat
7z x archive.7z
```

## Test

Linux smoke test (uses system `libbrotli-dev`):

```bash
g++ -O2 -std=c++14 tests/test_linux.cpp -lbrotlienc -lbrotlidec \
    -lbrotlicommon -o tests/test_linux
./tests/test_linux
```

Windows COM round-trip (run on Windows):

```cmd
tests\test_brotli.exe build\brotli.dll
echo %ERRORLEVEL%   :: 0 = pass
```

## Level mapping

7-Zip exposes `-mx` levels 0..9. We map them to Brotli `BROTLI_PARAM_QUALITY`:

| `-mx` | Brotli quality | Notes |
|------:|---------------:|---|
| 0     | 1              | brotli q=0 is faster but big ratio drop |
| 1, 2  | 3              | |
| 3, 4  | 5              | |
| 5, 6  | 7              | default for 7-Zip |
| 7, 8  | 9              | |
| 9     | 11             | maximum (slow) |

## Implementation notes

- `src/BrotliCoder.cpp` is built around brotli's
  `BrotliEncoderCompressStream` / `BrotliDecoderDecompressStream`.
  We pump 64 KiB ping-pong buffers between 7-Zip's
  `ISequentialInStream` / `ISequentialOutStream` and brotli's
  state-machine API.
- The encoder reads input until EOF, then issues
  `BROTLI_OPERATION_FINISH` and drains output with
  `BrotliEncoderHasMoreOutput` + `BrotliEncoderTakeOutput`.
- The decoder loops until brotli returns `BROTLI_DECODER_RESULT_SUCCESS`.
  Truncated input — `NEEDS_MORE_INPUT` with no bytes available —
  is treated as a corrupt stream.

## Reference

- Brotli upstream: <https://github.com/google/brotli>
- RFC 7932: <https://datatracker.ietf.org/doc/html/rfc7932>
