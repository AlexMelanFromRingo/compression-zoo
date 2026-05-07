# `zpaq` — 7-Zip codec plugin for ZPAQ

ZPAQ (Matt Mahoney's "context-mixing journaling archiver" algorithm) wrapped
as a 7-Zip codec. ZPAQ at level 5 typically compresses 15–20 % better than
LZMA2 ultra at the cost of ~10× encode time.

| Property | Value |
|---|---|
| Method name | `ZPAQ` |
| Method ID | `0x4F71103` (community allocation) |
| Levels | 1–5 (5 is the strongest) |
| Encode speed | ~1–4 MB/s (level 5, single thread) |
| Decode speed | similar to encode |
| License (wrapper) | MIT |
| License (libzpaq) | Public domain (Matt Mahoney) |

## Status

Build skeleton in place; the `Makefile` and the SDK include paths are
verified. The actual codec source (`src/ZpaqCoder.cpp` + a vendored
copy of `libzpaq.cpp`/`libzpaq.h`) is the next step.

## Build (when done)

From the repo root, on Linux/WSL with MinGW-w64:

```bash
make -C plugins/zpaq
cp plugins/zpaq/build/zpaq.dll "/mnt/<drive>/Programs/7-Zip/Codecs/"
```

Then in a Windows shell:

```cmd
7z a archive.7z -m0=zpaq -mx5 input.dat
7z x archive.7z
```

## Implementation plan

1. Vendor `libzpaq.h` and `libzpaq.cpp` from
   <http://mattmahoney.net/dc/zpaq.html> (latest 7.15) into `upstream/`.
2. Implement `CZpaqEncoder` and `CZpaqDecoder` in `src/ZpaqCoder.cpp`,
   each implementing `ICompressCoder` and adapting between
   `ISequentialInStream`/`ISequentialOutStream` and libzpaq's
   `libzpaq::Reader` / `libzpaq::Writer` virtual classes.
3. Add `ICompressSetCoderProperties` to read the level (1–5).
4. Register via `REGISTER_CODEC_E(Zpaq, CZpaqDecoder, CZpaqEncoder,
   0x4F71103, "ZPAQ")`.
5. Compose the plugin DLL by linking together our codec source +
   `plugins/sdk/CPP/7zip/Compress/{CodecExports,DllExportsCompress}.cpp`
   + the vendored libzpaq.

## Reference

- libzpaq API and the algorithm itself: <http://mattmahoney.net/dc/zpaq.html>
- Older 7-Zip ZPAQ plugin (for reference, not a dependency):
  <https://encode.su/threads/2120-ZPAQ-7-Zip-plugin>
