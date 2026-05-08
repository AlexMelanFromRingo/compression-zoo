# `zpaq` — 7-Zip codec plugin for ZPAQ

ZPAQ (Matt Mahoney's "context-mixing journaling archiver" algorithm)
wrapped as a 7-Zip codec. ZPAQ at level 5 typically compresses 15–20 %
better than LZMA2 ultra at the cost of ~10× encode time.

| Property | Value |
|---|---|
| Method name | `ZPAQ` |
| Method ID | `0x4F71103` (community allocation) |
| Levels | 7-Zip 1–9 → libzpaq method `0`..`5` |
| Encode speed | ~1–4 MB/s (level 5, single thread) |
| Decode speed | similar to encode |
| License (wrapper) | MIT |
| License (libzpaq) | Public domain (Matt Mahoney) |

## Status

Built and tested. The DLL ships as a 64-bit Windows codec plugin and
round-trips correctly through 7-Zip's `ICompressCoder` ABI.

  * `tests/test_linux` — 12/12 in-process round-trips through libzpaq's
    Reader/Writer adapters (the same buffer-pump that lives in
    `src/ZpaqCoder.cpp`). Linux smoke; doesn't exercise the COM ABI.
  * `tests/test_zpaq.exe` — Windows-side COM round-trip: `LoadLibrary`,
    `GetNumberOfMethods`, `CreateObject(encoder)` /
    `CreateObject(decoder)`, in-memory encode/decode of 8 KiB buffer.
    Reported `ROUND-TRIP OK` on Windows 10/11 with the DLL placed in
    `Codecs/`.

Tests for the level mapping and progress reporting still need filling
in; everything else is wired up.

## Build

From `plugins/zpaq/`, on Linux/WSL with MinGW-w64:

```bash
make                      # builds build/zpaq.dll (~1 MiB)
make install              # copies to $CODECS_DIR (default /mnt/l/Programs/7-Zip/Codecs)
```

Then in 7-Zip:

```cmd
7z a archive.7z -m0=zpaq -mx5 input.dat
7z x archive.7z
```

## Test

```bash
g++ -O2 -std=c++14 tests/test_linux.cpp upstream/libzpaq.cpp \
    -DNOJIT -Dunix -o tests/test_linux
./tests/test_linux             # Linux smoke

# Windows (built once on Linux):
x86_64-w64-mingw32-g++ -O2 -std=c++14 tests/test_zpaq.cpp \
    -o tests/test_zpaq.exe -loleaut32 -luuid
# then on a Windows host:
tests\test_zpaq.exe build\zpaq.dll
```

## Level mapping

7-Zip exposes `-mx` levels 0..9. We map them to libzpaq's method
strings `"0"`..`"5"`:

| `-mx` | libzpaq method | Notes |
|------:|:--------------:|---|
| 0     | `0`            | store / fast |
| 1, 2  | `1`            | |
| 3, 4  | `2`            | |
| 5, 6  | `3`            | default |
| 7, 8  | `4`            | |
| 9     | `5`            | maximum |

## Implementation notes

- `src/ZpaqCoder.cpp` adapts `ISequentialInStream` /
  `ISequentialOutStream` to libzpaq's `Reader` / `Writer` virtual
  classes; libzpaq pulls bytes synchronously through these.
- libzpaq signals errors by calling `libzpaq::error(const char*)`. We
  define it to throw a `CZpaqError` and catch in `Code()`, mapping to
  `E_FAIL` / `E_OUTOFMEMORY`.

## Reference

- libzpaq API and the algorithm itself: <http://mattmahoney.net/dc/zpaq.html>
- Older 7-Zip ZPAQ plugin (different codebase, same method ID):
  <https://encode.su/threads/2120-ZPAQ-7-Zip-plugin>
