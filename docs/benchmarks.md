# Benchmarks

Wall-clock measurements on a single CPU thread, MinGW-W64 13 (host),
g++ 13 (Linux harness). For CMIX, see the dedicated note below — it
gets its own row only on tiny inputs because of its time/RAM cost.

Reproduce with:

```bash
cd plugins/zpaq && make tests/zpaq_cli  # one-time; cli compiles libzpaq
cd plugins/bsc  && make tests/bsc_cli   # one-time
cd plugins/cmix && make tests/test_linux
scripts/bench.sh <input>
```

The CLI binaries used to drive the codecs are
`plugins/zpaq/tests/zpaq_cli`, `plugins/bsc/tests/bsc_cli`, and
`plugins/cmix/tests/test_linux`. They use the same libzpaq /
libbsc / cmix code that the corresponding 7-Zip plugin DLLs do —
they're just stripped down to "stdin -> stdout".

## Source code (111 KiB C++ file)

`plugins/bsc/upstream/libbsc/coder/qlfc/qlfc.cpp`, 111 242 bytes.

| codec                  |   out_size |  ratio |    time |
|------------------------|-----------:|-------:|--------:|
| xz -9e (LZMA2)         |       8076 |   7.26% |   0.06 s |
| zstd --ultra -22       |       8396 |   7.55% |   0.80 s |
| zpaq level 1           |      12718 |  11.43% |   0.01 s |
| zpaq level 3           |       9466 |   8.51% |   0.02 s |
| **zpaq level 5**       |   **6864** | **6.17%** |   0.52 s |
| bsc level 1            |       8736 |   7.85% |   0.02 s |
| bsc level 5            |       8322 |   7.48% |   0.02 s |
| bsc level 9            |       8114 |   7.29% |   0.02 s |
| cmix                   | (skipped — too slow on >4 KiB) |  |  |

Headline: **ZPAQ level 5 beats LZMA2 ultra by ~15 %** on natural source
code at the cost of ~10× encode time.

## Highly repetitive text (1 MiB)

1 MiB of `"The quick brown fox jumps over the lazy dog. "` repeated.

| codec                  |   out_size |  ratio |    time |
|------------------------|-----------:|-------:|--------:|
| xz -9e (LZMA2)         |        328 |   0.03% |   0.04 s |
| **zstd --ultra -22**   |    **146** | **0.01%** |   0.66 s |
| zpaq level 1           |        544 |   0.05% |   0.02 s |
| zpaq level 3           |        351 |   0.03% |   0.05 s |
| zpaq level 5           |        481 |   0.05% |   5.18 s |
| bsc level 1            |        352 |   0.03% |   0.04 s |
| **bsc level 5**        |    **206** | **0.02%** |   0.02 s |
| bsc level 9            |        212 |   0.02% |   0.02 s |

On extremely repetitive inputs the LZ77 family (zstd) and BWT family
(bsc) win on ratio. zpaq's overhead from per-block ZPAQL initialisation
is more visible.

## CMIX, tiny inputs

CMIX is impractical for general use (hours per GiB, >25 GiB peak RSS
on big inputs); we only benchmark it on toy inputs to confirm the
plugin works.

```
$ printf 'ABCDEFGH' | plugins/cmix/tests/test_linux encode | wc -c
13
$ printf 'ABCDEFGH' | plugins/cmix/tests/test_linux encode \
       | plugins/cmix/tests/test_linux decode
ABCDEFGH
```

For the same 8-byte input zpaq produces 80+ bytes of overhead. CMIX's
ratio advantage shows up at much larger sizes, but those are also where
its time/RAM cost makes it impractical for everyday use.

## Choosing a codec

Approximate guidance based on the numbers above:

| Want | Pick |
|---|---|
| Drop-in best-effort | LZMA2 (built into 7-Zip) |
| ~15% better than LZMA2 on text/code, can wait ~10× longer | **zpaq -mx5** (this repo) |
| Fast and slightly better than LZMA2 on repetitive data | **bsc -mx5** (this repo) |
| Maximum ratio, time and RAM budget unlimited | CMIX (this repo) |

Plain `7z a` only loads codec DLLs that 7-Zip recognises by method ID;
make sure the plugin is in your `Codecs/` directory before invoking it
(`-m0=zpaq`, `-m0=bsc`, `-m0=cmix`).
