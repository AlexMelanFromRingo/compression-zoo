# compression-zoo

A monorepo of high-ratio lossless compression tooling. Two halves:

  1. Four 7-Zip codec plugin DLLs (ZPAQ, libbsc, CMIX, Brotli) that
     give 7-Zip better compression methods than its built-in LZMA2.
  2. A memory-safe Rust port of the algorithms behind those plugins
     and behind 7-Zip itself, with `#![forbid(unsafe_code)]`.

## Status at a glance

| Component | Built | Tested | Installed |
|---|---|---|---|
| `plugins/zpaq/` (`zpaq.dll`) | ✓ 998 KiB | 12/12 round-trip cases | `/Codecs/zpaq.dll` |
| `plugins/bsc/` (`bsc.dll`) | ✓ 5.9 MiB | 10/10 round-trip cases | `/Codecs/bsc.dll` |
| `plugins/cmix/` (`cmix.dll`) | ✓ 148 MiB | cross-process round-trip (see plugin README for limits) | `/Codecs/cmix.dll` |
| `plugins/brotli/` (`brotli.dll`) | ✓ 1.4 MiB | 12/12 Linux smoke; Windows test built, awaiting host | pending |
| `rust/sevenz-rs/` | LZMA, LZMA2, PPMd7, PPMd8, BCJ, AES, hashes, CRC | 46 unit tests + bytewise C cross-check | — |
| `rust/bsc-rs/` | adler32, format, LZP, BWT forward+inverse, range coder, predictor, QLFC static/adaptive/fast (encode + decode), top-level compress + decompress | 47 unit tests + 30/30 Rust enc → Rust+libbsc dec at levels 1/3/5/7/9 (and 40/40 libbsc enc → Rust dec previously) | — |
| `rust/zpaq-rs/` | I/O traits, arith coder, SHA-1, block format, ZPAQL VM (256 opcodes), Predictor (8 components), PostProcessor PASS+PROG/PCOMP, top-level decompress | 22 unit tests + 12/18 end-to-end vs libzpaq (methods 0/1/2/3 all fixtures pass, 4/5 fail with a subtle predictor divergence) | — |
| `rust/cmix-rs/` | skeleton | — | — |

### Method IDs (recorded in `docs/method-ids.md`)

| ID          | Codec |
|-------------|-------|
| `0x4F71102` | Brotli (consistent with the existing community ID) |
| `0x4F71103` | ZPAQ (consistent with the existing community ID) |
| `0x4F71200` | libbsc (proposed) |
| `0x4F71201` | CMIX (proposed) |

## Why bother — measured ratios

From `docs/benchmarks.md`, on a 111 KiB C++ source file:

| codec               | out_size | ratio  |
|---------------------|---------:|-------:|
| xz -9e (LZMA2)      |     8076 |  7.26% |
| zstd --ultra -22    |     8396 |  7.55% |
| **zpaq level 5**    | **6864** | **6.17%** |
| bsc level 9         |     8114 |  7.29% |

**ZPAQ level 5 compresses ~15 % smaller than LZMA2 ultra** in
exchange for ~10× encode time. CMIX gives an additional ~2× ratio
improvement but at hours of CPU per gigabyte and >25 GiB peak RSS,
so it's mostly useful for tiny corpora or as a benchmark.

## Quick start (use the plugins)

You need MinGW-w64 to cross-compile the DLLs (`apt install
mingw-w64`). On Windows you'd use MSVC; the upstream code is C++14
and doesn't need anything fancy.

```bash
make -C plugins/zpaq   && make -C plugins/zpaq   install \
    CODECS_DIR="/mnt/<drive>/Programs/7-Zip/Codecs"
make -C plugins/bsc    && make -C plugins/bsc    install
make -C plugins/cmix   && make -C plugins/cmix   install
make -C plugins/brotli && make -C plugins/brotli install
```

Then in 7-Zip:

```cmd
7z a archive.7z -m0=zpaq   -mx5 input
7z x archive.7z

7z a archive.7z -m0=bsc    -mx5 input
7z a archive.7z -m0=brotli -mx9 input
7z a archive.7z -m0=cmix         input    :: be patient
```

## Quick start (Rust port)

The crate workspace lives at `rust/`:

```bash
cd rust && cargo test --release
```

That runs 46 sevenz-rs tests (LZMA / LZMA2 / PPMd7 / PPMd8 /
BCJ / AES / hashes / CRC) and 13 bsc-rs tests (Adler-32 + the
`bsc_block_info` header parser). All pass and all cross-check
byte-for-byte against the C reference.

## Components

### `plugins/` — Windows codec DLLs for 7-Zip

Each plugin is a 64-bit Windows DLL implementing the 7-Zip codec
interface (`ICompressCoder`). Drop the DLL into the `Codecs/`
subdirectory of your 7-Zip install and the new method ID becomes
available from the GUI and CLI.

### `rust/sevenz-rs/` — Memory-safe Rust port of 7-Zip algorithms

Direct port of the C reference implementations from `7zip/C/` with
`#![forbid(unsafe_code)]`. Used as the foundation for the future
"variant A" goal of bringing the Rust port to feature parity with
7-Zip's GUI (container, profiles, multithreading).

### `rust/bsc-rs/`, `rust/zpaq-rs/`, `rust/cmix-rs/`

Per-plugin Rust ports. Currently `bsc-rs` has the Adler-32 and block
header parser; `zpaq-rs` and `cmix-rs` are skeleton crates pending
implementation.

## Licensing

This repo aggregates components with different upstream licenses. The
top-level project tooling (build scripts, docs) is MIT. Each
subdirectory under `plugins/` and `rust/` has its own `LICENSE`
reflecting the upstream algorithm:

- `plugins/zpaq/`, `rust/zpaq-rs/` — MIT/Unlicense (wrappers) over
  public-domain libzpaq
- `plugins/bsc/`, `rust/bsc-rs/` — Apache-2.0
- `plugins/cmix/`, `rust/cmix-rs/` — GPL-3.0 (note: linking with the
  DLL imposes GPL-3.0 on derived works that statically link it;
  loading it as a 7-Zip plugin at runtime is fine)
- `plugins/brotli/` — MIT (wrapper) over MIT-licensed Google Brotli
- `plugins/sdk/` — LGPL-2.1+ / BSD (vendored 7-Zip plugin SDK
  headers, Igor Pavlov)
- `rust/sevenz-rs/` — LGPL-2.1+ matching 7-Zip upstream, with
  public-domain fallback for files derived from the LZMA SDK only

See each subdirectory's `LICENSE` and `README.md` for specifics.
