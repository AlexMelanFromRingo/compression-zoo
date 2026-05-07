# compression-zoo

A monorepo of high-ratio lossless compression tooling: 7-Zip codec plugins
wrapping algorithms that compress better than LZMA2, plus an in-progress
memory-safe Rust port of the 7-Zip / LZMA SDK.

## Components

### `plugins/` — Windows codec DLLs for 7-Zip

Each plugin is a 64-bit Windows DLL implementing the 7-Zip codec interface
(`ICompressCoder`). Drop the DLL into the `Codecs/` subdirectory of your
7-Zip install and the new method ID becomes available from the GUI and CLI.

| Plugin | Algorithm | Method ID | License | Notes |
|---|---|---|---|---|
| `zpaq` | ZPAQ (Matt Mahoney) | `0x4F71103` | MIT (wrapper); ZPAQ is public domain | Levels 1–5; level 5 ≈ 15–20 % better ratio than LZMA2 ultra |
| `bsc` | libbsc (Ilya Grebnov) | TBD | Apache-2.0 | BWT + ST + LZP; modest improvement, comparable speed |
| `cmix` | CMIX (Byron Knoll) | TBD | GPL-3.0 | Top-of-leaderboard ratio; **hours of CPU and >25 GB RAM per GB** |

See `docs/method-ids.md` for the community method-ID registry and
`docs/benchmarks.md` for measured ratios.

### `rust/sevenz-rs` — Memory-safe Rust port of 7-Zip algorithms

Port of the C reference implementations from `7zip/C/` with
`#![forbid(unsafe_code)]`. Currently implements:

- LZMA encoder/decoder, LZMA2 framing
- PPMd7 (PPMdH) and PPMd8 (PPMdI) encoders/decoders
- BCJ family (x86, ARM, ARM64, IA64, PPC, SPARC, RISC-V, ARM-Thumb) and BCJ2
- AES (CBC + CTR), MD5, SHA-1, SHA-256, SHA-512, Blake2s, XXH64
- CRC-32 (slicing-by-N) and CRC-64

The Rust port is a research/teaching artefact, cross-checked byte-for-byte
against the C reference. The longer-term plan is to bring it up to feature
parity with the 7-Zip GUI (container, profiles, multithreading) and to
re-implement each plugin in Rust.

## Quick start (plugins)

Cross-compiling from Linux/WSL:

```bash
sudo apt install mingw-w64
cd plugins/zpaq && make
cp build/zpaq.dll "/mnt/<drive>/Program Files/7-Zip/Codecs/"
```

In 7-Zip CLI: `7z a archive.7z -m0=zpaq -mx5 file`. In the GUI, ZPAQ appears
in the *Method* dropdown after the DLL is loaded.

## Licensing

This repo aggregates components with different upstream licenses. The
top-level project tooling (build scripts, docs) is MIT. Each subdirectory
under `plugins/` has its own `LICENSE` reflecting the upstream algorithm:

- `plugins/zpaq/` — MIT (wrapper) over public domain
- `plugins/bsc/` — Apache-2.0
- `plugins/cmix/` — GPL-3.0 (note: linking with this DLL imposes GPL-3.0
  on derived works that statically link it; loading it as a 7-Zip plugin at
  runtime is fine)
- `plugins/sdk/` — LGPL-2.1+ / BSD (vendored 7-Zip plugin SDK headers,
  Igor Pavlov)
- `rust/sevenz-rs/` — LGPL-2.1+ to match 7-Zip upstream, with public-domain
  fallback for files that derive from the LZMA SDK only

See each subdirectory's `LICENSE` and `README.md` for specifics.

## Status

Work in progress. The Rust port is functionally complete for the listed
algorithms; container-level features and the plugin DLLs are still being
built. Not yet recommended for production use.
