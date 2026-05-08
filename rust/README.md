# Rust components

A workspace of memory-safe Rust ports for the compression algorithms
used by the plugins in `../plugins/` and by 7-Zip itself. Every crate
sets `#![forbid(unsafe_code)]` at the crate root.

## Crates

### `sevenz-rs/` — 7-Zip / LZMA SDK algorithms

Direct port of the C reference implementations from `7zip/C/`. Used as
the foundation for an eventual full Rust 7-Zip clone.

  * **Done**: LZMA, LZMA2, PPMd7, PPMd8, BCJ, BCJ2, AES, Delta, CRC32,
    CRC64, hash drivers.
  * **Tested**: 46 unit tests + bytewise C cross-check.

### `bsc-rs/` — libbsc port

  * **Done**: Adler-32, block header parser, LZP encode + decode,
    BWT forward (suffix-array via prefix doubling) + inverse
    (sentinel-based), libbsc range coder + tables, predictor,
    `QlfcStatisticalModel1` + `QlfcStatisticalModel2`, QLFC
    **static + adaptive + fast** decoders **and** encoders, QLFC
    transform (rank + run). Top-level `compress` and `decompress`
    cover BWT + any QLFC variant + optional LZP.
  * **Cross-checks**: bit-for-bit wire-compatible with libbsc at
    levels 1, 3, 5, 7, 9. Rust enc → Rust dec **30/30** and Rust enc
    → `bsc_cli d` **30/30** (5 levels × 6 fixtures: "Hello", hosts,
    services, qlfc.cpp 111K, random 1K/64K). libbsc enc → Rust dec
    **40/40** verified previously.
  * **TODO**: Schindler transform forward + inverse (for ST3..ST8
    sorter modes; not used by the bsc plugin's default settings),
    libsais SA-IS port for faster forward BWT.

### `zpaq-rs/` — libzpaq port

  * **Done**: I/O traits, 32-bit binary arithmetic coder, SHA-1
    (FIPS 180-2 vectors verified), block-format reader (magic finder,
    ZPAQL header, segment headers, end markers), ZPAQL VM (full 256-
    opcode interpreter), Predictor (all 8 components — CONS, CM, ICM,
    MATCH, AVG, MIX2, MIX, ISSE, SSE — including the squash/stretch
    sigmoid lookup tables), PostProcessor with PASS + PROG/PCOMP
    states, top-level `decompress` for both stored and modeled
    blocks. **22 unit tests pass.**
  * **End-to-end results** vs libzpaq's `compress(method)`:
      - method 0 (store): all fixtures decompress with SHA-1 verified.
      - method 1 (LZ77, n=0): all fixtures (after fixing PostProcessor
        ph/pm propagation; before the fix, M was sized 1 byte and all
        LZ77 back-references aliased to offset 0, producing classic
        RLE-style "repeat last byte" corruption).
      - method 2: matches method 1 coverage.
      - method 3: all fixtures (LZ77 + ICM stack).
      - methods 4 & 5 (modeled, n=8 / n=23): the arithmetic decoder
        signals `Corrupt` 7-9 input bytes in. The predictor diverges
        from the encoder somewhere in the
        ICM/ISSE/MATCH/MIX update math; my port matches libzpaq
        line-by-line per spot-check, but evidently some integer
        precision detail is off. Needs a debug build of libzpaq with
        a `predict()` trace to bisect.
  * **Bugs already fixed in this port**:
      - `find()` row-eviction comparison was `<=` (mine) vs `<`
        (libzpaq). Affects which colliding row gets evicted.
      - ICM update did `(pn as i32) >> 8` (arithmetic shift) instead
        of `(pn >> 8) as i32` (logical shift). Diverges for
        `pn >= 2^31`.
      - PostProcessor stored-mode path passed ph=0/pm=0 to the PCOMP
        VM init instead of `header.ph`/`header.pm`. M collapsed to
        1 byte → all writes aliased.
      - ZPAQL `*B<>A` / `*C<>A` (opcodes 32/40) zeroed the high 24
        bits of A instead of preserving them.
  * **TODO**: track down the remaining method-4/5 predictor bug,
    add encode side, JIT path is intentionally skipped.

### `cmix-rs/` — CMIX port

  * **Done**: skeleton + Cargo crate.
  * **TODO**: practically all of CMIX. The upstream is a C++ codebase
    with hundreds of `static`-scope mutable variables and a deep
    PAQ8-derived ensemble; a faithful port is a large project on its
    own and is parked behind the simpler ones.

## Quick check

```bash
cd rust && cargo test --release
```

Builds all four crates, runs every unit test, and runs each crate's
cross-check binary (e.g. `lzp_xcheck`, `unbwt_xcheck`) against
matching C harness binaries when available.
