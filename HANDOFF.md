# Session handoff — Rust port continuation

Drop this entire file into a new chat as the first message. It tells
the next agent (or future-you) exactly what's done, what's broken,
and where to pick up.

---

## Repo layout

```
/home/young-developer/Rust/compression-zoo/
├── plugins/                   ← C++ 7-Zip codec DLLs (built)
│   ├── zpaq/   bsc/   cmix/   brotli/
│   └── upstream sources are vendored under each plugin's upstream/
├── rust/                      ← memory-safe Rust ports
│   ├── sevenz-rs/             ← LZMA/LZMA2/PPMd/BCJ/AES (mature)
│   ├── bsc-rs/                ← libbsc port (~feature-complete)
│   ├── zpaq-rs/               ← libzpaq port (decompress mostly works)
│   └── cmix-rs/               ← skeleton only
└── HANDOFF.md                 ← this file
```

## Build / test commands

```bash
cd /home/young-developer/Rust/compression-zoo/rust
cargo test --release            # 118 unit tests across all crates
cargo build --release           # builds all bin targets
```

Plugin DLLs (cross-compiled with MinGW-w64):

```bash
make -C plugins/{zpaq,bsc,cmix,brotli}
```

Test harness binaries (already built, in `/tmp/`):
- `/tmp/zpaq_make`  — wraps `libzpaq::compress(in, out, method)`
- `/tmp/qlfc_xcheck` — exercises bsc QLFC encoder
- `/tmp/rc_xcheck`   — exercises bsc range coder
- `/tmp/bwt_xcheck`  — exercises libsais BWT
- `/tmp/st_xcheck`   — exercises bsc ST encode/decode

If any are missing, source files for them are at `/tmp/*.cpp`.

## Status summary

### `bsc-rs` — DONE (encode + decode for all 5 libbsc levels)

Files:
```
rust/bsc-rs/src/
  adler32.rs   bwt.rs (forward+inverse)   coder_tables.rs
  format.rs    libbsc.rs (compress/decompress)
  lzp.rs       predictor.rs (ProbabilityCounter, ProbabilityMixer)
  qlfc.rs (static/adaptive/fast — encoder AND decoder)
  qlfc_model.rs (Model1, Model2)   rangecoder.rs   st.rs (inverse)
```

Test results:
- 49 unit tests pass.
- **30/30** Rust enc → Rust+libbsc dec (5 levels × 6 fixtures).
- **20/20** ST inverse vs libbsc (k=3..6 × 5 fixtures).
- **6/6** unbwt vs libsais.
- **8/8** range coder bidirectional vs libbsc.

Remaining work (low priority):
- Forward ST (most archives use BWT, not ST).
- libsais SA-IS port for faster forward BWT (current is prefix-doubling, O(n log² n)).

### `zpaq-rs` — DONE for decompress (methods 0–5 round-trip)

Files:
```
rust/zpaq-rs/src/
  arith.rs                  ← 32-bit binary arith coder (4 unit tests)
  io.rs                     ← Reader/Writer traits
  sha1.rs                   ← FIPS-verified
  format.rs                 ← block magic + ZPAQL header + segments
  zpaql.rs                  ← 256-opcode VM interpreter
  predictor.rs              ← 8 components: CONS/CM/ICM/MATCH/AVG/MIX2/MIX/ISSE/SSE
  predictor_tables.rs       ← squash/stretch/dt/dt2k tables
  state_table.rs            ← SNS[1024] next-state table
  decompress.rs             ← top-level decompress + PostProcessor
  bin/zpaq_decompress.rs    ← end-to-end CLI
  bin/zpaq_inspect.rs       ← block/segment header dumper
```

Test results:
- 23 unit tests pass.
- End-to-end vs `/tmp/zpaq_make`:
  ```
  ok  m0 (store)        — all fixtures
  ok  m1 (LZ77)         — all fixtures
  ok  m2                — all fixtures
  ok  m3                — all fixtures
  ok  m4 (n=8 model)    — all fixtures (incl. random 1KB and 100KB)
  ok  m5 (n=23 model)   — all fixtures
  ```
  Round-trips libzpaq.cpp (273 KB text) at 16.67% with m5.

Bugs fixed historically (don't re-introduce):
1. `decompress.rs::decompress_block` stored path passed `ph=0/pm=0` to
   `pp_write`. Should pass `header.ph`/`header.pm`. Without this fix,
   PCOMP's M array collapses to 1 byte and LZ77 back-refs alias.
2. `predictor.rs::find` had `<=` instead of `<` in the second
   row-eviction comparison. Must match libzpaq exactly:
   ```rust
   else if p1 < p2 { h1 } else { h2 }   // not p1 <= p2
   ```
3. ICM update was `(pn as i32) >> 8` (arithmetic) instead of
   `(pn >> 8) as i32` (logical). Diverges for `pn >= 2^31`.
4. ZPAQL `*B<>A` / `*C<>A` (opcodes 32/40) zeroed A's high 24 bits.
   Must preserve them: `self.a = (self.a & !0xFFu32) | (m_old as u32)`.
5. **MIX update/predict masked the cm index by `cm_mask = total - 1`**
   where `total = m * (1<<sb)`. When `m` is not a power of two (e.g.
   m=7, total=1792), `total-1` is **not** a valid bit mask — `448 &
   1791 = 192`, so writes intended for cm[448..454] landed at
   cm[192..198]. Fix: don't mask; upstream uses raw `cm[cxt+j]` and
   relies on the construction `cxt = (h & (c-1)) * m` keeping
   `cxt+m ≤ cm.size()`. (This was the "method 4/5" bug.)

Remaining zpaq-rs work:
- ZPAQ encode side (Encoder + predictor train + compile config strings).
- JIT path (intentionally skipped — interpret-only is fine).
- More archive-format tests (multi-block, multi-segment).

### `cmix-rs` — NOT STARTED

`rust/cmix-rs/src/lib.rs` is still a one-line skeleton.

Realistic scope: CMIX is ~30K lines of intricate C++ (PAQ8-derived
ensemble of dozens of sub-models with a logistic-mixer stack and
heavy preprocessing). A faithful port is multi-week work and very
bug-prone — every per-byte state divergence corrupts the bit stream.

The plugin (`plugins/cmix/cmix.dll`) works fine for archive use
today; the Rust port is purely for memory safety / portability and
isn't on the critical path.

Suggested approach when starting:
1. Vendor `plugins/cmix/upstream` is already there.
2. Start with `predictor.cpp` — the per-byte mix-of-models entry
   point. Its inputs/outputs are well-defined.
3. Expect to need ~60% of CMIX's source ported before a single-byte
   round-trip works (all the static state matters).
4. Use the same per-step trace bisection as for ZPAQ to find any
   divergence early.

### Plugins (C++) — DONE

All four DLLs built and round-trip-tested:
- `plugins/zpaq/` — ZPAQ via libzpaq, levels 1–5.
- `plugins/bsc/`  — libbsc, levels 1–9.
- `plugins/cmix/` — CMIX (slow but works).
- `plugins/brotli/` — Google Brotli, levels 1–11.

Method IDs (community-aligned):
- 0x4F71102 Brotli, 0x4F71103 ZPAQ (existing IDs reused for
  cross-plugin compatibility).
- 0x4F71200 libbsc, 0x4F71201 CMIX (proposed new).

## Memories / preferences captured (auto-memory)

`~/.claude/projects/.../memory/`:
- "Read upstream first": port from the actual `.cpp`/`.h` source,
  not derived guesses. Multiple bugs this session came from
  trusting my mental model instead of grepping libzpaq.
- The user accepts terse Russian and English; prefers fewer
  follow-up questions, more autonomous progress on long arcs.

## Suggested ordering for next session

1. **ZPAQ encode side.** Mirror of decompress: `predictor.update`
   already works; need `Encoder` (symmetric of `Decoder`, already
   in `arith.rs`), `Compiler` for config strings (~500 lines), and
   `Compresser` driver.
2. **libsais SA-IS Rust port.** Standalone, well-bounded (libsais.c
   is ~5K lines, no global state). Replaces my prefix-doubling SA
   in `bsc-rs/src/bwt.rs::suffix_array`.
3. **CMIX-rs.** Start only if explicitly asked. Multi-session.

## Tests at handoff

```
$ cargo test --release
test result: ok. 49 passed   (bsc-rs)
test result: ok. 23 passed   (zpaq-rs)
test result: ok. 46 passed   (sevenz-rs)
   total: 118 unit tests passing

cross-language:
  bsc-rs encode + decode  30/30
  bsc-rs ST inverse       20/20
  bsc-rs unbwt vs libsais  6/6
  bsc-rs range coder       8/8
  zpaq-rs methods 0..5    36/36 short fixtures + 12/12 random binaries
  Brotli plugin           12/12
```

## How to verify nothing regressed before you start

```bash
cd /home/young-developer/Rust/compression-zoo/rust
cargo test --release 2>&1 | grep "test result.*passed"
# Should show 49, 23, 46 (in some order)

# bsc-rs end-to-end:
TMP=$(mktemp -d); ENC=target/release/bsc_compress; DEC=target/release/bsc_decompress
printf 'hello' > "$TMP/in"
for L in 1 3 5 7 9; do
  $ENC $L < "$TMP/in" | $DEC > "$TMP/out" && cmp -s "$TMP/in" "$TMP/out" \
    && echo "ok L=$L" || echo "REGRESSION L=$L"
done

# zpaq-rs end-to-end (all six methods):
for M in 0 1 2 3 4 5; do
  printf 'Hello, ZPAQ!' | /tmp/zpaq_make $M > "$TMP/c.zpaq"
  target/release/zpaq_decompress < "$TMP/c.zpaq" > "$TMP/out" 2>/dev/null
  cmp -s <(printf 'Hello, ZPAQ!') "$TMP/out" && echo "ok m$M" || echo "REGRESSION m$M"
done
```
