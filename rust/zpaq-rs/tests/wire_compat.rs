//! Wire-compat snapshot fixtures for `zpaq-rs`.
//!
//! Pins the byte-level output of:
//!
//!   * `expand_digit_method(method, data)` — the digit→"x..." table.
//!   * `compile(cfg).header` / `compile(cfg).pcomp` for canonical
//!     configs (covers the Compiler).
//!   * `compress_method(out, data, method)` — full encode pipeline
//!     (Compiler + LZBuffer + arith coder + format).
//!
//! Each snapshot hashes the output with SHA-1 (using zpaq-rs's own
//! SHA-1 implementation, which is part of the codec's checksum
//! contract anyway) and asserts that today's output equals a
//! committed digest.
//!
//! These are snapshot tests — when the format genuinely changes, the
//! committed digest needs updating and the change should be reviewed
//! for wire-compat regressions against upstream libzpaq.
//!
//! Independent decode-by-upstream is exercised separately by the
//! cross-language harness; this file just locks in the byte stream
//! we currently produce.

use zpaq_rs::compiler::compile;
use zpaq_rs::compress::{compress_method, expand_digit_method};
use zpaq_rs::io::{SliceReader, VecWriter};
use zpaq_rs::decompress::decompress;
use zpaq_rs::lzbuffer::{preprocess, LzArgs};
use zpaq_rs::sha1::sha1_of;

/// Hex-encode a byte slice, lowercase.
fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for &v in b { s.push_str(&format!("{:02x}", v)); }
    s
}

/// Short SHA-1 prefix used in fixtures. Cuts noise in test output —
/// 16 hex chars = 64 bits is collision-resistant for a fixed corpus.
fn sha1_short(data: &[u8]) -> String { hex(&sha1_of(data))[..16].to_string() }

/// Method strings → expected `expand_digit_method(...)` output for
/// a 100-byte input. Asserts the upstream-equivalent shape.
#[test]
fn expand_digit_method_pins_canonical_shapes() {
    let data = vec![0u8; 100];
    let cases: &[(&str, &str)] = &[
        ("0",       "00,0"),
        ("1",       "x0,1,5,0,3,20"),
        ("2",       "x0,1,4,0,7,21,1"),
        ("3",       "x0,2,12,0,7,21,1c0,0,511i2"),
        ("4",       "x0,0ci1,1,1,1,2am"),
        ("5",       "x0,0w1i1c256ci1,1,1,1,1,1,2ac0,2,0,255i1\
                     c0,3,0,0,255i1c0,4,0,0,0,255i1mm16ts19t0"),
        ("20,0,0",  "x0,0"),
        ("20,128,1", "x0,1,4,0,7,21,1"),
    ];
    for (input, expected) in cases {
        let got = expand_digit_method(input, &data)
            .unwrap_or_else(|| panic!("expand({}) returned None", input));
        assert_eq!(&got, expected,
            "expand_digit_method({:?}) drift: got {:?}, want {:?}",
            input, got, expected);
    }
}

/// Compiled bytecode for the `min.cfg`-equivalent config. Pins the
/// HCOMP encoding produced by the Rust compiler. (Also covered by
/// the unit test `compile_min_cfg_matches_canned` against `MIN_CFG`,
/// but reasserting via the digest gives a one-line drift signal.)
#[test]
fn compile_min_cfg_header_digest() {
    let cfg = r#"
        comp 1 2 0 0 2
          0 icm 16
          1 isse 19 0
        hcomp
          *b=a a=0 d=0 hash b-- hash *d=a d++ b-- hash b-- hash *d=a halt
        post 0 end
    "#;
    let cc = compile(cfg).expect("compile min.cfg");
    assert!(cc.pcomp.is_none(), "min.cfg has no pcomp");
    let digest = sha1_short(&cc.header);
    assert_eq!(digest, "d3757a28670bc4d6",
        "min.cfg header digest drifted (header={} bytes)",
        cc.header.len());
}

/// LZBuffer preprocess output digests for `(input, level, e8e9)`
/// combinations. The preprocessor is deterministic for a given
/// `(input, args)` pair — these fixtures pin the exact byte-stream
/// our LZ77 / BWT / E8E9 implementations emit, independent of the
/// outer block format.
#[test]
fn lzbuffer_preprocess_snapshot_fixtures() {
    let repeat: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".repeat(40);
    let bin: Vec<u8>    = (0..2048).map(|i| (i ^ 0xA5) as u8).collect();

    // (label, data, level (1..=3), e8e9, expected sha1-short, length)
    let cases: &[(&str, &[u8], u32, bool, &str, usize)] = &[
        ("repeat", &repeat, 1, false, "7f1f5299988c5ca8",   51),
        ("repeat", &repeat, 1, true,  "7f1f5299988c5ca8",   51),
        ("repeat", &repeat, 2, false, "ac8bc1d1fcf69ec7",  136),
        ("repeat", &repeat, 2, true,  "ac8bc1d1fcf69ec7",  136),
        ("repeat", &repeat, 3, false, "24cd9941e899d5af", 1805),
        ("repeat", &repeat, 3, true,  "24cd9941e899d5af", 1805),
        ("binary", &bin,    1, false, "2b35d7b883fae1c9",  267),
        ("binary", &bin,    1, true,  "2b35d7b883fae1c9",  267),
        ("binary", &bin,    2, false, "5a5d003a5ecab96f",  346),
        ("binary", &bin,    2, true,  "5a5d003a5ecab96f",  346),
        ("binary", &bin,    3, false, "bed0ddd01ed5aa48", 2053),
        ("binary", &bin,    3, true,  "bed0ddd01ed5aa48", 2053),
    ];

    for (label, data, level, e8e9, want_digest, want_len) in cases {
        let args = LzArgs {
            log_block_mib: 4,
            level_flag:    level | if *e8e9 { 4 } else { 0 },
            min_match:     if *level == 1 { 4 } else { 1 },
            min_match2:    0,
            log_bucket:    0,
            log_ht_size:   16,
        };
        let out = preprocess(data, args);
        let digest = sha1_short(&out);
        assert_eq!(out.len(), *want_len,
            "preprocess size drift for ({}, L={}, e8e9={}): {} (want {})",
            label, level, e8e9, out.len(), want_len);
        assert_eq!(&digest, want_digest,
            "preprocess digest drift for ({}, L={}, e8e9={}): {} (want {})",
            label, level, e8e9, digest, want_digest);
    }
}

/// Snapshot of the full encoded byte stream for a curated set of
/// (input, method) pairs. Each entry locks in:
///
///   * the framing produced by `Compresser` (block magic, post-process
///     header, segment headers, EOF markers, terminators);
///   * the LZBuffer / E8E9 preprocessor output (when applicable);
///   * the arith-coded predictor output (when applicable);
///   * the Compiler-emitted HCOMP bytecode embedded in the block
///     header.
///
/// In addition, each snapshot is verified to round-trip back through
/// the Rust decoder — so a digest drift accompanied by a successful
/// round-trip is "format intentionally changed; update the digest";
/// a digest drift with a failing round-trip is a real regression.
#[test]
fn compress_method_snapshot_fixtures() {
    let repeat: Vec<u8> = b"Hello, ZPAQ! Hello, ZPAQ!".repeat(8);
    let periodic: Vec<u8> = (0..1024).map(|i| (i % 7) as u8).collect();

    // (label, data, method, sha1-short, byte length)
    let cases: &[(&str, &[u8], &str, &str, usize)] = &[
        ("repeat",   &repeat,   "0", "5af0c18aa463355f", 242),
        ("repeat",   &repeat,   "1", "9ffa57cd7e8bde71", 377),
        ("repeat",   &repeat,   "3", "bcb53235af835ec5", 256),
        ("repeat",   &repeat,   "4", "eb9111fe21b3d418", 132),
        ("repeat",   &repeat,   "5", "77536f1fe1b1218e", 400),
        ("periodic", &periodic, "0", "49daa31a87ddad98", 1066),
        ("periodic", &periodic, "1", "0be57740654514fb", 369),
        ("periodic", &periodic, "3", "771aca82782bb869", 251),
        ("periodic", &periodic, "4", "d1107566639b2e21", 119),
        ("periodic", &periodic, "5", "9952e232557587ff", 341),
    ];

    for (label, data, method, want_digest, want_len) in cases {
        let out = compress_method(VecWriter::new(), data, method)
            .unwrap_or_else(|e| panic!(
                "compress_method({:?}, {}, {}) failed: {:?}",
                label, data.len(), method, e));
        let bytes = out.into_inner();
        let digest = sha1_short(&bytes);
        assert_eq!(bytes.len(), *want_len,
            "wire size drift for ({}, m={}): {} bytes (want {})",
            label, method, bytes.len(), want_len);
        assert_eq!(&digest, want_digest,
            "wire digest drift for ({}, m={}): {} (want {})",
            label, method, digest, want_digest);

        // Round-trip safety check — every snapshot must decode back
        // to the original input.
        let mut r = SliceReader::new(&bytes);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap_or_else(|e|
            panic!("decode failed for ({}, m={}): {:?}", label, method, e));
        assert_eq!(w.into_inner(), *data,
            "round-trip mismatch for ({}, m={})", label, method);
    }
}
