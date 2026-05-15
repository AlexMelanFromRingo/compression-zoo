//! `runner.cpp` — encode / decode driver.
//!
//! Provides byte-stream encode/decode entry points around the
//! arithmetic coder and top-level [`Predictor`]. Mirrors the
//! essentials of upstream's `cmix -c` / `cmix -d` paths minus the
//! file-type detection (lives in [`crate::preprocess`]).
//!
//! Header layout (independent of upstream's vocab-bitmap header):
//!
//! * Bytes `[0..8]`: big-endian u64 = original input byte length.
//! * Bytes `[8..]`:  raw arith-coder output.

#![forbid(unsafe_code)]

use std::io::{Read, Write, Result as IoResult, ErrorKind, Error as IoError};

use crate::coder::{ByteSink, ByteSource, Decoder, Encoder};
use crate::orchestrator::{CmixPredictor, Config};
use crate::predictor::Predictor;

const HEADER_LEN: usize = 8;

struct WriteSink<W: Write> { w: W }
impl<W: Write> ByteSink for WriteSink<W> {
    fn put(&mut self, b: u8) {
        // The arith-coder's ByteSink is infallible by contract.
        // Treat I/O errors as fatal here — callers should ensure the
        // sink can hold the encoded output (in-memory `Vec<u8>` or a
        // fully writable file).
        self.w.write_all(&[b]).expect("encoder sink write failed");
    }
}

struct ReadSource<R: Read> { r: R }
impl<R: Read> ByteSource for ReadSource<R> {
    fn get(&mut self) -> u8 {
        let mut b = [0u8; 1];
        match self.r.read(&mut b) {
            Ok(1) => b[0],
            _     => 0, // EOF — upstream's `is->good()` fallback
        }
    }
}

/// paq8 memory level used by the runner's default `encode`/`decode`.
///
/// Upstream cmix runs paq8 at level 11 (~25 GiB RAM). cmix-rs defaults
/// to level 0 so the encode/decode pipeline stays usable on a typical
/// dev machine; callers that have the RAM can use the `*_leveled`
/// entry points with a higher level. The paq8 level must match
/// between encode and decode (it is *not* stored in the header — it
/// is a property of the codec configuration, like upstream).
pub const DEFAULT_PAQ8_LEVEL: u32 = 0;

/// Encode `input` into `output` using the top-level [`Predictor`].
/// Returns the number of input bytes consumed.
pub fn encode<R: Read, W: Write>(
    mut input: R, mut output: W,
) -> IoResult<u64> {
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    encode_bytes(&buf, &mut output)
}

/// Encode `bytes` straight into `output` at [`DEFAULT_PAQ8_LEVEL`].
pub fn encode_bytes<W: Write>(
    bytes: &[u8], output: &mut W,
) -> IoResult<u64> {
    encode_bytes_leveled(bytes, output, DEFAULT_PAQ8_LEVEL)
}

/// Encode `bytes` into `output` with an explicit paq8 memory level.
pub fn encode_bytes_leveled<W: Write>(
    bytes: &[u8], output: &mut W, paq8_level: u32,
) -> IoResult<u64> {
    let length = bytes.len() as u64;
    let mut hdr = [0u8; HEADER_LEN];
    for i in 0..HEADER_LEN {
        hdr[i] = ((length >> ((HEADER_LEN - 1 - i) * 8)) & 0xff) as u8;
    }
    output.write_all(&hdr)?;

    let sink = WriteSink { w: &mut *output };
    let mut enc = Encoder::new(sink, Predictor::with_paq8_level(paq8_level));
    for &byte in bytes {
        for i in (0..8).rev() {
            enc.encode(((byte >> i) & 1) as i32);
        }
    }
    enc.flush();
    Ok(length)
}

/// Decode `input` (an encoded stream produced by [`encode`]) into
/// `output`. Returns the number of decoded bytes.
pub fn decode<R: Read, W: Write>(
    input: R, output: W,
) -> IoResult<u64> {
    decode_leveled(input, output, DEFAULT_PAQ8_LEVEL)
}

/// Decode with an explicit paq8 memory level — must match the level
/// used at encode time.
pub fn decode_leveled<R: Read, W: Write>(
    mut input: R, mut output: W, paq8_level: u32,
) -> IoResult<u64> {
    let mut hdr = [0u8; HEADER_LEN];
    input.read_exact(&mut hdr)?;
    let mut length: u64 = 0;
    for &b in &hdr {
        length = (length << 8) | (b as u64);
    }
    if length > u32::MAX as u64 * 16 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "decoded length implausibly large — likely corrupted header",
        ));
    }
    let src = ReadSource { r: input };
    let mut dec = Decoder::new(src, Predictor::with_paq8_level(paq8_level));
    let mut byte_buf = [0u8; 1];
    for _ in 0..length {
        let mut byte: u8 = 0;
        for _ in 0..8 {
            let bit = dec.decode();
            byte = (byte << 1) | (bit as u8);
        }
        byte_buf[0] = byte;
        output.write_all(&byte_buf)?;
    }
    Ok(length)
}

// ============================================================
// Full-orchestrator entry points
// ============================================================

/// Build the 256-bit vocab bitmap from a byte slice — `vocab[i] =
/// true` iff `bytes` contains byte `i`. Tightening the vocab shrinks
/// the LSTM ByteMixer (`vocab_size × vocab_size` per layer) and zeros
/// out impossible byte positions in every byte-aware model. Header
/// layout for the encoded stream stores the vocab as a 32-byte
/// bitmask after the length.
fn discover_vocab(bytes: &[u8]) -> [bool; 256] {
    let mut v = [false; 256];
    for &b in bytes { v[b as usize] = true; }
    // Empty input → default to ASCII so the predictor builds something.
    if bytes.is_empty() {
        for i in 0..128 { v[i] = true; }
    }
    v
}

fn vocab_to_bitmask(v: &[bool; 256]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..256 {
        if v[i] { out[i / 8] |= 1 << (i % 8); }
    }
    out
}

fn vocab_from_bitmask(bm: &[u8; 32]) -> [bool; 256] {
    let mut v = [false; 256];
    for i in 0..256 {
        if (bm[i / 8] >> (i % 8)) & 1 != 0 { v[i] = true; }
    }
    v
}

const VOCAB_HEADER_LEN: usize = 32;

/// Encode `bytes` into `output` using the full [`CmixPredictor`]
/// orchestrator (3-layer mixer tree + SSE + LSTM byte mixer + every
/// upstream sub-model). Memory profile is set by `config`. The
/// vocab is derived from `bytes` and written into the header so the
/// decoder reconstructs an identical predictor.
pub fn encode_bytes_full<W: Write>(
    bytes: &[u8], output: &mut W, config: Config,
) -> IoResult<u64> {
    encode_bytes_full_pretrained(bytes, output, config, &[])
}

/// Like [`encode_bytes_full`] but pretrains the predictor's bit-level
/// model bank on `pretrain` before encoding (mirrors upstream
/// `preprocessor::Pretrain` which warms the predictor on a dictionary
/// file). The decoder must call [`decode_full_pretrained`] with the
/// same pretrain buffer to reach an identical predictor state.
pub fn encode_bytes_full_pretrained<W: Write>(
    bytes: &[u8], output: &mut W, config: Config, pretrain: &[u8],
) -> IoResult<u64> {
    let length = bytes.len() as u64;
    let mut hdr = [0u8; HEADER_LEN];
    for i in 0..HEADER_LEN {
        hdr[i] = ((length >> ((HEADER_LEN - 1 - i) * 8)) & 0xff) as u8;
    }
    output.write_all(&hdr)?;

    let vocab256 = discover_vocab(bytes);
    let bitmask = vocab_to_bitmask(&vocab256);
    output.write_all(&bitmask)?;

    let mut predictor = CmixPredictor::new(vocab256.to_vec(), config);
    for &b in pretrain {
        for i in (0..8).rev() {
            predictor.pretrain(((b >> i) & 1) as i32);
        }
    }

    let sink = WriteSink { w: &mut *output };
    let mut enc = Encoder::new(sink, predictor);
    for &byte in bytes {
        for i in (0..8).rev() {
            enc.encode(((byte >> i) & 1) as i32);
        }
    }
    enc.flush();
    Ok(length)
}

/// Decode a stream produced by [`encode_bytes_full`]. `config` must
/// match the encode-time configuration exactly — it is *not* stored
/// in the header, like upstream cmix. The vocab IS stored (32-byte
/// bitmask after the 8-byte length) so the predictor is reconstructed
/// identically.
pub fn decode_full<R: Read, W: Write>(
    input: R, output: W, config: Config,
) -> IoResult<u64> {
    decode_full_pretrained(input, output, config, &[])
}

/// Like [`decode_full`] but pretrains the predictor on `pretrain`
/// before decoding — must match the encoder's pretrain bytes exactly.
pub fn decode_full_pretrained<R: Read, W: Write>(
    mut input: R, mut output: W, config: Config, pretrain: &[u8],
) -> IoResult<u64> {
    let mut hdr = [0u8; HEADER_LEN];
    input.read_exact(&mut hdr)?;
    let mut length: u64 = 0;
    for &b in &hdr { length = (length << 8) | (b as u64); }
    if length > u32::MAX as u64 * 16 {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "decoded length implausibly large — likely corrupted header",
        ));
    }
    let mut vbm = [0u8; VOCAB_HEADER_LEN];
    input.read_exact(&mut vbm)?;
    let vocab256 = vocab_from_bitmask(&vbm);

    let mut predictor = CmixPredictor::new(vocab256.to_vec(), config);
    for &b in pretrain {
        for i in (0..8).rev() {
            predictor.pretrain(((b >> i) & 1) as i32);
        }
    }

    let src = ReadSource { r: input };
    let mut dec = Decoder::new(src, predictor);
    let mut byte_buf = [0u8; 1];
    for _ in 0..length {
        let mut byte: u8 = 0;
        for _ in 0..8 {
            let bit = dec.decode();
            byte = (byte << 1) | (bit as u8);
        }
        byte_buf[0] = byte;
        output.write_all(&byte_buf)?;
    }
    Ok(length)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(plain: &[u8]) {
        let mut encoded = Vec::new();
        encode_bytes(plain, &mut encoded).unwrap();
        let mut decoded = Vec::new();
        decode(&encoded[..], &mut decoded).unwrap();
        assert_eq!(decoded, plain,
            "round-trip must reproduce input exactly");
    }

    /// Empty input — header still emitted, decoder reads zero bytes.
    #[test]
    #[ignore = "Predictor::new allocates several GB — heavy test"]
    fn round_trip_empty() { round_trip(b""); }

    /// Single byte. Smallest non-trivial case.
    #[test]
    #[ignore = "Predictor::new allocates several GB — heavy test"]
    fn round_trip_single_byte() { round_trip(b"A"); }

    /// Short ASCII text — exercises 8 bits × 13 bytes through the
    /// full byte_boundary + per-bit pipeline.
    #[test]
    #[ignore = "Predictor::new allocates several GB — heavy test"]
    fn round_trip_hello_world() { round_trip(b"Hello, World!"); }

    /// Random-ish binary bytes.
    #[test]
    #[ignore = "Predictor::new allocates several GB — heavy test"]
    fn round_trip_pseudo_random_bytes() {
        let mut data = Vec::with_capacity(64);
        let mut x: u32 = 0xC0FFEE;
        for _ in 0..64 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            data.push((x >> 24) as u8);
        }
        round_trip(&data);
    }

    /// Round-trip through the full orchestrator at `Config::tiny()`.
    /// The orchestrator is the upstream-faithful predictor — three-
    /// layer mixer tree + every model bank from `predictor.cpp`. The
    /// tiny config disables PAQ8/FXCM/SSE/LSTM so the test stays
    /// laptop-friendly (peak heap < 256 MB).
    #[test]
    fn round_trip_orchestrator_tiny() {
        let plain = b"Hello, full orchestrator!";
        let mut encoded = Vec::new();
        encode_bytes_full(plain, &mut encoded, Config::tiny()).unwrap();
        let mut decoded = Vec::new();
        decode_full(&encoded[..], &mut decoded, Config::tiny()).unwrap();
        assert_eq!(decoded, plain,
            "orchestrator round-trip must be exact");
    }

    /// Pretrained round-trip — encoder + decoder both warm their
    /// predictors on the same English sentence before the payload.
    /// Round-trip must still be bit-exact.
    #[test]
    fn round_trip_orchestrator_tiny_pretrained() {
        let pretrain = b"the quick brown fox jumps over the lazy dog";
        let plain    = b"the quick brown fox";
        let mut encoded = Vec::new();
        encode_bytes_full_pretrained(plain, &mut encoded,
            Config::tiny(), pretrain).unwrap();
        let mut decoded = Vec::new();
        decode_full_pretrained(&encoded[..], &mut decoded,
            Config::tiny(), pretrain).unwrap();
        assert_eq!(decoded, plain);
    }

    /// Mismatched pretrain corrupts the decode — guards against
    /// callers silently producing wrong output when one side
    /// pretrains and the other doesn't.
    #[test]
    fn round_trip_pretrained_mismatch_corrupts_decode() {
        let plain = b"the quick brown fox";
        let mut encoded = Vec::new();
        encode_bytes_full_pretrained(plain, &mut encoded,
            Config::tiny(), b"some pretrain bytes").unwrap();
        let mut decoded = Vec::new();
        // Decoder uses empty pretrain → predictor diverges → output
        // is garbage. We just verify it's not equal (no panic, no
        // crash; the decoder still produces *length* bytes).
        decode_full_pretrained(&encoded[..], &mut decoded,
            Config::tiny(), b"").unwrap();
        assert_ne!(decoded, plain);
        assert_eq!(decoded.len(), plain.len());
    }

    /// Round-trip with PAQ8 + LSTM ByteMixer enabled (`Config::medium`).
    /// Heavier (~4 GB peak), so marked `#[ignore]` for normal CI runs.
    #[test] #[ignore = "Config::medium allocates ~4 GB — heavy test"]
    fn round_trip_orchestrator_medium() {
        let plain = b"The quick brown fox jumps over the lazy dog.";
        let mut encoded = Vec::new();
        encode_bytes_full(plain, &mut encoded, Config::medium()).unwrap();
        let mut decoded = Vec::new();
        decode_full(&encoded[..], &mut decoded, Config::medium()).unwrap();
        assert_eq!(decoded, plain);
    }

    /// Upstream parity sanity check on a 134-byte repeated-text input.
    /// Reference upstream cmix -c (compiled from plugins/cmix/upstream
    /// at this commit) compresses the same input to **59 bytes**
    /// (cross-entropy 3.522 bits/byte). cmix-rs currently uses only
    /// fxcmv1 in its top-level Predictor — full upstream parity will
    /// require finishing the paq8/PPMD/LSTM mixer wiring (#7, full
    /// #8). Until then, this test enforces a 2× upper bound on
    /// compressed payload so a regression in fxcmv1 quality is
    /// caught early.
    #[test]
    #[ignore = "Predictor::new allocates several GB — heavy test"]
    fn upstream_parity_size_bound() {
        let plain: &[u8] =
            b"The quick brown fox jumps over the lazy dog. \
The quick brown fox jumps over the lazy dog. \
The quick brown fox jumps over the lazy dog.";
        assert_eq!(plain.len(), 134);
        let mut encoded = Vec::new();
        encode_bytes(plain, &mut encoded).unwrap();
        // 8-byte header + payload. Upstream payload is 59 bytes —
        // allow up to 2× = 118 bytes of payload until full Predictor
        // lands.
        let payload = encoded.len() - 8;
        assert!(payload <= 118,
            "cmix-rs payload {} bytes exceeds 2× upstream baseline (59)",
            payload);
        // Round-trip must still be exact.
        let mut decoded = Vec::new();
        decode(&encoded[..], &mut decoded).unwrap();
        assert_eq!(decoded, plain);
    }
}
