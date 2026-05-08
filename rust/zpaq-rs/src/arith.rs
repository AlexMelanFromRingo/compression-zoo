//! Binary arithmetic coder, ported from
//! `plugins/zpaq/upstream/libzpaq.cpp:2090-2155`.
//!
//! The coder operates on a 32-bit `[low, high]` range with a
//! 16-bit probability of "1" (range `0..=65535`). On the wire each
//! byte shifts out the highest 8 bits when `low` and `high` agree on
//! that byte (libzpaq calls it "shift out identical leading bytes"
//! with the standard `(high ^ low) < 0x01000000` test).
//!
//! ZPAQ's Decoder is bit-by-bit but the predictor produces 16-bit
//! probabilities, so this coder is **not** byte-aligned the way the
//! libbsc range coder is — every bit can shift the range arbitrarily.
//!
//! The coder also depends on a special "0/1 byte" framing for segment
//! initialisation: when `curr == 0` the decoder seeds itself by
//! reading 4 bytes; if those four are also `0`, the segment is
//! considered empty (libzpaq uses this as the EOF marker for stored
//! segments). Our port keeps the same semantics.
//!
//! The encoder/decoder pair are not part of any larger module yet —
//! the predictor, ZPAQL VM and block-format reader will land in
//! follow-up commits.

#![allow(dead_code)]

use crate::io::{Reader, Writer};

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ArithError {
    /// Decoder reached end of input mid-byte (`get` returned `None`
    /// while we still need more data to renormalise the range).
    UnexpectedEof,
    /// Decoder's `curr` fell outside `[low, high]` — typically a
    /// corrupt archive or a probability source out of sync with the
    /// encoder.
    Corrupt,
}

// ----- Decoder ----------------------------------------------------

/// libzpaq-compatible binary arithmetic decoder. Initialised at
/// segment start by [`Decoder::init_segment`] (which reads the first
/// four bytes of the wire); thereafter [`Decoder::decode`] returns
/// individual bits at caller-supplied 16-bit probabilities.
pub struct Decoder {
    low: u32,
    high: u32,
    curr: u32,
}

impl Decoder {
    pub fn new() -> Self {
        Self { low: 1, high: 0xFFFF_FFFF, curr: 0 }
    }

    /// Reset state to "block start, modeled mode" (see
    /// `Decoder::init` in upstream).
    pub fn init_modeled(&mut self) {
        self.low = 1;
        self.high = 0xFFFF_FFFF;
        self.curr = 0;
    }

    /// Reset state to "block start, stored mode" (`isModeled() ==
    /// false` — used when the predictor has zero components and
    /// libzpaq's Decoder::decompress does run-length pass-through).
    pub fn init_stored(&mut self) {
        self.low = 0;
        self.high = 0;
        self.curr = 0;
    }

    /// Pulls 4 bytes off the input and stuffs them into `curr` as a
    /// big-endian word. Mirrors the "for i=0..4 curr = curr<<8 | get"
    /// idiom that runs at the start of every modeled segment.
    pub fn fill_curr<R: Reader>(&mut self, input: &mut R) -> Result<(), ArithError> {
        for _ in 0..4 {
            let c = input.get().ok_or(ArithError::UnexpectedEof)?;
            self.curr = (self.curr << 8) | c as u32;
        }
        Ok(())
    }

    /// Decode one bit at probability `p_one_in_65536`
    /// (P(bit==1) = p / 65536).
    pub fn decode<R: Reader>(&mut self, input: &mut R, p: u32) -> Result<u32, ArithError> {
        debug_assert!(p < 65536);
        debug_assert!(self.high > self.low && self.low > 0);
        if self.curr < self.low || self.curr > self.high {
            return Err(ArithError::Corrupt);
        }

        // Split the range. The C uses U64 arithmetic; we widen too.
        let mid = self.low.wrapping_add(
            (((self.high - self.low) as u64 * p as u64) >> 16) as u32,
        );
        debug_assert!(self.high > mid && mid >= self.low);

        let y: u32;
        if self.curr <= mid {
            y = 1;
            self.high = mid;
        } else {
            y = 0;
            self.low = mid + 1;
        }

        // Renormalise: shift out identical leading bytes.
        while (self.high ^ self.low) < 0x0100_0000 {
            self.high = (self.high << 8) | 0xFF;
            self.low = self.low << 8;
            if self.low == 0 {
                self.low = 1; // "low+=(low==0)" — never let low underflow to 0.
            }
            let c = input.get().ok_or(ArithError::UnexpectedEof)?;
            self.curr = (self.curr << 8) | c as u32;
        }

        Ok(y)
    }

    pub fn curr(&self) -> u32 { self.curr }
    pub fn low(&self) -> u32  { self.low }
    pub fn high(&self) -> u32 { self.high }
}

impl Default for Decoder {
    fn default() -> Self { Self::new() }
}

// ----- Encoder ---------------------------------------------------

/// libzpaq-compatible binary arithmetic encoder.
///
/// The encoder mirrors `Decoder` exactly (same `low`/`high`,
/// renormalisation, and `low+=(low==0)` quirk). After encoding all
/// segment bits the caller must call [`Encoder::flush`] which writes
/// 4 finalisation bytes (`high >> 24` truncated each step) so the
/// decoder's initial 4-byte fill_curr lands on a value inside
/// `[low, high]`.
pub struct Encoder {
    low: u32,
    high: u32,
}

impl Encoder {
    pub fn new() -> Self {
        Self { low: 1, high: 0xFFFF_FFFF }
    }

    /// Reset state to start a new modeled segment.
    pub fn init(&mut self) {
        self.low = 1;
        self.high = 0xFFFF_FFFF;
    }

    /// Encode one bit at probability `p_one_in_65536`.
    pub fn encode<W: Writer>(&mut self, out: &mut W, y: u32, p: u32) {
        debug_assert!(p < 65536);
        debug_assert!(y == 0 || y == 1);
        debug_assert!(self.high > self.low && self.low > 0);

        let mid = self.low.wrapping_add(
            (((self.high - self.low) as u64 * p as u64) >> 16) as u32,
        );
        if y != 0 {
            self.high = mid;
        } else {
            self.low = mid + 1;
        }

        while (self.high ^ self.low) < 0x0100_0000 {
            // Shift out the matching top byte.
            out.put((self.high >> 24) as u8);
            self.high = (self.high << 8) | 0xFF;
            self.low = self.low << 8;
            if self.low == 0 {
                self.low = 1;
            }
        }
    }

    /// Emit four finalisation bytes so the decoder's `curr` lands on
    /// a value inside the current `[low, high]` range. Mirrors the
    /// libzpaq Encoder destructor / `compress` flush logic in
    /// `Encoder::compress(c=-1)`.
    pub fn flush<W: Writer>(&mut self, out: &mut W) {
        // libzpaq writes `low >> 24` four times (the encoder sends
        // `low` as a sentinel because it's the smallest valid value
        // in the range and the decoder reads exactly 4 bytes into
        // `curr`).
        let low = self.low;
        out.put((low >> 24) as u8);
        out.put((low >> 16) as u8);
        out.put((low >> 8) as u8);
        out.put(low as u8);
    }
}

impl Default for Encoder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceReader, VecWriter};

    /// Round-trip: encode a stream of bits at fair-coin probability
    /// (32768) and check the decoder reads them back.
    fn roundtrip(bits: &[u32], probs: &[u32]) {
        assert_eq!(bits.len(), probs.len());
        let mut out = VecWriter::new();
        let mut enc = Encoder::new();
        for (b, p) in bits.iter().zip(probs.iter()) {
            enc.encode(&mut out, *b, *p);
        }
        enc.flush(&mut out);

        let wire = out.into_inner();
        let mut input = SliceReader::new(&wire);
        let mut dec = Decoder::new();
        dec.init_modeled();
        dec.fill_curr(&mut input).expect("fill_curr");
        for (b, p) in bits.iter().zip(probs.iter()) {
            let got = dec.decode(&mut input, *p).expect("decode");
            assert_eq!(got, *b, "bit mismatch");
        }
    }

    #[test]
    fn roundtrip_fair_coin() {
        let bits: Vec<u32> = (0..1000u32).map(|i| i & 1).collect();
        let probs = vec![32768u32; bits.len()];
        roundtrip(&bits, &probs);
    }

    #[test]
    fn roundtrip_skewed_zero() {
        // Strong bias toward 0: most bits are 0.
        let mut bits: Vec<u32> = vec![0; 2000];
        bits[100] = 1; bits[1500] = 1; bits[1900] = 1;
        let probs = vec![1024u32; bits.len()]; // ~1.5% chance of 1
        roundtrip(&bits, &probs);
    }

    #[test]
    fn roundtrip_skewed_one() {
        // Strong bias toward 1.
        let mut bits: Vec<u32> = vec![1; 2000];
        bits[100] = 0; bits[1500] = 0;
        let probs = vec![64500u32; bits.len()]; // ~98.4% chance of 1
        roundtrip(&bits, &probs);
    }

    #[test]
    fn roundtrip_pseudo_random() {
        let mut bits: Vec<u32> = Vec::with_capacity(8000);
        let mut probs: Vec<u32> = Vec::with_capacity(8000);
        let mut x: u32 = 0x12345678;
        for _ in 0..8000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            bits.push(x & 1);
            // probabilities scattered through the (1..65535) range.
            probs.push(((x >> 1) % 65534) + 1);
        }
        roundtrip(&bits, &probs);
    }
}
