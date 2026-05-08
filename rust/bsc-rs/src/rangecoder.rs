//! Carryless range coder, ported from
//! `plugins/bsc/upstream/libbsc/coder/common/rangecoder.h`.
//!
//! Bit-for-bit compatible with libbsc's encoder/decoder pair: writes
//! and reads 16-bit big-endian-on-the-wire (i.e. **native-endian**
//! `u16` values). libbsc's reference uses native-endian unaligned
//! writes; the Windows DLL we ship is x86_64 little-endian, so on the
//! wire each "short" appears as two LE bytes. This module follows
//! that convention exactly.
//!
//! Probability scale defaults to 12 bits (`P = 12`, range 0..=4096).

#![allow(dead_code)]

use std::convert::TryInto;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum RangeCoderError {
    /// Decoder ran out of input bytes mid-symbol.
    UnexpectedEof,
    /// Encoder output buffer was too small for the produced stream.
    OutputOverflow,
}

const TOP: u32 = 0x10000;
const FF_THRESHOLD: u32 = 0xffff_0000;

// =====================================================================
// Encoder
// =====================================================================

/// Streaming range encoder writing into a caller-provided `Vec<u8>`.
/// Every 16 bits of "low" become two bytes in native-endian order on
/// the wire (matches libbsc's `*ari_output++ = s` with `ari_output`
/// being `unsigned short *`).
pub struct RangeEncoder<'a> {
    out: &'a mut Vec<u8>,
    /// 64-bit low register; the high 32 bits hold the carry.
    low: u64,
    /// Pending 0xff carry-cache count.
    ffnum: u32,
    /// Most recently shifted-out 16-bit chunk awaiting carry resolution.
    cache: u32,
    /// Range register.
    range: u32,
    /// Hard cap on output size; if we'd exceed this, we mark `eob`.
    output_eob: usize,
    eob: bool,
}

impl<'a> RangeEncoder<'a> {
    /// `output_size` mirrors libbsc's `ari_outputEOB = output + size - 16`
    /// — the encoder bails out early if it would write past this point.
    pub fn new(out: &'a mut Vec<u8>, output_size: usize) -> Self {
        let start = out.len();
        let cap = output_size.saturating_sub(16);
        Self {
            out,
            low: 0,
            ffnum: 0,
            cache: 0,
            range: 0xffff_ffff,
            output_eob: start + cap,
            eob: false,
        }
    }

    #[inline]
    pub fn check_eob(&self) -> bool {
        self.out.len() >= self.output_eob
    }

    #[inline]
    fn output_short(&mut self, s: u16) {
        self.out.extend_from_slice(&s.to_ne_bytes());
    }

    /// Set `low.u.low32 <<= 16` while preserving the carry bit
    /// (high 32 bits of the u64). Mirrors libbsc's union access.
    #[inline]
    fn shift_low_low32(&mut self) {
        let low32 = self.low as u32;
        let carry = (self.low >> 32) as u32;
        let low32_new = low32.wrapping_shl(16);
        self.low = ((carry as u64) << 32) | (low32_new as u64);
    }

    /// Slow path: shift out one 16-bit chunk, resolving carries.
    /// Returns the new range << 16.
    fn shift_low_slow(&mut self) -> u32 {
        let low32 = self.low as u32;
        let carry = (self.low >> 32) as u32;
        if low32 < FF_THRESHOLD || carry != 0 {
            self.output_short((self.cache.wrapping_add(carry)) as u16);
            if self.ffnum != 0 {
                let s = (carry.wrapping_sub(1)) as u16;
                while self.ffnum > 0 {
                    self.output_short(s);
                    self.ffnum -= 1;
                }
            }
            self.cache = low32 >> 16;
            // Clear carry but keep low32 (we will shift it next).
            self.low = low32 as u64;
        } else {
            self.ffnum += 1;
        }
        self.shift_low_low32();
        self.range << 16
    }

    /// Fast/slow combined; matches libbsc's `ShiftLow()` exactly.
    fn shift_low(&mut self) -> u32 {
        let low32 = self.low as u32;
        let carry = (self.low >> 32) as u32;
        if self.ffnum == 0 && low32 < FF_THRESHOLD {
            self.output_short(self.cache.wrapping_add(carry) as u16);
            self.cache = low32 >> 16;
            // Fast-path: clear carry AND set low32 = (low32 << 16).
            self.low = low32.wrapping_shl(16) as u64;
            return self.range << 16;
        }
        self.shift_low_slow()
    }

    /// Encode a 0 bit at probability `prob` (12-bit scale, 0..=4095).
    #[inline]
    pub fn encode_bit_0(&mut self, prob: u32) {
        self.encode_bit_0_p(prob, 12);
    }

    /// Encode a 1 bit at probability `prob` (12-bit scale).
    #[inline]
    pub fn encode_bit_1(&mut self, prob: u32) {
        self.encode_bit_1_p(prob, 12);
    }

    /// `EncodeBit0<P>(prob)` — generalised over the probability scale.
    #[inline]
    pub fn encode_bit_0_p(&mut self, prob: u32, p: u32) {
        if self.range < TOP {
            self.range = self.shift_low();
        }
        self.range = (self.range >> p) * prob;
    }

    /// `EncodeBit1<P>(prob)` — generalised over the probability scale.
    #[inline]
    pub fn encode_bit_1_p(&mut self, prob: u32, p: u32) {
        if self.range < TOP {
            self.range = self.shift_low();
        }
        let r = (self.range >> p) * prob;
        self.low = self.low.wrapping_add(r as u64);
        self.range -= r;
    }

    /// Encode `bit` (0 or 1) at probability `prob` (12-bit scale).
    #[inline]
    pub fn encode_bit_prob(&mut self, bit: u32, prob: u32) {
        if bit != 0 {
            self.encode_bit_1(prob);
        } else {
            self.encode_bit_0(prob);
        }
    }

    /// `EncodeBit<P>(bit, prob)` — generalised over the probability scale.
    #[inline]
    pub fn encode_bit_p(&mut self, bit: u32, prob: u32, p: u32) {
        if bit != 0 {
            self.encode_bit_1_p(prob, p);
        } else {
            self.encode_bit_0_p(prob, p);
        }
    }

    /// Encode a single bit at the implicit 0.5 probability (== 2048).
    #[inline]
    pub fn encode_bit(&mut self, bit: u32) {
        self.encode_bit_prob(bit, 2048);
    }

    /// Encode 8 bits MSB-first using the implicit 2048 probability.
    pub fn encode_byte(&mut self, byte: u32) {
        for bit in (0..8).rev() {
            self.encode_bit((byte >> bit) & 1);
        }
    }

    /// Encode 32 bits MSB-first using the implicit 2048 probability.
    pub fn encode_word(&mut self, word: u32) {
        for bit in (0..32).rev() {
            self.encode_bit((word >> bit) & 1);
        }
    }

    /// Flush the encoder; returns the number of bytes written into `out`.
    pub fn finish(mut self) -> usize {
        let start_len = self.output_eob;
        let _ = start_len;
        if self.range < TOP {
            self.shift_low();
        }
        self.shift_low();
        self.shift_low();
        self.shift_low();
        self.out.len() // caller can subtract the original start themselves
    }
}

// =====================================================================
// Decoder
// =====================================================================

/// Streaming range decoder reading from a caller-provided byte slice.
/// Treats every two bytes as a native-endian `u16` "short", matching
/// libbsc's `ari_input` of type `unsigned short *`.
pub struct RangeDecoder<'a> {
    input: &'a [u8],
    pos: usize,
    code: u32,
    range: u32,
}

impl<'a> RangeDecoder<'a> {
    /// libbsc reads three 16-bit shorts up front, so `input` must have
    /// at least 6 bytes (we don't enforce that — `next_short` returns
    /// 0 past EOF, mirroring libbsc's behaviour of reading uninitialized
    /// memory at the tail; in practice the encoder guarantees enough
    /// padding in `FinishEncoder`).
    pub fn new(input: &'a [u8]) -> Self {
        let mut me = Self { input, pos: 0, code: 0, range: 0xffff_ffff };
        let s1 = me.next_short();
        let s2 = me.next_short();
        let s3 = me.next_short();
        me.code = (s1 as u32) << 16 | s2 as u32;
        // Third short is the LOW 16 bits of `code` after another shift.
        me.code = (me.code << 16) | s3 as u32;
        me
    }

    #[inline]
    fn next_short(&mut self) -> u16 {
        if self.pos + 2 <= self.input.len() {
            let s = u16::from_ne_bytes(self.input[self.pos..self.pos + 2].try_into().unwrap());
            self.pos += 2;
            s
        } else {
            // Past end of stream, libbsc reads garbage; tests should
            // always provide enough padding via `RangeEncoder::finish`.
            0
        }
    }

    /// Returns the decoded bit at probability `prob` (12-bit scale).
    /// Mirrors `DecodeBit(P=12)`.
    #[inline]
    pub fn decode_bit_prob(&mut self, prob: u32) -> u32 {
        self.decode_bit_p(prob, 12)
    }

    /// `DecodeBit<P>(prob)` — generalised over the probability scale.
    #[inline]
    pub fn decode_bit_p(&mut self, prob: u32, p: u32) -> u32 {
        if self.range < TOP {
            self.range <<= 16;
            self.code = (self.code << 16) | self.next_short() as u32;
        }
        let r = (self.range >> p) * prob;
        let bit = (self.code >= r) as u32;
        if bit != 0 {
            self.range -= r;
            self.code -= r;
        } else {
            self.range = r;
        }
        bit
    }

    /// `PeakBit<P>(prob)` — returns the bit that would be decoded
    /// without consuming it. Used by the fast QLFC decoder which then
    /// commits via `decode_bit_0_p` / `decode_bit_1_p`.
    #[inline]
    pub fn peak_bit_p(&mut self, prob: u32, p: u32) -> u32 {
        if self.range < TOP {
            self.range <<= 16;
            self.code = (self.code << 16) | self.next_short() as u32;
        }
        (self.code >= (self.range >> p) * prob) as u32
    }

    /// `DecodeBit0<P>(prob)` — commit a 0 after a peek.
    #[inline]
    pub fn decode_bit_0_p(&mut self, prob: u32, p: u32) {
        self.range = (self.range >> p) * prob;
    }

    /// `DecodeBit1<P>(prob)` — commit a 1 after a peek.
    #[inline]
    pub fn decode_bit_1_p(&mut self, prob: u32, p: u32) {
        let r = (self.range >> p) * prob;
        self.code -= r;
        self.range -= r;
    }

    /// Decode a bit at the implicit 0.5 probability (== 2048).
    #[inline]
    pub fn decode_bit(&mut self) -> u32 {
        self.decode_bit_prob(2048)
    }

    /// Decode 8 bits MSB-first using the implicit 2048 probability.
    pub fn decode_byte(&mut self) -> u32 {
        let mut byte: u32 = 0;
        for _ in 0..8 {
            byte = byte + byte + self.decode_bit();
        }
        byte
    }

    /// Decode 32 bits MSB-first using the implicit 2048 probability.
    pub fn decode_word(&mut self) -> u32 {
        let mut word: u32 = 0;
        for _ in 0..32 {
            word = word + word + self.decode_bit();
        }
        word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_then_dec_bytes(bytes: &[u8]) {
        let mut out = Vec::new();
        let cap = bytes.len() * 2 + 1024; // generous
        out.reserve(cap);
        let mut enc = RangeEncoder::new(&mut out, cap);
        for &b in bytes {
            enc.encode_byte(b as u32);
        }
        let _ = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for &b in bytes {
            let got = dec.decode_byte() as u8;
            assert_eq!(got, b, "byte mismatch at offset");
        }
    }

    fn enc_then_dec_bits(bits: &[u32], probs: &[u32]) {
        assert_eq!(bits.len(), probs.len());
        let mut out = Vec::new();
        let cap = bits.len() / 2 + 1024;
        out.reserve(cap);
        let mut enc = RangeEncoder::new(&mut out, cap);
        for (b, p) in bits.iter().zip(probs.iter()) {
            enc.encode_bit_prob(*b, *p);
        }
        let _ = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for (b, p) in bits.iter().zip(probs.iter()) {
            let got = dec.decode_bit_prob(*p);
            assert_eq!(got, *b);
        }
    }

    #[test]
    fn roundtrip_empty_byte_stream() {
        enc_then_dec_bytes(&[]);
    }

    #[test]
    fn roundtrip_single_byte() {
        for b in 0u8..=255 {
            enc_then_dec_bytes(&[b]);
        }
    }

    #[test]
    fn roundtrip_short_byte_stream() {
        enc_then_dec_bytes(b"Hello, libbsc range coder!");
    }

    #[test]
    fn roundtrip_64k_pseudo_random_bytes() {
        let mut v = vec![0u8; 65536];
        let mut x: u32 = 0xC0FFEE;
        for b in v.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        enc_then_dec_bytes(&v);
    }

    #[test]
    fn roundtrip_mixed_probabilities() {
        // Drive the carry path: a long run of 1s at low probability
        // forces ffnum to grow.
        let mut bits = Vec::new();
        let mut probs = Vec::new();
        for _ in 0..2000 { bits.push(1); probs.push(64); }
        for _ in 0..2000 { bits.push(0); probs.push(4032); }
        for i in 0..2000u32 { bits.push(i & 1); probs.push(2048); }
        enc_then_dec_bits(&bits, &probs);
    }

    #[test]
    fn roundtrip_word() {
        let words = [0u32, 1, 0xffff_ffff, 0xdead_beef, 0x1234_5678];
        let mut out = Vec::new();
        let cap = 1024;
        out.reserve(cap);
        let mut enc = RangeEncoder::new(&mut out, cap);
        for w in words { enc.encode_word(w); }
        let _ = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for w in words {
            assert_eq!(dec.decode_word(), w);
        }
    }
}
