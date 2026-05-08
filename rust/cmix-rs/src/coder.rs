//! Arithmetic coder — port of `coder/{encoder,decoder}.{h,cpp}`.
//!
//! Mirrors CMIX's float-probability binary arithmetic coder. Each
//! `Encode`/`Decode` call asks the [`Predictor`] for the bit-1
//! probability, splits the `[x1, x2]` range, and shifts out
//! identical leading bytes. The discretisation `1 + 65534*p` keeps
//! the probability in `[1, 65535]` so the range never collapses.

#![allow(dead_code)]

/// Predictor contract used by both [`Encoder`] and [`Decoder`].
/// `predict()` returns the bit-1 probability and `perceive(bit)`
/// updates the model with the bit just (en|de)coded.
pub trait Predictor {
    fn predict(&mut self) -> f32;
    fn perceive(&mut self, bit: i32);
}

/// Output sink. Implementations supply byte-at-a-time writing.
pub trait ByteSink {
    fn put(&mut self, b: u8);
}

/// Input source for the decoder.
pub trait ByteSource {
    /// Returns the next byte (or 0 at EOF — mirrors upstream's
    /// `if (!is->good()) return 0;`).
    fn get(&mut self) -> u8;
}

#[inline]
fn discretize(p: f32) -> u32 { 1 + (65534.0 * p) as u32 }

pub struct Encoder<W: ByteSink, P: Predictor> {
    out: W,
    x1: u32,
    x2: u32,
    pred: P,
}

impl<W: ByteSink, P: Predictor> Encoder<W, P> {
    pub fn new(out: W, pred: P) -> Self {
        Self { out, x1: 0, x2: 0xFFFF_FFFF, pred }
    }

    pub fn encode(&mut self, bit: i32) {
        let p = discretize(self.pred.predict());
        let xmid = self.x1
            + ((self.x2 - self.x1) >> 16) * p
            + (((self.x2 - self.x1) & 0xFFFF) * p >> 16);
        if bit != 0 { self.x2 = xmid; } else { self.x1 = xmid + 1; }
        self.pred.perceive(bit);
        while ((self.x1 ^ self.x2) & 0xFF00_0000) == 0 {
            self.out.put((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }

    pub fn flush(&mut self) {
        while ((self.x1 ^ self.x2) & 0xFF00_0000) == 0 {
            self.out.put((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
        self.out.put((self.x2 >> 24) as u8);
    }

    pub fn into_inner(self) -> (W, P) { (self.out, self.pred) }
}

pub struct Decoder<R: ByteSource, P: Predictor> {
    src: R,
    x1: u32,
    x2: u32,
    x:  u32,
    pred: P,
}

impl<R: ByteSource, P: Predictor> Decoder<R, P> {
    pub fn new(mut src: R, pred: P) -> Self {
        let mut x = 0u32;
        for _ in 0..4 {
            x = (x << 8) | src.get() as u32;
        }
        Self { src, x1: 0, x2: 0xFFFF_FFFF, x, pred }
    }

    pub fn decode(&mut self) -> i32 {
        let p = discretize(self.pred.predict());
        let xmid = self.x1
            + ((self.x2 - self.x1) >> 16) * p
            + (((self.x2 - self.x1) & 0xFFFF) * p >> 16);
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit == 1 { self.x2 = xmid; } else { self.x1 = xmid + 1; }
        self.pred.perceive(bit);
        while ((self.x1 ^ self.x2) & 0xFF00_0000) == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | self.src.get() as u32;
        }
        bit
    }
}

// ----- Tiny in-memory adapters for tests --------------------------------

pub struct VecSink(pub Vec<u8>);
impl ByteSink for VecSink {
    fn put(&mut self, b: u8) { self.0.push(b); }
}

pub struct SliceSource<'a> { pub data: &'a [u8], pub pos: usize }
impl<'a> ByteSource for SliceSource<'a> {
    fn get(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial predictor that always says 0.5 and ignores the bit.
    /// Used to verify the arith coder's range-tracking machinery
    /// independently of the (much larger) predictor port.
    struct Half;
    impl Predictor for Half {
        fn predict(&mut self) -> f32 { 0.5 }
        fn perceive(&mut self, _bit: i32) {}
    }

    fn round_trip(bits: &[i32]) {
        let mut enc = Encoder::new(VecSink(Vec::new()), Half);
        for &b in bits { enc.encode(b); }
        enc.flush();
        let (sink, _) = enc.into_inner();
        let mut dec = Decoder::new(
            SliceSource { data: &sink.0, pos: 0 },
            Half,
        );
        for &b in bits {
            assert_eq!(dec.decode(), b, "decoded bit mismatched");
        }
    }

    #[test] fn rt_alternating()    { round_trip(&[0,1,0,1,0,1,0,1]); }
    #[test] fn rt_all_ones()       { round_trip(&[1; 64]); }
    #[test] fn rt_all_zeros()      { round_trip(&[0; 64]); }
    #[test] fn rt_pseudo_random()  {
        let mut bits = Vec::with_capacity(1024);
        let mut x: u32 = 0xC0FFEE;
        for _ in 0..1024 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            bits.push(((x >> 24) & 1) as i32);
        }
        round_trip(&bits);
    }
}
