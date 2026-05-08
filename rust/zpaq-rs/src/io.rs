//! Mirrors libzpaq's `Reader`/`Writer` abstract base classes
//! (`plugins/zpaq/upstream/libzpaq.h:120-130`). Implementations may
//! provide either a single-byte path or a block path; the trait
//! offers default block implementations that loop over the byte
//! method, matching libzpaq's defaults.
//!
//! The block I/O methods take `&mut Vec<u8>` for the output side
//! (Writer::write) so callers don't have to track lengths separately.

#![allow(dead_code)]

/// Source of bytes. `get` returns the next byte (0..=255) or `None`
/// at end of stream.
pub trait Reader {
    /// Read one byte; return `None` at EOF.
    fn get(&mut self) -> Option<u8>;

    /// Try to fill `buf`; return the number of bytes actually read
    /// (`< buf.len()` only at EOF).
    fn read(&mut self, buf: &mut [u8]) -> usize {
        for (i, slot) in buf.iter_mut().enumerate() {
            match self.get() {
                Some(b) => *slot = b,
                None => return i,
            }
        }
        buf.len()
    }
}

/// Sink of bytes.
pub trait Writer {
    /// Append a single byte.
    fn put(&mut self, c: u8);

    /// Append `buf` in one go. Default loops over `put`.
    fn write(&mut self, buf: &[u8]) {
        for &b in buf { self.put(b); }
    }
}

// --- Adapters ----------------------------------------------------

/// `Reader` that wraps a `&[u8]` slice.
pub struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    pub fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    pub fn position(&self) -> usize { self.pos }
}

impl<'a> Reader for SliceReader<'a> {
    fn get(&mut self) -> Option<u8> {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    fn read(&mut self, buf: &mut [u8]) -> usize {
        let n = (self.data.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        n
    }
}

/// `Writer` that appends to an owned `Vec<u8>`.
pub struct VecWriter {
    pub buf: Vec<u8>,
}

impl VecWriter {
    pub fn new() -> Self { Self { buf: Vec::new() } }
    pub fn into_inner(self) -> Vec<u8> { self.buf }
}

impl Default for VecWriter {
    fn default() -> Self { Self::new() }
}

impl Writer for VecWriter {
    fn put(&mut self, c: u8) { self.buf.push(c); }
    fn write(&mut self, buf: &[u8]) { self.buf.extend_from_slice(buf); }
}
