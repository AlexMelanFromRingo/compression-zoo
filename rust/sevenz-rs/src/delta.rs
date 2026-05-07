//! Delta filter — port of `7zip/C/Delta.c`.
//!
//! The Delta filter subtracts each byte from the byte that appeared `delta`
//! positions earlier in the (logical) stream. With suitable `delta` values
//! (e.g. `2` for 16-bit PCM audio, `4` for RGBA) the residual stream is more
//! compressible by an entropy coder.
//!
//! The filter is fully streaming: a [`Delta`] instance keeps the last `delta`
//! bytes of context so successive [`Delta::encode`] / [`Delta::decode`] calls
//! produce identical output to a single call on the concatenated input.

/// Maximum context size, matching the C `DELTA_STATE_SIZE` constant.
pub const STATE_SIZE: usize = 256;

/// Streaming Delta filter.
#[derive(Clone, Debug)]
pub struct Delta {
    state: [u8; STATE_SIZE],
    delta: usize,
}

impl Delta {
    /// Create a fresh filter with the given context width.
    ///
    /// `delta` must be in `1..=STATE_SIZE` (the LZMA SDK enforces the same
    /// bounds on its filter property byte).
    pub fn new(delta: usize) -> Self {
        assert!(
            (1..=STATE_SIZE).contains(&delta),
            "delta must be in 1..=256 (got {delta})"
        );
        Self { state: [0; STATE_SIZE], delta }
    }

    /// Reset the context to all-zero (equivalent to `Delta_Init`).
    pub fn reset(&mut self) {
        self.state = [0; STATE_SIZE];
    }

    /// Configured context width.
    #[inline]
    pub fn delta(&self) -> usize {
        self.delta
    }

    /// In-place delta encoding. Equivalent to `Delta_Encode`.
    pub fn encode(&mut self, data: &mut [u8]) {
        encode(&mut self.state, self.delta, data);
    }

    /// In-place delta decoding. Equivalent to `Delta_Decode`.
    pub fn decode(&mut self, data: &mut [u8]) {
        decode(&mut self.state, self.delta, data);
    }
}

/// Free-function `Delta_Encode`.
pub fn encode(state: &mut [u8; STATE_SIZE], delta: usize, data: &mut [u8]) {
    let size = data.len();
    if size == 0 {
        return;
    }
    debug_assert!((1..=STATE_SIZE).contains(&delta));

    // Save the previous context — encoding mutates `state` and then `data`.
    let mut temp = [0u8; STATE_SIZE];
    temp[..delta].copy_from_slice(&state[..delta]);

    if size <= delta {
        // Streaming case where we don't have a full `delta` of new data.
        for i in 0..size {
            let b = data[i];
            data[i] = b.wrapping_sub(temp[i]);
            temp[i] = b;
        }
        // state[k] = temp[(size + k) mod delta]  for k in 0..delta
        for k in 0..delta {
            let idx = (size + k) % delta;
            state[k] = temp[idx];
        }
        return;
    }

    // Save the trailing `delta` bytes of plaintext for the next call.
    state[..delta].copy_from_slice(&data[size - delta..size]);

    // Subtract from the back so each `data[j-delta]` is still the original byte.
    for j in (delta..size).rev() {
        data[j] = data[j].wrapping_sub(data[j - delta]);
    }
    for j in (0..delta).rev() {
        data[j] = data[j].wrapping_sub(temp[j]);
    }
}

/// Free-function `Delta_Decode`.
pub fn decode(state: &mut [u8; STATE_SIZE], delta: usize, data: &mut [u8]) {
    let size = data.len();
    if size == 0 {
        return;
    }
    debug_assert!((1..=STATE_SIZE).contains(&delta));

    if size <= delta {
        for i in 0..size {
            data[i] = data[i].wrapping_add(state[i]);
        }
        // Shift state left by `size`, then append the new `size` bytes.
        state.copy_within(size..delta, 0);
        state[delta - size..delta].copy_from_slice(&data[..size]);
        return;
    }

    // First `delta` bytes use the saved context.
    for i in 0..delta {
        data[i] = data[i].wrapping_add(state[i]);
    }
    // Then chain forwards through the buffer.
    for j in delta..size {
        data[j] = data[j].wrapping_add(data[j - delta]);
    }
    // Save the trailing `delta` bytes for the next call.
    state[..delta].copy_from_slice(&data[size - delta..size]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(delta: usize, chunks: &[&[u8]]) {
        let original: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();

        // Encode chunk-by-chunk
        let mut enc = Delta::new(delta);
        let mut encoded: Vec<u8> = Vec::with_capacity(original.len());
        for c in chunks {
            let mut buf = c.to_vec();
            enc.encode(&mut buf);
            encoded.extend_from_slice(&buf);
        }

        // Encoding the whole buffer in one go must produce the same output.
        let mut single = original.clone();
        Delta::new(delta).encode(&mut single);
        assert_eq!(encoded, single, "chunked vs single encode mismatch");

        // Decode chunk-by-chunk and verify we get back the original.
        let mut dec = Delta::new(delta);
        let mut decoded: Vec<u8> = Vec::with_capacity(encoded.len());
        let mut offset = 0;
        for c in chunks {
            let mut buf = encoded[offset..offset + c.len()].to_vec();
            dec.decode(&mut buf);
            decoded.extend_from_slice(&buf);
            offset += c.len();
        }
        assert_eq!(decoded, original, "round-trip failed");
    }

    #[test]
    fn empty_input() {
        let mut d = Delta::new(4);
        let mut buf: [u8; 0] = [];
        d.encode(&mut buf);
        d.decode(&mut buf);
    }

    #[test]
    fn delta_one_constant_stream_is_zero_after_first_byte() {
        // delta=1 on a constant stream: first byte stays, rest become 0.
        let mut buf = vec![0x42u8; 16];
        Delta::new(1).encode(&mut buf);
        assert_eq!(buf[0], 0x42);
        for &b in &buf[1..] {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn round_trip_various_sizes() {
        let data: Vec<u8> = (0..1024u32).map(|i| (i * 31 + 7) as u8).collect();
        for &delta in &[1usize, 2, 3, 4, 5, 8, 16, 64, 256] {
            // single-shot
            round_trip(delta, &[&data]);
            // mid-split — exercises continuity across calls
            let mid = data.len() / 2;
            round_trip(delta, &[&data[..mid], &data[mid..]]);
            // chunks shorter and longer than delta, contiguous and non-empty
            let a = 1usize.min(data.len());
            let b = (delta.saturating_sub(1).max(a)).min(data.len());
            let c = (delta + 1).min(data.len());
            round_trip(delta, &[&data[..a], &data[a..b], &data[b..c], &data[c..]]);
        }
    }

    #[test]
    fn small_then_large_chunks() {
        // Tail-end behaviour where size<=delta then size>delta — exercises the
        // state shift + append logic.
        let data: Vec<u8> = (0..500u32).map(|i| (i ^ 0x5a) as u8).collect();
        round_trip(4, &[&data[..2], &data[2..3], &data[3..7], &data[7..]]);
        round_trip(8, &[&data[..1], &data[1..8], &data[8..16], &data[16..]]);
    }
}
