//! Port of libbsc's LZP (Lempel-Ziv predictor) preprocessor.
//!
//! Mirrors the SCALAR / portable path in `plugins/bsc/upstream/libbsc/
//! lzp/lzp.cpp` — the SSE/AArch64 fast paths in upstream are pure
//! optimisations and produce the same byte stream as the scalar code.
//! We don't reproduce them.
//!
//! Wire format
//! -----------
//!
//! * The first 4 bytes of input are copied verbatim to output.
//! * After that the encoder maintains a 32-bit "context" of the last
//!   four output bytes and a hash table mapping context -> previous
//!   output position. Whenever the next input byte equals
//!   `LZP_MATCH_FLAG` (0xF2) and the hash table reports a previous
//!   occurrence of this context, the encoder either:
//!     - emits the flag plus an escape byte 0xFF for a real 0xF2
//!       literal, or
//!     - emits the flag plus a length encoded as one or more
//!       cumulative bytes (each `254` means "add 254 and read another
//!       length byte"; anything else terminates).
//! * Other input bytes pass through verbatim, with the side effect of
//!   updating the hash table at the current context.
//!
//! Decoder is the mirror image: scan the hash table at every output
//! position, and on flag-byte input, either emit one literal flag
//! (escape) or copy `len` bytes from the previously-recorded output
//! position.



pub const LZP_MATCH_FLAG: u8 = 0xF2;

/// Errors returned by [`decode_block`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LzpError {
    /// `input` is shorter than 4 bytes (LZP always requires the
    /// first 4 bytes verbatim).
    UnexpectedEob,
    /// `hash_size` is outside the libbsc-allowed range [10, 28] or
    /// `min_len` is outside [4, 255].
    BadParameter,
    /// While decoding, an output buffer position would be referenced
    /// from a position that hasn't been written yet, or the input ran
    /// out mid-token.
    DataCorrupt,
}

fn validate_params(hash_size: i32, min_len: i32) -> Result<(), LzpError> {
    if hash_size < 10 || hash_size > 28 { return Err(LzpError::BadParameter); }
    if min_len < 4 || min_len > 255 { return Err(LzpError::BadParameter); }
    Ok(())
}

#[inline]
fn build_context(prev4: &[u8]) -> u32 {
    debug_assert_eq!(prev4.len(), 4);
    // (context = output[-1] | (output[-2] << 8) | (output[-3] << 16) | (output[-4] << 24))
    // where output[-1] is the most recent byte. prev4 is in chronological order
    // so prev4[3] is the most recent.
    (prev4[3] as u32)
        | ((prev4[2] as u32) << 8)
        | ((prev4[1] as u32) << 16)
        | ((prev4[0] as u32) << 24)
}

#[inline]
fn ctx_hash(context: u32, mask: u32) -> usize {
    (((context >> 15) ^ context ^ (context >> 3)) & mask) as usize
}

/// Encode `input` with LZP and append the encoded bytes to `output`.
/// Returns the number of bytes appended.
///
/// Mirrors the scalar slow-path of `bsc_lzp_encode_block` from upstream
/// (we deliberately do NOT replicate the SSE/AArch64 fast paths — they
/// produce identical wire output, just faster). The encoder finds
/// matches by walking forward from the previously-recorded position
/// `value` and comparing bytes; if `match_len >= min_len` the byte
/// stream gets `FLAG, len_bytes...`, where `len_bytes` encodes
/// `match_len - min_len` as a sequence terminated by any byte != 254.
/// A literal `FLAG` byte in the input that would otherwise be confused
/// with a match is escaped as `FLAG, 255`.
pub fn encode_block(
    input: &[u8],
    output: &mut Vec<u8>,
    hash_size: i32,
    min_len: i32,
) -> Result<usize, LzpError> {
    validate_params(hash_size, min_len)?;
    if input.len() < 4 {
        return Err(LzpError::UnexpectedEob);
    }

    let mask: u32 = (1u32 << hash_size as u32) - 1;
    let mut lookup: Vec<i32> = vec![0; (mask as usize) + 1];

    let out_start = output.len();
    output.extend_from_slice(&input[..4]);
    let mut ip = 4usize;

    while ip < input.len() {
        // Predictor sees the last 4 bytes of decoder-mirror, which
        // equals input[..ip] by construction (decoder reconstructs
        // input exactly).
        let context = build_context(&input[ip - 4..ip]);
        let index = ctx_hash(context, mask);
        let value = lookup[index];
        lookup[index] = ip as i32;

        if value > 0 {
            let v = value as usize;
            // Find longest match starting at ip vs v. Overlap (v+m
            // crossing ip) is fine: the decoder copies byte-by-byte,
            // so a periodic pattern like "ABCDABCD..." encodes as one
            // big match — the standard LZ77 run-length idiom.
            let mut m = 0usize;
            let max = input.len() - ip;
            while m < max && v + m < input.len() && input[v + m] == input[ip + m] {
                m += 1;
            }
            if (m as i32) >= min_len {
                output.push(LZP_MATCH_FLAG);
                let mut k = m as i32 - min_len;
                while k >= 254 {
                    output.push(254);
                    k -= 254;
                }
                output.push(k as u8);
                ip += m;
                continue;
            }
            if input[ip] == LZP_MATCH_FLAG {
                // Escape: literal flag in original.
                output.push(LZP_MATCH_FLAG);
                output.push(0xFF);
                ip += 1;
                continue;
            }
        }
        output.push(input[ip]);
        ip += 1;
    }

    Ok(output.len() - out_start)
}

/// Wire-format compatible with libbsc's `bsc_lzp_compress` for the
/// single-block case: prepends a 1-byte `nBlocks=1` header before the
/// LZP block bytes.
///
/// Returns the number of bytes appended to `output`.
pub fn compress(
    input: &[u8],
    output: &mut Vec<u8>,
    hash_size: i32,
    min_len: i32,
) -> Result<usize, LzpError> {
    let start = output.len();
    output.push(1u8); // nBlocks = 1
    encode_block(input, output, hash_size, min_len)?;
    Ok(output.len() - start)
}

/// Inverse of [`compress`]. Reads the leading `nBlocks` byte and then
/// decodes a single block. Multi-block inputs (nBlocks > 1) are not
/// supported yet.
pub fn decompress(
    input: &[u8],
    output: &mut Vec<u8>,
    hash_size: i32,
    min_len: i32,
) -> Result<usize, LzpError> {
    if input.is_empty() {
        return Err(LzpError::UnexpectedEob);
    }
    let n_blocks = input[0];
    if n_blocks == 1 {
        decode_block(&input[1..], output, hash_size, min_len)
    } else {
        // TODO: multi-block split; libbsc switches to it for very
        // large inputs. We keep single-block for now.
        Err(LzpError::BadParameter)
    }
}

/// Decode an LZP-encoded block into `output`. Returns the number of
/// bytes written.
///
/// Mirrors `bsc_lzp_decode_block` in upstream. `output` must be large
/// enough; we don't have an upper bound up front so the caller usually
/// allocates with the original (pre-LZP) data size + slack.
pub fn decode_block(
    input: &[u8],
    output: &mut Vec<u8>,
    hash_size: i32,
    min_len: i32,
) -> Result<usize, LzpError> {
    validate_params(hash_size, min_len)?;
    if input.len() < 4 { return Err(LzpError::UnexpectedEob); }

    let mask: u32 = (1u32 << hash_size as u32) - 1;
    let mut lookup: Vec<i32> = vec![0; (mask as usize) + 1];

    let out_start = output.len();
    output.extend_from_slice(&input[..4]);
    let mut ip = 4usize; // input cursor

    while ip < input.len() {
        // Build context from last four output bytes.
        let written = output.len() - out_start;
        // We always have at least 4 bytes of output here because we
        // copy the first 4 input bytes above and the loop body never
        // shrinks output.
        let prev4_end = output.len();
        let context = build_context(&output[prev4_end - 4..prev4_end]);
        let index = ctx_hash(context, mask);
        let value = lookup[index];
        lookup[index] = written as i32;

        let byte = input[ip];
        if byte == LZP_MATCH_FLAG && value > 0 {
            ip += 1;
            if ip >= input.len() { return Err(LzpError::DataCorrupt); }
            if input[ip] != 0xFF {
                // Length-encoded match. Read minLen + sum_of_bytes
                // until a non-254 byte.
                let mut len: i32 = min_len;
                loop {
                    if ip >= input.len() { return Err(LzpError::DataCorrupt); }
                    let b = input[ip];
                    len = len.wrapping_add(b as i32);
                    ip += 1;
                    if b != 254 { break; }
                }
                if len <= 0 { return Err(LzpError::DataCorrupt); }
                let reference = value as usize;
                let copy_len = len as usize;
                if reference + copy_len > output.len() - out_start + copy_len {
                    // The reference range may overlap with the
                    // destination; we copy byte-by-byte.
                }
                if reference >= output.len() - out_start {
                    return Err(LzpError::DataCorrupt);
                }
                for k in 0..copy_len {
                    let src_idx = out_start + reference + k;
                    if src_idx >= output.len() {
                        return Err(LzpError::DataCorrupt);
                    }
                    let v = output[src_idx];
                    output.push(v);
                }
            } else {
                // Escape: a real 0xF2 in the original.
                ip += 1;
                output.push(LZP_MATCH_FLAG);
            }
        } else {
            output.push(byte);
            ip += 1;
        }
    }

    Ok(output.len() - out_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip helper: encode `input` with our port, decode with
    /// our port, verify the output matches.
    fn round_trip(input: &[u8], hash_size: i32, min_len: i32) -> usize {
        let mut encoded = Vec::with_capacity(input.len() + 64);
        encode_block(input, &mut encoded, hash_size, min_len).expect("encode");
        let mut decoded = Vec::with_capacity(input.len());
        decode_block(&encoded, &mut decoded, hash_size, min_len).expect("decode");
        assert_eq!(&decoded[..], input, "lzp round-trip differs");
        encoded.len()
    }

    #[test]
    fn rejects_bad_params() {
        let mut out = Vec::new();
        assert_eq!(decode_block(&[1, 2, 3, 4], &mut out, 9, 4),
                   Err(LzpError::BadParameter));
        assert_eq!(decode_block(&[1, 2, 3, 4], &mut out, 29, 4),
                   Err(LzpError::BadParameter));
        assert_eq!(decode_block(&[1, 2, 3, 4], &mut out, 15, 3),
                   Err(LzpError::BadParameter));
        assert_eq!(decode_block(&[1, 2, 3, 4], &mut out, 15, 256),
                   Err(LzpError::BadParameter));
    }

    #[test]
    fn rejects_short_input() {
        let mut out = Vec::new();
        assert_eq!(decode_block(&[], &mut out, 15, 4),
                   Err(LzpError::UnexpectedEob));
        assert_eq!(decode_block(&[1, 2, 3], &mut out, 15, 4),
                   Err(LzpError::UnexpectedEob));
    }

    #[test]
    fn passes_through_short_unique_input() {
        // Random-looking 32 bytes — any 4-byte context is unique so no
        // LZP matches happen and the encoded form is the same as the
        // input (plus 0xF2 escape if any of the bytes is 0xF2 — none
        // here).
        let input: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8,
            10, 11, 12, 13, 14, 15, 16, 17,
            20, 21, 22, 23, 24, 25, 26, 27,
            30, 31, 32, 33, 34, 35, 36, 37,
        ];
        round_trip(&input, 15, 6);
    }

    #[test]
    fn round_trip_repetitive() {
        let mut input = Vec::new();
        for _ in 0..32 {
            input.extend_from_slice(b"ABCD-EFGH-");
        }
        round_trip(&input, 15, 6);
    }

    #[test]
    fn round_trip_through_flag_byte() {
        // A literal 0xF2 inside the input forces the escape path.
        let mut input = vec![1, 2, 3, 4, 5, 6];
        input.push(LZP_MATCH_FLAG);
        input.extend_from_slice(b"after the flag");
        round_trip(&input, 15, 6);
    }

    #[test]
    fn round_trip_lots_of_repetition() {
        // 4 KiB of repeating ASCII. LZP should knock this down a lot.
        let mut input = Vec::with_capacity(4096);
        while input.len() < 4096 {
            input.extend_from_slice(b"the quick brown fox jumps over the lazy dog\n");
        }
        input.truncate(4096);
        let encoded_size = round_trip(&input, 15, 6);
        // LZP after the first occurrence of the phrase should turn the
        // remainder mostly into match tokens. Sanity-check that we got
        // *some* compression.
        assert!(encoded_size < input.len() / 2,
            "expected significant LZP compression, got {} -> {}",
            input.len(), encoded_size);
    }

    #[test]
    fn round_trip_random_no_compression() {
        // Pseudo-random bytes have effectively zero LZP-detectable
        // structure; output should be roughly size-of-input with a few
        // escapes for incidental 0xF2 bytes.
        let mut input = vec![0u8; 4096];
        let mut x = 0x9E3779B1u32;
        for b in input.iter_mut() {
            x = x.wrapping_mul(2654435769).wrapping_add(1);
            *b = (x >> 24) as u8;
        }
        let encoded_size = round_trip(&input, 15, 6);
        // Random data shouldn't grow by more than ~5% (escape bytes).
        assert!(encoded_size < input.len() + input.len() / 16,
            "random grew too much: {} -> {}", input.len(), encoded_size);
    }

    #[test]
    fn long_match_crosses_254_boundary() {
        // Force a >= 254-byte match so the length encoding chains
        // 254-bytes. Pattern: 8-byte prefix to fill the hash, then a
        // long run of identical "ABCD" so the encoder finds a huge
        // match shortly after the first occurrence.
        let mut input = Vec::with_capacity(2048);
        input.extend_from_slice(b"PREFIX..");
        for _ in 0..512 {
            input.extend_from_slice(b"ABCD");
        }
        let encoded_size = round_trip(&input, 15, 6);
        assert!(encoded_size < 200, "long-match encoding should compress hard, got {}", encoded_size);
    }
}
