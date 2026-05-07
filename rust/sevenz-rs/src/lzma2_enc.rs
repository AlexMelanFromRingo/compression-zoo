//! LZMA2 encoder — port of `7zip/C/Lzma2Enc.c`.
//!
//! Produces a single LZMA2 chunk per call: control byte 0xE0 ("LZMA, reset
//! state + new properties + reset dictionary"), 5-byte LZMA properties tail,
//! then the LZMA payload.  The encoder works in single-block mode (no
//! threading, no chunk splitting) — sufficient for one-shot encoding.

use crate::lzma_enc::{encode_props, Encoder, EncoderConfig};

/// Pack a single LZMA2 dictionary-property byte from a dict size, using the
/// C `Lzma2EncProps_GetDictSize` helper's inverse.
pub fn dic_size_to_prop(mut dict_size: u32) -> u8 {
    // The valid set is `(2 | (i & 1)) << ((i / 2) + 11)`. Find the smallest
    // `i` whose canonical size is ≥ dict_size, exactly mirroring the C
    // `Lzma2EncProps_Normalize`.
    if dict_size == 0xFFFF_FFFF {
        return 40;
    }
    if dict_size < (1 << 12) {
        dict_size = 1 << 12;
    }
    for i in 0u8..=40 {
        let s = (2u32 | (i as u32 & 1)) << ((i / 2) + 11);
        if dict_size <= s {
            return i;
        }
    }
    40
}

// LZMA2 single-chunk size limits (per the on-wire encoding).
const MAX_UNPACK_PER_CHUNK: usize = 1 << 21; // 2 MiB
const MAX_PACK_PER_CHUNK: usize = 1 << 16; // 64 KiB

/// Encode `data` as a sequence of independent LZMA2 chunks plus the
/// end-of-stream byte.  Each chunk uses control byte `0xE0` (LZMA, reset
/// state + new properties + reset dictionary) so chunks are independent;
/// this trades some compression for simplicity but is fully spec-compatible.
pub fn encode_one_shot(data: &[u8], cfg: EncoderConfig) -> Result<(u8, Vec<u8>), &'static str> {
    let prop_byte = dic_size_to_prop(cfg.dict_size);
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        // Pick a chunk size that keeps both unpack and pack within limits.
        let mut size = (data.len() - off).min(MAX_UNPACK_PER_CHUNK);
        // Try the requested size; if the compressed payload exceeds 64 KiB,
        // halve and retry. Worst case we end up at single-byte chunks (which
        // always compress to a few bytes).
        let mut payload;
        loop {
            let mut lz_cfg = cfg;
            lz_cfg.write_end_mark = false;
            let mut enc = Encoder::new(lz_cfg);
            payload = enc.encode(&data[off..off + size])?;
            if payload.len() <= MAX_PACK_PER_CHUNK {
                break;
            }
            if size == 1 {
                return Err("LZMA2 chunk pack-size overflow even at 1 byte");
            }
            size /= 2;
        }
        write_lzma_chunk(&cfg, &data[off..off + size], &payload, &mut out);
        off += size;
    }
    out.push(0); // end-of-stream marker
    Ok((prop_byte, out))
}

fn write_lzma_chunk(cfg: &EncoderConfig, raw: &[u8], payload: &[u8], out: &mut Vec<u8>) {
    let unp = (raw.len() - 1) as u32;
    let pck = (payload.len() - 1) as u32;
    let control = 0xE0u8 | (((unp >> 16) & 0x1F) as u8);
    out.push(control);
    out.push((unp >> 8) as u8);
    out.push((unp & 0xFF) as u8);
    out.push((pck >> 8) as u8);
    out.push((pck & 0xFF) as u8);
    let props = encode_props(cfg);
    out.push(props[0]);
    out.extend_from_slice(payload);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_via_our_decoder() {
        let data = b"Hello LZMA2! Testing testing 123 hello hello world.".to_vec();
        let cfg = EncoderConfig::default();
        let (prop, framed) = encode_one_shot(&data, cfg).unwrap();
        let decoded = crate::lzma2_dec::decode_one_shot(prop, &framed).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn empty_input() {
        let cfg = EncoderConfig::default();
        let (prop, framed) = encode_one_shot(&[], cfg).unwrap();
        assert_eq!(framed, vec![0]);
        let decoded = crate::lzma2_dec::decode_one_shot(prop, &framed).unwrap();
        assert!(decoded.is_empty());
    }
}
