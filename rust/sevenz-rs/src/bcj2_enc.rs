//! BCJ2 encoder — port of `7zip/C/Bcj2Enc.c`, one-shot variant.
//!
//! Splits an x86 byte stream into four sub-streams: `main`, `call`, `jump`,
//! `rc` (range-coded "is this a real branch?" bits). Inverse of
//! [`crate::bcj2_dec::decode_one_shot`].

const TOP_VALUE: u32 = 1 << 24;
const NUM_BIT_MODEL_TOTAL_BITS: u32 = 11;
const BIT_MODEL_TOTAL: u32 = 1 << NUM_BIT_MODEL_TOTAL_BITS;
const NUM_MOVE_BITS: u32 = 5;
const NUM_SHIFT_BITS: u32 = 24;
pub const RELAT_LIMIT_DEFAULT: u32 = 0x0F << 24;

#[derive(Debug)]
pub struct Output {
    pub main: Vec<u8>,
    pub call: Vec<u8>,
    pub jump: Vec<u8>,
    pub rc: Vec<u8>,
}

/// One-shot BCJ2 encode. `ip` is the virtual program counter at the start
/// of `data` (use 0 if unknown). Returns the four sub-streams.
pub fn encode_one_shot(data: &[u8], mut ip: u64) -> Output {
    let file_ip64: u64 = ip;
    let file_size_minus1: u64 = u64::MAX - 1; // unlimited

    let mut probs = [(BIT_MODEL_TOTAL >> 1) as u16; 2 + 256];
    let mut range: u32 = 0xFFFF_FFFF;
    let mut low: u64 = 0;
    let mut cache: u8 = 0;
    let mut cache_size: u32 = 1;

    let mut main = Vec::with_capacity(data.len());
    let mut call: Vec<u8> = Vec::new();
    let mut jump: Vec<u8> = Vec::new();
    let mut rc: Vec<u8> = Vec::new();

    let mut shift_low = |low_in: &mut u64,
                         cache_in: &mut u8,
                         cache_size_in: &mut u32,
                         rc: &mut Vec<u8>| {
        let l = *low_in as u32;
        let high = (*low_in >> 32) as u32;
        if l < 0xFF00_0000 || high != 0 {
            // Emit cache_size bytes: first is cache+high, rest are 0xFF+high.
            let extra = high as u8;
            let mut first = true;
            let mut cs = *cache_size_in;
            while cs > 0 {
                if first {
                    rc.push((*cache_in).wrapping_add(extra));
                    first = false;
                } else {
                    rc.push((0xFFu8).wrapping_add(extra));
                }
                cs -= 1;
            }
            *cache_in = (l >> 24) as u8;
            *cache_size_in = 0;
        }
        *cache_size_in += 1;
        *low_in = (l << 8) as u64;
    };

    let mut v: u32 = 0; // context for marker detection
    let mut i = 0usize;
    while i < data.len() {
        // Range-coder normalize.
        if range < TOP_VALUE {
            shift_low(&mut low, &mut cache, &mut cache_size, &mut rc);
            range <<= 8;
        }

        let b = data[i];
        main.push(b);
        v = (v << NUM_SHIFT_BITS) | b as u32;
        let is_e8_e9 = ((b as u32 + (0x100 - 0xe8)) & 0xfe) == 0;
        let is_0f_8x = ((v.wrapping_sub((0x0f << NUM_SHIFT_BITS) + 0x80)) & 0xFFFF_FFF0u32)
            == 0;
        i += 1;
        ip = ip.wrapping_add(1);
        if !is_e8_e9 && !is_0f_8x {
            continue;
        }

        // Need 4 more bytes for displacement.
        let mut conv_flag = false;
        if data.len() - i >= 4 {
            let relat = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
            let ip_rel = ip.wrapping_sub(file_ip64);
            // Mirrors C's CONV_FLAG logic at v23.00.
            let extra = (((v + 0x20) >> 5) & 1) as u64;
            if ip_rel > extra {
                let target = ip_rel
                    .wrapping_add(4)
                    .wrapping_add(relat as i32 as i64 as u64);
                if target <= file_size_minus1 {
                    if (relat.wrapping_add(RELAT_LIMIT_DEFAULT) >> 1) < RELAT_LIMIT_DEFAULT {
                        conv_flag = true;
                    }
                }
            }
        }

        // Range-coded bit.
        let c = ((v.wrapping_add(0x17)) >> 6) & 1;
        let prob_idx =
            (((0u32.wrapping_sub(c)) & ((v >> NUM_SHIFT_BITS) & 0xff))
                .wrapping_add(c)
                .wrapping_add((v >> 5) & 1)) as usize;
        let mut ttt = probs[prob_idx] as u32;
        let bound = (range >> NUM_BIT_MODEL_TOTAL_BITS) * ttt;
        if !conv_flag {
            range = bound;
            ttt = ttt + ((BIT_MODEL_TOTAL - ttt) >> NUM_MOVE_BITS);
            probs[prob_idx] = ttt as u16;
            continue;
        }
        low = low.wrapping_add(bound as u64);
        range -= bound;
        ttt -= ttt >> NUM_MOVE_BITS;
        probs[prob_idx] = ttt as u16;

        // Real CALL/JUMP — pull 4 bytes, write absolute target to call/jump
        // stream as big-endian, advance.
        let call_or_jump = (((v.wrapping_add(0x57)) >> 6) & 1) + 1;
        let relat = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        let absol = (ip as u32).wrapping_add(4).wrapping_add(relat);
        let stream_buf = if call_or_jump == 1 { &mut call } else { &mut jump };
        stream_buf.extend_from_slice(&absol.to_be_bytes());
        i += 4;
        ip = ip.wrapping_add(4);
        v = relat >> 24;
    }

    // Flush range coder: 5 shift_low calls.
    for _ in 0..5 {
        shift_low(&mut low, &mut cache, &mut cache_size, &mut rc);
    }
    Output { main, call, jump, rc }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_via_our_decoder() {
        let data = vec![
            0x55u8, 0x48, 0x89, 0xe5,
            0xe8, 0x10, 0x00, 0x00, 0x00,
            0x90, 0x90, 0x90, 0x90,
            0xe9, 0x20, 0x00, 0x00, 0x00,
            0xc3, 0x00, 0x00, 0x00,
        ];
        let enc = encode_one_shot(&data, 0);
        let mut out = Vec::new();
        crate::bcj2_dec::decode_one_shot_to(&enc.main, &enc.call, &enc.jump, &enc.rc, &mut out)
            .unwrap();
        assert_eq!(out, data);
    }
}
