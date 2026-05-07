//! BCJ2 decoder — port of `7zip/C/Bcj2.c`.
//!
//! BCJ2 takes an x86 instruction stream pre-split into four sub-streams by
//! the encoder:
//! * `Main`: bytes that are *not* part of converted CALL/JUMP instructions;
//! * `Call`: 32-bit big-endian absolute target addresses for CALL (`E8`);
//! * `Jump`: 32-bit big-endian absolute target addresses for `E9` / `0F 8x`;
//! * `Rc`:   range-coded stream that decides for each ambiguous E8/E9/0F8x
//!           whether it's real or data.

const TOP_VALUE: u32 = 1 << 24;
const NUM_BIT_MODEL_TOTAL_BITS: u32 = 11;
const BIT_MODEL_TOTAL: u32 = 1 << NUM_BIT_MODEL_TOTAL_BITS;
const NUM_MOVE_BITS: u32 = 5;
const NUM_SHIFT_BITS: u32 = 24;

/// Decode the four BCJ2 sub-streams in one shot. `out` is appended.
pub fn decode_one_shot_to(
    main: &[u8],
    call: &[u8],
    jump: &[u8],
    rc: &[u8],
    out: &mut Vec<u8>,
) -> Result<(), &'static str> {
    if call.len() & 3 != 0 || jump.len() & 3 != 0 {
        return Err("call/jump stream lengths must be multiples of 4");
    }
    let mut probs = [(BIT_MODEL_TOTAL >> 1) as u16; 2 + 256];
    let mut main_pos = 0usize;
    let mut call_pos = 0usize;
    let mut jump_pos = 0usize;
    let mut rc_pos = 0usize;

    let mut ip: u32 = 0;
    let mut temp: u32 = 0;
    let mut range: u32 = 0;
    let mut code: u32 = 0;

    // Range-coder warm-up: the first 5 bytes of the RC stream are the
    // initial code (with first byte mandated zero by encoder).
    for r in 0..5 {
        if r == 1 && code != 0 {
            return Err("BCJ2 RC stream invalid first byte");
        }
        if rc_pos >= rc.len() {
            return Err("BCJ2 RC stream truncated during init");
        }
        code = (code << 8) | rc[rc_pos] as u32;
        rc_pos += 1;
    }
    if code == 0xFFFF_FFFF {
        return Err("BCJ2 RC stream invalid initial code");
    }
    range = 0xFFFF_FFFF;

    loop {
        // (1) RC normalize once per "outer" iteration (= once per marker found
        //     or terminal exit).
        if range < TOP_VALUE {
            if rc_pos >= rc.len() {
                // No more RC data — end of stream.
                return Ok(());
            }
            range <<= 8;
            code = (code << 8) | rc[rc_pos] as u32;
            rc_pos += 1;
        }

        // (2) Drain main stream, emitting every byte, until a marker candidate
        //     is found.
        let mut found_marker = false;
        while main_pos < main.len() {
            let b = main[main_pos];
            main_pos += 1;
            out.push(b);
            ip = ip.wrapping_add(1);
            temp = (temp << NUM_SHIFT_BITS) | b as u32;

            // E8/E9 candidate?
            if ((b as u32 + (0x100 - 0xe8)) & 0xfe) == 0 {
                found_marker = true;
                break;
            }
            // 0F 8x candidate? `temp` has prev_byte at bits 24..31 and current
            // at bits 0..7, with bits 8..23 = 0 (since the shift is 24).
            if ((temp.wrapping_sub((0x0f << NUM_SHIFT_BITS) + 0x80)) & 0xFFFF_FFF0u32)
                == 0
            {
                found_marker = true;
                break;
            }
        }
        if !found_marker {
            // Main stream exhausted without finding another marker.
            return Ok(());
        }

        // (3) Range-coded bit: real CALL/JUMP or data?
        let c = ((temp.wrapping_add(0x17)) >> 6) & 1;
        let prob_idx = (((0u32.wrapping_sub(c)) & ((temp >> NUM_SHIFT_BITS) & 0xff))
            .wrapping_add(c)
            .wrapping_add((temp >> 5) & 1)) as usize;
        let mut ttt = probs[prob_idx] as u32;
        let bound = (range >> NUM_BIT_MODEL_TOTAL_BITS) * ttt;
        if code < bound {
            // Bit 0: not a real call/jump, just keep scanning.
            range = bound;
            ttt = ttt + ((BIT_MODEL_TOTAL - ttt) >> NUM_MOVE_BITS);
            probs[prob_idx] = ttt as u16;
            continue;
        }
        range -= bound;
        code -= bound;
        ttt -= ttt >> NUM_MOVE_BITS;
        probs[prob_idx] = ttt as u16;

        // (4) Real CALL or JUMP — pull 4 BE bytes from the corresponding
        //     stream.  The encoded value is the absolute target; convert to
        //     relative by subtracting `ip + 4`.
        let cj_is_call = (((temp.wrapping_add(0x57)) >> 6) & 1) == 0;
        let (cj_buf, cj_pos) = if cj_is_call {
            (call, &mut call_pos)
        } else {
            (jump, &mut jump_pos)
        };
        if *cj_pos + 4 > cj_buf.len() {
            return Err("BCJ2 CALL/JUMP stream truncated");
        }
        let v = u32::from_be_bytes(cj_buf[*cj_pos..*cj_pos + 4].try_into().unwrap());
        *cj_pos += 4;
        ip = ip.wrapping_add(4);
        let rel = v.wrapping_sub(ip);

        out.push(rel as u8);
        out.push((rel >> 8) as u8);
        out.push((rel >> 16) as u8);
        out.push((rel >> 24) as u8);
        temp = rel >> 24;
    }
}

/// Convenience wrapper.
pub fn decode_one_shot(
    main: &[u8],
    call: &[u8],
    jump: &[u8],
    rc: &[u8],
) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::new();
    decode_one_shot_to(main, call, jump, rc, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_main_init_only() {
        // 5 zero bytes of RC: valid init, range=0xFFFFFFFF code=0.
        let out = decode_one_shot(&[], &[], &[], &[0; 5]).unwrap();
        assert!(out.is_empty());
    }
}
