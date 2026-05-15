//! `MatchModel` — paq8.cpp:3521-3693.
//!
//! Finds the longest recent match of the current context in the
//! history buffer and predicts the next byte from it. Backed by a
//! `U32` position table + 3 StateMap32 + 3 SmallStationaryContextMap
//! + 3 StationaryMap + an IndirectContext bit-history.

#![allow(dead_code)]

use super::context_map::{SmallStationaryContextMap, StationaryMap};
use super::apm::StateMap32;
use super::mixer::Mixer;
use super::stats::ModelStats;
use super::substrate::{
    combine64, finalize64, hash5, ilog2, Buf, Ilog, Squash, Stretch,
};
use super::util::IndirectContext;

const MAX_LEN:    u32 = 0xFFFF;
const MAX_EXTEND: u32 = 0;
const MIN_LEN:    u32 = 5;
const STEP_SIZE:  u32 = 2;
const DELTA_LEN:  u32 = 5;
const NUM_CTXS:   usize = 3;
const NUM_HASHES: usize = 3;

pub struct MatchModel {
    table:      Vec<u32>,
    state_maps: Vec<StateMap32>,
    scm:        Vec<SmallStationaryContextMap>,
    maps:       Vec<StationaryMap>,
    i_ctx:      IndirectContext,
    hashes:     [u32; NUM_HASHES],
    ctx:        [u32; NUM_CTXS],
    length:     u32,
    index:      u32,
    mask:       u32,
    hashbits:   u32,
    expected_byte: u8,
    delta:      bool,
}

impl MatchModel {
    /// `Size` is the table size in bytes (upstream `MEM()*2`).
    pub fn new(size: u64, dt: [i32; 1024]) -> Self {
        let entries = (size / 4) as usize;
        let entries = entries.next_power_of_two();
        let state_maps = vec![
            StateMap32::new(56 * 256, dt),
            StateMap32::new(8 * 256 * 256 + 1, dt),
            StateMap32::new(256 * 256, dt),
        ];
        let scm = vec![
            SmallStationaryContextMap::new(8, 8),
            SmallStationaryContextMap::new(11, 1),
            SmallStationaryContextMap::new(8, 8),
        ];
        let maps = vec![
            StationaryMap::new(16, 8, 0),
            StationaryMap::new(22, 1, 0),
            StationaryMap::new(4, 1, 0),
        ];
        Self {
            table: vec![0u32; entries],
            state_maps,
            scm,
            maps,
            i_ctx: IndirectContext::new(19, 1, 8),
            hashes: [0; NUM_HASHES],
            ctx: [0; NUM_CTXS],
            length: 0,
            index: 0,
            mask: (entries - 1) as u32,
            hashbits: ilog2(entries as u32),
            expected_byte: 0,
            delta: false,
        }
    }

    fn update(&mut self, buffer: &Buf, c0: u32, y: i32,
               stats: &mut ModelStats) {
        self.delta = false;
        // Update hashes (highest order first).
        let mut min_len = MIN_LEN + (NUM_HASHES as u32 - 1) * STEP_SIZE;
        for i in 0..NUM_HASHES {
            let mut h: u64 = 0;
            let mut j = min_len;
            while j > 0 {
                h = combine64(h, buffer.at(j) as u64);
                j -= 1;
            }
            self.hashes[i] = finalize64(h, self.hashbits);
            min_len -= STEP_SIZE;
        }
        // Extend or find a new match.
        if self.length != 0 {
            self.index += 1;
            if self.length < MAX_LEN { self.length += 1; }
        } else {
            let mut min_len = MIN_LEN + (NUM_HASHES as u32 - 1) * STEP_SIZE;
            let mut best_len = 0u32;
            let mut best_index = 0u32;
            for i in 0..NUM_HASHES {
                if self.length >= min_len { break; }
                self.index = self.table[(self.hashes[i] & self.mask) as usize];
                if self.index > 0 {
                    self.length = 0;
                    while self.length < (min_len + MAX_EXTEND)
                        && buffer.at(self.length + 1)
                            == buffer.abs(self.index.wrapping_sub(self.length).wrapping_sub(1))
                    {
                        self.length += 1;
                    }
                    if self.length > best_len {
                        best_len = self.length;
                        best_index = self.index;
                    }
                }
                min_len -= STEP_SIZE;
            }
            if best_len >= MIN_LEN {
                self.length = best_len - (MIN_LEN - 1);
                self.index = best_index;
            } else {
                self.length = 0;
                self.index = 0;
            }
        }
        // Store current position into all hash slots.
        for i in 0..NUM_HASHES {
            self.table[(self.hashes[i] & self.mask) as usize] = buffer.pos;
        }
        self.expected_byte = buffer.abs(self.index);
        self.i_ctx.add(y as u32);
        self.i_ctx.set(((buffer.at(1) as u32) << 8) | self.expected_byte as u32);
        self.scm[0].set(self.expected_byte as u32);
        self.scm[1].set(self.expected_byte as u32);
        self.scm[2].set(buffer.pos);
        self.maps[0].set_direct(((self.expected_byte as u32) << 8)
            | buffer.at(1) as u32);
        self.maps[1].set(hash5(
            self.expected_byte as u64, c0 as u64,
            buffer.at(1) as u64, buffer.at(2) as u64,
            3.min(ilog2(self.length + 1)) as u64));
        self.maps[2].set_direct(self.i_ctx.get());
        stats.r#match.expected_byte =
            if self.length > 0 { self.expected_byte } else { 0 };
    }

    /// `Predict` — paq8.cpp:3632-3692. Returns `length`.
    pub fn predict(&mut self, m: &mut Mixer, buffer: &Buf,
                   c0: u32, bpos: i32, y: i32,
                   ilog: &Ilog, dt: &[i32; 1024],
                   squash: &Squash, stretch: &Stretch,
                   stats: &mut ModelStats) -> u32 {
        if bpos == 0 {
            self.update(buffer, c0, y, stats);
        } else {
            let b = (c0 << (8 - bpos)) as u8;
            self.scm[1].set(((bpos as u32) << 8)
                | (self.expected_byte ^ b) as u32);
            self.maps[1].set(hash5(
                self.expected_byte as u64, c0 as u64,
                buffer.at(1) as u64, buffer.at(2) as u64,
                3.min(ilog2(self.length + 1)) as u64));
            self.i_ctx.add(y as u32);
            self.i_ctx.set(((bpos as u32) << 16)
                | ((buffer.at(1) as u32) << 8)
                | (self.expected_byte ^ b) as u32);
            self.maps[2].set_direct(self.i_ctx.get());
        }
        let expected_bit = ((self.expected_byte >> (7 - bpos)) & 1) as i32;

        if self.length > 0 {
            let is_match = if bpos == 0 {
                buffer.at(1) == buffer.abs(self.index.wrapping_sub(1))
            } else {
                ((self.expected_byte as u32 + 256) >> (8 - bpos)) == c0
            };
            if !is_match {
                self.delta = (self.length + MIN_LEN) > DELTA_LEN;
                self.length = 0;
            }
        }

        self.ctx = [0; NUM_CTXS];
        if self.length > 0 {
            if self.length <= 16 {
                self.ctx[0] = (self.length - 1) * 2 + expected_bit as u32;
            } else {
                self.ctx[0] = 24
                    + ((self.length - 1).min(63) >> 2) * 2
                    + expected_bit as u32;
            }
            self.ctx[0] = (self.ctx[0] << 8) | c0;
            self.ctx[1] = (((self.expected_byte as u32) << 11)
                | ((bpos as u32) << 8)
                | buffer.at(1) as u32) + 1;
            let sign = 2 * expected_bit - 1;
            m.add((sign * ((self.length.min(32) as i32) << 5)) as i16);
            m.add((sign * ((ilog.get(((self.length) & 0xffff) as u16) as i32) << 2)) as i16);
        } else {
            m.add(0);
            m.add(0);
        }

        if self.delta {
            self.ctx[2] = ((self.expected_byte as u32) << 8) | c0;
        }

        for i in 0..NUM_CTXS {
            let c = self.ctx[i];
            let p = self.state_maps[i].p(c, 1023, y);
            if c != 0 {
                m.add(((stretch.get(p) + 1) >> 1) as i16);
            } else {
                m.add(0);
            }
        }

        self.scm[0].mix(m, y, 7, 1, 4, squash, stretch);
        self.scm[1].mix(m, y, 6, 1, 4, squash, stretch);
        self.scm[2].mix(m, y, 5, 1, 4, squash, stretch);
        self.maps[0].mix(m, y, 1, 4, 255, dt, squash, stretch);
        self.maps[1].mix(m, y, 1, 4, 1023, dt, squash, stretch);
        self.maps[2].mix(m, y, 1, 4, 1023, dt, squash, stretch);

        stats.r#match.length = self.length;
        self.length
    }

    pub fn length(&self) -> u32 { self.length }
    pub fn expected_byte(&self) -> u8 { self.expected_byte }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    #[test]
    fn match_model_finds_repeated_pattern() {
        let dt = build_dt();
        let il = Ilog::new();
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        let mut mm = MatchModel::new(1 << 16, dt);
        let mut buf = Buf::new();
        buf.set_size(1 << 16);
        let mut stats = ModelStats::new();
        let mut mixer = Mixer::new(64, 4, 0);

        // Feed "abcdefghij..." repeated. Upstream order: model all 8
        // bits of a byte, THEN commit it to the buffer — so bpos==0
        // of the next byte sees the previous byte at `buffer(1)`.
        let pattern = b"abcdefghij";
        for _round in 0..8 {
            for &byte in pattern {
                for bp in 0..8 {
                    let bit = ((byte >> (7 - bp)) & 1) as i32;
                    let c0 = if bp == 0 { 1u32 }
                        else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                    let _ = mm.predict(&mut mixer, &buf, c0, bp, bit,
                        &il, &dt, &sq, &st, &mut stats);
                }
                buf.push(byte);
            }
        }
        // By the last rounds a match should have been found.
        assert!(mm.length() > 0 || stats.r#match.length > 0,
            "expected the repeated pattern to produce a match");
    }
}
