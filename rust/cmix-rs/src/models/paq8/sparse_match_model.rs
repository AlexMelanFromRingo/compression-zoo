//! `SparseMatchModel` — paq8.cpp:3695-3843.
//!
//! Like [`MatchModel`](super::match_model::MatchModel) but searches
//! for matches at sparse strides / with bit masks / with offsets,
//! and tracks which of the 4 sparse configs found the active match
//! via an MTF list.

#![allow(dead_code)]

use super::context_map::StationaryMap;
use super::mixer::Mixer;
use super::stats::ModelStats;
use super::substrate::{combine64, finalize64, hash5, ilog2, Buf, Squash, Stretch};
use super::util::{IndirectContext, MtfList};

const MAX_LEN:     u32 = 0xFFFF;
const MIN_LEN:     u32 = 3;
const NUM_HASHES:  usize = 4;

#[derive(Clone, Copy)]
struct SparseConfig {
    offset:    u32,
    stride:    u32,
    deletions: u32,
    min_len:   u32,
    bit_mask:  u32,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self { offset: 0, stride: 1, deletions: 0,
               min_len: MIN_LEN, bit_mask: 0xFF }
    }
}

pub struct SparseMatchModel {
    table:      Vec<u32>,
    maps:       Vec<StationaryMap>,
    i_ctx8:     IndirectContext,
    i_ctx16:    IndirectContext,
    list:       MtfList,
    sparse:     [SparseConfig; NUM_HASHES],
    hashes:     [u32; NUM_HASHES],
    hash_index: u32,
    length:     u32,
    index:      u32,
    mask:       u32,
    hashbits:   u32,
    expected_byte: u8,
    valid:      bool,
}

impl SparseMatchModel {
    pub fn new(size: u64) -> Self {
        let entries = ((size / 4) as usize).next_power_of_two();
        let maps = vec![
            StationaryMap::new(22, 1, 0),
            StationaryMap::new(14, 4, 0),
            StationaryMap::new(8, 1, 0),
            StationaryMap::new(19, 1, 0),
        ];
        let mut sparse = [SparseConfig::default(); NUM_HASHES];
        sparse[0].min_len = 5; sparse[0].bit_mask = 0xDF;
        sparse[1].offset = 1;  sparse[1].min_len = 4;
        sparse[2].stride = 2;  sparse[2].min_len = 4; sparse[2].bit_mask = 0xDF;
        sparse[3].min_len = 5; sparse[3].bit_mask = 0xF;
        Self {
            table: vec![0u32; entries],
            maps,
            i_ctx8:  IndirectContext::new(19, 1, 8),
            i_ctx16: IndirectContext::new(16, 8, 16),
            list:    MtfList::new(NUM_HASHES),
            sparse,
            hashes:  [0; NUM_HASHES],
            hash_index: 0,
            length:  0,
            index:   0,
            mask:    (entries - 1) as u32,
            hashbits: ilog2(entries as u32),
            expected_byte: 0,
            valid:   false,
        }
    }

    fn update(&mut self, buffer: &Buf, c0: u32, y: i32) {
        // Update sparse hashes.
        for i in 0..NUM_HASHES {
            let cfg = self.sparse[i];
            let mut h: u64 = 0;
            let mut k = cfg.offset + 1;
            for _ in 0..cfg.min_len {
                h = combine64(h, (buffer.at(k) as u32 & cfg.bit_mask) as u64);
                k += cfg.stride;
            }
            self.hashes[i] = finalize64(h, self.hashbits);
        }
        // Extend or find a new match.
        if self.length != 0 {
            self.index += 1;
            if self.length < MAX_LEN { self.length += 1; }
        } else {
            let mut i = self.list.get_first();
            while i >= 0 {
                let cfg = self.sparse[i as usize];
                self.index = self.table[(self.hashes[i as usize] & self.mask) as usize];
                if self.index > 0 {
                    let mut offset = cfg.offset + 1;
                    while self.length < cfg.min_len
                        && ((buffer.at(offset)
                            ^ buffer.abs(self.index.wrapping_sub(offset)))
                            as u32 & cfg.bit_mask) == 0
                    {
                        self.length += 1;
                        offset += cfg.stride;
                    }
                    if self.length >= cfg.min_len {
                        self.length -= cfg.min_len - 1;
                        self.index += cfg.deletions;
                        self.hash_index = i as u32;
                        self.list.move_to_front(i);
                        break;
                    }
                }
                self.length = 0;
                self.index = 0;
                i = self.list.get_next();
            }
        }
        // Store position information.
        for i in 0..NUM_HASHES {
            self.table[(self.hashes[i] & self.mask) as usize] = buffer.pos;
        }
        self.expected_byte = buffer.abs(self.index);
        if self.valid {
            self.i_ctx8.add(y as u32);
            self.i_ctx16.add(buffer.at(1) as u32);
        }
        self.valid = self.length > 1;
        if self.valid {
            self.maps[0].set(hash5(
                self.expected_byte as u64, c0 as u64,
                buffer.at(1) as u64, buffer.at(2) as u64,
                (ilog2(self.length + 1) * NUM_HASHES as u32
                    + self.hash_index) as u64));
            self.maps[1].set_direct(((self.expected_byte as u32) << 8)
                | buffer.at(1) as u32);
            let comb = ((buffer.at(1) as u32) << 8) | self.expected_byte as u32;
            self.i_ctx8.set(comb);
            self.i_ctx16.set(comb);
            self.maps[2].set_direct(self.i_ctx8.get());
            self.maps[3].set_direct(self.i_ctx16.get());
        }
    }

    /// `Predict` — paq8.cpp:3803-3842. Returns `length`.
    pub fn predict(&mut self, m: &mut Mixer, buffer: &Buf,
                   c0: u32, bpos: i32, y: i32, dt: &[i32; 1024],
                   squash: &Squash, stretch: &Stretch,
                   _stats: &mut ModelStats) -> u32 {
        let b = (c0 << (8 - bpos)) as u8;
        if bpos == 0 {
            self.update(buffer, c0, y);
        } else if self.valid {
            self.maps[0].set(hash5(
                self.expected_byte as u64, c0 as u64,
                buffer.at(1) as u64, buffer.at(2) as u64,
                (ilog2(self.length + 1) * NUM_HASHES as u32
                    + self.hash_index) as u64));
            if bpos == 4 {
                self.maps[1].set_direct(0x10000
                    | (((self.expected_byte ^ ((c0 << 4) as u8)) as u32) << 8)
                    | buffer.at(1) as u32);
            }
            self.i_ctx8.add(y as u32);
            self.i_ctx8.set(((bpos as u32) << 16)
                | ((buffer.at(1) as u32) << 8)
                | (self.expected_byte ^ b) as u32);
            self.maps[2].set_direct(self.i_ctx8.get());
            self.maps[3].set_direct(((bpos as u32) << 16)
                | (self.i_ctx16.get()
                    ^ ((b as u32) | ((b as u32) << 8))));
        }

        // Check if the next bit matches the prediction.
        if self.length > 0
            && (((self.expected_byte ^ b) as u32
                & self.sparse[self.hash_index as usize].bit_mask)
                >> (8 - bpos)) != 0
        {
            self.length = 0;
        }

        if self.valid {
            if self.length > 1
                && ((self.sparse[self.hash_index as usize].bit_mask
                    >> (7 - bpos)) & 1) > 0
            {
                let expected_bit = ((self.expected_byte >> (7 - bpos)) & 1) as i32;
                let sign = 2 * expected_bit - 1;
                m.add((sign * (((self.length - 1).min(64) as i32) << 4)) as i16);
                m.add((sign
                    * (1i32 << (self.length.saturating_sub(2)).min(3))
                    * ((self.length - 1).min(8) as i32) << 4) as i16);
                m.add((sign * 512) as i16);
            } else {
                m.add(0); m.add(0); m.add(0);
            }
            for i in 0..4 {
                self.maps[i].mix(m, y, 1, 2, 1023, dt, squash, stretch);
            }
        } else {
            for _ in 0..11 { m.add(0); }
        }

        m.set(((self.hash_index << 6)
            | ((bpos as u32) << 3)
            | 7.min(self.length)) as u32,
            NUM_HASHES as u32 * 64);
        m.set(((self.hash_index << 11)
            | (7.min(ilog2(self.length + 1)) << 8)
            | (c0 ^ ((self.expected_byte as u32) >> (8 - bpos)))) as u32,
            NUM_HASHES as u32 * 2048);

        self.length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    #[test]
    fn sparse_match_model_runs_without_panic() {
        let dt = build_dt();
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        let mut smm = SparseMatchModel::new(1 << 16);
        let mut buf = Buf::new();
        buf.set_size(1 << 16);
        let mut stats = ModelStats::new();
        let mut mixer = Mixer::new(64, 4, 0);

        let pattern = b"a-b-c-d-e-f-g-h-";
        for _round in 0..6 {
            for &byte in pattern {
                for bp in 0..8 {
                    let bit = ((byte >> (7 - bp)) & 1) as i32;
                    let c0 = if bp == 0 { 1u32 }
                        else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                    let _ = smm.predict(&mut mixer, &buf, c0, bp, bit,
                        &dt, &sq, &st, &mut stats);
                }
                buf.push(byte);
            }
        }
        // No assertion on match length — sparse matches are
        // input-dependent; this is a no-panic smoke test.
    }
}
