//! LZMA decoder — port of `7zip/C/LzmaDec.c`.
//!
//! Public API:
//! * [`Properties`] — parse the 5-byte LZMA properties header.
//! * [`Decoder`] — streaming decoder that owns its dictionary.
//! * [`decode_one_shot`] — one-call decoder for a full LZMA stream.
//!
//! The implementation faithfully mirrors the SDK control flow so it produces
//! bit-identical output to the C reference; the only freedom we take is using
//! safe slicing instead of raw pointer arithmetic.

use std::convert::TryInto;

// ====================================================================
// Constants (same names/values as LzmaDec.c)
// ====================================================================

const K_TOP_VALUE: u32 = 1 << 24;
const K_NUM_BIT_MODEL_TOTAL_BITS: u32 = 11;
const K_BIT_MODEL_TOTAL: u32 = 1 << K_NUM_BIT_MODEL_TOTAL_BITS;
const K_NUM_MOVE_BITS: u32 = 5;

const RC_INIT_SIZE: usize = 5;
pub const PROPS_SIZE: usize = 5;
pub const REQUIRED_INPUT_MAX: usize = 20;

const K_NUM_POS_BITS_MAX: u32 = 4;
const K_NUM_POS_STATES_MAX: usize = 1 << K_NUM_POS_BITS_MAX;

const K_LEN_NUM_LOW_BITS: u32 = 3;
const K_LEN_NUM_LOW_SYMBOLS: u32 = 1 << K_LEN_NUM_LOW_BITS;
const K_LEN_NUM_HIGH_BITS: u32 = 8;
const K_LEN_NUM_HIGH_SYMBOLS: u32 = 1 << K_LEN_NUM_HIGH_BITS;

const LEN_LOW: usize = 0;
const LEN_HIGH: usize = LEN_LOW + 2 * (K_NUM_POS_STATES_MAX << K_LEN_NUM_LOW_BITS);
const K_NUM_LEN_PROBS: usize = LEN_HIGH + K_LEN_NUM_HIGH_SYMBOLS as usize;
const LEN_CHOICE: usize = LEN_LOW;
const LEN_CHOICE2: usize = LEN_LOW + (1 << K_LEN_NUM_LOW_BITS);

const K_NUM_STATES: u32 = 12;
const K_NUM_STATES2: u32 = 16;
const K_NUM_LIT_STATES: u32 = 7;

const K_START_POS_MODEL_INDEX: u32 = 4;
const K_END_POS_MODEL_INDEX: u32 = 14;
const K_NUM_FULL_DISTANCES: usize = 1 << (K_END_POS_MODEL_INDEX as usize >> 1);

const K_NUM_POS_SLOT_BITS: u32 = 6;
const K_NUM_LEN_TO_POS_STATES: u32 = 4;

const K_NUM_ALIGN_BITS: u32 = 4;
const K_ALIGN_TABLE_SIZE: usize = 1 << K_NUM_ALIGN_BITS;

const K_MATCH_MIN_LEN: u32 = 2;
const K_MATCH_SPEC_LEN_START: u32 =
    K_MATCH_MIN_LEN + K_LEN_NUM_LOW_SYMBOLS * 2 + K_LEN_NUM_HIGH_SYMBOLS;
const K_MATCH_SPEC_LEN_ERROR_DATA: u32 = 1 << 9;
const K_MATCH_SPEC_LEN_ERROR_FAIL: u32 = K_MATCH_SPEC_LEN_ERROR_DATA - 1;

const LZMA_LIT_SIZE: usize = 0x300;

// Probability layout indices into the prob array, with kStartOffset = 1664.
// We rebase to non-negative indices.
const SPEC_POS: usize = 0;
const IS_REP0_LONG: usize = SPEC_POS + K_NUM_FULL_DISTANCES;
const REP_LEN_CODER: usize =
    IS_REP0_LONG + (K_NUM_STATES2 as usize) * K_NUM_POS_STATES_MAX;
const LEN_CODER: usize = REP_LEN_CODER + K_NUM_LEN_PROBS;
const IS_MATCH: usize = LEN_CODER + K_NUM_LEN_PROBS;
const ALIGN_OFF: usize =
    IS_MATCH + (K_NUM_STATES2 as usize) * K_NUM_POS_STATES_MAX;
const IS_REP: usize = ALIGN_OFF + K_ALIGN_TABLE_SIZE;
const IS_REP_G0: usize = IS_REP + K_NUM_STATES as usize;
const IS_REP_G1: usize = IS_REP_G0 + K_NUM_STATES as usize;
const IS_REP_G2: usize = IS_REP_G1 + K_NUM_STATES as usize;
const POS_SLOT: usize = IS_REP_G2 + K_NUM_STATES as usize;
const LITERAL: usize = POS_SLOT + (K_NUM_LEN_TO_POS_STATES as usize) * (1 << K_NUM_POS_SLOT_BITS);

// LITERAL is the offset where the literal context probs start. The full
// probs array is LITERAL + LZMA_LIT_SIZE * (1 << (lc + lp)).
const _: () = assert!(LITERAL == 1984, "bad layout");

const LZMA_DIC_MIN: u32 = 1 << 12;

// Range-coder threshold that catches ill-formed initial range states.
const K_BAD_REP_CODE: u32 = 0xC0000000 - 0x400;

// ====================================================================
// Public types
// ====================================================================

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Properties {
    pub lc: u8,
    pub lp: u8,
    pub pb: u8,
    pub dic_size: u32,
}

impl Properties {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        if data.len() < PROPS_SIZE {
            return Err(Error::Unsupported);
        }
        let d = data[0];
        if d >= 9 * 5 * 5 {
            return Err(Error::Unsupported);
        }
        let lc = d % 9;
        let d = d / 9;
        let pb = d / 5;
        let lp = d % 5;
        let dic_size = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
        let dic_size = dic_size.max(LZMA_DIC_MIN);
        Ok(Self { lc, lp, pb, dic_size })
    }

    fn num_probs(self) -> usize {
        LITERAL + LZMA_LIT_SIZE * (1usize << (self.lc + self.lp))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FinishMode {
    Any,
    End,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Status {
    NotSpecified,
    FinishedWithMark,
    NotFinished,
    NeedsMoreInput,
    MaybeFinishedWithoutMark,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    Data,
    Unsupported,
    InputEof,
    Fail,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Data => f.write_str("LZMA data error"),
            Error::Unsupported => f.write_str("unsupported LZMA properties"),
            Error::InputEof => f.write_str("unexpected end of LZMA input"),
            Error::Fail => f.write_str("internal LZMA decoder failure"),
        }
    }
}
impl std::error::Error for Error {}

// ====================================================================
// Decoder state
// ====================================================================

#[derive(Debug)]
pub struct Decoder {
    pub prop: Properties,
    probs: Vec<u16>,
    dic: Vec<u8>,
    dic_pos: usize,
    range: u32,
    code: u32,
    processed_pos: u32,
    check_dic_size: u32,
    reps: [u32; 4],
    state: u32,
    remain_len: u32,
    temp_buf: [u8; REQUIRED_INPUT_MAX],
    temp_buf_size: usize,
}

#[inline(always)]
fn dummy_to_match(d: Dummy) -> bool {
    matches!(d, Dummy::Match)
}

impl Decoder {
    /// Allocate decoder state for the given properties, with the dictionary
    /// owned internally.
    pub fn new(prop: Properties) -> Self {
        let dic_buf_size = round_up_dic_size(prop.dic_size);
        let mut d = Self {
            prop,
            probs: vec![0u16; prop.num_probs()],
            dic: vec![0u8; dic_buf_size],
            dic_pos: 0,
            range: 0,
            code: 0,
            processed_pos: 0,
            check_dic_size: 0,
            reps: [0; 4],
            state: 0,
            remain_len: 0,
            temp_buf: [0; REQUIRED_INPUT_MAX],
            temp_buf_size: 0,
        };
        d.init();
        d
    }

    /// Reset to the initial state — equivalent to `LzmaDec_Init`.
    pub fn init(&mut self) {
        self.dic_pos = 0;
        self.remain_len = K_MATCH_SPEC_LEN_START + 2;
        self.processed_pos = 0;
        self.check_dic_size = 0;
        self.temp_buf_size = 0;
    }

    /// Equivalent of `LzmaDec_InitDicAndState`: re-initialise the range
    /// coder, optionally resetting the dictionary state and/or the LZMA
    /// state. Used by LZMA2 between chunks.
    pub fn init_dic_and_state(&mut self, init_dic: bool, init_state: bool) {
        self.remain_len = K_MATCH_SPEC_LEN_START + 1;
        self.temp_buf_size = 0;
        if init_dic {
            self.processed_pos = 0;
            self.check_dic_size = 0;
            self.remain_len = K_MATCH_SPEC_LEN_START + 2;
        }
        if init_state {
            self.remain_len = K_MATCH_SPEC_LEN_START + 2;
        }
    }

    pub fn dic_buf_size(&self) -> usize {
        self.dic.len()
    }

    pub fn dic_pos(&self) -> usize {
        self.dic_pos
    }

    /// Mutable view over the raw dictionary buffer — for LZMA2 uncompressed
    /// chunks that copy bytes directly into the dictionary.
    pub fn dic_mut(&mut self) -> &mut [u8] {
        &mut self.dic
    }

    /// Write a slice directly into the dictionary as an uncompressed chunk
    /// (mirrors `LzmaDec_UpdateWithUncompressed`).
    pub fn append_uncompressed(&mut self, src: &[u8]) {
        let n = src.len();
        self.dic[self.dic_pos..self.dic_pos + n].copy_from_slice(src);
        self.dic_pos += n;
        if self.check_dic_size == 0
            && (self.prop.dic_size - self.processed_pos) <= n as u32
        {
            self.check_dic_size = self.prop.dic_size;
        }
        self.processed_pos += n as u32;
    }

    /// Replace the LZMA properties without reallocating the dictionary —
    /// used by LZMA2 when a chunk has a "set new prop" header byte.
    pub fn set_lc_lp_pb(&mut self, lc: u8, lp: u8, pb: u8) {
        self.prop.lc = lc;
        self.prop.lp = lp;
        self.prop.pb = pb;
        let n = self.prop.num_probs();
        if n != self.probs.len() {
            self.probs.resize(n, 0);
        }
    }

    /// Streaming decoder — analogue of `LzmaDec_DecodeToDic`.
    /// Reads from `src` (returning the number of bytes consumed in `consumed`)
    /// and decodes into the internal dictionary up to `dic_limit`.
    pub fn decode_to_dic(
        &mut self,
        dic_limit: usize,
        src: &[u8],
        consumed: &mut usize,
        finish_mode: FinishMode,
    ) -> Result<Status, Error> {
        let mut in_size = src.len();
        let mut src_off = 0usize;
        *consumed = 0;
        let status;

        if self.remain_len > K_MATCH_SPEC_LEN_START {
            if self.remain_len > K_MATCH_SPEC_LEN_START + 2 {
                return Err(if self.remain_len == K_MATCH_SPEC_LEN_ERROR_FAIL {
                    Error::Fail
                } else {
                    Error::Data
                });
            }
            // Range coder warm-up: gather first 5 bytes into temp_buf.
            while in_size > 0 && self.temp_buf_size < RC_INIT_SIZE {
                self.temp_buf[self.temp_buf_size] = src[src_off];
                self.temp_buf_size += 1;
                src_off += 1;
                *consumed += 1;
                in_size -= 1;
            }
            if self.temp_buf_size != 0 && self.temp_buf[0] != 0 {
                return Err(Error::Data);
            }
            if self.temp_buf_size < RC_INIT_SIZE {
                return Ok(Status::NeedsMoreInput);
            }
            self.code = ((self.temp_buf[1] as u32) << 24)
                | ((self.temp_buf[2] as u32) << 16)
                | ((self.temp_buf[3] as u32) << 8)
                | (self.temp_buf[4] as u32);
            if self.check_dic_size == 0
                && self.processed_pos == 0
                && self.code >= K_BAD_REP_CODE
            {
                return Err(Error::Data);
            }
            self.range = 0xFFFF_FFFF;
            self.temp_buf_size = 0;

            if self.remain_len > K_MATCH_SPEC_LEN_START + 1 {
                let n = self.prop.num_probs();
                for i in 0..n {
                    self.probs[i] = (K_BIT_MODEL_TOTAL >> 1) as u16;
                }
                self.reps = [1, 1, 1, 1];
                self.state = 0;
            }
            self.remain_len = 0;
        }

        loop {
            if self.remain_len == K_MATCH_SPEC_LEN_START {
                if self.code != 0 {
                    return Err(Error::Data);
                }
                return Ok(Status::FinishedWithMark);
            }

            self.write_rem(dic_limit);

            // (remain_len == 0 || dic_pos == dic_limit)

            let mut check_end_mark_now = false;
            if self.dic_pos >= dic_limit {
                if self.remain_len == 0 && self.code == 0 {
                    return Ok(Status::MaybeFinishedWithoutMark);
                }
                if finish_mode == FinishMode::Any {
                    return Ok(Status::NotFinished);
                }
                if self.remain_len != 0 {
                    return Err(Error::Data);
                }
                check_end_mark_now = true;
            }

            // remain_len == 0
            if self.temp_buf_size == 0 {
                let dummy_processed_opt;
                let buf_limit;
                if in_size < REQUIRED_INPUT_MAX || check_end_mark_now {
                    let mut buf_out_idx = src_off + in_size;
                    let dummy_res = match self.try_dummy(src, src_off, &mut buf_out_idx) {
                        Some(r) => r,
                        None => {
                            // DUMMY_INPUT_EOF
                            if in_size >= REQUIRED_INPUT_MAX {
                                self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                                return Err(Error::Fail);
                            }
                            *consumed += in_size;
                            self.temp_buf[..in_size]
                                .copy_from_slice(&src[src_off..src_off + in_size]);
                            self.temp_buf_size = in_size;
                            return Ok(Status::NeedsMoreInput);
                        }
                    };
                    let dummy_processed = buf_out_idx - src_off;
                    if dummy_processed > REQUIRED_INPUT_MAX {
                        self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                        return Err(Error::Fail);
                    }
                    if check_end_mark_now && !dummy_to_match(dummy_res) {
                        *consumed += dummy_processed;
                        self.temp_buf[..dummy_processed]
                            .copy_from_slice(&src[src_off..src_off + dummy_processed]);
                        self.temp_buf_size = dummy_processed;
                        status = Status::NotFinished;
                        let _ = status;
                        return Err(Error::Data);
                    }
                    buf_limit = src_off; // decode exactly one symbol
                    dummy_processed_opt = Some(dummy_processed);
                } else {
                    buf_limit = src_off + in_size - REQUIRED_INPUT_MAX;
                    dummy_processed_opt = None;
                }

                let mut buf_pos = src_off;
                let res = self.decode_real2(dic_limit, src, buf_limit, &mut buf_pos);
                let processed = buf_pos - src_off;
                match dummy_processed_opt {
                    None => {
                        if processed > in_size {
                            self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                            return Err(Error::Fail);
                        }
                    }
                    Some(dp) => {
                        if dp != processed {
                            self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                            return Err(Error::Fail);
                        }
                    }
                }
                src_off += processed;
                in_size -= processed;
                *consumed += processed;
                if let Err(e) = res {
                    self.remain_len = K_MATCH_SPEC_LEN_ERROR_DATA;
                    return Err(e);
                }
                continue;
            }

            // We have data buffered in temp_buf.
            let mut rem = self.temp_buf_size;
            let mut ahead = 0usize;
            while rem < REQUIRED_INPUT_MAX && ahead < in_size {
                self.temp_buf[rem] = src[src_off + ahead];
                rem += 1;
                ahead += 1;
            }

            let dummy_processed_opt;
            if rem < REQUIRED_INPUT_MAX || check_end_mark_now {
                let mut buf_out_idx = rem;
                // Build a temporary buffer for try_dummy by passing temp_buf as src.
                // Take a snapshot so the borrow is released before the mutable
                // access in decode_real2.
                let temp_buf_copy = self.temp_buf;
                let temp_slice = &temp_buf_copy[..rem];
                let dummy_res = match self.try_dummy(temp_slice, 0, &mut buf_out_idx) {
                    Some(r) => r,
                    None => {
                        if rem >= REQUIRED_INPUT_MAX {
                            self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                            return Err(Error::Fail);
                        }
                        self.temp_buf_size = rem;
                        *consumed += ahead;
                        return Ok(Status::NeedsMoreInput);
                    }
                };
                let dp = buf_out_idx;
                if dp < self.temp_buf_size {
                    self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                    return Err(Error::Fail);
                }
                if check_end_mark_now && !dummy_to_match(dummy_res) {
                    *consumed += dp - self.temp_buf_size;
                    self.temp_buf_size = dp;
                    return Err(Error::Data);
                }
                dummy_processed_opt = Some(dp);
            } else {
                dummy_processed_opt = None;
            }

            // Decode one symbol from temp_buf.
            let temp_copy = self.temp_buf;
            let mut buf_pos = 0usize;
            let res = self.decode_real2(dic_limit, &temp_copy, 0, &mut buf_pos);
            let processed = buf_pos;
            let saved_temp_size = self.temp_buf_size;
            match dummy_processed_opt {
                None => {
                    if processed > REQUIRED_INPUT_MAX {
                        self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                        return Err(Error::Fail);
                    }
                    if processed < saved_temp_size {
                        self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                        return Err(Error::Fail);
                    }
                }
                Some(dp) => {
                    if dp != processed {
                        self.remain_len = K_MATCH_SPEC_LEN_ERROR_FAIL;
                        return Err(Error::Fail);
                    }
                }
            }
            let advanced = processed - saved_temp_size;
            src_off += advanced;
            in_size -= advanced;
            *consumed += advanced;
            self.temp_buf_size = 0;
            if let Err(e) = res {
                self.remain_len = K_MATCH_SPEC_LEN_ERROR_DATA;
                return Err(e);
            }
        }
    }

    /// Buffer interface — analogue of `LzmaDec_DecodeToBuf`.
    pub fn decode_to_buf(
        &mut self,
        dest: &mut [u8],
        src: &[u8],
        consumed_in: &mut usize,
        consumed_out: &mut usize,
        finish_mode: FinishMode,
    ) -> Result<Status, Error> {
        let mut out_size = dest.len();
        let mut in_size = src.len();
        *consumed_in = 0;
        *consumed_out = 0;
        let mut src_off = 0usize;
        let mut dst_off = 0usize;
        loop {
            if self.dic_pos == self.dic_buf_size() {
                self.dic_pos = 0;
            }
            let dic_pos = self.dic_pos;
            let (out_size_cur, cur_finish_mode) = if out_size > self.dic_buf_size() - dic_pos {
                (self.dic_buf_size(), FinishMode::Any)
            } else {
                (dic_pos + out_size, finish_mode)
            };
            let mut consumed_inner = 0usize;
            let res = self.decode_to_dic(
                out_size_cur,
                &src[src_off..src_off + in_size],
                &mut consumed_inner,
                cur_finish_mode,
            );
            src_off += consumed_inner;
            in_size -= consumed_inner;
            *consumed_in += consumed_inner;
            let produced = self.dic_pos - dic_pos;
            dest[dst_off..dst_off + produced]
                .copy_from_slice(&self.dic[dic_pos..dic_pos + produced]);
            dst_off += produced;
            out_size -= produced;
            *consumed_out += produced;
            let status = res?;
            if produced == 0 || out_size == 0 {
                return Ok(status);
            }
        }
    }

    fn write_rem(&mut self, dic_limit: usize) {
        let mut len = self.remain_len as usize;
        if len == 0 {
            return;
        }
        let rem = dic_limit - self.dic_pos;
        if rem < len {
            len = rem;
            if len == 0 {
                return;
            }
        }
        if self.check_dic_size == 0
            && (self.prop.dic_size - self.processed_pos) <= len as u32
        {
            self.check_dic_size = self.prop.dic_size;
        }
        self.processed_pos += len as u32;
        self.remain_len -= len as u32;
        let dic_buf_size = self.dic_buf_size();
        let rep0 = self.reps[0] as usize;
        for _ in 0..len {
            let src_pos = if self.dic_pos < rep0 {
                self.dic_pos + dic_buf_size - rep0
            } else {
                self.dic_pos - rep0
            };
            self.dic[self.dic_pos] = self.dic[src_pos];
            self.dic_pos += 1;
        }
    }

    /// Adjusts limit per `LzmaDec_DecodeReal2`, then runs the inner decoder.
    fn decode_real2(
        &mut self,
        dic_limit: usize,
        buf: &[u8],
        buf_limit: usize,
        buf_pos: &mut usize,
    ) -> Result<(), Error> {
        let mut limit = dic_limit;
        if self.check_dic_size == 0 {
            let rem = self.prop.dic_size - self.processed_pos;
            if (limit - self.dic_pos) as u32 > rem {
                limit = self.dic_pos + rem as usize;
            }
        }
        let res = self.decode_real(limit, buf, buf_limit, buf_pos);
        if self.check_dic_size == 0 && self.processed_pos >= self.prop.dic_size {
            self.check_dic_size = self.prop.dic_size;
        }
        res
    }

    /// The hot path — `LZMA_DECODE_REAL` in C.
    fn decode_real(
        &mut self,
        limit: usize,
        buf: &[u8],
        buf_limit: usize,
        buf_pos: &mut usize,
    ) -> Result<(), Error> {
        let mut state = self.state;
        let mut rep0 = self.reps[0];
        let mut rep1 = self.reps[1];
        let mut rep2 = self.reps[2];
        let mut rep3 = self.reps[3];
        let pb_mask = (1u32 << self.prop.pb) - 1;
        let lc = self.prop.lc as u32;
        let lp_mask: u32 = (0x100u32 << self.prop.lp) - (0x100u32 >> lc);
        let dic_buf_size = self.dic_buf_size();
        let mut dic_pos = self.dic_pos;
        let mut processed_pos = self.processed_pos;
        let check_dic_size = self.check_dic_size;
        let mut len: u32 = 0;
        let mut p = *buf_pos;
        let mut range = self.range;
        let mut code = self.code;

        macro_rules! read_byte {
            () => {{
                let b = buf[p] as u32;
                p += 1;
                b
            }};
        }
        macro_rules! normalize {
            () => {
                if range < K_TOP_VALUE {
                    range <<= 8;
                    code = (code << 8) | read_byte!();
                }
            };
        }

        loop {
            let pos_state = ((processed_pos & pb_mask) << 4) as usize;
            let combined_ps = pos_state + state as usize;

            // IsMatch[state * 16 + posState]
            let prob_idx = IS_MATCH + combined_ps;
            let mut ttt = self.probs[prob_idx] as u32;
            normalize!();
            let bound = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;

            if code < bound {
                // Literal
                range = bound;
                self.probs[prob_idx] = (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;

                let mut prob_off = LITERAL;
                if processed_pos != 0 || check_dic_size != 0 {
                    let prev_byte = if dic_pos == 0 {
                        self.dic[dic_buf_size - 1]
                    } else {
                        self.dic[dic_pos - 1]
                    } as u32;
                    let lit_state = (((processed_pos << 8) + prev_byte) & lp_mask) << lc;
                    prob_off += 3 * lit_state as usize;
                }
                processed_pos += 1;

                let mut symbol: u32 = 1;
                if state < K_NUM_LIT_STATES {
                    state -= if state < 4 { state } else { 3 };
                    while symbol < 0x100 {
                        let p_idx = prob_off + symbol as usize;
                        let t = self.probs[p_idx] as u32;
                        normalize!();
                        let b = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                        if code < b {
                            range = b;
                            self.probs[p_idx] = (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
                            symbol = symbol + symbol;
                        } else {
                            range -= b;
                            code -= b;
                            self.probs[p_idx] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
                            symbol = symbol + symbol + 1;
                        }
                        let _ = t;
                    }
                } else {
                    let mut match_byte = (if dic_pos < rep0 as usize {
                        self.dic[dic_pos + dic_buf_size - rep0 as usize]
                    } else {
                        self.dic[dic_pos - rep0 as usize]
                    }) as u32;
                    let mut offs: u32 = 0x100;
                    state -= if state < 10 { 3 } else { 6 };
                    while symbol < 0x100 {
                        match_byte += match_byte;
                        let bit = offs;
                        offs &= match_byte;
                        let p_idx = prob_off + (offs + bit + symbol) as usize;
                        let t = self.probs[p_idx] as u32;
                        normalize!();
                        let b = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                        if code < b {
                            range = b;
                            self.probs[p_idx] =
                                (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
                            symbol = symbol + symbol;
                            offs ^= bit;
                        } else {
                            range -= b;
                            code -= b;
                            self.probs[p_idx] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
                            symbol = symbol + symbol + 1;
                        }
                        let _ = t;
                    }
                }
                self.dic[dic_pos] = symbol as u8;
                dic_pos += 1;
                if dic_pos >= limit || p >= buf_limit {
                    break;
                }
                continue;
            }

            // Match / Rep branch
            range -= bound;
            code -= bound;
            self.probs[prob_idx] = (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;

            let prob_idx_isrep = IS_REP + state as usize;
            ttt = self.probs[prob_idx_isrep] as u32;
            normalize!();
            let bound2 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
            let prob_len_base;
            if code < bound2 {
                // Non-rep match
                range = bound2;
                self.probs[prob_idx_isrep] =
                    (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;
                state += K_NUM_STATES;
                prob_len_base = LEN_CODER;
            } else {
                range -= bound2;
                code -= bound2;
                self.probs[prob_idx_isrep] = (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;

                let p_g0 = IS_REP_G0 + state as usize;
                ttt = self.probs[p_g0] as u32;
                normalize!();
                let b = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                if code < b {
                    range = b;
                    self.probs[p_g0] = (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;
                    let p_long = IS_REP0_LONG + combined_ps;
                    let t = self.probs[p_long] as u32;
                    normalize!();
                    let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                    if code < bnd {
                        range = bnd;
                        self.probs[p_long] =
                            (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
                        // Single-byte rep0
                        let src_pos = if dic_pos < rep0 as usize {
                            dic_pos + dic_buf_size - rep0 as usize
                        } else {
                            dic_pos - rep0 as usize
                        };
                        self.dic[dic_pos] = self.dic[src_pos];
                        dic_pos += 1;
                        processed_pos += 1;
                        state = if state < K_NUM_LIT_STATES { 9 } else { 11 };
                        if dic_pos >= limit || p >= buf_limit {
                            break;
                        }
                        continue;
                    }
                    range -= bnd;
                    code -= bnd;
                    self.probs[p_long] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
                } else {
                    range -= b;
                    code -= b;
                    self.probs[p_g0] = (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;

                    // Choose between rep1, rep2, rep3.
                    let p_g1 = IS_REP_G1 + state as usize;
                    ttt = self.probs[p_g1] as u32;
                    normalize!();
                    let bb = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                    let distance;
                    if code < bb {
                        range = bb;
                        self.probs[p_g1] =
                            (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;
                        distance = rep1;
                    } else {
                        range -= bb;
                        code -= bb;
                        self.probs[p_g1] = (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;
                        let p_g2 = IS_REP_G2 + state as usize;
                        ttt = self.probs[p_g2] as u32;
                        normalize!();
                        let bbb = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                        if code < bbb {
                            range = bbb;
                            self.probs[p_g2] =
                                (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;
                            distance = rep2;
                        } else {
                            range -= bbb;
                            code -= bbb;
                            self.probs[p_g2] = (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;
                            distance = rep3;
                            rep3 = rep2;
                        }
                        rep2 = rep1;
                    }
                    rep1 = rep0;
                    rep0 = distance;
                }
                state = if state < K_NUM_LIT_STATES { 8 } else { 11 };
                prob_len_base = REP_LEN_CODER;
            }

            // Length decode
            let mut local_len: u32 = 0;
            let p_choice = prob_len_base + LEN_CHOICE;
            ttt = self.probs[p_choice] as u32;
            normalize!();
            let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
            if code < bnd {
                range = bnd;
                self.probs[p_choice] =
                    (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;
                let probs_low = prob_len_base + LEN_LOW + pos_state;
                local_len = decode_tree_bits(
                    &mut self.probs,
                    probs_low,
                    K_LEN_NUM_LOW_BITS,
                    &mut range,
                    &mut code,
                    buf,
                    &mut p,
                );
            } else {
                range -= bnd;
                code -= bnd;
                self.probs[p_choice] = (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;
                let p_choice2 = prob_len_base + LEN_CHOICE2;
                ttt = self.probs[p_choice2] as u32;
                normalize!();
                let bnd2 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                if code < bnd2 {
                    range = bnd2;
                    self.probs[p_choice2] =
                        (ttt + ((K_BIT_MODEL_TOTAL - ttt) >> K_NUM_MOVE_BITS)) as u16;
                    let probs_mid =
                        prob_len_base + LEN_LOW + pos_state + (1 << K_LEN_NUM_LOW_BITS);
                    let n = decode_tree_bits(
                        &mut self.probs,
                        probs_mid,
                        K_LEN_NUM_LOW_BITS,
                        &mut range,
                        &mut code,
                        buf,
                        &mut p,
                    );
                    local_len = K_LEN_NUM_LOW_SYMBOLS + n;
                } else {
                    range -= bnd2;
                    code -= bnd2;
                    self.probs[p_choice2] =
                        (ttt - (ttt >> K_NUM_MOVE_BITS)) as u16;
                    let probs_high = prob_len_base + LEN_HIGH;
                    let n = decode_tree_bits(
                        &mut self.probs,
                        probs_high,
                        K_LEN_NUM_HIGH_BITS,
                        &mut range,
                        &mut code,
                        buf,
                        &mut p,
                    );
                    local_len = K_LEN_NUM_LOW_SYMBOLS * 2 + n;
                }
            }
            len = local_len;

            if state >= K_NUM_STATES {
                let probs_pos_slot = POS_SLOT
                    + ((if len < K_NUM_LEN_TO_POS_STATES {
                        len
                    } else {
                        K_NUM_LEN_TO_POS_STATES - 1
                    }) << K_NUM_POS_SLOT_BITS) as usize;
                let mut distance = decode_tree_bits(
                    &mut self.probs,
                    probs_pos_slot,
                    K_NUM_POS_SLOT_BITS,
                    &mut range,
                    &mut code,
                    buf,
                    &mut p,
                );

                if distance >= K_START_POS_MODEL_INDEX {
                    let pos_slot = distance;
                    let mut num_direct_bits = (pos_slot >> 1) - 1;
                    distance = 2 | (pos_slot & 1);
                    if pos_slot < K_END_POS_MODEL_INDEX {
                        distance <<= num_direct_bits;
                        let prob_off = SPEC_POS;
                        let mut m: u32 = 1;
                        distance += 1;
                        while num_direct_bits != 0 {
                            let p_idx = prob_off + distance as usize;
                            let t = self.probs[p_idx] as u32;
                            normalize!();
                            let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                            if code < bnd {
                                range = bnd;
                                self.probs[p_idx] =
                                    (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
                                distance += m;
                                m += m;
                            } else {
                                range -= bnd;
                                code -= bnd;
                                self.probs[p_idx] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
                                m += m;
                                distance += m;
                            }
                            num_direct_bits -= 1;
                        }
                        distance -= m;
                    } else {
                        num_direct_bits -= K_NUM_ALIGN_BITS;
                        loop {
                            normalize!();
                            range >>= 1;
                            code = code.wrapping_sub(range);
                            // sign-bit of (code - range), all-ones if borrow
                            let t = 0u32.wrapping_sub(code >> 31);
                            distance = (distance << 1) + t.wrapping_add(1);
                            code = code.wrapping_add(range & t);
                            num_direct_bits -= 1;
                            if num_direct_bits == 0 {
                                break;
                            }
                        }
                        let prob_off = ALIGN_OFF;
                        distance <<= K_NUM_ALIGN_BITS;
                        let mut i: u32 = 1;
                        // REV_BIT_CONST(prob,i,1); REV_BIT_CONST(prob,i,2); REV_BIT_CONST(prob,i,4);
                        // REV_BIT_LAST(prob,i,8)
                        let do_const = |dec: &mut Decoder,
                                             prob_off: usize,
                                             m_const: u32,
                                             i: &mut u32,
                                             range: &mut u32,
                                             code: &mut u32,
                                             buf: &[u8],
                                             p: &mut usize| {
                            let p_idx = prob_off + *i as usize;
                            let t = dec.probs[p_idx] as u32;
                            if *range < K_TOP_VALUE {
                                *range <<= 8;
                                *code = (*code << 8) | (buf[*p] as u32);
                                *p += 1;
                            }
                            let bnd = (*range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                            if *code < bnd {
                                *range = bnd;
                                dec.probs[p_idx] =
                                    (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
                                *i += m_const;
                            } else {
                                *range -= bnd;
                                *code -= bnd;
                                dec.probs[p_idx] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
                                *i += m_const * 2;
                            }
                        };
                        do_const(self, prob_off, 1, &mut i, &mut range, &mut code, buf, &mut p);
                        do_const(self, prob_off, 2, &mut i, &mut range, &mut code, buf, &mut p);
                        do_const(self, prob_off, 4, &mut i, &mut range, &mut code, buf, &mut p);
                        // REV_BIT_LAST: bit-1 path subtracts m, bit-0 path no-op
                        let p_idx = prob_off + i as usize;
                        let t = self.probs[p_idx] as u32;
                        if range < K_TOP_VALUE {
                            range <<= 8;
                            code = (code << 8) | read_byte!();
                        }
                        let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                        if code < bnd {
                            range = bnd;
                            self.probs[p_idx] =
                                (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
                            // bit==0 → REV_BIT_LAST: i -= m  (m=8)
                            i = i.wrapping_sub(8);
                        } else {
                            range -= bnd;
                            code -= bnd;
                            self.probs[p_idx] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
                            // bit==1 → no change
                        }
                        distance |= i;
                        if distance == 0xFFFF_FFFF {
                            len = K_MATCH_SPEC_LEN_START;
                            state -= K_NUM_STATES;
                            break;
                        }
                    }
                }
                rep3 = rep2;
                rep2 = rep1;
                rep1 = rep0;
                rep0 = distance + 1;
                state = if state < K_NUM_STATES + K_NUM_LIT_STATES {
                    K_NUM_LIT_STATES
                } else {
                    K_NUM_LIT_STATES + 3
                };
                let limit_dist = if check_dic_size == 0 { processed_pos } else { check_dic_size };
                if distance >= limit_dist {
                    len += K_MATCH_SPEC_LEN_ERROR_DATA + K_MATCH_MIN_LEN;
                    break;
                }
            }

            len += K_MATCH_MIN_LEN;
            // Copy `len` bytes from rep0 ago into the dictionary, capped by limit.
            let rem = limit - dic_pos;
            if rem == 0 {
                break;
            }
            let mut cur_len = if (rem as u32) < len { rem as u32 } else { len };
            let mut pos = if dic_pos < rep0 as usize {
                dic_pos + dic_buf_size - rep0 as usize
            } else {
                dic_pos - rep0 as usize
            };
            processed_pos += cur_len;
            len -= cur_len;
            if (cur_len as usize) <= dic_buf_size - pos {
                // C does a byte-by-byte forward copy where source may overlap
                // destination (rep-1 run-length).  We must mirror that.
                let n = cur_len as usize;
                for i in 0..n {
                    self.dic[dic_pos + i] = self.dic[pos + i];
                }
                dic_pos += n;
            } else {
                while cur_len != 0 {
                    self.dic[dic_pos] = self.dic[pos];
                    dic_pos += 1;
                    pos += 1;
                    if pos == dic_buf_size {
                        pos = 0;
                    }
                    cur_len -= 1;
                }
            }
            if dic_pos >= limit || p >= buf_limit {
                break;
            }
        }

        // Final normalize and store back
        if range < K_TOP_VALUE {
            range <<= 8;
            code = (code << 8) | read_byte!();
        }

        self.range = range;
        self.code = code;
        self.remain_len = len;
        self.dic_pos = dic_pos;
        self.processed_pos = processed_pos;
        self.reps[0] = rep0;
        self.reps[1] = rep1;
        self.reps[2] = rep2;
        self.reps[3] = rep3;
        self.state = state;
        *buf_pos = p;
        if len >= K_MATCH_SPEC_LEN_ERROR_DATA {
            return Err(Error::Data);
        }
        Ok(())
    }

    /// Returns `Some(dummy)` on success, `None` on `DUMMY_INPUT_EOF`.  The
    /// "buf_out_idx" is updated to the position past the last consumed byte.
    fn try_dummy(&self, buf: &[u8], buf_off: usize, buf_out_idx: &mut usize) -> Option<Dummy> {
        let mut range = self.range;
        let mut code = self.code;
        let buf_limit = *buf_out_idx;
        let mut p = buf_off;
        let mut state = self.state;
        let res: Dummy;

        macro_rules! normalize_check {
            () => {
                if range < K_TOP_VALUE {
                    if p >= buf_limit {
                        return None;
                    }
                    range <<= 8;
                    code = (code << 8) | (buf[p] as u32);
                    p += 1;
                }
            };
        }

        let pos_state = ((self.processed_pos & ((1u32 << self.prop.pb) - 1)) << 4) as usize;
        let combined_ps = pos_state + state as usize;
        let prob_idx = IS_MATCH + combined_ps;
        let mut ttt = self.probs[prob_idx] as u32;
        normalize_check!();
        let bound = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
        if code < bound {
            range = bound;
            // Literal
            let mut prob_off = LITERAL;
            if self.check_dic_size != 0 || self.processed_pos != 0 {
                let lp_mask: u32 = (1u32 << self.prop.lp) - 1;
                let prev = if self.dic_pos == 0 {
                    self.dic[self.dic_buf_size() - 1]
                } else {
                    self.dic[self.dic_pos - 1]
                } as u32;
                prob_off += LZMA_LIT_SIZE
                    * (((self.processed_pos & lp_mask) << self.prop.lc as u32)
                        + (prev >> (8 - self.prop.lc as u32))) as usize;
            }
            if state < K_NUM_LIT_STATES {
                let mut symbol: u32 = 1;
                while symbol < 0x100 {
                    let p_idx = prob_off + symbol as usize;
                    let t = self.probs[p_idx] as u32;
                    normalize_check!();
                    let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                    if code < bnd {
                        range = bnd;
                        symbol = symbol + symbol;
                    } else {
                        range -= bnd;
                        code -= bnd;
                        symbol = symbol + symbol + 1;
                    }
                }
            } else {
                let mut match_byte = (if self.dic_pos < self.reps[0] as usize {
                    self.dic[self.dic_pos + self.dic_buf_size() - self.reps[0] as usize]
                } else {
                    self.dic[self.dic_pos - self.reps[0] as usize]
                }) as u32;
                let mut offs: u32 = 0x100;
                let mut symbol: u32 = 1;
                while symbol < 0x100 {
                    match_byte += match_byte;
                    let bit = offs;
                    offs &= match_byte;
                    let p_idx = prob_off + (offs + bit + symbol) as usize;
                    let t = self.probs[p_idx] as u32;
                    normalize_check!();
                    let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                    if code < bnd {
                        range = bnd;
                        symbol = symbol + symbol;
                        offs ^= bit;
                    } else {
                        range -= bnd;
                        code -= bnd;
                        symbol = symbol + symbol + 1;
                    }
                }
            }
            res = Dummy::Lit;
        } else {
            range -= bound;
            code -= bound;
            let p_isrep = IS_REP + state as usize;
            ttt = self.probs[p_isrep] as u32;
            normalize_check!();
            let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
            let prob_len_base;
            let mut got_rep0_short = false;
            if code < bnd {
                range = bnd;
                state = 0;
                prob_len_base = LEN_CODER;
                res = Dummy::Match;
            } else {
                range -= bnd;
                code -= bnd;
                let p_g0 = IS_REP_G0 + state as usize;
                ttt = self.probs[p_g0] as u32;
                normalize_check!();
                let bnd2 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                if code < bnd2 {
                    range = bnd2;
                    let p_long = IS_REP0_LONG + combined_ps;
                    let t = self.probs[p_long] as u32;
                    normalize_check!();
                    let bnd3 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                    if code < bnd3 {
                        range = bnd3;
                        got_rep0_short = true;
                    } else {
                        range -= bnd3;
                        code -= bnd3;
                    }
                } else {
                    range -= bnd2;
                    code -= bnd2;
                    let p_g1 = IS_REP_G1 + state as usize;
                    ttt = self.probs[p_g1] as u32;
                    normalize_check!();
                    let bnd_g1 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                    if code < bnd_g1 {
                        range = bnd_g1;
                    } else {
                        range -= bnd_g1;
                        code -= bnd_g1;
                        let p_g2 = IS_REP_G2 + state as usize;
                        ttt = self.probs[p_g2] as u32;
                        normalize_check!();
                        let bnd_g2 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                        if code < bnd_g2 {
                            range = bnd_g2;
                        } else {
                            range -= bnd_g2;
                            code -= bnd_g2;
                        }
                    }
                }
                if got_rep0_short {
                    *buf_out_idx = p;
                    return Some(Dummy::Rep);
                }
                state = K_NUM_STATES;
                prob_len_base = REP_LEN_CODER;
                res = Dummy::Rep;
            }

            // Length
            let p_choice = prob_len_base + LEN_CHOICE;
            ttt = self.probs[p_choice] as u32;
            normalize_check!();
            let bnd_lc = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
            let limit_bits;
            let probs_len_base;
            let offset;
            if code < bnd_lc {
                range = bnd_lc;
                probs_len_base = prob_len_base + LEN_LOW + pos_state;
                offset = 0u32;
                limit_bits = K_LEN_NUM_LOW_BITS;
            } else {
                range -= bnd_lc;
                code -= bnd_lc;
                let p_choice2 = prob_len_base + LEN_CHOICE2;
                ttt = self.probs[p_choice2] as u32;
                normalize_check!();
                let bnd_lc2 = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * ttt;
                if code < bnd_lc2 {
                    range = bnd_lc2;
                    probs_len_base =
                        prob_len_base + LEN_LOW + pos_state + (1 << K_LEN_NUM_LOW_BITS);
                    offset = K_LEN_NUM_LOW_SYMBOLS;
                    limit_bits = K_LEN_NUM_LOW_BITS;
                } else {
                    range -= bnd_lc2;
                    code -= bnd_lc2;
                    probs_len_base = prob_len_base + LEN_HIGH;
                    offset = K_LEN_NUM_LOW_SYMBOLS * 2;
                    limit_bits = K_LEN_NUM_HIGH_BITS;
                }
            }
            // Decode tree
            let mut local_len: u32 = 1;
            for _ in 0..limit_bits {
                let p_idx = probs_len_base + local_len as usize;
                let t = self.probs[p_idx] as u32;
                normalize_check!();
                let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                if code < bnd {
                    range = bnd;
                    local_len = local_len + local_len;
                } else {
                    range -= bnd;
                    code -= bnd;
                    local_len = local_len + local_len + 1;
                }
            }
            let len = local_len - (1 << limit_bits) + offset;

            if state < 4 {
                let probs_pos_slot = POS_SLOT
                    + ((if len < K_NUM_LEN_TO_POS_STATES - 1 {
                        len
                    } else {
                        K_NUM_LEN_TO_POS_STATES - 1
                    }) << K_NUM_POS_SLOT_BITS) as usize;
                let mut pos_slot: u32 = 1;
                for _ in 0..K_NUM_POS_SLOT_BITS {
                    let p_idx = probs_pos_slot + pos_slot as usize;
                    let t = self.probs[p_idx] as u32;
                    normalize_check!();
                    let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                    if code < bnd {
                        range = bnd;
                        pos_slot = pos_slot + pos_slot;
                    } else {
                        range -= bnd;
                        code -= bnd;
                        pos_slot = pos_slot + pos_slot + 1;
                    }
                }
                pos_slot -= 1u32 << K_NUM_POS_SLOT_BITS;
                if pos_slot >= K_START_POS_MODEL_INDEX {
                    let mut num_direct_bits = (pos_slot >> 1) - 1;
                    let prob_off;
                    if pos_slot < K_END_POS_MODEL_INDEX {
                        prob_off = SPEC_POS + ((2 | (pos_slot & 1)) << num_direct_bits) as usize;
                    } else {
                        num_direct_bits -= K_NUM_ALIGN_BITS;
                        loop {
                            normalize_check!();
                            range >>= 1;
                            let cmp = (code.wrapping_sub(range) >> 31).wrapping_sub(1);
                            code = code.wrapping_sub(range & cmp);
                            num_direct_bits -= 1;
                            if num_direct_bits == 0 {
                                break;
                            }
                        }
                        prob_off = ALIGN_OFF;
                        num_direct_bits = K_NUM_ALIGN_BITS;
                    }
                    let mut i: u32 = 1;
                    let mut m: u32 = 1;
                    for _ in 0..num_direct_bits {
                        let p_idx = prob_off + i as usize;
                        let t = self.probs[p_idx] as u32;
                        normalize_check!();
                        let bnd = (range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
                        if code < bnd {
                            range = bnd;
                            i += m;
                            m += m;
                        } else {
                            range -= bnd;
                            code -= bnd;
                            m += m;
                            i += m;
                        }
                    }
                }
            }
        }

        if range < K_TOP_VALUE {
            if p >= buf_limit {
                return None;
            }
            // The actual normalize would consume one byte; we don't bother
            // computing range/code here since they're unused after.
            p += 1;
        }
        let _ = (state, ttt);
        *buf_out_idx = p;
        Some(res)
    }
}

#[derive(Copy, Clone, Debug)]
enum Dummy {
    Lit,
    Match,
    Rep,
}

#[inline(always)]
fn decode_tree_bits(
    probs: &mut [u16],
    base: usize,
    num_bits: u32,
    range: &mut u32,
    code: &mut u32,
    buf: &[u8],
    p: &mut usize,
) -> u32 {
    let mut i: u32 = 1;
    for _ in 0..num_bits {
        if *range < K_TOP_VALUE {
            *range <<= 8;
            *code = (*code << 8) | (buf[*p] as u32);
            *p += 1;
        }
        let idx = base + i as usize;
        let t = probs[idx] as u32;
        let bnd = (*range >> K_NUM_BIT_MODEL_TOTAL_BITS) * t;
        if *code < bnd {
            *range = bnd;
            probs[idx] = (t + ((K_BIT_MODEL_TOTAL - t) >> K_NUM_MOVE_BITS)) as u16;
            i = i + i;
        } else {
            *range -= bnd;
            *code -= bnd;
            probs[idx] = (t - (t >> K_NUM_MOVE_BITS)) as u16;
            i = i + i + 1;
        }
    }
    i - (1u32 << num_bits)
}

fn round_up_dic_size(dict_size: u32) -> usize {
    let dict_size = dict_size as usize;
    let mask: usize = if dict_size >= (1 << 30) {
        (1 << 22) - 1
    } else if dict_size >= (1 << 22) {
        (1 << 20) - 1
    } else {
        (1 << 12) - 1
    };
    let v = (dict_size + mask) & !mask;
    v.max(dict_size)
}

// ====================================================================
// One-shot helpers
// ====================================================================

/// Decode a complete LZMA stream.  `props` is the 5-byte LZMA properties
/// header, `src` is the compressed payload.  Returns the uncompressed bytes.
pub fn decode_one_shot(props: &[u8], src: &[u8]) -> Result<Vec<u8>, Error> {
    let prop = Properties::parse(props)?;
    let mut dec = Decoder::new(prop);
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    let mut src_off = 0;
    loop {
        let mut consumed_in = 0usize;
        let mut consumed_out = 0usize;
        let status = dec.decode_to_buf(
            &mut buf,
            &src[src_off..],
            &mut consumed_in,
            &mut consumed_out,
            FinishMode::Any,
        )?;
        src_off += consumed_in;
        out.extend_from_slice(&buf[..consumed_out]);
        match status {
            Status::FinishedWithMark | Status::MaybeFinishedWithoutMark => break,
            Status::NeedsMoreInput => return Err(Error::InputEof),
            Status::NotFinished | Status::NotSpecified => {
                if consumed_in == 0 && consumed_out == 0 {
                    return Err(Error::Data);
                }
            }
        }
    }
    Ok(out)
}

/// Decode a "known size" LZMA stream — the first 5 bytes are LZMA properties,
/// the next 8 bytes are the little-endian uncompressed size (or 0xFFFF…FFFF
/// for unknown), then compressed data.  This is the format produced by the C
/// `7lzma e` reference command.
pub fn decode_lzma_alone(stream: &[u8]) -> Result<Vec<u8>, Error> {
    if stream.len() < 13 {
        return Err(Error::InputEof);
    }
    let props = &stream[..5];
    let unpacked: u64 = u64::from_le_bytes(stream[5..13].try_into().unwrap());
    let body = &stream[13..];
    let prop = Properties::parse(props)?;
    let mut dec = Decoder::new(prop);
    if unpacked == u64::MAX {
        // No size field — decode until end mark.
        decode_one_shot(props, body)
    } else {
        let unpacked = unpacked as usize;
        let mut out = vec![0u8; unpacked];
        let mut consumed_in = 0usize;
        let mut consumed_out = 0usize;
        let _ = dec.decode_to_buf(
            &mut out,
            body,
            &mut consumed_in,
            &mut consumed_out,
            FinishMode::End,
        )?;
        if consumed_out != unpacked {
            return Err(Error::Data);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_props_known() {
        // Default 7-Zip LZMA: lc=3, lp=0, pb=2, dic=0x800000
        let p = Properties::parse(&[0x5d, 0x00, 0x00, 0x80, 0x00]).unwrap();
        assert_eq!(p.lc, 3);
        assert_eq!(p.lp, 0);
        assert_eq!(p.pb, 2);
        assert_eq!(p.dic_size, 0x00800000);
    }

    #[test]
    fn parse_props_invalid() {
        // Property byte >= 9*5*5 = 225
        assert!(matches!(
            Properties::parse(&[226, 0, 0, 0x80, 0]),
            Err(Error::Unsupported)
        ));
    }

    // Real round-trip tests live in the C-cross-check binary.
}
