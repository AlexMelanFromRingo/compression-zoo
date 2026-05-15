//! `contextModel2` + `Predictor` — paq8.cpp:8102-8385.
//!
//! The paq8 integration point: wires every sub-model into a single
//! per-bit prediction. `Paq8Predictor::update(y)` mirrors upstream's
//! `Predictor::update()` — byte-boundary state advance, the
//! `contextModel2` mixer pipeline, and the file-type-aware APM
//! cascade.

#![allow(dead_code)]

use super::apm::{Apm, Apm1, StateMap32};
use super::context_map::{ContextMap2, RunContextMap};
use super::dmc::DmcForest;
use super::exe_model::ExeModel;
use super::file_models::{
    AudioModel, ImgModel, JpegModel,
};
use super::match_model::MatchModel;
use super::mixer::Mixer;
use super::small_models::{
    DistanceModel, IndirectModel, LinearPredictionModel, NestModel,
    PicModel, RecordModel, RecordModel1, SparseModel, SparseModel1, WordModel,
};
use super::sparse_match_model::SparseMatchModel;
use super::state::Paq8State;
use super::stats::Filetype;
use super::substrate::{combine64, finalize64, hash3, hash4, ilog2, mem};
use super::text_model::TextModel;
use super::xml_model::XmlModel;

const NUM_INPUTS: usize = 1552;
const NUM_SETS:   usize = 28;

/// `WRT_mpw` — paq8.cpp:3869.
const WRT_MPW: [u32; 16] =
    [4, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
/// `WRT_mtt` — paq8.cpp:3870.
const WRT_MTT: [u32; 16] =
    [0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7];

/// `AsciiGroupC0` — paq8.cpp:3043-3051. Quantised partial-byte
/// ASCII group, indexed by `(1<<bpos)-2+(c0 & ((1<<bpos)-1))`.
const ASCII_GROUP_C0: [u8; 254] = [
    0, 10,
    0, 1, 10, 10,
    0, 4, 2, 3, 10, 10, 10, 10,
    0, 0, 5, 4, 2, 2, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10,
    0, 0, 0, 0, 5, 5, 9, 4, 2, 2, 2, 2, 3, 3, 3, 3,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    0, 0, 0, 0, 0, 0, 0, 0, 5, 8, 8, 5, 9, 9, 6, 5,
    2, 2, 2, 2, 2, 2, 2, 8, 3, 3, 3, 3, 3, 3, 3, 8,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    7, 8, 8, 8, 8, 8, 5, 5, 9, 9, 9, 9, 9, 7, 8, 5,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 8,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
];

/// Top-level paq8 predictor — owns every sub-model + the mixer.
pub struct Paq8Predictor {
    pub state: Paq8State,

    // contextModel2 components.
    cm: ContextMap2,
    text_model: TextModel,
    match_model: MatchModel,
    sparse_match_model: SparseMatchModel,
    dmc_forest: DmcForest,
    rcm7: RunContextMap,
    rcm9: RunContextMap,
    rcm10: RunContextMap,
    state_maps: [StateMap32; 2],
    m: Mixer,
    cxt: [u64; 16],
    ft2: Filetype,
    filetype: Filetype,
    size: i32,
    info: i32,

    // small models.
    sparse_model: SparseModel,
    sparse_model1: SparseModel1,
    distance_model: DistanceModel,
    pic_model: PicModel,
    record_model: RecordModel,
    record_model1: RecordModel1,
    word_model: WordModel,
    nest_model: NestModel,
    indirect_model: IndirectModel,
    xml_model: XmlModel,
    exe_model: ExeModel,
    linear_prediction_model: LinearPredictionModel,

    // file-type detection models (return false on text).
    jpeg_model: JpegModel,
    img_model: ImgModel,
    audio_model: AudioModel,

    // APM cascade.
    text_apms:    [Apm; 4],
    text_apm1s:   [Apm1; 3],
    generic_apm1s: [Apm1; 7],

    x5: u32,
}

impl Paq8Predictor {
    pub fn new(level: u32) -> Self {
        let state = Paq8State::new(level);
        let dt = state.dt;
        let m = mem(level);
        let stretch = &state.stretch;
        let p = Self {
            cm: ContextMap2::new(m * 16, 10, dt),
            text_model: TextModel::new(m * 16, dt),
            match_model: MatchModel::new(m * 2, dt),
            sparse_match_model: SparseMatchModel::new(m / 2),
            dmc_forest: DmcForest::new(level, dt),
            rcm7:  RunContextMap::new(m as usize),
            rcm9:  RunContextMap::new(m as usize),
            rcm10: RunContextMap::new(m as usize),
            state_maps: [
                StateMap32::new(256, dt),
                StateMap32::new(256 * 256, dt),
            ],
            m: Mixer::new(NUM_INPUTS, NUM_SETS, 32),
            cxt: [0; 16],
            ft2: Filetype::Default,
            filetype: Filetype::Default,
            size: 0,
            info: 0,
            sparse_model:  SparseModel::new(m, dt),
            sparse_model1: SparseModel1::new(m, dt),
            distance_model: DistanceModel::new(m, dt),
            pic_model: PicModel::new(dt),
            record_model: RecordModel::new(m, dt),
            record_model1: RecordModel1::new(dt),
            word_model: WordModel::new(m, dt),
            nest_model: NestModel::new(m, dt),
            indirect_model: IndirectModel::new(m, dt),
            xml_model: XmlModel::new(m, dt),
            exe_model: ExeModel::new(m, dt),
            linear_prediction_model: LinearPredictionModel::new(),
            jpeg_model: JpegModel::new(),
            img_model: ImgModel::new(),
            audio_model: AudioModel::new(),
            text_apms: [
                Apm::new(0x10000, dt), Apm::new(0x10000, dt),
                Apm::new(0x10000, dt), Apm::new(0x10000, dt),
            ],
            text_apm1s: [
                Apm1::new(0x10000, stretch), Apm1::new(0x10000, stretch),
                Apm1::new(0x10000, stretch),
            ],
            generic_apm1s: [
                Apm1::new(0x2000, stretch), Apm1::new(0x10000, stretch),
                Apm1::new(0x10000, stretch), Apm1::new(0x10000, stretch),
                Apm1::new(0x10000, stretch), Apm1::new(0x10000, stretch),
                Apm1::new(0x10000, stretch),
            ],
            x5: 0,
            state,
        };
        p
    }

    /// `contextModel2` — paq8.cpp:8102-8207. Returns the stage-1
    /// mixed prediction `pr0` in `[0, 4095]`.
    fn context_model2(&mut self) -> i32 {
        let s = &mut self.state;
        let bpos = s.bpos;
        let y = s.y;
        let c0 = s.c0;
        let c4 = s.c4;

        if bpos == 0 {
            self.size -= 1;
            s.blpos += 1;
            if self.size == -1 {
                self.info = 0;
                self.ft2 = filetype_from_byte(s.buf.at(1));
            }
            if self.size == -5 && !has_info(self.ft2) {
                self.size = ((s.buf.at(4) as i32) << 24)
                    | ((s.buf.at(3) as i32) << 16)
                    | ((s.buf.at(2) as i32) << 8)
                    | (s.buf.at(1) as i32);
                s.blpos = 0;
            }
            if self.size == -9 {
                self.size = ((s.buf.at(8) as i32) << 24)
                    | ((s.buf.at(7) as i32) << 16)
                    | ((s.buf.at(6) as i32) << 8)
                    | (s.buf.at(5) as i32);
                self.info = ((s.buf.at(4) as i32) << 24)
                    | ((s.buf.at(3) as i32) << 16)
                    | ((s.buf.at(2) as i32) << 8)
                    | (s.buf.at(1) as i32);
                s.blpos = 0;
                if self.ft2 == Filetype::Text && self.info != 0 {
                    self.size = self.info - 8;
                }
            }
            if s.blpos == 0 { self.filetype = self.ft2; }
            if self.size == 0 { self.filetype = Filetype::Default; }
            s.stats.filetype = self.filetype;
        }

        self.m.update(y);
        self.m.add(64);

        if bpos == 0 {
            let b = (c4 & 0xFF) as u8;
            self.cxt[15] = if b.is_ascii_alphabetic() {
                combine64(self.cxt[15], b.to_ascii_lowercase() as u64)
            } else {
                0
            };
            self.cm.set(self.cxt[15]);
            for i in (1..=14).rev() {
                self.cxt[i] = combine64(self.cxt[i - 1], b as u64);
            }
            for i in 0..7 {
                self.cm.set(self.cxt[i]);
            }
            self.rcm7.set(self.cxt[7], s.buf.at(1));
            self.cm.set(self.cxt[8]);
            self.rcm9.set(self.cxt[10], s.buf.at(1));
            self.rcm10.set(self.cxt[12], s.buf.at(1));
            self.cm.set(self.cxt[14]);
        }

        let sm0 = self.state_maps[0].p(c0, 1023, y);
        self.m.add(((self.state.stretch.get(sm0) + 1) >> 1) as i16);
        let sm1_cx = c0 | ((self.state.buf.at(1) as u32) << 8);
        let sm1 = self.state_maps[1].p(sm1_cx, 1023, y);
        self.m.add(((self.state.stretch.get(sm1) + 1) >> 1) as i16);

        let order = {
            let st = &self.state;
            self.cm.mix(&mut self.m, st.y, st.bpos, &st.ilog,
                        &st.squash, &st.stretch)
        };
        {
            let st = &self.state;
            self.rcm7.mix(&mut self.m, st.c0, st.bpos, &st.ilog);
            self.rcm9.mix(&mut self.m, st.c0, st.bpos, &st.ilog);
            self.rcm10.mix(&mut self.m, st.c0, st.bpos, &st.ilog);
        }

        let match_len = {
            let st_ptr = &mut self.state as *mut Paq8State;
            // SAFETY-free split: match_model needs &Buf + &mut stats;
            // both live in `state`. We pass the fields explicitly.
            let _ = st_ptr;
            self.predict_match()
        };
        let ismatch = self.state.ilog.get((match_len & 0xffff) as u16) as i32;

        // File-type dispatch — image / jpeg / audio. For text the
        // image models are never reached and jpeg/img/audio detect
        // nothing, so contextModel2 falls through.
        match self.filetype {
            Filetype::Image1 | Filetype::Image4 | Filetype::Image8
            | Filetype::Image8Gray | Filetype::Image24 | Filetype::Image32 => {
                // Image filetypes return early in upstream; for the
                // text-focused port these branches are unreachable
                // (the preprocessor never emits them yet).
                return self.m.p(self.state.y, &self.state.squash,
                                &self.state.stretch);
            }
            _ => {}
        }
        let jpeg_hit = self.filetype != Filetype::Exe
            && self.jpeg_model.mix(&mut self.state, &mut self.m);
        let img_hit = self.size > 0
            && self.img_model.mix(&mut self.state, &mut self.m);
        let audio_hit = self.audio_model.mix(&mut self.state, &mut self.m);
        if jpeg_hit || img_hit || audio_hit {
            return self.m.p(self.state.y, &self.state.squash,
                            &self.state.stretch);
        }

        // The text / generic sub-model bank.
        self.predict_sparse_match();
        self.sparse_model.mix(&mut self.state, &mut self.m, ismatch, order);
        self.sparse_model1.mix(&mut self.state, &mut self.m, ismatch, order);
        self.distance_model.mix(&mut self.state, &mut self.m);
        self.pic_model.mix(&mut self.state, &mut self.m);
        let is_text = matches!(self.filetype,
            Filetype::Default | Filetype::Text);
        self.record_model.mix(&mut self.state, &mut self.m, is_text);
        self.record_model1.mix(&mut self.state, &mut self.m);
        self.word_model.mix(&mut self.state, &mut self.m);
        self.nest_model.mix(&mut self.state, &mut self.m);
        self.indirect_model.mix(&mut self.state, &mut self.m);
        self.dmc_forest.mix(&mut self.state, &mut self.m);
        self.xml_model.mix(&mut self.state, &mut self.m);
        self.predict_text();
        self.exe_model.mix(&mut self.state, &mut self.m, true);
        self.linear_prediction_model.mix(&mut self.state, &mut self.m);

        // Stage-1 mixer set contexts (paq8.cpp:8187-8204).
        let st = &self.state;
        let order_set = 0.max(order - 3);
        self.m.set(((order_set << 3) as u32) | bpos as u32, 64);
        let order2 = 0.max(order - 5) as u32;
        let d = c0 << (8 - bpos);
        let mut c = (d + if bpos == 1 { st.b3 / 2 } else { 0 }) & 192;
        if bpos == 0 { c = (st.words.wrapping_mul(16)) & 192; }
        let c1 = st.buf.at(1) as u32;
        self.m.set(order2 * 256 + (st.w4 & 240) + (st.b2 >> 4), 1536);
        self.m.set(order2 * 256 + (st.w4 & 3) * 64
            + ((st.words >> 1) & 63), 1536);
        self.m.set((bpos as u32) * 256 + c1, 2048);
        self.m.set(5.min(bpos as u32) * 256 + (st.tt & 63) + c, 1536);
        self.m.set(order2 * 256 + ((d | (c1 >> bpos)) & 248)
            + bpos as u32, 1536);
        self.m.set((bpos as u32) * 256
            + ((((st.words << bpos) & 255) >> bpos) | (d & 255)), 2048);
        let pr_q = (st.last_prediction / 16) as u32;
        self.m.set(pr_q, 256);
        self.m.set(c0, 256);

        self.m.p(self.state.y, &self.state.squash, &self.state.stretch)
    }

    fn predict_match(&mut self) -> u32 {
        // Borrow-split: MatchModel needs &Buf and &mut stats. Move the
        // buffer out, run, move it back.
        let buf = std::mem::take(&mut self.state.buf);
        let st = &mut self.state;
        let len = self.match_model.predict(
            &mut self.m, &buf, st.c0, st.bpos, st.y,
            &st.ilog, &st.dt, &st.squash, &st.stretch, &mut st.stats);
        self.state.buf = buf;
        len
    }

    fn predict_sparse_match(&mut self) {
        let buf = std::mem::take(&mut self.state.buf);
        let st = &mut self.state;
        self.sparse_match_model.predict(
            &mut self.m, &buf, st.c0, st.bpos, st.y,
            &st.dt, &st.squash, &st.stretch, &mut st.stats);
        self.state.buf = buf;
    }

    fn predict_text(&mut self) {
        let buf = std::mem::take(&mut self.state.buf);
        let st = &mut self.state;
        self.text_model.predict(
            &mut self.m, &buf, buf.pos, st.c0, st.bpos, st.y, st.grp0,
            &st.ilog, &st.squash, &st.stretch, &mut st.stats);
        self.state.buf = buf;
    }

    /// `Predictor::update()` — paq8.cpp:8249-8363.
    pub fn update(&mut self, y: i32) {
        self.state.y = y;
        let pr = self.state.last_prediction;
        self.state.c0 = self.state.c0 * 2 + y as u32;
        self.state.stats.misses = (self.state.stats.misses << 1)
            | (((pr >> 11) != y) as u64);

        if self.state.c0 >= 256 {
            let c0_full = self.state.c0;
            self.state.buf.push((c0_full & 0xff) as u8);
            self.state.c0 = c0_full - 256;
            let c0 = self.state.c0;
            self.state.c4 = (self.state.c4 << 8) + c0;
            let mut i = WRT_MPW[(c0 >> 4) as usize];
            self.state.w4 = self.state.w4.wrapping_mul(4).wrapping_add(i);
            if self.state.b2 == 3 { i = 2; }
            self.state.w5 = self.state.w5.wrapping_mul(4).wrapping_add(i);
            self.state.b3 = self.state.b2;
            self.state.b2 = c0;
            self.state.x4 = self.state.x4.wrapping_mul(256).wrapping_add(c0);
            self.x5 = (self.x5 << 8).wrapping_add(c0);
            if c0 == b'.' as u32 || c0 == b'!' as u32 || c0 == b'?' as u32
                || c0 == b'/' as u32 || c0 == b')' as u32
            {
                self.state.w5 = (self.state.w5 << 8) | 0x3ff;
                self.state.f4 = (self.state.f4 & 0xffff_fff0) + 2;
                self.x5 = (self.x5 << 8).wrapping_add(c0);
                self.state.x4 = self.state.x4.wrapping_mul(256)
                    .wrapping_add(c0);
                if c0 != b'!' as u32 {
                    self.state.w4 |= 12;
                    self.state.tt = (self.state.tt & 0xffff_fff8) + 1;
                    self.state.b3 = b'.' as u32;
                }
            }
            let mut c0m = c0;
            if c0m == 32 { c0m -= 1; }
            self.state.tt = self.state.tt.wrapping_mul(8)
                .wrapping_add(WRT_MTT[(c0m >> 4) as usize]);
            self.state.f4 = self.state.f4.wrapping_mul(16)
                .wrapping_add(c0 >> 4);
            self.state.c0 = 1;
            self.state.x5 = self.x5; // mirror into state if needed
        }
        self.state.bpos = (self.state.bpos + 1) & 7;
        let bpos = self.state.bpos;
        self.state.grp0 = if bpos > 0 {
            let idx = (1usize << bpos) - 2
                + (self.state.c0 as usize & ((1usize << bpos) - 1));
            ASCII_GROUP_C0[idx.min(ASCII_GROUP_C0.len() - 1)]
        } else {
            0
        };

        let pr0 = self.context_model2();
        self.state.add_prediction(pr0);

        // APM cascade (paq8.cpp:8281-8359).
        let pr_final = self.apm_cascade(pr0);
        self.state.last_prediction = pr_final;
        self.state.reset_predictions();
    }

    fn apm_cascade(&mut self, pr0: i32) -> i32 {
        let y = self.state.y;
        let c0 = self.state.c0;
        let c4 = self.state.c4;
        let bpos = self.state.bpos as u32;
        let blpos = self.state.blpos;
        let st = &self.state.stats;
        let misses = st.misses;
        let match_len = st.r#match.length;
        let expected = st.r#match.expected_byte;
        let text_mask = st.text.mask;
        let text_first = st.text.first_letter;
        let squash = &self.state.squash;
        let stretch = &self.state.stretch;

        match self.filetype {
            Filetype::Text => {
                let limit = (0x3FF >> ((blpos < 0xFFF) as u32 * 2)) as u32;
                let mut pr = self.text_apms[0].p(pr0,
                    (c0 << 8) | ((text_mask as u32) & 0xF)
                        | (((misses as u32) & 0xF) << 4),
                    limit, y, stretch);
                let pr1 = self.text_apms[1].p(pr0, finalize64(hash4(
                    bpos as u64, (misses & 3) as u64, (c4 & 0xffff) as u64,
                    (text_mask >> 4) as u64), 16), limit, y, stretch);
                let pr2 = self.text_apms[2].p(pr0, finalize64(hash3(
                    c0 as u64, expected as u64,
                    3.min(ilog2(match_len + 1)) as u64), 16),
                    limit, y, stretch);
                let pr3 = self.text_apms[3].p(pr0, finalize64(hash3(
                    c0 as u64, (c4 & 0xffff) as u64, text_first as u64), 16),
                    limit, y, stretch);
                let pr0b = (pr0 + pr1 + pr2 + pr3 + 2) >> 2;
                let pr1b = self.text_apm1s[0].p(pr0b, finalize64(hash3(
                    expected as u64, 3.min(ilog2(match_len + 1)) as u64,
                    (c4 & 0xff) as u64), 16), 7, y, stretch);
                let pr2b = self.text_apm1s[1].p(pr, finalize64(
                    super::substrate::hash2(c0 as u64,
                        (c4 & 0x00ffffff) as u64), 16), 6, y, stretch);
                let pr3b = self.text_apm1s[2].p(pr, finalize64(
                    super::substrate::hash2(c0 as u64,
                        (c4 & 0xffffff00) as u64), 16), 6, y, stretch);
                pr = (pr + pr1b + pr2b + pr3b + 2) >> 2;
                pr = (pr + pr0b + 1) >> 1;
                let _ = squash;
                pr
            }
            _ => {
                // Generic APM1 cascade (paq8.cpp:8341-8358).
                let c1 = self.state.buf.at(1) as u32;
                let mut pr = self.generic_apm1s[0].p(pr0,
                    (3u32.min(ilog2(match_len + 1)) << 11)
                        | (c0 << 3) | ((misses as u32) & 0x7),
                    7, y, stretch);
                let ctx1 = c0 ^ ((c1 << 8) & 0xffff);
                let ctx2 = c0 ^ finalize64(
                    super::substrate::hash1((c4 & 0xffff) as u64), 16);
                let ctx3 = c0 ^ finalize64(
                    super::substrate::hash1((c4 & 0xffffff) as u64), 16);
                let pr1 = self.generic_apm1s[1].p(pr0, ctx1 & 0xffff, 7, y, stretch);
                let pr2 = self.generic_apm1s[2].p(pr0, ctx2 & 0xffff, 7, y, stretch);
                let pr3 = self.generic_apm1s[3].p(pr0, ctx3 & 0xffff, 7, y, stretch);
                let pr0b = (pr0 + pr1 + pr2 + pr3 + 2) >> 2;
                let pr1b = self.generic_apm1s[4].p(pr,
                    ((expected as u32) << 8) | c1, 7, y, stretch);
                let pr2b = self.generic_apm1s[5].p(pr, ctx2 & 0xffff, 7, y, stretch);
                let pr3b = self.generic_apm1s[6].p(pr, ctx3 & 0xffff, 7, y, stretch);
                pr = (pr + pr1b + pr2b + pr3b + 2) >> 2;
                pr = (pr + pr0b + 1) >> 1;
                let _ = squash;
                pr
            }
        }
    }

    /// Current bit-1 probability as a float in `[0, 1]`.
    pub fn predict(&self) -> f32 {
        (self.state.last_prediction as f32 + 0.5) / 4096.0
    }
}

fn filetype_from_byte(b: u8) -> Filetype {
    match b {
        0 => Filetype::Default,
        1 => Filetype::Hdr,
        2 => Filetype::Jpeg,
        3 => Filetype::Exe,
        4 => Filetype::Text,
        5 => Filetype::Image1,
        6 => Filetype::Image4,
        7 => Filetype::Image8,
        8 => Filetype::Image8Gray,
        9 => Filetype::Image24,
        10 => Filetype::Image32,
        11 => Filetype::Audio,
        _ => Filetype::Default,
    }
}

fn has_info(ft: Filetype) -> bool {
    matches!(ft, Filetype::Text | Filetype::Image1 | Filetype::Image4
        | Filetype::Image8 | Filetype::Image8Gray
        | Filetype::Image24 | Filetype::Image32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Paq8Predictor at production level allocates several GB"]
    fn paq8_predictor_runs_through_text_at_production_level() {
        let mut p = Paq8Predictor::new(11);
        for &byte in b"Hello, paq8 contextModel2 integration!" {
            for bp in (0..8).rev() {
                p.update(((byte >> bp) & 1) as i32);
                let pr = p.predict();
                assert!(pr >= 0.0 && pr <= 1.0);
            }
        }
    }

    #[test]
    fn paq8_predictor_runs_through_text_at_small_level() {
        // level 0 keeps allocations to a few hundred MiB at most.
        let mut p = Paq8Predictor::new(0);
        for &byte in b"The quick brown fox. 12345\nAnother line!" {
            for bp in (0..8).rev() {
                p.update(((byte >> bp) & 1) as i32);
            }
            let pr = p.predict();
            assert!(pr >= 0.0 && pr <= 1.0, "predict out of range: {}", pr);
        }
    }
}
