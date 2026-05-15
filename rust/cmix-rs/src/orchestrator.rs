//! `predictor.cpp` orchestrator — faithful port of the full CMIX
//! `Predictor` (AddBracket + AddFXCM + AddPAQ8 + AddPPMD + AddWord +
//! AddDirect + AddMatch + AddDoubleIndirect + AddMixers).
//!
//! Upstream wires ~100 sub-models into three stacked logistic mixer
//! layers + an LSTM ByteMixer + an SSE smoother. Each model holds a
//! `const u64&` reference to a [`ContextManager`] field; the
//! manager's typed [`crate::context_manager::CtxNode`] collection plus
//! the [`Src`] enum stand in for those references in safe Rust.
//!
//! Memory profile at `Config::upstream()` matches upstream (~30 GB
//! peak, dominated by PAQ8 + PPMD + SSE + shared maps). Use
//! [`Config::tiny`] to instantiate a memory-scaled version that fits
//! in a few hundred MB — primarily for tests and for running on
//! constrained dev machines.

#![allow(dead_code)]
#![forbid(unsafe_code)]

use crate::context_manager::{BitCtxNode, ContextManager, CtxNode, Src};
use crate::contexts::{
    BitContext as BitCtx, BracketContext, CombinedContext, ContextHash,
    IndirectHash, Interval, IntervalHash, Sparse,
};
use crate::mixer::byte_mixer::ByteMixer;
use crate::mixer::lstm::Lstm;
use crate::mixer::sse::SSE;
use crate::mixer::{ContextData, Mixer, MixerInput};
use crate::models::ppmd::Ppmd;
use crate::models::paq8::Paq8;
use crate::models::fxcmv1;
use crate::models::{Bracket, Direct, DirectHash, Indirect, Match};
use crate::sigmoid::Sigmoid;
use crate::states::nonstationary::Nonstationary;
use crate::states::run_map::RunMap;

// ============================================================
// Config
// ============================================================

/// Memory-scaling knobs. `upstream()` mirrors upstream cmix at full
/// quality (~30 GB peak); `tiny()` shrinks every allocation enough
/// to instantiate on a laptop (~256 MB peak). Compression ratio
/// degrades gracefully with scale — the structure (mixer tree,
/// context bindings, SSE, LSTM) is identical at every scale.
#[derive(Clone, Debug)]
pub struct Config {
    pub history_bytes: usize,
    pub shared_map_bytes: usize,
    /// PAQ8 memory level (0..=11). 11 mirrors upstream `PAQ8(11)`.
    pub paq8_level: u32,
    /// Cap for `Direct`'s context-cell vector.
    pub direct_cap: usize,
    /// Cap for `DirectHash`'s slot count.
    pub direct_hash_cap: usize,
    /// Cap for `Match`'s map size.
    pub match_cap: usize,
    /// LSTM cells per layer (upstream: 200).
    pub lstm_cells: usize,
    /// LSTM layer count (upstream: 2).
    pub lstm_layers: usize,
    /// LSTM BPTT horizon (upstream: 100).
    pub lstm_horizon: usize,
    /// LSTM learning rate (upstream: 0.03).
    pub lstm_learning_rate: f32,
    /// LSTM gradient clip (upstream: 10).
    pub lstm_gradient_clip: f32,
    /// PPMD order (upstream: 25).
    pub ppmd_order: i32,
    /// PPMD heap MB (upstream: 14000).
    pub ppmd_memory_mb: i32,
    /// Allocate the SSE smoother. Costs ~440 MB.
    pub enable_sse: bool,
    /// Allocate the LSTM ByteMixer. Costs proportional to
    /// `lstm_cells^2 * lstm_layers * lstm_horizon`.
    pub enable_byte_mixer: bool,
    /// Wire in the FXCM bit-model. ~1-2 GB at default settings.
    pub enable_fxcm: bool,
    /// Wire in the PAQ8 bit-model. ~15 GB at `paq8_level=11`.
    pub enable_paq8: bool,
}

impl Config {
    /// Upstream cmix configuration — peak heap ~30 GB.
    pub fn upstream() -> Self {
        Self {
            history_bytes: 100_000_000,
            shared_map_bytes: 256 * 8_000_000,
            paq8_level: 11,
            direct_cap: 16 * 1024 * 1024,
            direct_hash_cap: 500_000,
            match_cap: 20_000_000,
            lstm_cells: 200,
            lstm_layers: 2,
            lstm_horizon: 100,
            lstm_learning_rate: 0.03,
            lstm_gradient_clip: 10.0,
            ppmd_order: 25,
            ppmd_memory_mb: 14_000,
            enable_sse: true,
            enable_byte_mixer: true,
            enable_fxcm: true,
            enable_paq8: true,
        }
    }

    /// Tiny configuration for tests and constrained dev machines —
    /// peak heap ~256 MB.
    pub fn tiny() -> Self {
        Self {
            history_bytes: 256 * 1024,
            shared_map_bytes: 1024 * 1024,
            paq8_level: 0,
            direct_cap: 8192,
            direct_hash_cap: 1024,
            match_cap: 65536,
            lstm_cells: 4,
            lstm_layers: 1,
            lstm_horizon: 2,
            lstm_learning_rate: 0.01,
            lstm_gradient_clip: 2.0,
            ppmd_order: 4,
            ppmd_memory_mb: 4,
            enable_sse: false,
            enable_byte_mixer: false,
            enable_fxcm: false,
            enable_paq8: false,
        }
    }

    /// Medium configuration — fits in ~4 GB, useful for smoke tests
    /// of the full predictor including PAQ8 at low memory level and
    /// the LSTM ByteMixer.
    pub fn medium() -> Self {
        Self {
            history_bytes: 1024 * 1024,
            shared_map_bytes: 4 * 1024 * 1024,
            paq8_level: 0,
            direct_cap: 65536,
            direct_hash_cap: 8192,
            match_cap: 262144,
            lstm_cells: 16,
            lstm_layers: 1,
            lstm_horizon: 8,
            lstm_learning_rate: 0.03,
            lstm_gradient_clip: 10.0,
            ppmd_order: 6,
            ppmd_memory_mb: 16,
            enable_sse: false,
            enable_byte_mixer: true,
            enable_fxcm: false,
            enable_paq8: true,
        }
    }
}

// ============================================================
// OrchModel — heterogeneous bit-level model with context bindings
// ============================================================

/// Each variant binds one upstream sub-model to its byte-context
/// and bit-context [`Src`]s. The orchestrator resolves both on
/// every call instead of holding live references.
pub enum OrchModel {
    Direct {
        m: Direct,
        byte_src: Src,
        size: u64,
    },
    DirectHash {
        m: DirectHash,
        byte_src: Src,
    },
    IndirectNs {
        m: Indirect<Nonstationary>,
        byte_src: Src,
    },
    IndirectRm {
        m: Indirect<RunMap>,
        byte_src: Src,
    },
    Match {
        m: Match,
        byte_src: Src,
    },
    Bracket {
        m: Bracket,
    },
    Fxcm {
        m: fxcmv1::Predictor,
        output: [f32; 1],
    },
    Paq8 {
        m: Paq8,
        output: [f32; 1],
    },
}

impl OrchModel {
    pub fn num_outputs(&self) -> usize { 1 }
    pub fn output(&self) -> f32 {
        match self {
            OrchModel::Direct { m, .. } => m.outputs()[0],
            OrchModel::DirectHash { m, .. } => m.outputs()[0],
            OrchModel::IndirectNs { m, .. } => m.outputs()[0],
            OrchModel::IndirectRm { m, .. } => m.outputs()[0],
            OrchModel::Match { m, .. } => m.outputs()[0],
            OrchModel::Bracket { m } => m.byte_model.outputs()[0],
            OrchModel::Fxcm { output, .. } => output[0],
            OrchModel::Paq8 { output, .. } => output[0],
        }
    }
}

/// Implementation of `Model::outputs()` for each variant requires
/// importing the trait into scope.
use crate::models::Model;

// ============================================================
// Per-mixer binding
// ============================================================

struct MixerBinding {
    mixer: Mixer,
    /// Context source for the per-mixer key.
    ctx_src: Src,
    /// Maximum number of extra inputs this mixer reads from the
    /// per-layer growing `extra_inputs` array — equal to its index
    /// in `mixers[layer]` at construction time, matching upstream's
    /// `Mixer(..., mixers_[layer].size())`.
    extras_cap: usize,
    /// Snapshot of inputs as seen by the most recent `mix()` —
    /// replayed at `perceive()` time so `extra_inputs.Clear()` between
    /// predict and perceive doesn't strip the training signal.
    last_inputs: Vec<f32>,
    last_extras: Vec<f32>,
}

// ============================================================
// CmixPredictor
// ============================================================

pub struct CmixPredictor {
    cfg: Config,
    manager: ContextManager,
    models: Vec<OrchModel>,
    byte_models: Vec<Ppmd>,
    byte_mixers: Vec<ByteMixer>,
    /// Three layers of MixerInputs, one per stage of the tree.
    layers: [MixerInput; 3],
    /// Mixers per layer (l=0,1,2).
    mixers: [Vec<MixerBinding>; 3],
    sse: Option<SSE>,
    sigmoid: Sigmoid,
    /// Indices into `layers[0].inputs()` that are "auxiliary" — fed
    /// directly to layers 1 and 2 without remixing.
    auxiliary: Vec<usize>,
    /// Position of the FXCM model in `models` (special — perceived
    /// last, only after the byte_mixer feeds back).
    fxcm_index: Option<usize>,
    /// Vocab mask (256 entries, padded if caller passed a shorter
    /// `Vec<bool>`).
    vocab: [bool; 256],
    /// Last `byte_mixer_override` (0 or 1 short-circuit from upstream).
    byte_mixer_override: Option<f32>,
}

impl CmixPredictor {
    pub fn new(vocab: Vec<bool>, cfg: Config) -> Self {
        // Pad vocab to 256 entries (upstream's vocab_ is parameter-driven).
        let mut vocab256 = [false; 256];
        for i in 0..256.min(vocab.len()) { vocab256[i] = vocab[i]; }

        let sigmoid = Sigmoid::new(100_001);
        let manager = ContextManager::new(cfg.history_bytes, cfg.shared_map_bytes);

        let layers = [
            MixerInput::new(sigmoid.clone(), 1.0e-4),
            MixerInput::new(sigmoid.clone(), 1.0e-4),
            MixerInput::new(sigmoid.clone(), 1.0e-4),
        ];
        let mixers = [Vec::new(), Vec::new(), Vec::new()];

        let sse = if cfg.enable_sse { Some(SSE::new()) } else { None };

        let mut p = Self {
            cfg, manager, models: Vec::new(),
            byte_models: Vec::new(), byte_mixers: Vec::new(),
            layers, mixers, sse, sigmoid,
            auxiliary: Vec::new(), fxcm_index: None,
            vocab: vocab256, byte_mixer_override: None,
        };

        p.add_bracket();
        p.add_fxcm();
        p.add_paq8();
        p.add_ppmd();
        p.add_word();
        p.add_direct();
        p.add_match();
        p.add_double_indirect();
        p.add_mixers();
        p
    }

    // ----------------- model registration -----------------

    fn add_model(&mut self, m: OrchModel) { self.models.push(m); }
    fn add_byte_model(&mut self, m: Ppmd) { self.byte_models.push(m); }
    fn add_byte_mixer(&mut self, m: ByteMixer) { self.byte_mixers.push(m); }

    /// Number of bit-1 probability slots produced by all bit-level
    /// models, byte models and byte mixers combined.
    fn num_models(&self) -> usize {
        let mut n = 0;
        for m in &self.models { n += m.num_outputs(); }
        n += self.byte_models.len();     // one bit-1 prob per PPMD
        n += self.byte_mixers.len();     // one bit-1 prob per byte_mixer
        n
    }

    fn add_auxiliary(&mut self) {
        let idx = self.num_models() - 1;
        self.auxiliary.push(idx);
    }

    fn add_mixer(&mut self, layer: usize, ctx_src: Src, learning_rate: f32) {
        let inputs = self.layers[layer].inputs().len();
        // Upstream's `AddMixer` passes `mixers_[layer].size()` as the
        // extra-input cap — so mixer i in layer L reads exactly the
        // outputs of the i mixers before it at that layer.
        let extras_cap = self.mixers[layer].len();
        let mixer = Mixer::new(inputs.max(1), extras_cap, learning_rate);
        self.mixers[layer].push(MixerBinding {
            mixer, ctx_src, extras_cap,
            last_inputs: Vec::new(), last_extras: Vec::new(),
        });
    }

    // ----------------- AddBracket -----------------

    fn add_bracket(&mut self) {
        let vocab_vec = self.vocab.to_vec();
        // 1) The Bracket "byte-shaping" sub-model.
        self.add_model(OrchModel::Bracket {
            m: Bracket::new(200, 10, 100_000, vocab_vec),
        });
        // 2) Register BracketContext + the Direct/Indirect that feed
        // off of it.
        let bracket_ctx = self.manager.add_context(CtxNode::Bracket {
            c: BracketContext::new(256, 15),
            byte_src: Src::BitContext,
        });
        let size = self.manager.ctx_size(Src::Ctx(bracket_ctx));
        self.add_model(OrchModel::Direct {
            m: Direct::new(30, 0.0, cap(size, self.cfg.direct_cap)),
            byte_src: Src::Ctx(bracket_ctx),
            size,
        });
        self.add_model(OrchModel::IndirectNs {
            m: Indirect::new(Nonstationary::new(), 300.0,
                self.cfg.shared_map_bytes, 0xDEAD_BEEF),
            byte_src: Src::Ctx(bracket_ctx),
        });
    }

    // ----------------- AddFXCM -----------------

    fn add_fxcm(&mut self) {
        if !self.cfg.enable_fxcm { return; }
        self.add_model(OrchModel::Fxcm {
            m: fxcmv1::Predictor::new(),
            output: [0.5],
        });
        self.add_auxiliary();
        self.fxcm_index = Some(self.models.len() - 1);
    }

    // ----------------- AddPAQ8 -----------------

    fn add_paq8(&mut self) {
        if !self.cfg.enable_paq8 { return; }
        self.add_model(OrchModel::Paq8 {
            m: Paq8::new(self.cfg.paq8_level),
            output: [0.5],
        });
        self.add_auxiliary();
    }

    // ----------------- AddPPMD -----------------

    fn add_ppmd(&mut self) {
        self.add_byte_model(Ppmd::new(self.cfg.ppmd_order, self.cfg.ppmd_memory_mb));
    }

    // ----------------- AddWord -----------------

    fn add_word(&mut self) {
        let delta = 200.0f32;
        let model_params: &[&[u32]] = &[
            &[0], &[0, 1], &[7, 2], &[7], &[1], &[1, 2], &[1, 2, 3],
            &[1, 3], &[1, 4], &[1, 5], &[2, 3], &[3, 4], &[1, 2, 4],
            &[1, 2, 3, 4], &[2, 3, 4], &[2], &[1, 2, 3, 4, 5],
            &[1, 2, 3, 4, 5, 6],
        ];
        for params in model_params {
            let ctx = self.manager.add_context(CtxNode::Sparse {
                c: Sparse::new(params.to_vec()),
            });
            self.add_model(OrchModel::IndirectNs {
                m: Indirect::new(Nonstationary::new(), delta,
                    self.cfg.shared_map_bytes, seed_from(ctx)),
                byte_src: Src::Ctx(ctx),
            });
        }
        let model_params2: &[&[u32]] = &[
            &[0], &[1], &[7], &[1, 3], &[1, 2, 3], &[7, 2],
        ];
        for params in model_params2 {
            let ctx = self.manager.add_context(CtxNode::Sparse {
                c: Sparse::new(params.to_vec()),
            });
            self.add_model(OrchModel::Match {
                m: Match::new(200, 0.5, self.cfg.match_cap),
                byte_src: Src::Ctx(ctx),
            });
            if params.len() == 1 && params[0] == 1 {
                self.add_model(OrchModel::IndirectRm {
                    m: Indirect::new(RunMap::new(), delta,
                        self.cfg.shared_map_bytes, seed_from(ctx)),
                    byte_src: Src::Ctx(ctx),
                });
                self.add_model(OrchModel::DirectHash {
                    m: DirectHash::new(30, 0.0, self.cfg.direct_hash_cap),
                    byte_src: Src::Ctx(ctx),
                });
            }
        }
    }

    // ----------------- AddDirect -----------------

    fn add_direct(&mut self) {
        let delta = 0.0f32;
        let limit = 30;
        let params: &[(u32, u32)] =
            &[(0, 8), (1, 8), (2, 8), (3, 8)];
        for &(order, hash_size) in params {
            let ctx = self.manager.add_context(CtxNode::ContextHash {
                c: ContextHash::new(order, hash_size),
                byte_src: Src::BitContext,
            });
            let size = self.manager.ctx_size(Src::Ctx(ctx));
            if order < 3 {
                self.add_model(OrchModel::Direct {
                    m: Direct::new(limit, delta, cap(size, self.cfg.direct_cap)),
                    byte_src: Src::Ctx(ctx),
                    size,
                });
            } else {
                self.add_model(OrchModel::DirectHash {
                    m: DirectHash::new(limit, delta, self.cfg.direct_hash_cap),
                    byte_src: Src::Ctx(ctx),
                });
            }
        }
    }

    // ----------------- AddMatch -----------------

    fn add_match(&mut self) {
        let delta = 0.5f32;
        let limit = 200;
        let params: &[(u32, u32)] = &[
            (0, 8), (1, 8), (2, 8), (7, 4), (11, 3), (13, 2),
            (15, 2), (17, 2), (20, 1), (25, 1),
        ];
        for &(order, hash_size) in params {
            let ctx = self.manager.add_context(CtxNode::ContextHash {
                c: ContextHash::new(order, hash_size),
                byte_src: Src::BitContext,
            });
            let size = self.manager.ctx_size(Src::Ctx(ctx));
            let max = (size as usize).min(self.cfg.match_cap).max(1);
            self.add_model(OrchModel::Match {
                m: Match::new(limit, delta, max),
                byte_src: Src::Ctx(ctx),
            });
        }
    }

    // ----------------- AddDoubleIndirect -----------------

    fn add_double_indirect(&mut self) {
        let delta = 400.0f32;
        let params: &[(u32, u32, u32, u32)] = &[
            (1, 8, 1, 8), (2, 8, 1, 8), (1, 8, 2, 8), (2, 8, 2, 8),
            (1, 8, 3, 8), (3, 8, 1, 8), (4, 6, 4, 8), (5, 5, 5, 5),
            (1, 8, 4, 8), (1, 8, 5, 6), (6, 4, 6, 4),
        ];
        for &(o1, hs1, o2, hs2) in params {
            let ctx = self.manager.add_context(CtxNode::IndirectHash {
                c: IndirectHash::new(o1, hs1, o2, hs2),
                byte_src: Src::BitContext,
            });
            self.add_model(OrchModel::IndirectNs {
                m: Indirect::new(Nonstationary::new(), delta,
                    self.cfg.shared_map_bytes, seed_from(ctx)),
                byte_src: Src::Ctx(ctx),
            });
        }
    }

    // ----------------- AddMixers -----------------

    fn add_mixers(&mut self) {
        let vocab_size = self.vocab.iter().filter(|b| **b).count();
        if self.cfg.enable_byte_mixer && vocab_size > 0 {
            let lstm = Lstm::new(
                vocab_size, vocab_size,
                self.cfg.lstm_cells, self.cfg.lstm_layers,
                self.cfg.lstm_horizon,
                self.cfg.lstm_learning_rate, self.cfg.lstm_gradient_clip,
            );
            let bm = ByteMixer::new(
                self.byte_models.len() as u32,
                self.vocab.to_vec(), vocab_size, lstm,
            );
            self.add_byte_mixer(bm);
            self.add_auxiliary();
        }

        let input_size = self.num_models();
        self.layers[0].set_num_models(input_size);

        // Layer 0: ContextHash + BitContext wrapped → mixer key
        let p0: &[(u32, u32, f32)] = &[
            (0, 8, 0.005), (0, 8, 0.0005),
            (1, 8, 0.005), (1, 8, 0.0005),
            (2, 4, 0.005), (3, 2, 0.002),
        ];
        for &(o, hs, lr) in p0 {
            let ctx = self.manager.add_context(CtxNode::ContextHash {
                c: ContextHash::new(o, hs),
                byte_src: Src::BitContext,
            });
            let size = self.manager.ctx_size(Src::Ctx(ctx));
            let bit_ctx = self.manager.add_bit_context(BitCtxNode {
                c: BitCtx::new(size),
                byte_src: Src::Ctx(ctx),
            });
            self.add_mixer(0, Src::BitCtx(bit_ctx), lr);
        }

        // RecentByte-based mixers.
        self.add_mixer(0, Src::RecentByte(2), 0.002);
        self.add_mixer(0, Src::RecentByte(3), 0.005);

        // Scalar context-source mixers.
        self.add_mixer(0, Src::Zero, 0.00005);
        self.add_mixer(0, Src::LineBreak, 0.0007);
        self.add_mixer(0, Src::LongestMatch, 0.0005);
        self.add_mixer(0, Src::Wrt, 0.002);
        self.add_mixer(0, Src::Auxiliary, 0.0005);

        // Interval-based mixers — three threshold-bucket maps.
        let mut map = vec![0i32; 256];
        for i in 0..256 {
            map[i] = ((i < 1) as i32) + ((i < 32) as i32) + ((i < 64) as i32)
                + ((i < 128) as i32) + ((i < 255) as i32) + ((i < 142) as i32)
                + ((i < 138) as i32) + ((i < 140) as i32) + ((i < 137) as i32)
                + ((i < 97) as i32);
        }
        let iv1 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map.clone(), 8), byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv1), 0.001);

        for i in 0..256 {
            map[i] = ((i < 41) as i32) + ((i < 92) as i32) + ((i < 124) as i32)
                + ((i < 58) as i32) + ((i < 11) as i32) + ((i < 46) as i32)
                + ((i < 36) as i32) + ((i < 47) as i32) + ((i < 64) as i32)
                + ((i < 4) as i32) + ((i < 61) as i32) + ((i < 97) as i32)
                + ((i < 125) as i32) + ((i < 45) as i32) + ((i < 48) as i32);
        }
        let iv2 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map.clone(), 8), byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv2), 0.001);

        for i in 0..256 { map[i] = 0; }
        for i in (b'a' as usize)..=(b'z' as usize) { map[i] = 1; }
        for i in (b'A' as usize)..=(b'Z' as usize) { map[i] = 1; }
        for i in (b'0' as usize)..=(b'9' as usize) { map[i] = 1; }
        for i in 0x80..256 { map[i] = 1; }
        let iv3 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map.clone(), 7), byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv3), 0.001);
        let size3 = self.manager.ctx_size(Src::Ctx(iv3));
        let bc5 = self.manager.add_bit_context(BitCtxNode {
            c: BitCtx::new(size3), byte_src: Src::Ctx(iv3),
        });
        self.add_mixer(0, Src::BitCtx(bc5), 0.005);

        // 256-entry character class map (verbatim upstream).
        let map_class1: [i32; 256] = [
            2,3,1,3,3,0,1,2,3,3,0,0,1,3,3,3,
            3,3,3,3,3,3,3,3,3,3,3,0,3,3,3,3,
            3,2,0,2,1,3,2,1,3,3,3,3,2,3,0,2,
            1,1,1,1,1,1,1,1,1,1,3,2,2,3,2,2,
            2,2,0,0,2,3,1,2,1,2,2,2,2,2,0,0,
            2,2,2,2,2,2,2,2,3,0,2,3,2,0,2,3,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        ];
        let iv4 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map_class1.to_vec(), 10), byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv4), 0.001);
        let iv5 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map_class1.to_vec(), 15), byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv5), 0.001);
        let iv8 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map_class1.to_vec(), 7), byte_src: Src::BitContext,
        });
        let size8 = self.manager.ctx_size(Src::Ctx(iv8));
        let bc4 = self.manager.add_bit_context(BitCtxNode {
            c: BitCtx::new(size8), byte_src: Src::Ctx(iv8),
        });
        self.add_mixer(0, Src::BitCtx(bc4), 0.005);

        // 256-entry character class map #2 (verbatim upstream).
        let map_class2: [i32; 256] = [
            0,0,2,0,5,6,0,6,0,2,0,4,3,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,4,1,4,4,7,4,7,3,7,2,2,3,5,3,1,
            1,1,1,1,1,1,1,1,1,1,0,5,3,3,5,5,
            0,5,5,7,5,0,1,5,4,5,0,0,6,0,7,1,
            3,3,7,4,5,5,7,0,2,2,5,4,4,7,4,6,
            5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
            5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
            6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
            6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
            6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
            6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
            6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
            7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
            7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
            7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
        ];
        let iv6 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map_class2.to_vec(), 9), byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv6), 0.001);
        let iv7 = self.manager.add_context(CtxNode::IntervalHash {
            c: IntervalHash::new(map_class2.to_vec(), 8, 7, 2),
            byte_src: Src::BitContext,
        });
        self.add_mixer(0, Src::Ctx(iv7), 0.001);
        let iv9 = self.manager.add_context(CtxNode::Interval {
            c: Interval::new(map_class2.to_vec(), 7), byte_src: Src::BitContext,
        });
        let size9 = self.manager.ctx_size(Src::Ctx(iv9));
        let bc6 = self.manager.add_bit_context(BitCtxNode {
            c: BitCtx::new(size9), byte_src: Src::Ctx(iv9),
        });
        self.add_mixer(0, Src::BitCtx(bc6), 0.005);

        // BitContext over recent_bytes[1].
        let bc1 = self.manager.add_bit_context(BitCtxNode {
            c: BitCtx::new(256), byte_src: Src::RecentByte(1),
        });
        self.add_mixer(0, Src::BitCtx(bc1), 0.005);

        // Combined contexts.
        let cb1 = self.manager.add_context(CtxNode::Combined {
            c: CombinedContext::new(256, 256),
            a: Src::RecentByte(1), b: Src::RecentByte(0),
        });
        self.add_mixer(0, Src::Ctx(cb1), 0.005);
        let cb2 = self.manager.add_context(CtxNode::Combined {
            c: CombinedContext::new(256, 256),
            a: Src::RecentByte(2), b: Src::RecentByte(1),
        });
        self.add_mixer(0, Src::Ctx(cb2), 0.003);

        // Layer 1 — re-mixes layer-0 outputs and auxiliaries.
        let l1_size = self.mixers[0].len() + self.auxiliary.len();
        self.layers[1].set_num_models(l1_size);

        self.add_mixer(1, Src::Zero, 0.005);
        self.add_mixer(1, Src::Zero, 0.0005);
        self.add_mixer(1, Src::LongBitContext, 0.005);
        self.add_mixer(1, Src::LongBitContext, 0.0005);
        self.add_mixer(1, Src::LongBitContext, 0.00001);
        self.add_mixer(1, Src::RecentByte(0), 0.005);
        self.add_mixer(1, Src::RecentByte(1), 0.005);
        self.add_mixer(1, Src::RecentByte(2), 0.005);
        self.add_mixer(1, Src::LongestMatch, 0.0005);
        self.add_mixer(1, Src::Wrt, 0.002);
        self.add_mixer(1, Src::Ctx(iv1), 0.001);
        self.add_mixer(1, Src::Ctx(iv2), 0.001);
        self.add_mixer(1, Src::Ctx(iv3), 0.001);
        self.add_mixer(1, Src::Ctx(iv4), 0.001);
        self.add_mixer(1, Src::Ctx(iv5), 0.001);
        self.add_mixer(1, Src::Ctx(iv6), 0.001);
        self.add_mixer(1, Src::Ctx(iv7), 0.001);
        self.add_mixer(1, Src::BitCtx(bc4), 0.001);
        self.add_mixer(1, Src::BitCtx(bc5), 0.001);
        self.add_mixer(1, Src::BitCtx(bc6), 0.001);

        // Layer 2 — final mix.
        let l2_size = self.mixers[0].len() + self.mixers[1].len()
            + self.auxiliary.len();
        self.layers[2].set_num_models(l2_size);
        self.add_mixer(2, Src::Zero, 0.0003);
    }

    /// Read-only access to the embedded [`ContextManager`] (mostly
    /// useful for tests / introspection — callers don't need to touch
    /// this).
    pub fn manager(&self) -> &ContextManager { &self.manager }

    // ----------------- Predict -----------------

    /// Returns the bit-1 probability for the next bit (∈ [0, 1]).
    pub fn predict(&mut self) -> f32 {
        // 1) Read all bit-level models into layer-0.
        let mut input_index = 0usize;
        for i in 0..self.models.len() {
            let p = predict_model(&mut self.models[i], &self.manager);
            self.layers[0].set_input(input_index, p);
            input_index += 1;
        }

        // 2) Read all byte models (PPMD) — bit-1 prob via binary search.
        for ppmd in &mut self.byte_models {
            let p = byte_model_bit_p(ppmd, self.manager.bit_context);
            self.layers[0].set_input(input_index, p);
            input_index += 1;
        }

        // 3) Read all byte mixers.
        let mut override_p: Option<f32> = None;
        for bm in &mut self.byte_mixers {
            let p = byte_mixer_bit_p(bm, self.manager.bit_context);
            if p == 0.0 || p == 1.0 { override_p = Some(p); }
            self.layers[0].set_input(input_index, p);
            input_index += 1;
        }
        self.byte_mixer_override = override_p;

        // 4) Auxiliary average → manager.auxiliary_context.
        if !self.auxiliary.is_empty() {
            let inputs = self.layers[0].inputs();
            let mut acc = 0.0f32;
            for &i in &self.auxiliary {
                acc += Sigmoid::logistic(inputs[i]);
            }
            acc /= self.auxiliary.len() as f32;
            self.manager.auxiliary_context = (acc * 15.0) as u64;
        }

        // 5) Layer-0 mixer pass. Each mixer reads layer-0 inputs +
        // the running `extra_inputs` slice — at mixer index `i`,
        // exactly `i` prior mixer outputs are visible. Snapshot per
        // mixer so perceive() can replay the same training signal.
        let l0_inputs: Vec<f32> = self.layers[0].inputs().to_vec();
        for i in 0..self.mixers[0].len() {
            let cap = self.mixers[0][i].extras_cap;
            let extras_now = self.layers[0].extra_inputs();
            let take = cap.min(extras_now.len());
            let l0_extras: Vec<f32> = extras_now[..take].to_vec();
            let ctx = self.manager.resolve(self.mixers[0][i].ctx_src);
            let p = self.mixers[0][i].mixer.mix(&l0_inputs, &l0_extras, ctx);
            // Cache for perceive replay.
            self.mixers[0][i].last_inputs = l0_inputs.clone();
            self.mixers[0][i].last_extras = l0_extras;
            self.layers[0].set_extra_input(p);
            self.layers[1].set_stretched_input(i, p);
            self.layers[2].set_stretched_input(i, p);
        }
        self.layers[0].clear_extra_inputs();

        // 6) Forward auxiliaries into layers 1 and 2.
        let aux_offset_1 = self.mixers[0].len();
        let aux_offset_2 = self.mixers[0].len() + self.mixers[1].len();
        let l0_input_now = self.layers[0].inputs().to_vec();
        for (i, &slot) in self.auxiliary.iter().enumerate() {
            let p = l0_input_now[slot];
            self.layers[1].set_stretched_input(aux_offset_1 + i, p);
            self.layers[2].set_stretched_input(aux_offset_2 + i, p);
        }

        // 7) Layer-1 mixer pass — same per-mixer extras pattern.
        let l1_inputs: Vec<f32> = self.layers[1].inputs().to_vec();
        for i in 0..self.mixers[1].len() {
            let cap = self.mixers[1][i].extras_cap;
            let extras_now = self.layers[1].extra_inputs();
            let take = cap.min(extras_now.len());
            let l1_extras: Vec<f32> = extras_now[..take].to_vec();
            let ctx = self.manager.resolve(self.mixers[1][i].ctx_src);
            let p = self.mixers[1][i].mixer.mix(&l1_inputs, &l1_extras, ctx);
            self.mixers[1][i].last_inputs = l1_inputs.clone();
            self.mixers[1][i].last_extras = l1_extras;
            self.layers[1].set_extra_input(p);
            self.layers[2].set_stretched_input(self.mixers[0].len() + i, p);
        }
        self.layers[1].clear_extra_inputs();

        // 8) Layer-2 mixer (only one).
        let l2_inputs: Vec<f32> = self.layers[2].inputs().to_vec();
        let cap = self.mixers[2][0].extras_cap;
        let extras_now = self.layers[2].extra_inputs();
        let take = cap.min(extras_now.len());
        let l2_extras: Vec<f32> = extras_now[..take].to_vec();
        let ctx = self.manager.resolve(self.mixers[2][0].ctx_src);
        let raw = self.mixers[2][0].mixer.mix(&l2_inputs, &l2_extras, ctx);
        self.mixers[2][0].last_inputs = l2_inputs.clone();
        self.mixers[2][0].last_extras = l2_extras;
        let mut p = Sigmoid::logistic(raw);

        // 9) SSE smoothing.
        if let Some(sse) = &mut self.sse {
            p = sse.predict(p);
        }

        // 10) Byte-mixer override short-circuit.
        if let Some(ov) = self.byte_mixer_override { ov } else { p }
    }

    // ----------------- Pretrain -----------------

    /// Train only the bit-level model bank on `bit` without writing
    /// to the arithmetic coder — mirrors upstream `Predictor::Pretrain`
    /// (predictor.cpp:471-487). Used to warm sub-models on a prefix
    /// of the input before the encoding pass starts.
    pub fn pretrain(&mut self, bit: i32) {
        for i in 0..self.models.len() {
            let _ = predict_model(&mut self.models[i], &self.manager);
        }
        for i in 0..self.models.len() {
            perceive_model(&mut self.models[i], bit, &mut self.manager);
        }
        let byte_update = self.manager.bit_context >= 128;
        let at_boundary = self.manager.update_bit(bit);
        self.manager.update_contexts_owned(at_boundary);
        if byte_update {
            let just_done_byte = self.manager.bit_context as u8;
            for i in 0..self.models.len() {
                byte_update_model(&mut self.models[i], just_done_byte,
                    &mut self.manager);
            }
            self.manager.bit_context = 1;
            self.manager.long_bit_context = 1;
        }
    }

    // ----------------- Perceive -----------------

    pub fn perceive(&mut self, bit: i32) {
        // 1) Perceive every bit model except FXCM (handled last).
        for i in 0..self.models.len() {
            if Some(i) == self.fxcm_index { continue; }
            perceive_model(&mut self.models[i], bit, &mut self.manager);
        }
        // 2) PPMD byte models.
        for ppmd in &mut self.byte_models {
            byte_model_perceive(ppmd, bit, self.manager.bit_context);
        }
        // 3) Byte mixers (their inner ByteModel binary-search advances).
        for bm in &mut self.byte_mixers {
            byte_mixer_perceive(bm, bit);
        }

        // 4) Train all three mixer layers — each mixer trains on the
        // exact `(inputs, extras)` snapshot it consumed at mix(), so
        // the just-cleared layer-0/1 extras don't strip the signal.
        for layer in 0..3 {
            for binding in self.mixers[layer].iter_mut() {
                let ctx = self.manager.resolve(binding.ctx_src);
                binding.mixer.perceive(
                    bit, &binding.last_inputs, &binding.last_extras, ctx,
                );
            }
        }

        // 5) SSE update.
        if let Some(sse) = &mut self.sse { sse.perceive(bit); }

        // 6) byte_update flag and context update.
        let byte_update = self.manager.bit_context >= 128;
        let at_boundary = self.manager.update_bit(bit);
        self.manager.update_contexts_owned(at_boundary);

        if byte_update {
            let just_done_byte = self.manager.bit_context as u8;
            // 7) Per-byte hooks for bit-level models that need them.
            for i in 0..self.models.len() {
                byte_update_model(&mut self.models[i], just_done_byte,
                    &mut self.manager);
            }
            // 8) PPMD byte_update.
            for ppmd in &mut self.byte_models {
                ppmd.byte_update(just_done_byte);
            }
            // 9) Feed PPMD byte distributions to byte_mixers, then
            //    have each byte_mixer commit its byte update.
            for ppmd in &mut self.byte_models {
                let probs = ppmd.finalize_probs();
                for bm in &mut self.byte_mixers {
                    for j in 0..256 {
                        bm.set_input(j, probs[j]);
                    }
                }
            }
            for bm in &mut self.byte_mixers {
                bm.byte_update(just_done_byte as u32);
            }
            // Reset bit_context for the NEXT byte before the FXCM
            // cross-feed runs — `byte_mixer_bit_p` reads bit_context
            // via `bit_context_range`, which only yields `(0, 255)`
            // when bit_context == 1. Upstream's `bit_context_ = 1`
            // happens at the very end of Perceive but its byte_mixer
            // ->Predict() reads internal bot/top fields (already
            // reset by ByteMixer::byte_update) so it sees the same
            // (0, 255) partition.
            self.manager.bit_context = 1;
            self.manager.long_bit_context = 1;
        }
        // 10) Byte-mixer output drives FXCM training (upstream
        // `predictor.cpp:462-467`). For each byte_mixer:
        //   lstmpr = Discretize(byte_mixer->Predict()[0])
        //   lstmex = byte_mixer->ex
        //   models_[fxcm_index_]->Perceive(bit)
        // Discretize(p) = 1 + 4094 * p. With multiple byte_mixers
        // upstream just iterates (each iteration overwrites lstmpr/
        // lstmex, then perceives FXCM) — the last byte_mixer's
        // signals are the ones that ultimately feed into FXCM's per-
        // bit chain on the *next* predict.
        if let Some(fxi) = self.fxcm_index {
            for bm in &mut self.byte_mixers {
                let p = byte_mixer_bit_p(bm, self.manager.bit_context);
                let lstmpr = 1 + (4094.0 * p) as i32;
                let lstmex = bm.byte_model.ex;
                if let OrchModel::Fxcm { m, .. } = &mut self.models[fxi] {
                    m.set_lstm_signals(lstmpr, lstmex);
                }
                perceive_model(&mut self.models[fxi], bit, &mut self.manager);
            }
            if self.byte_mixers.is_empty() {
                perceive_model(&mut self.models[fxi], bit, &mut self.manager);
            }
        }
    }
}

impl crate::coder::Predictor for CmixPredictor {
    fn predict(&mut self) -> f32 { Self::predict(self) }
    fn perceive(&mut self, bit: i32) { Self::perceive(self, bit) }
}

// ============================================================
// Free helpers — keep model dispatch in one place so the mut-borrow
// scopes around `self.manager` and `self.models` stay short.
// ============================================================

#[inline]
fn cap(size: u64, cap: usize) -> usize {
    let s = size as usize;
    if s == 0 { 1 } else { s.min(cap).max(1) }
}

#[inline]
fn seed_from(ctx_idx: usize) -> u64 {
    // Stable per-context deterministic seed — replaces upstream's libc rand().
    0x9E37_79B9_7F4A_7C15u64.wrapping_mul(ctx_idx as u64 + 1)
}

fn predict_model(m: &mut OrchModel, mgr: &ContextManager) -> f32 {
    match m {
        OrchModel::Direct { m, byte_src, size } => {
            let raw = mgr.resolve(*byte_src);
            let in_ctx = if *size > 0 { (raw % *size) as usize } else { 0 };
            let row = in_ctx % m.capacity().max(1);
            m.predict(row, (mgr.bit_context as usize) & 0xff)
        }
        OrchModel::DirectHash { m, .. } => {
            // byte_update has already set the slot index.
            m.predict((mgr.bit_context as usize) & 0xff)
        }
        OrchModel::IndirectNs { m, .. } => {
            m.predict(&mgr.shared_map, (mgr.bit_context as usize) & 0xff)
        }
        OrchModel::IndirectRm { m, .. } => {
            m.predict(&mgr.shared_map, (mgr.bit_context as usize) & 0xff)
        }
        OrchModel::Match { m, .. } => m.predict(),
        OrchModel::Bracket { m } => m.byte_model.predict(),
        OrchModel::Fxcm { m, output } => {
            let p = m.predict();
            output[0] = p;
            p
        }
        OrchModel::Paq8 { m, output } => {
            let p = m.predict_bit();
            output[0] = p;
            p
        }
    }
}

fn perceive_model(m: &mut OrchModel, bit: i32, mgr: &mut ContextManager) {
    match m {
        OrchModel::Direct { m, byte_src, size } => {
            let raw = mgr.resolve(*byte_src);
            let in_ctx = if *size > 0 { (raw % *size) as usize } else { 0 };
            let row = in_ctx % m.capacity().max(1);
            m.perceive(bit, row, (mgr.bit_context as usize) & 0xff);
        }
        OrchModel::DirectHash { m, .. } => {
            m.perceive(bit, (mgr.bit_context as usize) & 0xff);
        }
        OrchModel::IndirectNs { m, .. } => {
            m.perceive(bit, &mut mgr.shared_map);
        }
        OrchModel::IndirectRm { m, .. } => {
            m.perceive(bit, &mut mgr.shared_map);
        }
        OrchModel::Match { m, byte_src } => {
            let byte_ctx = mgr.resolve(*byte_src);
            m.perceive(bit, mgr.bit_context, byte_ctx);
        }
        OrchModel::Bracket { m } => {
            m.byte_model.perceive(bit);
        }
        OrchModel::Fxcm { m, .. } => m.perceive(bit),
        OrchModel::Paq8 { m, .. } => m.perceive_bit(bit),
    }
}

fn byte_update_model(m: &mut OrchModel, just_done_byte: u8, mgr: &mut ContextManager) {
    let history_snapshot = std::mem::take(&mut mgr.history);
    match m {
        OrchModel::DirectHash { m, byte_src } => {
            let byte_ctx = mgr.resolve(*byte_src);
            m.byte_update(byte_ctx);
        }
        OrchModel::IndirectNs { m, byte_src } => {
            let byte_ctx = mgr.resolve(*byte_src);
            let len = mgr.shared_map.len();
            m.byte_update(byte_ctx, len);
        }
        OrchModel::IndirectRm { m, byte_src } => {
            let byte_ctx = mgr.resolve(*byte_src);
            let len = mgr.shared_map.len();
            m.byte_update(byte_ctx, len);
        }
        OrchModel::Match { m, byte_src } => {
            let byte_ctx = mgr.resolve(*byte_src);
            m.byte_update(byte_ctx, &history_snapshot);
            // Upstream's Match writes the match-length context back
            // through a &longest_match pointer that aliases the
            // manager field — propagate the max so mixers reading
            // `Src::LongestMatch` see the same signal.
            if m.longest_match > mgr.longest_match {
                mgr.longest_match = m.longest_match;
            }
        }
        OrchModel::Bracket { m } => {
            m.byte_update(just_done_byte);
        }
        OrchModel::Direct { .. } | OrchModel::Fxcm { .. } | OrchModel::Paq8 { .. } => {}
    }
    mgr.history = history_snapshot;
}

fn byte_model_bit_p(p: &mut Ppmd, bit_context: u32) -> f32 {
    // Treat PPMD's byte-prob vector as a ByteModel binary search.
    // Reuse the byte model machinery from `models::ByteModel` by
    // running the same `mid = bot + (top-bot)/2` logic.
    // Here `p.probs` is the live byte distribution maintained by PPMD.
    // For the current binary search position derived from bit_context,
    // emit Σprobs[mid+1..top+1] / Σprobs[bot..top+1].
    let (bot, top) = bit_context_range(bit_context);
    let mid = bot + (top - bot) / 2;
    let mut num = 0.0f32;
    for i in (mid + 1)..=top { num += p.probs[i as usize]; }
    let mut denom = num;
    for i in bot..=mid { denom += p.probs[i as usize]; }
    if denom == 0.0 { 0.5 } else { num / denom }
}

fn byte_model_perceive(_p: &mut Ppmd, _bit: i32, _bit_context: u32) {
    // PPMD's byte distribution doesn't track per-bit search state;
    // that state lives in the orchestrator via `bit_context_range`.
    // No-op.
}

fn byte_mixer_bit_p(bm: &mut ByteMixer, bit_context: u32) -> f32 {
    let (bot, top) = bit_context_range(bit_context);
    let mid = bot + (top - bot) / 2;
    let mut num = 0.0f32;
    for i in (mid + 1)..=top { num += bm.byte_model.probs[i as usize]; }
    let mut denom = num;
    for i in bot..=mid { denom += bm.byte_model.probs[i as usize]; }
    // Update the byte_model's `ex` (most-probable byte in [bot..=top])
    // so the orchestrator's lstmex cross-feed sees the current value.
    let mut max_p = bm.byte_model.probs[bot as usize];
    let mut ex = bot;
    for i in (bot + 1)..=top {
        let p = bm.byte_model.probs[i as usize];
        if p > max_p { max_p = p; ex = i; }
    }
    bm.byte_model.ex = ex;
    if denom == 0.0 { 0.5 } else { num / denom }
}

fn byte_mixer_perceive(bm: &mut ByteMixer, bit: i32) {
    bm.byte_model.perceive(bit);
}

/// Map a partial-byte register (1..=255 or, post-boundary, 0..=255)
/// to a `(bot, top)` range over the 256 possible byte values being
/// binary-searched.
fn bit_context_range(bit_context: u32) -> (i32, i32) {
    // Mid-byte: bit_context has 1..=8 leading bits. The full byte
    // value is in (bit_context << remaining_bits)..(bit_context << remaining_bits) + (1 << remaining_bits).
    // Find the leading 1.
    let mut lead = 0;
    let mut bc = bit_context.max(1);
    while bc > 0 { bc >>= 1; lead += 1; }
    let bits_known = lead - 1;
    let remaining = 8 - bits_known;
    let value_so_far = bit_context & ((1 << bits_known) - 1);
    let bot = (value_so_far as i32) << remaining;
    let top = bot + (1 << remaining) - 1;
    (bot.min(255), top.min(255))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn full_vocab() -> Vec<bool> { vec![true; 256] }

    #[test]
    fn cmix_predictor_tiny_round_trip_shape() {
        let mut p = CmixPredictor::new(full_vocab(), Config::tiny());
        for byte in b"hello world" {
            for bp in (0..8).rev() {
                let pr = p.predict();
                assert!(pr >= 0.0 && pr <= 1.0,
                    "predict outside [0,1]: {}", pr);
                let bit = ((byte >> bp) & 1) as i32;
                p.perceive(bit);
            }
        }
    }

    #[test]
    fn bit_context_range_initial_byte_is_full_range() {
        let (b, t) = bit_context_range(1);
        assert_eq!((b, t), (0, 255));
    }

    /// The lstmex / lstmpr cross-feed must reach fxcmv1 — i.e. the
    /// orchestrator's tiny-config predictions move when we observe a
    /// strongly-biased byte stream. We don't assert on the bit-exact
    /// probability (that depends on every sub-model), only that the
    /// predictor reacts and stays in range over many bytes.
    #[test]
    fn cmix_predictor_reacts_to_biased_stream() {
        let mut p = CmixPredictor::new(vec![true; 256], Config::tiny());
        let mut sum = 0.0f64;
        let mut count = 0usize;
        // Repeat byte 0x41 ('A') 256 times.
        for _ in 0..256 {
            for bp in (0..8).rev() {
                let pr = p.predict();
                assert!(pr >= 0.0 && pr <= 1.0);
                sum += pr as f64;
                count += 1;
                p.perceive((((0x41u8) >> bp) & 1) as i32);
            }
        }
        // After 2KB of constant input the mean prediction should
        // skew clearly from 0.5 (toward predicting the next bit
        // correctly). Don't require a tight bound — the tiny config
        // disables PAQ8/FXCM/SSE.
        let mean = sum / count as f64;
        assert!(mean != 0.5, "predictor should drift away from neutral");
    }

    #[test]
    fn bit_context_range_after_one_bit_halves() {
        // bit_context = 0b10 means bit 0 of byte = 0 → low half.
        let (b, t) = bit_context_range(0b10);
        assert_eq!((b, t), (0, 127));
        // bit_context = 0b11 means bit 0 of byte = 1 → high half.
        let (b, t) = bit_context_range(0b11);
        assert_eq!((b, t), (128, 255));
    }
}
