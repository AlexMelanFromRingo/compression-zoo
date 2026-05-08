//! QLFC statistical model — port of
//! `plugins/bsc/upstream/libbsc/coder/qlfc/qlfc_model.{h,cpp}`.
//!
//! The C struct nests very large arrays (`[256][256]` mantissa
//! tables, an array-of-eight of those, plus per-character mixers).
//! In Rust we keep the *small* (≤ a few KiB) tables as fixed-size
//! arrays, but spill the multi-megabyte ones into `Vec<i16>` with a
//! documented stride to keep stack frames small and avoid relying on
//! placement-new optimisations.
//!
//! Probability values are 12-bit signed (0..4095, occasionally
//! negative when libbsc treats them as differential / stretched);
//! they're initialised to 2048 (50 %) in the static model.

#![allow(dead_code)]

use crate::predictor::ProbabilityMixer;

pub const ALPHABET_SIZE: usize = 256;

// ===================================================================
// Tuning constants — these come from
// upstream/libbsc/coder/qlfc/qlfc_model.h and are used by the QLFC
// encoder/decoder. Reproduced verbatim so the predictor updates match
// the C reference bit-for-bit.
// ===================================================================

// Adaptive (M) — encoder/decoder updates.
pub const M_RANK_TS_TH0: i32 =    1; pub const M_RANK_TS_AR0: i32 =   57;
pub const M_RANK_TS_TH1: i32 = -111; pub const M_RANK_TS_AR1: i32 =   31;
pub const M_RANK_TC_TH0: i32 =  291; pub const M_RANK_TC_AR0: i32 =  250;
pub const M_RANK_TC_TH1: i32 =  154; pub const M_RANK_TC_AR1: i32 =  528;
pub const M_RANK_TP_TH0: i32 =  375; pub const M_RANK_TP_AR0: i32 =  163;
pub const M_RANK_TP_TH1: i32 =  313; pub const M_RANK_TP_AR1: i32 =  639;
pub const M_RANK_TM_TH0: i32 =  -41; pub const M_RANK_TM_AR0: i32 =   96;
pub const M_RANK_TM_TH1: i32 =   53; pub const M_RANK_TM_AR1: i32 =   49;
pub const M_RANK_TM_LR0: i32 =   20; pub const M_RANK_TM_LR1: i32 =   47;
pub const M_RANK_TM_LR2: i32 =   27;

pub const M_RANK_ES_TH0: i32 = -137; pub const M_RANK_ES_AR0: i32 =   17;
pub const M_RANK_ES_TH1: i32 =  482; pub const M_RANK_ES_AR1: i32 =   40;
pub const M_RANK_EC_TH0: i32 =   61; pub const M_RANK_EC_AR0: i32 =  192;
pub const M_RANK_EC_TH1: i32 =  200; pub const M_RANK_EC_AR1: i32 =  133;
pub const M_RANK_EP_TH0: i32 =   54; pub const M_RANK_EP_AR0: i32 = 1342;
pub const M_RANK_EP_TH1: i32 =  578; pub const M_RANK_EP_AR1: i32 = 1067;
pub const M_RANK_EM_TH0: i32 =  -11; pub const M_RANK_EM_AR0: i32 =  318;
pub const M_RANK_EM_TH1: i32 =  144; pub const M_RANK_EM_AR1: i32 =  848;
pub const M_RANK_EM_LR0: i32 =   49; pub const M_RANK_EM_LR1: i32 =   41;
pub const M_RANK_EM_LR2: i32 =   40;

pub const M_RANK_MS_TH0: i32 = -145; pub const M_RANK_MS_AR0: i32 =   18;
pub const M_RANK_MS_TH1: i32 =  114; pub const M_RANK_MS_AR1: i32 =   24;
pub const M_RANK_MC_TH0: i32 =  -43; pub const M_RANK_MC_AR0: i32 =   69;
pub const M_RANK_MC_TH1: i32 =  -36; pub const M_RANK_MC_AR1: i32 =   78;
pub const M_RANK_MP_TH0: i32 =   -2; pub const M_RANK_MP_AR0: i32 = 1119;
pub const M_RANK_MP_TH1: i32 =   11; pub const M_RANK_MP_AR1: i32 = 1181;
pub const M_RANK_MM_TH0: i32 = -203; pub const M_RANK_MM_AR0: i32 =   20;
pub const M_RANK_MM_TH1: i32 = -271; pub const M_RANK_MM_AR1: i32 =   15;
pub const M_RANK_MM_LR0: i32 =  263; pub const M_RANK_MM_LR1: i32 =  175;
pub const M_RANK_MM_LR2: i32 =   17;

pub const M_RANK_PS_TH0: i32 =  -99; pub const M_RANK_PS_AR0: i32 =   32;
pub const M_RANK_PS_TH1: i32 =  318; pub const M_RANK_PS_AR1: i32 =   42;
pub const M_RANK_PC_TH0: i32 =   17; pub const M_RANK_PC_AR0: i32 =  101;
pub const M_RANK_PC_TH1: i32 = 1116; pub const M_RANK_PC_AR1: i32 =  246;
pub const M_RANK_PP_TH0: i32 =   22; pub const M_RANK_PP_AR0: i32 =  964;
pub const M_RANK_PP_TH1: i32 =   -2; pub const M_RANK_PP_AR1: i32 = 1110;
pub const M_RANK_PM_TH0: i32 = -194; pub const M_RANK_PM_AR0: i32 =   21;
pub const M_RANK_PM_TH1: i32 = -129; pub const M_RANK_PM_AR1: i32 =   20;
pub const M_RANK_PM_LR0: i32 =  480; pub const M_RANK_PM_LR1: i32 =  202;
pub const M_RANK_PM_LR2: i32 =   17;

pub const M_RUN_TS_TH0: i32 =  -93; pub const M_RUN_TS_AR0: i32 =   34;
pub const M_RUN_TS_TH1: i32 =   -4; pub const M_RUN_TS_AR1: i32 =   51;
pub const M_RUN_TC_TH0: i32 =  139; pub const M_RUN_TC_AR0: i32 =  423;
pub const M_RUN_TC_TH1: i32 =  244; pub const M_RUN_TC_AR1: i32 =  162;
pub const M_RUN_TP_TH0: i32 =  275; pub const M_RUN_TP_AR0: i32 =  450;
pub const M_RUN_TP_TH1: i32 =   -6; pub const M_RUN_TP_AR1: i32 =  579;
pub const M_RUN_TM_TH0: i32 =  -68; pub const M_RUN_TM_AR0: i32 =   25;
pub const M_RUN_TM_TH1: i32 =    1; pub const M_RUN_TM_AR1: i32 =   64;
pub const M_RUN_TM_LR0: i32 =   15; pub const M_RUN_TM_LR1: i32 =   50;
pub const M_RUN_TM_LR2: i32 =   78;

pub const M_RUN_ES_TH0: i32 = -116; pub const M_RUN_ES_AR0: i32 =   31;
pub const M_RUN_ES_TH1: i32 =   43; pub const M_RUN_ES_AR1: i32 =   45;
pub const M_RUN_EC_TH0: i32 =  165; pub const M_RUN_EC_AR0: i32 =  222;
pub const M_RUN_EC_TH1: i32 =   30; pub const M_RUN_EC_AR1: i32 =  324;
pub const M_RUN_EP_TH0: i32 =  315; pub const M_RUN_EP_AR0: i32 =  857;
pub const M_RUN_EP_TH1: i32 =  109; pub const M_RUN_EP_AR1: i32 =  867;
pub const M_RUN_EM_TH0: i32 =  -14; pub const M_RUN_EM_AR0: i32 =  215;
pub const M_RUN_EM_TH1: i32 =   61; pub const M_RUN_EM_AR1: i32 =   73;
pub const M_RUN_EM_LR0: i32 =   35; pub const M_RUN_EM_LR1: i32 =   37;
pub const M_RUN_EM_LR2: i32 =   42;

pub const M_RUN_MS_TH0: i32 = -176; pub const M_RUN_MS_AR0: i32 =   14;
pub const M_RUN_MS_TH1: i32 = -141; pub const M_RUN_MS_AR1: i32 =   21;
pub const M_RUN_MC_TH0: i32 =   84; pub const M_RUN_MC_AR0: i32 =  172;
pub const M_RUN_MC_TH1: i32 =   37; pub const M_RUN_MC_AR1: i32 =  263;
pub const M_RUN_MP_TH0: i32 =    2; pub const M_RUN_MP_AR0: i32 =   15;
pub const M_RUN_MP_TH1: i32 = -197; pub const M_RUN_MP_AR1: i32 =   20;
pub const M_RUN_MM_TH0: i32 =  -27; pub const M_RUN_MM_AR0: i32 =  142;
pub const M_RUN_MM_TH1: i32 = -146; pub const M_RUN_MM_AR1: i32 =   27;
pub const M_RUN_MM_LR0: i32 =   51; pub const M_RUN_MM_LR1: i32 =   44;
pub const M_RUN_MM_LR2: i32 =   80;

// "Fast" / static (F) — decoder-only path, no probability updates
// for the threshold variants (no `*_AR_*` past the listed ones).
pub const F_RANK_TS_TH0: i32 = -116; pub const F_RANK_TS_AR0: i32 =   33;
pub const F_RANK_TS_TH1: i32 =  -78; pub const F_RANK_TS_AR1: i32 =   34;
pub const F_RANK_TC_TH0: i32 =   -2; pub const F_RANK_TC_AR0: i32 =  282;
pub const F_RANK_TC_TH1: i32 =   12; pub const F_RANK_TC_AR1: i32 =  274;
pub const F_RANK_TP_TH0: i32 =    4; pub const F_RANK_TP_AR0: i32 =  697;
pub const F_RANK_TP_TH1: i32 =   55; pub const F_RANK_TP_AR1: i32 = 1185;
pub const F_RANK_TM_LR0: i32 =   17; pub const F_RANK_TM_LR1: i32 =   14;
pub const F_RANK_TM_LR2: i32 =    1;

pub const F_RANK_ES_TH0: i32 = -177; pub const F_RANK_ES_AR0: i32 =   23;
pub const F_RANK_ES_TH1: i32 = -370; pub const F_RANK_ES_AR1: i32 =   11;
pub const F_RANK_EC_TH0: i32 =  -14; pub const F_RANK_EC_AR0: i32 =  271;
pub const F_RANK_EC_TH1: i32 =    3; pub const F_RANK_EC_AR1: i32 =  308;
pub const F_RANK_EP_TH0: i32 =   -3; pub const F_RANK_EP_AR0: i32 =  788;
pub const F_RANK_EP_TH1: i32 =  135; pub const F_RANK_EP_AR1: i32 = 1364;
pub const F_RANK_EM_LR0: i32 =   22; pub const F_RANK_EM_LR1: i32 =    6;
pub const F_RANK_EM_LR2: i32 =    4;

pub const F_RANK_MS_TH0: i32 = -254; pub const F_RANK_MS_AR0: i32 =   16;
pub const F_RANK_MS_TH1: i32 = -177; pub const F_RANK_MS_AR1: i32 =   20;
pub const F_RANK_MC_TH0: i32 =  -55; pub const F_RANK_MC_AR0: i32 =   73;
pub const F_RANK_MC_TH1: i32 =  -54; pub const F_RANK_MC_AR1: i32 =   74;
pub const F_RANK_MP_TH0: i32 =   -6; pub const F_RANK_MP_AR0: i32 =  575;
pub const F_RANK_MP_TH1: i32 = 1670; pub const F_RANK_MP_AR1: i32 = 1173;
pub const F_RANK_MM_LR0: i32 =   15; pub const F_RANK_MM_LR1: i32 =   10;
pub const F_RANK_MM_LR2: i32 =    7;

pub const F_RANK_PS_TH0: i32 = -126; pub const F_RANK_PS_AR0: i32 =   32;
pub const F_RANK_PS_TH1: i32 = -126; pub const F_RANK_PS_AR1: i32 =   32;
pub const F_RANK_PC_TH0: i32 =  -33; pub const F_RANK_PC_AR0: i32 =  120;
pub const F_RANK_PC_TH1: i32 =  -25; pub const F_RANK_PC_AR1: i32 =  157;
pub const F_RANK_PP_TH0: i32 =   -6; pub const F_RANK_PP_AR0: i32 =  585;
pub const F_RANK_PP_TH1: i32 =  150; pub const F_RANK_PP_AR1: i32 =  275;
pub const F_RANK_PM_LR0: i32 =   16; pub const F_RANK_PM_LR1: i32 =   11;
pub const F_RANK_PM_LR2: i32 =    5;

pub const F_RUN_TS_TH0: i32 =  -68; pub const F_RUN_TS_AR0: i32 =   38;
pub const F_RUN_TS_TH1: i32 = -112; pub const F_RUN_TS_AR1: i32 =   36;
pub const F_RUN_TC_TH0: i32 =   -4; pub const F_RUN_TC_AR0: i32 =  221;
pub const F_RUN_TC_TH1: i32 =  -13; pub const F_RUN_TC_AR1: i32 =  231;
pub const F_RUN_TP_TH0: i32 =    0; pub const F_RUN_TP_AR0: i32 =    0;
pub const F_RUN_TP_TH1: i32 =    0; pub const F_RUN_TP_AR1: i32 =    0;
pub const F_RUN_TM_LR0: i32 =   14; pub const F_RUN_TM_LR1: i32 =   18;
pub const F_RUN_TM_LR2: i32 =    0;

pub const F_RUN_ES_TH0: i32 =  -90; pub const F_RUN_ES_AR0: i32 =   45;
pub const F_RUN_ES_TH1: i32 =  -92; pub const F_RUN_ES_AR1: i32 =   44;
pub const F_RUN_EC_TH0: i32 =   -3; pub const F_RUN_EC_AR0: i32 =  325;
pub const F_RUN_EC_TH1: i32 =  -11; pub const F_RUN_EC_AR1: i32 =  341;
pub const F_RUN_EP_TH0: i32 =   24; pub const F_RUN_EP_AR0: i32 =  887;
pub const F_RUN_EP_TH1: i32 =   -4; pub const F_RUN_EP_AR1: i32 =  765;
pub const F_RUN_EM_LR0: i32 =   14; pub const F_RUN_EM_LR1: i32 =   15;
pub const F_RUN_EM_LR2: i32 =    3;

pub const F_RUN_MS_TH0: i32 = -275; pub const F_RUN_MS_AR0: i32 =   14;
pub const F_RUN_MS_TH1: i32 = -185; pub const F_RUN_MS_AR1: i32 =   22;
pub const F_RUN_MC_TH0: i32 =  -18; pub const F_RUN_MC_AR0: i32 =  191;
pub const F_RUN_MC_TH1: i32 =  -15; pub const F_RUN_MC_AR1: i32 =  241;
pub const F_RUN_MP_TH0: i32 =  -73; pub const F_RUN_MP_AR0: i32 =   54;
pub const F_RUN_MP_TH1: i32 = -214; pub const F_RUN_MP_AR1: i32 =   19;
pub const F_RUN_MM_LR0: i32 =    7; pub const F_RUN_MM_LR1: i32 =   15;
pub const F_RUN_MM_LR2: i32 =   10;

// ===================================================================
// QlfcStatisticalModel1 — adaptive / static decoder model.
// ===================================================================

/// `Rank.Mantissa[k]` for k in 0..8.
/// Indexing: state_model and char_model use `[outer * ALPHABET_SIZE + inner]`.
#[derive(Clone)]
pub struct RankMantissa {
    pub static_model: [i16; ALPHABET_SIZE],
    pub state_model:  Vec<i16>, // ALPHABET_SIZE * ALPHABET_SIZE
    pub char_model:   Vec<i16>, // ALPHABET_SIZE * ALPHABET_SIZE
}

impl RankMantissa {
    fn filled(v: i16) -> Self {
        Self {
            static_model: [v; ALPHABET_SIZE],
            state_model: vec![v; ALPHABET_SIZE * ALPHABET_SIZE],
            char_model:  vec![v; ALPHABET_SIZE * ALPHABET_SIZE],
        }
    }
}

#[derive(Clone)]
pub struct RankExponent {
    pub static_model: [i16; 8],
    pub state_model:  Vec<i16>, // ALPHABET_SIZE * 8
    pub char_model:   Vec<i16>, // ALPHABET_SIZE * 8
}

impl RankExponent {
    fn filled(v: i16) -> Self {
        Self {
            static_model: [v; 8],
            state_model: vec![v; ALPHABET_SIZE * 8],
            char_model:  vec![v; ALPHABET_SIZE * 8],
        }
    }
}

#[derive(Clone)]
pub struct RankEscape {
    pub static_model: [i16; ALPHABET_SIZE],
    pub state_model:  Vec<i16>, // ALPHABET_SIZE * ALPHABET_SIZE
    pub char_model:   Vec<i16>, // ALPHABET_SIZE * ALPHABET_SIZE
}

impl RankEscape {
    fn filled(v: i16) -> Self {
        Self {
            static_model: [v; ALPHABET_SIZE],
            state_model: vec![v; ALPHABET_SIZE * ALPHABET_SIZE],
            char_model:  vec![v; ALPHABET_SIZE * ALPHABET_SIZE],
        }
    }
}

#[derive(Clone)]
pub struct Rank {
    pub static_model: i16,
    pub state_model:  [i16; ALPHABET_SIZE],
    pub char_model:   [i16; ALPHABET_SIZE],
    pub exponent:     RankExponent,
    pub mantissa:     [RankMantissa; 8],
    pub escape:       RankEscape,
}

#[derive(Clone)]
pub struct RunMantissa {
    pub static_model: [i16; 32],
    pub state_model:  Vec<i16>, // ALPHABET_SIZE * 32
    pub char_model:   Vec<i16>, // ALPHABET_SIZE * 32
}

impl RunMantissa {
    fn filled(v: i16) -> Self {
        Self {
            static_model: [v; 32],
            state_model: vec![v; ALPHABET_SIZE * 32],
            char_model:  vec![v; ALPHABET_SIZE * 32],
        }
    }
}

#[derive(Clone)]
pub struct RunExponent {
    pub static_model: [i16; 32],
    pub state_model:  Vec<i16>, // ALPHABET_SIZE * 32
    pub char_model:   Vec<i16>, // ALPHABET_SIZE * 32
}

impl RunExponent {
    fn filled(v: i16) -> Self {
        Self {
            static_model: [v; 32],
            state_model: vec![v; ALPHABET_SIZE * 32],
            char_model:  vec![v; ALPHABET_SIZE * 32],
        }
    }
}

#[derive(Clone)]
pub struct Run {
    pub static_model: i16,
    pub state_model:  [i16; ALPHABET_SIZE],
    pub char_model:   [i16; ALPHABET_SIZE],
    pub exponent:     RunExponent,
    pub mantissa:     Vec<RunMantissa>, // 32 entries
}

#[derive(Clone)]
pub struct QlfcStatisticalModel1 {
    pub rank: Rank,
    pub run:  Run,

    pub mixer_of_rank:           Vec<ProbabilityMixer>, // ALPHABET_SIZE
    pub mixer_of_rank_exponent:  Vec<ProbabilityMixer>, // 8 * 8 -> [c*8 + b]
    pub mixer_of_rank_mantissa:  Vec<ProbabilityMixer>, // 8
    pub mixer_of_rank_escape:    Vec<ProbabilityMixer>, // ALPHABET_SIZE
    pub mixer_of_run:            Vec<ProbabilityMixer>, // ALPHABET_SIZE
    pub mixer_of_run_exponent:   Vec<ProbabilityMixer>, // 32 * 32 -> [c*32 + b]
    pub mixer_of_run_mantissa:   Vec<ProbabilityMixer>, // 32
}

// ===================================================================
// QlfcStatisticalModel2 — fast decoder model (no mixers, smaller).
// ===================================================================

#[derive(Clone)]
pub struct QlfcStatisticalModel2 {
    /// `Rank.Exponent[char][bit]` flattened as `[char * 8 + bit]`.
    pub rank_exponent: Vec<i16>, // ALPHABET_SIZE * 8
    /// `Rank.Mantissa[char][bit_rank_size][rank]` flattened as
    /// `[char * 8 * ALPHABET_SIZE + brs * ALPHABET_SIZE + rank]`.
    pub rank_mantissa: Vec<i16>, // ALPHABET_SIZE * 8 * ALPHABET_SIZE
    /// `Run.Exponent[char][bit]`.
    pub run_exponent: Vec<i16>,  // ALPHABET_SIZE * 32
    /// `Run.Mantissa[char][bit_run_size][context]`.
    pub run_mantissa: Vec<i16>,  // ALPHABET_SIZE * 32 * 32
}

impl QlfcStatisticalModel2 {
    /// `bsc_qlfc_init_model(QlfcStatisticalModel2*)` — Rank fields
    /// init to 4096 (13-bit half), Run fields init to 1024 (11-bit
    /// half), per upstream `bsc_qlfc_init_static_model`.
    pub fn boxed_init() -> Box<Self> {
        Box::new(Self {
            rank_exponent: vec![4096; ALPHABET_SIZE * 8],
            rank_mantissa: vec![4096; ALPHABET_SIZE * 8 * ALPHABET_SIZE],
            run_exponent:  vec![1024; ALPHABET_SIZE * 32],
            run_mantissa:  vec![1024; ALPHABET_SIZE * 32 * 32],
        })
    }
}

impl QlfcStatisticalModel1 {
    /// Build a fresh `Box<QlfcStatisticalModel1>` initialised to
    /// libbsc's static template (probabilities = 2048; mixers in
    /// their `Init()` state).
    pub fn boxed_init() -> Box<Self> {
        let mantissa: [RankMantissa; 8] = std::array::from_fn(|_| RankMantissa::filled(2048));
        let mut run_mantissa: Vec<RunMantissa> = Vec::with_capacity(32);
        for _ in 0..32 { run_mantissa.push(RunMantissa::filled(2048)); }

        let rank = Rank {
            static_model: 2048,
            state_model:  [2048; ALPHABET_SIZE],
            char_model:   [2048; ALPHABET_SIZE],
            exponent:     RankExponent::filled(2048),
            mantissa,
            escape:       RankEscape::filled(2048),
        };
        let run = Run {
            static_model: 2048,
            state_model:  [2048; ALPHABET_SIZE],
            char_model:   [2048; ALPHABET_SIZE],
            exponent:     RunExponent::filled(2048),
            mantissa:     run_mantissa,
        };

        let mixer_of_rank          = (0..ALPHABET_SIZE).map(|_| ProbabilityMixer::new()).collect();
        let mixer_of_rank_exponent = (0..8 * 8).map(|_| ProbabilityMixer::new()).collect();
        let mixer_of_rank_mantissa = (0..8).map(|_| ProbabilityMixer::new()).collect();
        let mixer_of_rank_escape   = (0..ALPHABET_SIZE).map(|_| ProbabilityMixer::new()).collect();
        let mixer_of_run           = (0..ALPHABET_SIZE).map(|_| ProbabilityMixer::new()).collect();
        let mixer_of_run_exponent  = (0..32 * 32).map(|_| ProbabilityMixer::new()).collect();
        let mixer_of_run_mantissa  = (0..32).map(|_| ProbabilityMixer::new()).collect();

        Box::new(Self {
            rank, run,
            mixer_of_rank,
            mixer_of_rank_exponent,
            mixer_of_rank_mantissa,
            mixer_of_rank_escape,
            mixer_of_run,
            mixer_of_run_exponent,
            mixer_of_run_mantissa,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxed_init_is_uniformly_2048() {
        let m = QlfcStatisticalModel1::boxed_init();
        assert_eq!(m.rank.static_model, 2048);
        assert!(m.rank.state_model.iter().all(|&v| v == 2048));
        assert!(m.rank.char_model.iter().all(|&v| v == 2048));
        assert!(m.rank.exponent.static_model.iter().all(|&v| v == 2048));
        assert!(m.rank.exponent.state_model.iter().all(|&v| v == 2048));
        assert!(m.rank.escape.static_model.iter().all(|&v| v == 2048));
        for k in 0..8 {
            assert!(m.rank.mantissa[k].static_model.iter().all(|&v| v == 2048));
            assert!(m.rank.mantissa[k].state_model.iter().all(|&v| v == 2048));
            assert!(m.rank.mantissa[k].char_model.iter().all(|&v| v == 2048));
        }
        for k in 0..32 {
            assert!(m.run.mantissa[k].static_model.iter().all(|&v| v == 2048));
        }
        assert_eq!(m.mixer_of_rank.len(), ALPHABET_SIZE);
        assert_eq!(m.mixer_of_run_exponent.len(), 32 * 32);
    }

    #[test]
    fn boxed_init_is_independent_per_call() {
        // Two calls must return distinct heap allocations; the
        // prior version used a single static template.
        let a = QlfcStatisticalModel1::boxed_init();
        let b = QlfcStatisticalModel1::boxed_init();
        let pa = &*a as *const _ as usize;
        let pb = &*b as *const _ as usize;
        assert_ne!(pa, pb);
    }
}
