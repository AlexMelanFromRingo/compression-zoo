//! Concrete `State`-machine implementations bundled with CMIX.
//!
//!   * [`run_map`] — the 256-state run-length tracker.
//!   * [`nonstationary`] — the 256-state PAQ-derived nonstationary
//!     map. The 512-byte transition table is hard-coded verbatim
//!     from upstream `nonstationary.cpp`.

pub mod run_map;
pub mod nonstationary;
