//! CMIX state-machine trait — port of `state.h`. Each implementation
//! is a finite state machine over an 8- or 9-bit state index that
//! tracks the model's belief about the next bit. Two concrete
//! implementations are bundled (see [`crate::states::nonstationary`]
//! and [`crate::states::run_map`]).

#![allow(dead_code)]

/// State machine over an integer state index.
pub trait State {
    /// Initial bit-1 probability for `state`, in `[0, 1]`.
    fn init_probability(&self, _state: i32) -> f32 { 0.5 }
    /// Transition `(state, bit) -> next_state`.
    fn next(&self, _state: i32, _bit: i32) -> i32 { 0 }
}
