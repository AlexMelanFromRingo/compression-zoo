//! LZMA2 decoder — port of `7zip/C/Lzma2Dec.c`.
//!
//! LZMA2 is a thin chunk framing on top of LZMA with the additional ability
//! to encode raw (uncompressed) chunks and reset state between chunks. The
//! one-byte control flag at the start of each chunk encodes:
//!
//! ```text
//! 00000000               — end of stream
//! 00000001 U U           — uncompressed, reset dictionary
//! 00000010 U U           — uncompressed, no reset
//! 100uuuuu U U P P       — LZMA, no reset
//! 101uuuuu U U P P       — LZMA, reset state
//! 110uuuuu U U P P S     — LZMA, reset state + new properties
//! 111uuuuu U U P P S     — LZMA, reset state + new properties + reset dictionary
//! ```

use crate::lzma_dec::{
    Decoder as LzmaDecoder, Error, FinishMode, Properties, Status, PROPS_SIZE,
};

const LZMA2_LCLP_MAX: u8 = 4;
const LZMA2_CONTROL_COPY_RESET_DIC: u8 = 1;

#[inline]
fn is_uncompressed_state(control: u8) -> bool {
    (control & (1 << 7)) == 0
}

fn dic_size_from_prop(prop: u8) -> u32 {
    (2u32 | (prop as u32 & 1)) << ((prop / 2) + 11)
}

/// Translate LZMA2 dictionary-property byte to a 5-byte LZMA properties
/// header (LCLP_MAX = 4 + dictionary size derived from `prop`).
fn old_props(prop: u8) -> Result<[u8; PROPS_SIZE], Error> {
    if prop > 40 {
        return Err(Error::Unsupported);
    }
    let dic_size = if prop == 40 { 0xFFFF_FFFFu32 } else { dic_size_from_prop(prop) };
    let mut out = [0u8; PROPS_SIZE];
    out[0] = LZMA2_LCLP_MAX;
    out[1..5].copy_from_slice(&dic_size.to_le_bytes());
    Ok(out)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum State2 {
    Control,
    Unpack0,
    Unpack1,
    Pack0,
    Pack1,
    Prop,
    Data,
    DataCont,
    Finished,
    Error,
}

#[derive(Debug)]
pub struct Lzma2Decoder {
    decoder: LzmaDecoder,
    state: State2,
    control: u8,
    need_init_level: u8,
    is_extra_mode: bool,
    pack_size: u32,
    unpack_size: u32,
}

impl Lzma2Decoder {
    /// Create a new LZMA2 decoder from a single dictionary-property byte.
    pub fn new(prop: u8) -> Result<Self, Error> {
        let props = old_props(prop)?;
        let lzma_prop = Properties::parse(&props)?;
        Ok(Self {
            decoder: LzmaDecoder::new(lzma_prop),
            state: State2::Control,
            control: 0,
            need_init_level: 0xE0,
            is_extra_mode: false,
            pack_size: 0,
            unpack_size: 0,
        })
    }

    /// Reset the decoder, equivalent to `Lzma2Dec_Init`.
    pub fn init(&mut self) {
        self.state = State2::Control;
        self.need_init_level = 0xE0;
        self.is_extra_mode = false;
        self.unpack_size = 0;
        self.decoder.init();
    }

    /// Streaming decode into the internal dictionary, analogue of
    /// `Lzma2Dec_DecodeToDic`.
    pub fn decode_to_dic(
        &mut self,
        dic_limit: usize,
        src: &[u8],
        consumed: &mut usize,
        finish_mode: FinishMode,
    ) -> Result<Status, Error> {
        let in_size = src.len();
        *consumed = 0;

        while self.state != State2::Error {
            if self.state == State2::Finished {
                return Ok(Status::FinishedWithMark);
            }
            let dic_pos = self.decoder.dic_pos();
            if dic_pos == dic_limit && finish_mode == FinishMode::Any {
                return Ok(Status::NotFinished);
            }
            if self.state != State2::Data && self.state != State2::DataCont {
                if *consumed == in_size {
                    return Ok(Status::NeedsMoreInput);
                }
                let b = src[*consumed];
                *consumed += 1;
                self.update_state(b)?;
                if dic_pos == dic_limit && self.state != State2::Finished {
                    self.state = State2::Error;
                    return Err(Error::Data);
                }
                continue;
            }

            let in_cur = in_size - *consumed;
            let mut out_cur = dic_limit - dic_pos;
            let mut cur_finish_mode = FinishMode::Any;
            if out_cur >= self.unpack_size as usize {
                out_cur = self.unpack_size as usize;
                cur_finish_mode = FinishMode::End;
            }

            if is_uncompressed_state(self.control) {
                if in_cur == 0 {
                    return Ok(Status::NeedsMoreInput);
                }
                if self.state == State2::Data {
                    let init_dic = self.control == LZMA2_CONTROL_COPY_RESET_DIC;
                    self.decoder.init_dic_and_state(init_dic, false);
                }
                let copy = in_cur.min(out_cur);
                if copy == 0 {
                    self.state = State2::Error;
                    return Err(Error::Data);
                }
                self.decoder
                    .append_uncompressed(&src[*consumed..*consumed + copy]);
                *consumed += copy;
                self.unpack_size -= copy as u32;
                self.state = if self.unpack_size == 0 {
                    State2::Control
                } else {
                    State2::DataCont
                };
                continue;
            }

            // LZMA chunk
            if self.state == State2::Data {
                let init_dic = self.control >= 0xE0;
                let init_state = self.control >= 0xA0;
                self.decoder.init_dic_and_state(init_dic, init_state);
                self.state = State2::DataCont;
            }
            let mut chunk_in = in_cur.min(self.pack_size as usize);
            let mut chunk_consumed = 0usize;
            let res = self.decoder.decode_to_dic(
                dic_pos + out_cur,
                &src[*consumed..*consumed + chunk_in],
                &mut chunk_consumed,
                cur_finish_mode,
            );
            *consumed += chunk_consumed;
            chunk_in -= chunk_consumed;
            self.pack_size -= chunk_consumed as u32;
            let new_dic = self.decoder.dic_pos();
            let produced = new_dic - dic_pos;
            self.unpack_size -= produced as u32;
            let status = match res {
                Ok(s) => s,
                Err(e) => {
                    self.state = State2::Error;
                    return Err(e);
                }
            };
            if status == Status::NeedsMoreInput {
                if self.pack_size == 0 {
                    self.state = State2::Error;
                    return Err(Error::Data);
                }
                return Ok(Status::NeedsMoreInput);
            }
            if chunk_consumed == 0 && produced == 0 {
                if status != Status::MaybeFinishedWithoutMark
                    || self.unpack_size != 0
                    || self.pack_size != 0
                {
                    self.state = State2::Error;
                    return Err(Error::Data);
                }
                self.state = State2::Control;
            }
            let _ = chunk_in;
        }

        self.state = State2::Error;
        Err(Error::Data)
    }

    fn update_state(&mut self, b: u8) -> Result<(), Error> {
        match self.state {
            State2::Control => {
                self.is_extra_mode = false;
                self.control = b;
                if b == 0 {
                    self.state = State2::Finished;
                    return Ok(());
                }
                if is_uncompressed_state(b) {
                    if b == LZMA2_CONTROL_COPY_RESET_DIC {
                        self.need_init_level = 0xC0;
                    } else if b > 2 || self.need_init_level == 0xE0 {
                        self.state = State2::Error;
                        return Err(Error::Data);
                    }
                } else {
                    if b < self.need_init_level {
                        self.state = State2::Error;
                        return Err(Error::Data);
                    }
                    self.need_init_level = 0;
                    self.unpack_size = ((b & 0x1F) as u32) << 16;
                }
                self.state = State2::Unpack0;
            }
            State2::Unpack0 => {
                self.unpack_size |= (b as u32) << 8;
                self.state = State2::Unpack1;
            }
            State2::Unpack1 => {
                self.unpack_size |= b as u32;
                self.unpack_size += 1;
                self.state = if is_uncompressed_state(self.control) {
                    State2::Data
                } else {
                    State2::Pack0
                };
            }
            State2::Pack0 => {
                self.pack_size = (b as u32) << 8;
                self.state = State2::Pack1;
            }
            State2::Pack1 => {
                self.pack_size |= b as u32;
                self.pack_size += 1;
                self.state = if (self.control & 0x40) != 0 {
                    State2::Prop
                } else {
                    State2::Data
                };
            }
            State2::Prop => {
                let mut bb = b;
                if bb >= 9 * 5 * 5 {
                    self.state = State2::Error;
                    return Err(Error::Data);
                }
                let lc = bb % 9;
                bb /= 9;
                let pb = bb / 5;
                let lp = bb % 5;
                if lc + lp > LZMA2_LCLP_MAX {
                    self.state = State2::Error;
                    return Err(Error::Data);
                }
                self.decoder.set_lc_lp_pb(lc, lp, pb);
                self.state = State2::Data;
            }
            _ => {
                self.state = State2::Error;
                return Err(Error::Data);
            }
        }
        Ok(())
    }
}

/// One-shot decode of an LZMA2 stream.  `prop` is the dictionary-property
/// byte (e.g. found in 7z headers), `src` the LZMA2-framed payload.
pub fn decode_one_shot(prop: u8, src: &[u8]) -> Result<Vec<u8>, Error> {
    let mut dec = Lzma2Decoder::new(prop)?;
    let mut out = Vec::new();
    let mut cursor = 0;
    let mut buf = vec![0u8; 4096];
    loop {
        if dec.decoder.dic_pos() == dec.decoder.dic_buf_size() {
            // Flush full buffer
            out.extend_from_slice(dec.decoder.dic_mut());
            // We can't reset dic_pos here without knowing the LZMA semantics —
            // instead we rely on the LZMA decoder's wrap behaviour.
        }
        let prev_pos = dec.decoder.dic_pos();
        let limit = dec.decoder.dic_buf_size();
        let mut consumed = 0;
        let status = dec.decode_to_dic(
            limit,
            &src[cursor..],
            &mut consumed,
            FinishMode::Any,
        )?;
        cursor += consumed;
        let produced = dec.decoder.dic_pos() - prev_pos;
        out.extend_from_slice(&dec.decoder.dic_mut()[prev_pos..prev_pos + produced]);
        let _ = buf;
        match status {
            Status::FinishedWithMark | Status::MaybeFinishedWithoutMark => return Ok(out),
            Status::NeedsMoreInput => return Err(Error::InputEof),
            Status::NotFinished | Status::NotSpecified => {
                if consumed == 0 && produced == 0 {
                    return Err(Error::Data);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dic_prop_size() {
        // prop=0: smallest dictionary (4 KiB).
        assert_eq!(dic_size_from_prop(0), 1 << 12);
        // prop=39: (2 | 1) << (19+11) = 3 << 30.
        assert_eq!(dic_size_from_prop(39), 3u32 << 30);
        // prop==40 is the special 0xFFFFFFFF case (handled in `old_props`).
        let p = old_props(40).unwrap();
        assert_eq!(p, [4, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn end_of_stream_only() {
        // LZMA2 stream with just the end-of-stream byte.
        let out = decode_one_shot(0, &[0u8]).unwrap();
        assert!(out.is_empty());
    }
}
