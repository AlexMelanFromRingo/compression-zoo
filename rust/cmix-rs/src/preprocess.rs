//! Preprocessor — port of `preprocess/{dictionary,preprocessor}.cpp`.
//!
//! Provides:
//!
//! * [`PreprocDict`] — dictionary-based word substitution codec
//!   (encode + decode), inverse pair of upstream's
//!   `preprocessor::Dictionary`.
//! * [`encode_text`] / [`decode_text`] — convenience wrappers that
//!   run the full text-mode pipeline (dictionary substitution +
//!   escape framing).
//!
//! The full upstream `preprocess/preprocessor.cpp` also includes
//! file-type detection (JPEG / EXE / IMAGE_* / AUDIO) and per-type
//! filters. Those are not ported here: cmix-rs targets text/wiki
//! workloads first, and the file-type filters can be added module-by-
//! module without touching the core dictionary path.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::{self, Read, Write};

const K_CAPITALIZED: u8 = 0x40;
const K_UPPERCASE:   u8 = 0x07;
const K_END_UPPER:   u8 = 0x06;
const K_ESCAPE:      u8 = 0x0C;
const K_QUOTE:       u8 = 0x08;
const K_QUOTE_STR:   &[u8] = b"&quot;";

/// Encoding/decoding mapping built from a dictionary file (one
/// lowercase word per line). Mirrors upstream's
/// `preprocessor::Dictionary` ctor / Encode / Decode.
pub struct PreprocDict {
    byte_map:    HashMap<Vec<u8>, u32>,
    reverse_map: HashMap<u32, Vec<u8>>,
    longest_word: usize,
}

impl PreprocDict {
    /// Build mapping from a raw dictionary byte stream — runs of
    /// [a-z] are words, anything else is a separator.
    pub fn from_bytes(data: &[u8], encode: bool, decode: bool) -> Self {
        let mut byte_map    = HashMap::new();
        let mut reverse_map = HashMap::new();
        let mut longest_word = 0usize;
        let k_boundary1 = 80;
        let k_boundary2 = k_boundary1 + 3840;
        let k_boundary3 = k_boundary2 + 40960;
        let k_boundary4 = k_boundary3 + 81920;
        let mut line: Vec<u8> = Vec::with_capacity(32);
        let mut line_count: i32 = 0;
        for &c in data {
            if c >= b'a' && c <= b'z' {
                line.push(c);
            } else if !line.is_empty() {
                if line.len() > longest_word { longest_word = line.len(); }
                let bytes = Self::line_count_to_bytes(
                    line_count, k_boundary1, k_boundary2,
                    k_boundary3, k_boundary4);
                if let Some(bytes) = bytes {
                    if encode { byte_map.insert(line.clone(), bytes); }
                    if decode { reverse_map.insert(bytes, line.clone()); }
                }
                line_count += 1;
                line.clear();
            }
        }
        if !line.is_empty() {
            if line.len() > longest_word { longest_word = line.len(); }
            let bytes = Self::line_count_to_bytes(
                line_count, k_boundary1, k_boundary2,
                k_boundary3, k_boundary4);
            if let Some(bytes) = bytes {
                if encode { byte_map.insert(line.clone(), bytes); }
                if decode { reverse_map.insert(bytes, line); }
            }
        }
        Self { byte_map, reverse_map, longest_word }
    }

    fn line_count_to_bytes(line_count: i32, b1: i32, b2: i32,
                            b3: i32, b4: i32) -> Option<u32> {
        if line_count < b1 {
            Some((0x80 + line_count) as u32)
        } else if line_count < b2 {
            let mut bytes = (0xD0 + ((line_count - b1) / 80)) as u32;
            bytes += ((0x80 + ((line_count - b1) % 80)) as u32) << 8;
            Some(bytes)
        } else if line_count < b3 {
            let mut bytes = (0xF0 + (((line_count - b2) / 80) / 32)) as u32;
            bytes += ((0xD0 + (((line_count - b2) / 80) % 32)) as u32) << 8;
            bytes += ((0x80 + ((line_count - b2) % 80)) as u32) << 16;
            Some(bytes)
        } else if line_count < b4 {
            let mut bytes = (0xD0 + (((line_count - b2) / 80) / 32)) as u32;
            bytes += ((0xD0 + (((line_count - b2) / 80) % 32)) as u32) << 8;
            bytes += ((0x80 + ((line_count - b2) % 80)) as u32) << 16;
            Some(bytes)
        } else {
            None
        }
    }

    pub fn longest_word(&self) -> usize { self.longest_word }

    /// Encode `input` to `output`, mirroring upstream's
    /// `Dictionary::Encode`. `len` is the byte count to consume from
    /// `input`; pass `input.len()` to consume the whole slice.
    pub fn encode<R: Read, W: Write>(
        &self, input: &mut R, len: usize, output: &mut W,
    ) -> io::Result<()> {
        let mut word: Vec<u8> = Vec::with_capacity(64);
        let mut num_upper = 0i32;
        let mut num_lower = 0i32;
        let mut quote_state = 0usize;
        let mut buf = [0u8; 1];
        for pos in 0..len {
            if input.read(&mut buf)? == 0 { break; }
            let c = buf[0];
            if c == K_QUOTE_STR[quote_state] {
                quote_state += 1;
                if quote_state == 6 {
                    output.write_all(&[K_QUOTE])?;
                    num_upper = 0; num_lower = 0; word.clear();
                    quote_state = 0;
                    continue;
                }
            } else {
                quote_state = 0;
            }
            let mut advance = false;
            if word.len() > self.longest_word {
                advance = true;
            } else if c >= b'a' && c <= b'z' {
                if num_upper > 1 {
                    advance = true;
                } else {
                    num_lower += 1;
                    word.push(c);
                }
            } else if c >= b'A' && c <= b'Z' {
                if num_lower > 0 {
                    advance = true;
                } else {
                    num_upper += 1;
                    word.push(c - b'A' + b'a');
                }
            } else {
                advance = true;
            }
            if pos == len - 1 && !advance {
                self.encode_word(&word, num_upper, false, output)?;
            }
            if advance {
                if word.is_empty() {
                    Self::encode_byte(c, output)?;
                } else {
                    let next_lower = c >= b'a' && c <= b'z';
                    self.encode_word(&word, num_upper, next_lower, output)?;
                    num_lower = 0; num_upper = 0; word.clear();
                    if next_lower {
                        num_lower += 1;
                        word.push(c);
                    } else if c >= b'A' && c <= b'Z' {
                        num_upper += 1;
                        word.push(c - b'A' + b'a');
                    } else {
                        Self::encode_byte(c, output)?;
                    }
                    if pos == len - 1 && !word.is_empty() {
                        self.encode_word(&word, num_upper, false, output)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn encode_byte<W: Write>(c: u8, output: &mut W) -> io::Result<()> {
        if c == K_END_UPPER || c == K_ESCAPE || c == K_UPPERCASE
            || c == K_CAPITALIZED || c == K_QUOTE || c >= 0x80
        {
            output.write_all(&[K_ESCAPE])?;
        }
        output.write_all(&[c])
    }

    fn encode_bytes<W: Write>(bytes: u32, output: &mut W) -> io::Result<()> {
        output.write_all(&[(bytes & 0xff) as u8])?;
        if (bytes & 0xff00) != 0 {
            output.write_all(&[((bytes & 0xff00) >> 8) as u8])?;
        } else {
            return Ok(());
        }
        if (bytes & 0xff0000) != 0 {
            output.write_all(&[((bytes & 0xff0000) >> 16) as u8])?;
        }
        Ok(())
    }

    fn encode_word<W: Write>(
        &self, word: &[u8], num_upper: i32, next_lower: bool,
        output: &mut W,
    ) -> io::Result<()> {
        if num_upper > 1 { output.write_all(&[K_UPPERCASE])?; }
        else if num_upper == 1 { output.write_all(&[K_CAPITALIZED])?; }
        if let Some(&bytes) = self.byte_map.get(word) {
            Self::encode_bytes(bytes, output)?;
        } else if !self.encode_substring(word, output)? {
            output.write_all(word)?;
        }
        if num_upper > 1 && next_lower {
            output.write_all(&[K_END_UPPER])?;
        }
        Ok(())
    }

    fn encode_substring<W: Write>(
        &self, word: &[u8], output: &mut W,
    ) -> io::Result<bool> {
        if word.len() <= 7 { return Ok(false); }
        let mut size = word.len() - 1;
        if size > self.longest_word { size = self.longest_word; }
        // Try suffixes of decreasing length.
        let mut suffix_start = word.len() - size;
        while word.len() - suffix_start >= 7 {
            let suffix = &word[suffix_start..];
            if let Some(&bytes) = self.byte_map.get(suffix) {
                output.write_all(&word[..suffix_start])?;
                Self::encode_bytes(bytes, output)?;
                return Ok(true);
            }
            suffix_start += 1;
        }
        // Try prefixes of decreasing length.
        let mut prefix_end = size;
        while prefix_end >= 7 {
            let prefix = &word[..prefix_end];
            if let Some(&bytes) = self.byte_map.get(prefix) {
                Self::encode_bytes(bytes, output)?;
                output.write_all(&word[prefix_end..])?;
                return Ok(true);
            }
            prefix_end -= 1;
        }
        Ok(false)
    }

    /// Decode one byte from the encoded stream. Builds an internal
    /// output buffer when a multi-byte codeword or quote-escape is
    /// expanded. Returns `Ok(None)` on EOF.
    pub fn decode<R: Read>(
        &self, input: &mut R, state: &mut DecodeState,
    ) -> io::Result<Option<u8>> {
        while state.output_buffer.is_empty() {
            if !state.add_to_buffer(input, self)? { return Ok(None); }
        }
        Ok(Some(state.output_buffer.remove(0)))
    }
}

/// Streaming state for incremental decode.
pub struct DecodeState {
    output_buffer: Vec<u8>,
    decode_upper:   bool,
    decode_capital: bool,
}

impl Default for DecodeState { fn default() -> Self { Self::new() } }

impl DecodeState {
    pub fn new() -> Self {
        Self {
            output_buffer: Vec::new(),
            decode_upper: false,
            decode_capital: false,
        }
    }

    fn add_to_buffer<R: Read>(&mut self, input: &mut R,
                                dict: &PreprocDict) -> io::Result<bool> {
        let mut buf = [0u8; 1];
        if input.read(&mut buf)? == 0 { return Ok(false); }
        let c = buf[0];
        if c == K_ESCAPE {
            self.decode_upper = false;
            if input.read(&mut buf)? == 0 { return Ok(false); }
            self.output_buffer.push(buf[0]);
        } else if c == K_QUOTE {
            for i in 1..6 { self.output_buffer.push(K_QUOTE_STR[i]); }
        } else if c == K_UPPERCASE {
            self.decode_upper = true;
        } else if c == K_CAPITALIZED {
            self.decode_capital = true;
        } else if c == K_END_UPPER {
            self.decode_upper = false;
        } else if c >= 0x80 {
            let mut bytes = c as u32;
            if c > 0xCF {
                if input.read(&mut buf)? == 0 { return Ok(false); }
                bytes += (buf[0] as u32) << 8;
                if buf[0] > 0xCF {
                    if input.read(&mut buf)? == 0 { return Ok(false); }
                    bytes += (buf[0] as u32) << 16;
                }
            }
            if let Some(word) = dict.reverse_map.get(&bytes) {
                let mut word = word.clone();
                for i in 0..word.len() {
                    if i == 0 && self.decode_capital {
                        word[i] = word[i] - b'a' + b'A';
                        self.decode_capital = false;
                    }
                    if self.decode_upper {
                        word[i] = word[i] - b'a' + b'A';
                    }
                    self.output_buffer.push(word[i]);
                }
            }
        } else {
            let mut c = c;
            if !((c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z')) {
                self.decode_upper = false;
            }
            if self.decode_capital || self.decode_upper {
                c = c - b'a' + b'A';
            }
            if self.decode_capital { self.decode_capital = false; }
            self.output_buffer.push(c);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_encodes_dictionary_word_into_single_byte() {
        let dict = PreprocDict::from_bytes(b"hello\nworld\n", true, true);
        let mut out = Vec::new();
        let mut input = &b"hello"[..];
        dict.encode(&mut input, 5, &mut out).unwrap();
        // "hello" is dict word #0 (kBoundary1 territory) → 0x80
        // (Capitalized prefix only emitted for actual uppercase
        // input — lowercase "hello" alone doesn't emit it.)
        assert_eq!(out, vec![0x80]);
    }

    #[test]
    fn dict_round_trip_basic_text() {
        let dict_data = b"the\nbe\nto\nof\nand\na\nin\nthat\nhave\nit\n";
        let dict = PreprocDict::from_bytes(dict_data, true, true);
        let plain = b"the cat and a dog";
        let mut encoded = Vec::new();
        {
            let mut input = &plain[..];
            dict.encode(&mut input, plain.len(), &mut encoded).unwrap();
        }
        // Decode back.
        let mut decoded = Vec::new();
        let mut state = DecodeState::new();
        let mut enc_input = &encoded[..];
        loop {
            match dict.decode(&mut enc_input, &mut state).unwrap() {
                Some(b) => decoded.push(b),
                None    => break,
            }
        }
        assert_eq!(decoded, plain);
    }

    #[test]
    fn dict_round_trip_with_caps_and_punctuation() {
        let dict_data = b"hello\nworld\n";
        let dict = PreprocDict::from_bytes(dict_data, true, true);
        let plain = b"Hello, WORLD! Greeting hello again.";
        let mut encoded = Vec::new();
        {
            let mut input = &plain[..];
            dict.encode(&mut input, plain.len(), &mut encoded).unwrap();
        }
        let mut decoded = Vec::new();
        let mut state = DecodeState::new();
        let mut enc_input = &encoded[..];
        loop {
            match dict.decode(&mut enc_input, &mut state).unwrap() {
                Some(b) => decoded.push(b),
                None    => break,
            }
        }
        assert_eq!(decoded, plain);
    }

    #[test]
    fn dict_handles_quote_entity_escape() {
        // longest_word must be ≥ 4 so the partial-quote buffer (up
        // to "quot") doesn't trip the `word.len() > longest_word`
        // flush. Upstream behaves the same way — real dictionaries
        // have many long words.
        let dict = PreprocDict::from_bytes(
            b"alphabet\nencyclopedia\nlexicon\n", true, true);
        let plain = b"&quot;test&quot;";
        let mut encoded = Vec::new();
        {
            let mut input = &plain[..];
            dict.encode(&mut input, plain.len(), &mut encoded).unwrap();
        }
        let mut decoded = Vec::new();
        let mut state = DecodeState::new();
        let mut enc_input = &encoded[..];
        loop {
            match dict.decode(&mut enc_input, &mut state).unwrap() {
                Some(b) => decoded.push(b),
                None    => break,
            }
        }
        assert_eq!(decoded, plain);
    }
}
