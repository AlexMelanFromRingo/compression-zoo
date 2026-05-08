// CmixCoder.cpp — wraps CMIX (Byron Knoll) as a 7-Zip codec.
//
// CMIX has no streaming chunked API: it needs the full input buffered up
// front (the header records the uncompressed length) and produces a single
// arithmetic-coded stream.  We buffer the whole thing in std::ostringstream
// during encode and feed bytes through Predictor + Encoder.  Decode is the
// mirror image.
//
// CMIX is heavy: peak RSS easily exceeds 25 GB on big inputs and encode
// time is hours/GB.  This codec is a niche "ratio over everything" choice;
// document this where it shows in 7-Zip's UI.
//
// We use the CMIX "no preprocessing" path equivalent to its `cmix -n`
// invocation (no English-language dictionary, no text mode, no PHDA9
// preprocessor).  vocab is set to all-256 unconditionally so we don't have
// to scan the input twice.

#include "StdAfx.h"

#include "../../sdk/CPP/Common/ComTry.h"

#include "CmixCoder.h"

#include "../upstream/src/predictor.h"
#include "../upstream/src/coder/encoder.h"
#include "../upstream/src/coder/decoder.h"

#include <sstream>
#include <vector>
#include <string>

namespace ZooCmix {

// Minimum input size that triggers vocab encoding in CMIX's runner.
static const unsigned long long kMinVocabFileSize = 10000ULL;

namespace {

HRESULT read_full(ISequentialInStream *in, Byte *buf, UInt32 want, UInt32 *got) {
  UInt32 total = 0;
  while (total < want) {
    UInt32 step = 0;
    HRESULT hr = in->Read(buf + total, want - total, &step);
    if (hr != S_OK) return hr;
    if (step == 0) break;
    total += step;
  }
  *got = total;
  return S_OK;
}

HRESULT write_full(ISequentialOutStream *out, const Byte *buf, UInt32 n) {
  UInt32 written = 0;
  while (written < n) {
    UInt32 step = 0;
    HRESULT hr = out->Write(buf + written, n - written, &step);
    if (hr != S_OK) return hr;
    if (step == 0) return E_FAIL;
    written += step;
  }
  return S_OK;
}

// Drain in-stream into a std::string.
HRESULT drain(ISequentialInStream *in, std::string *out) {
  out->clear();
  Byte buf[64 * 1024];
  for (;;) {
    UInt32 got = 0;
    HRESULT hr = in->Read(buf, sizeof(buf), &got);
    if (hr != S_OK) return hr;
    if (got == 0) break;
    out->append((const char*)buf, got);
  }
  return S_OK;
}

void write_header(std::ostream *os, unsigned long long length) {
  // 5-byte big-endian length; top bit of MSB is the "dictionary used"
  // flag, which we always leave zero.
  for (int i = 4; i >= 0; --i) {
    char c = (char)(length >> (8 * i));
    if (i == 4) c &= 0x7F;
    os->put(c);
  }
  // 32 bytes of vocab bitmap; we always say "all 256 bytes possible".
  if (length >= kMinVocabFileSize) {
    for (int i = 0; i < 32; ++i) os->put((char)0xFF);
  }
}

bool read_header(std::istream *is, unsigned long long *length, std::vector<bool> *vocab) {
  unsigned long long len = 0;
  for (int i = 4; i >= 0; --i) {
    int c = is->get();
    if (c == EOF) return false;
    unsigned char uc = (unsigned char)c;
    if (i == 4) uc &= 0x7F; // drop "dictionary used" flag
    len = (len << 8) | uc;
  }
  // Loop above doesn't shift by 8*i — let's recompute the right way:
  *length = len;
  vocab->assign(256, false);
  if (len < kMinVocabFileSize) {
    std::fill(vocab->begin(), vocab->end(), true);
  } else {
    for (int i = 0; i < 32; ++i) {
      int c = is->get();
      if (c == EOF) return false;
      unsigned char uc = (unsigned char)c;
      for (int j = 0; j < 8; ++j) {
        if (uc & (1 << j)) (*vocab)[(size_t)(i * 8 + j)] = true;
      }
    }
  }
  return true;
}

}  // anon

// fxcmv1.cpp references this global from runner.cpp. Define it here.
}  // close namespace temporarily so the global lives at file scope
extern "C" { /* not C-linkage; just a barrier */ }
char *dictionary_path = nullptr;
namespace ZooCmix {

Z7_COM7F_IMF(CEncoder::SetCoderProperties(
    const PROPID * /*propIDs*/, const PROPVARIANT * /*props*/, UInt32 /*numProps*/))
{
  // CMIX has no levels; accept and ignore.
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::Code(
    ISequentialInStream *inStream,
    ISequentialOutStream *outStream,
    const UInt64 * /*inSize*/, const UInt64 * /*outSize*/,
    ICompressProgressInfo *progress))
{
  COM_TRY_BEGIN
  std::string input;
  HRESULT hr = drain(inStream, &input);
  if (hr != S_OK) return hr;

  std::ostringstream encoded;

  // Write the same 5-byte length + 32-byte vocab header CMIX itself uses.
  // We always claim "all-256" vocab so we don't need a second pass.
  unsigned long long length = (unsigned long long)input.size();
  write_header(&encoded, length);

  std::vector<bool> vocab(256, true);
  Predictor p(vocab);
  Encoder e(&encoded, &p);

  for (size_t pos = 0; pos < input.size(); ++pos) {
    unsigned char c = (unsigned char)input[pos];
    for (int j = 7; j >= 0; --j) {
      e.Encode((c >> j) & 1);
    }
    if (progress && (pos & 0xFFF) == 0) {
      UInt64 in_so_far = (UInt64)pos;
      UInt64 out_so_far = (UInt64)encoded.tellp();
      progress->SetRatioInfo(&in_so_far, &out_so_far);
    }
  }
  e.Flush();

  std::string out_data = encoded.str();
  hr = write_full(outStream, (const Byte *)out_data.data(),
                  (UInt32)out_data.size());
  if (hr != S_OK) return hr;
  if (progress) {
    UInt64 in_total  = (UInt64)input.size();
    UInt64 out_total = (UInt64)out_data.size();
    progress->SetRatioInfo(&in_total, &out_total);
  }
  return S_OK;
  COM_TRY_END
}

Z7_COM7F_IMF(CDecoder::Code(
    ISequentialInStream *inStream,
    ISequentialOutStream *outStream,
    const UInt64 * /*inSize*/, const UInt64 * /*outSize*/,
    ICompressProgressInfo *progress))
{
  COM_TRY_BEGIN
  std::string input;
  HRESULT hr = drain(inStream, &input);
  if (hr != S_OK) return hr;

  std::istringstream encoded(input);
  unsigned long long length = 0;
  std::vector<bool> vocab;
  if (!read_header(&encoded, &length, &vocab)) return E_FAIL;

  Predictor p(vocab);
  Decoder d(&encoded, &p);

  std::vector<Byte> output;
  output.reserve((size_t)length);

  for (unsigned long long pos = 0; pos < length; ++pos) {
    int byte = 1;
    while (byte < 256) byte += byte + d.Decode();
    output.push_back((Byte)(byte & 0xFF));

    if (progress && (pos & 0xFFF) == 0) {
      UInt64 in_so_far  = (UInt64)encoded.tellg();
      UInt64 out_so_far = (UInt64)pos;
      progress->SetRatioInfo(&in_so_far, &out_so_far);
    }
  }

  hr = write_full(outStream, output.data(), (UInt32)output.size());
  if (hr != S_OK) return hr;
  if (progress) {
    UInt64 in_total  = (UInt64)input.size();
    UInt64 out_total = (UInt64)output.size();
    progress->SetRatioInfo(&in_total, &out_total);
  }
  return S_OK;
  COM_TRY_END
}

}  // namespace ZooCmix
