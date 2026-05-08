// BrotliCoder.cpp — wraps Brotli's streaming encoder/decoder as a 7-Zip
// codec. Pulls bytes through 64 KiB ping-pong buffers; brotli's
// BrotliEncoder*/BrotliDecoder* state objects are heap-allocated and
// destroyed in the same Code() call.

#include "StdAfx.h"

#include "../../sdk/CPP/Common/ComTry.h"
#include "BrotliCoder.h"

#include "../upstream/c/include/brotli/encode.h"
#include "../upstream/c/include/brotli/decode.h"

#include <vector>

namespace ZooBrotli {

namespace {

constexpr size_t kBufSize = 64 * 1024;

struct CProgress {
  ICompressProgressInfo *progress;
  UInt64 in_bytes;
  UInt64 out_bytes;
  unsigned tick;
  CProgress(ICompressProgressInfo *p) : progress(p), in_bytes(0), out_bytes(0), tick(0) {}
  void poke() {
    if (!progress) return;
    if ((++tick & 0x3F) == 0) {
      progress->SetRatioInfo(&in_bytes, &out_bytes);
    }
  }
};

// Write `n` bytes; loop until done. Return false on stream failure.
bool WriteAll(ISequentialOutStream *out, const uint8_t *buf, size_t n,
              CProgress *prog) {
  while (n > 0) {
    UInt32 done = 0;
    HRESULT hr = out->Write(buf, (UInt32)(n > 0xFFFF0000u ? 0xFFFF0000u : n), &done);
    if (hr != S_OK || done == 0) return false;
    buf += done;
    n   -= done;
    if (prog) { prog->out_bytes += done; prog->poke(); }
  }
  return true;
}

}  // anon

// ---------- Encoder -----------------------------------------------------

Z7_COM7F_IMF(CEncoder::SetCoderProperties(
    const PROPID *propIDs, const PROPVARIANT *props, UInt32 numProps))
{
  for (UInt32 i = 0; i < numProps; i++) {
    PROPID id = propIDs[i];
    const PROPVARIANT &p = props[i];
    if (id == NCoderPropID::kLevel) {
      UInt32 lvl = 0;
      if (p.vt == VT_UI4) lvl = p.ulVal;
      else if (p.vt == VT_UI8) lvl = (UInt32)p.uhVal.QuadPart;
      else continue;
      // 7-Zip 0..9  ->  brotli 0..11 (quality).
      // Reserve 10/11 for -mx9; -mx0 falls back to 1 (0 is faster but
      // the ratio drop is steep enough that users rarely want it).
      if (lvl == 0)       _quality = 1;
      else if (lvl >= 9)  _quality = 11;
      else if (lvl >= 7)  _quality = 9;
      else if (lvl >= 5)  _quality = 7;
      else if (lvl >= 3)  _quality = 5;
      else                _quality = 3;
    }
    // Other props (dict size etc.) — silently ignored.
  }
  return S_OK;
}

Z7_COM7F_IMF(CEncoder::Code(
    ISequentialInStream *inStream,
    ISequentialOutStream *outStream,
    const UInt64 * /*inSize*/, const UInt64 * /*outSize*/,
    ICompressProgressInfo *progress))
{
  COM_TRY_BEGIN
  BrotliEncoderState *state = BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
  if (!state) return E_OUTOFMEMORY;

  if (!BrotliEncoderSetParameter(state, BROTLI_PARAM_QUALITY,
                                 (uint32_t)_quality)) {
    BrotliEncoderDestroyInstance(state);
    return E_FAIL;
  }

  CProgress prog(progress);
  std::vector<uint8_t> inbuf(kBufSize);
  std::vector<uint8_t> outbuf(kBufSize);

  size_t avail_in = 0;
  const uint8_t *next_in = nullptr;
  bool input_eof = false;
  HRESULT err_hr = S_OK;

  while (!BrotliEncoderIsFinished(state)) {
    if (avail_in == 0 && !input_eof) {
      UInt32 got = 0;
      HRESULT hr = inStream->Read(inbuf.data(), (UInt32)inbuf.size(), &got);
      if (hr != S_OK) { err_hr = hr; break; }
      if (got == 0) {
        input_eof = true;
      } else {
        prog.in_bytes += got;
        prog.poke();
      }
      next_in = inbuf.data();
      avail_in = got;
    }

    BrotliEncoderOperation op =
        input_eof ? BROTLI_OPERATION_FINISH : BROTLI_OPERATION_PROCESS;
    size_t avail_out = outbuf.size();
    uint8_t *next_out = outbuf.data();

    if (!BrotliEncoderCompressStream(state, op,
                                     &avail_in, &next_in,
                                     &avail_out, &next_out, nullptr)) {
      err_hr = E_FAIL;
      break;
    }

    size_t produced = outbuf.size() - avail_out;
    if (produced > 0) {
      if (!WriteAll(outStream, outbuf.data(), produced, &prog)) {
        err_hr = E_FAIL; break;
      }
    }

    // Drain any data still buffered inside the encoder.
    while (BrotliEncoderHasMoreOutput(state)) {
      size_t take_size = 0;
      const uint8_t *take = BrotliEncoderTakeOutput(state, &take_size);
      if (take_size == 0) break;
      if (!WriteAll(outStream, take, take_size, &prog)) {
        err_hr = E_FAIL; break;
      }
    }
    if (err_hr != S_OK) break;
  }

  BrotliEncoderDestroyInstance(state);
  if (err_hr == S_OK && progress) {
    progress->SetRatioInfo(&prog.in_bytes, &prog.out_bytes);
  }
  return err_hr;
  COM_TRY_END
}

// ---------- Decoder -----------------------------------------------------

Z7_COM7F_IMF(CDecoder::Code(
    ISequentialInStream *inStream,
    ISequentialOutStream *outStream,
    const UInt64 * /*inSize*/, const UInt64 * /*outSize*/,
    ICompressProgressInfo *progress))
{
  COM_TRY_BEGIN
  BrotliDecoderState *state = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
  if (!state) return E_OUTOFMEMORY;

  CProgress prog(progress);
  std::vector<uint8_t> inbuf(kBufSize);
  std::vector<uint8_t> outbuf(kBufSize);

  size_t avail_in = 0;
  const uint8_t *next_in = nullptr;
  HRESULT err_hr = S_OK;
  BrotliDecoderResult result = BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT;

  while (result != BROTLI_DECODER_RESULT_SUCCESS) {
    if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT) {
      UInt32 got = 0;
      HRESULT hr = inStream->Read(inbuf.data(), (UInt32)inbuf.size(), &got);
      if (hr != S_OK) { err_hr = hr; break; }
      if (got == 0) {
        // Truncated stream. Brotli docs say SUCCESS is the only clean
        // terminator — anything else with no more input is corruption.
        err_hr = E_FAIL;
        break;
      }
      prog.in_bytes += got;
      prog.poke();
      next_in = inbuf.data();
      avail_in = got;
    }

    size_t avail_out = outbuf.size();
    uint8_t *next_out = outbuf.data();

    result = BrotliDecoderDecompressStream(state,
                                           &avail_in, &next_in,
                                           &avail_out, &next_out, nullptr);

    size_t produced = outbuf.size() - avail_out;
    if (produced > 0) {
      if (!WriteAll(outStream, outbuf.data(), produced, &prog)) {
        err_hr = E_FAIL; break;
      }
    }

    if (result == BROTLI_DECODER_RESULT_ERROR) {
      err_hr = E_FAIL;
      break;
    }
  }

  BrotliDecoderDestroyInstance(state);
  if (err_hr == S_OK && progress) {
    progress->SetRatioInfo(&prog.in_bytes, &prog.out_bytes);
  }
  return err_hr;
  COM_TRY_END
}

}  // namespace ZooBrotli
