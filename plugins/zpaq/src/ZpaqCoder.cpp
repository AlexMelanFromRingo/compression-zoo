// ZpaqCoder.cpp — wraps libzpaq's compress/decompress as a 7-Zip codec.
//
// libzpaq exposes two abstract base classes (Reader, Writer) that we adapt
// to 7-Zip's COM streams. compress/decompress are blocking calls that pull
// bytes through these adapters.
//
// libzpaq signals errors by calling a global libzpaq::error(const char*).
// We define it to throw a CZpaqError; CEncoder/CDecoder::Code() catches it
// and returns a HRESULT.

#include "StdAfx.h"

// Note: MyInitGuid.h is included in DllExportsCompress.cpp; including it
// again here would emit duplicate GUID definitions.
#include "../../sdk/CPP/Common/ComTry.h"

#include "ZpaqCoder.h"

#include "../upstream/libzpaq.h"

#include <stdexcept>
#include <string>

namespace ZooZpaq {

// ---------- Error plumbing ----------------------------------------------

struct CZpaqError : std::runtime_error {
  explicit CZpaqError(const char *m) : std::runtime_error(m) {}
};

// ---------- Stream adapters --------------------------------------------

namespace {

// Calls progress->SetRatioInfo() periodically with running totals.
struct CProgress {
  ICompressProgressInfo *progress;
  UInt64 in_bytes;
  UInt64 out_bytes;
  unsigned tick;
  CProgress(ICompressProgressInfo *p) : progress(p), in_bytes(0), out_bytes(0), tick(0) {}
  void poke() {
    if (!progress) return;
    if ((++tick & 0x3FF) == 0) {
      progress->SetRatioInfo(&in_bytes, &out_bytes);
    }
  }
};

class CInStreamReader : public libzpaq::Reader {
public:
  ISequentialInStream *stream;
  CProgress *prog;
  bool eof;
  CInStreamReader(ISequentialInStream *s, CProgress *p)
      : stream(s), prog(p), eof(false) {}

  // Single-byte path. Slow; libzpaq prefers read().
  int get() Z7_override {
    Byte b;
    UInt32 got = 0;
    HRESULT hr = stream->Read(&b, 1, &got);
    if (hr != S_OK) throw CZpaqError("input stream read failed");
    if (got == 0) { eof = true; return -1; }
    if (prog) { prog->in_bytes += 1; prog->poke(); }
    return b;
  }

  // Bulk path. May read fewer than n bytes; that's libzpaq-legal.
  int read(char *buf, int n) Z7_override {
    int total = 0;
    while (total < n) {
      UInt32 got = 0;
      HRESULT hr = stream->Read(buf + total, (UInt32)(n - total), &got);
      if (hr != S_OK) throw CZpaqError("input stream read failed");
      if (got == 0) { eof = true; break; }
      total += (int)got;
    }
    if (prog) { prog->in_bytes += (UInt64)total; prog->poke(); }
    return total;
  }
};

class COutStreamWriter : public libzpaq::Writer {
public:
  ISequentialOutStream *stream;
  CProgress *prog;
  COutStreamWriter(ISequentialOutStream *s, CProgress *p) : stream(s), prog(p) {}

  void put(int c) Z7_override {
    Byte b = (Byte)c;
    UInt32 done = 0;
    HRESULT hr = stream->Write(&b, 1, &done);
    if (hr != S_OK || done != 1) throw CZpaqError("output stream write failed");
    if (prog) { prog->out_bytes += 1; prog->poke(); }
  }

  void write(const char *buf, int n) Z7_override {
    int total = 0;
    while (total < n) {
      UInt32 done = 0;
      HRESULT hr = stream->Write(buf + total, (UInt32)(n - total), &done);
      if (hr != S_OK || done == 0) throw CZpaqError("output stream write failed");
      total += (int)done;
    }
    if (prog) { prog->out_bytes += (UInt64)n; prog->poke(); }
  }
};

}  // anon

// ---------- Encoder -----------------------------------------------------

Z7_COM7F_IMF(CEncoder::SetCoderProperties(
    const PROPID *propIDs, const PROPVARIANT *props, UInt32 numProps))
{
  for (UInt32 i = 0; i < numProps; i++) {
    const PROPVARIANT &p = props[i];
    PROPID id = propIDs[i];
    if (id == NCoderPropID::kLevel) {
      UInt32 lvl = 0;
      if (p.vt == VT_UI4) lvl = p.ulVal;
      else if (p.vt == VT_UI8) lvl = (UInt32)p.uhVal.QuadPart;
      else continue;
      // 7-Zip levels are 1..9; ZPAQ has 0..5 useful range.
      if (lvl == 0)  _level = 1;
      else if (lvl >= 9) _level = 5;
      else if (lvl >= 7) _level = 4;
      else if (lvl >= 5) _level = 3;
      else if (lvl >= 3) _level = 2;
      else _level = 1;
    }
    // Other props (dict size, etc.) — silently accept and ignore for now.
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
  CProgress prog(progress);
  CInStreamReader  rd(inStream,  &prog);
  COutStreamWriter wr(outStream, &prog);
  char method[2] = { (char)('0' + _level), 0 };
  try {
    libzpaq::compress(&rd, &wr, method);
  } catch (const CZpaqError &) {
    return E_FAIL;
  } catch (const std::bad_alloc &) {
    return E_OUTOFMEMORY;
  } catch (...) {
    return E_FAIL;
  }
  if (progress) progress->SetRatioInfo(&prog.in_bytes, &prog.out_bytes);
  return S_OK;
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
  CProgress prog(progress);
  CInStreamReader  rd(inStream,  &prog);
  COutStreamWriter wr(outStream, &prog);
  try {
    libzpaq::decompress(&rd, &wr);
  } catch (const CZpaqError &) {
    return E_FAIL;
  } catch (const std::bad_alloc &) {
    return E_OUTOFMEMORY;
  } catch (...) {
    return E_FAIL;
  }
  if (progress) progress->SetRatioInfo(&prog.in_bytes, &prog.out_bytes);
  return S_OK;
  COM_TRY_END
}

}  // namespace ZooZpaq

// ---------- Required libzpaq global -------------------------------------

namespace libzpaq {
  void error(const char *msg) {
    throw ZooZpaq::CZpaqError(msg ? msg : "libzpaq: unknown error");
  }
}
