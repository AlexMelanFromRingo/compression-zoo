// BscCoder.cpp — wraps libbsc as a 7-Zip codec.
//
// Block-based: encoder reads up to _block_size bytes, calls bsc_compress
// (or bsc_store on NOT_COMPRESSIBLE), and emits a 28-byte header +
// payload. The decoder peeks at each header via bsc_block_info to learn
// block/data sizes and recovers the original.
//
// libbsc itself is single-threaded in our build (no LIBBSC_FEATURE_*
// extras at construction time); CUDA / OpenMP are off.

#include "StdAfx.h"

#include "../../sdk/CPP/Common/ComTry.h"

#include "BscCoder.h"

extern "C" {
#include "../upstream/libbsc/libbsc.h"
}

#include <vector>
#include <cstring>

namespace ZooBsc {

// Default per-block size = 25 MB. Configurable via SetCoderProperties
// (kDictionarySize). Values are clamped to a sensible range.
static const unsigned kDefaultBlockSize = 25u * 1024u * 1024u;
static const unsigned kMinBlockSize     = 1u  * 1024u * 1024u;
static const unsigned kMaxBlockSize     = 1024u * 1024u * 1024u; // 1 GiB

// Map 7-Zip's level (1..9) to a (lzpHashSize, lzpMinLen, blockSorter, coder)
// quadruple. Higher levels enable LZP and choose stronger sorters.
struct CParams {
  int lzp_hash;     // 0 = no LZP
  int lzp_min_len;  // 0 = no LZP
  int sorter;       // LIBBSC_BLOCKSORTER_*
  int coder;        // LIBBSC_CODER_*
};

static CParams params_for_level(int level) {
  if (level <= 1) return { 0,  0, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_FAST };
  if (level <= 3) return { 14, 64, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_FAST };
  if (level <= 5) return { 15, 72, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_STATIC };
  if (level <= 7) return { 16, 96, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_STATIC };
  return            { 17, 128, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE };
}

static const int kFeatures = LIBBSC_FEATURE_FASTMODE; // no MT, no CUDA

// Run bsc_init exactly once per process.
namespace {
struct CInit {
  CInit() { bsc_init(kFeatures); }
};
static CInit g_init;
}

CEncoder::CEncoder() : _level(5), _block_size(kDefaultBlockSize) {}

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
      if (lvl < 1) lvl = 1;
      if (lvl > 9) lvl = 9;
      _level = (int)lvl;
    } else if (id == NCoderPropID::kDictionarySize) {
      UInt64 ds = 0;
      if      (p.vt == VT_UI4) ds = p.ulVal;
      else if (p.vt == VT_UI8) ds = p.uhVal.QuadPart;
      else continue;
      if (ds < kMinBlockSize) ds = kMinBlockSize;
      if (ds > kMaxBlockSize) ds = kMaxBlockSize;
      _block_size = (unsigned)ds;
    }
  }
  return S_OK;
}

namespace {

// Helper to fully read up to `want` bytes from a 7-Zip stream.
// Returns the number actually read (may be < want at EOF).
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

}  // anon

Z7_COM7F_IMF(CEncoder::Code(
    ISequentialInStream *inStream,
    ISequentialOutStream *outStream,
    const UInt64 * /*inSize*/, const UInt64 * /*outSize*/,
    ICompressProgressInfo *progress))
{
  COM_TRY_BEGIN
  CParams pp = params_for_level(_level);

  std::vector<Byte> in_buf;
  std::vector<Byte> out_buf;

  in_buf.resize(_block_size);
  out_buf.resize((size_t)_block_size + LIBBSC_HEADER_SIZE);

  UInt64 total_in = 0;
  UInt64 total_out = 0;

  while (true) {
    UInt32 nIn = 0;
    HRESULT hr = read_full(inStream, in_buf.data(), _block_size, &nIn);
    if (hr != S_OK) return hr;
    if (nIn == 0) break;

    int n = (int)nIn;
    int rc = bsc_compress(in_buf.data(), out_buf.data(), n,
        pp.lzp_hash, pp.lzp_min_len, pp.sorter, pp.coder, kFeatures);

    if (rc == LIBBSC_NOT_COMPRESSIBLE) {
      // libbsc says "data won't compress; store it verbatim".
      rc = bsc_store(in_buf.data(), out_buf.data(), n, kFeatures);
    }
    if (rc < LIBBSC_NO_ERROR) return E_FAIL;

    hr = write_full(outStream, out_buf.data(), (UInt32)rc);
    if (hr != S_OK) return hr;

    total_in  += (UInt64)n;
    total_out += (UInt64)rc;
    if (progress) progress->SetRatioInfo(&total_in, &total_out);
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
  std::vector<Byte> compressed;
  std::vector<Byte> decompressed;

  UInt64 total_in = 0;
  UInt64 total_out = 0;

  Byte header[LIBBSC_HEADER_SIZE];

  while (true) {
    UInt32 hdr_got = 0;
    HRESULT hr = read_full(inStream, header, LIBBSC_HEADER_SIZE, &hdr_got);
    if (hr != S_OK) return hr;
    if (hdr_got == 0) break; // clean EOF
    if (hdr_got != LIBBSC_HEADER_SIZE) return E_FAIL;

    int blockSize = 0, dataSize = 0;
    int rc = bsc_block_info(header, LIBBSC_HEADER_SIZE,
                            &blockSize, &dataSize, kFeatures);
    if (rc != LIBBSC_NO_ERROR) return E_FAIL;
    if (blockSize <= LIBBSC_HEADER_SIZE || dataSize < 0) return E_FAIL;

    if ((size_t)blockSize > compressed.size()) compressed.resize((size_t)blockSize);
    if ((size_t)dataSize  > decompressed.size()) decompressed.resize((size_t)dataSize);

    std::memcpy(compressed.data(), header, LIBBSC_HEADER_SIZE);
    UInt32 want = (UInt32)(blockSize - LIBBSC_HEADER_SIZE);
    UInt32 rest_got = 0;
    hr = read_full(inStream, compressed.data() + LIBBSC_HEADER_SIZE, want, &rest_got);
    if (hr != S_OK) return hr;
    if (rest_got != want) return E_FAIL;

    rc = bsc_decompress(compressed.data(), blockSize,
                        decompressed.data(), dataSize, kFeatures);
    if (rc != LIBBSC_NO_ERROR) return E_FAIL;

    hr = write_full(outStream, decompressed.data(), (UInt32)dataSize);
    if (hr != S_OK) return hr;

    total_in  += (UInt64)blockSize;
    total_out += (UInt64)dataSize;
    if (progress) progress->SetRatioInfo(&total_in, &total_out);
  }
  return S_OK;
  COM_TRY_END
}

}  // namespace ZooBsc
