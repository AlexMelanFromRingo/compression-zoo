// test_linux.cpp — Linux smoke test for libbsc + our block-framing.
//
// Simulates the encoder loop in src/BscCoder.cpp (chunk into blocks of
// _block_size, bsc_compress / bsc_store, write 28-byte header + payload)
// and the decoder loop (read header, bsc_block_info, read payload,
// bsc_decompress) end-to-end on a few input shapes.
//
// Build:
//   g++ -O2 -std=c++14 tests/test_linux.cpp \
//       -Iupstream/libbsc \
//       upstream/libbsc/adler32/adler32.cpp \
//       upstream/libbsc/bwt/bwt.cpp \
//       upstream/libbsc/coder/coder.cpp \
//       upstream/libbsc/coder/qlfc/qlfc.cpp \
//       upstream/libbsc/coder/qlfc/qlfc_model.cpp \
//       upstream/libbsc/filters/detectors.cpp \
//       upstream/libbsc/filters/preprocessing.cpp \
//       upstream/libbsc/libbsc/libbsc.cpp \
//       upstream/libbsc/lzp/lzp.cpp \
//       upstream/libbsc/platform/platform.cpp \
//       upstream/libbsc/st/st.cpp \
//       upstream/libbsc/bwt/libsais/libsais.c \
//       -o tests/test_linux

// libbsc.h uses size_t without including <stddef.h>, so we pull a
// standard header in first.
#include <cstddef>

extern "C" {
#include "../upstream/libbsc/libbsc.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int kFeatures = LIBBSC_FEATURE_FASTMODE;

namespace {

struct CParams {
  int lzp_hash, lzp_min_len, sorter, coder;
};

CParams params_for_level(int level) {
  if (level <= 1) return { 0,  0, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_FAST };
  if (level <= 3) return { 14, 64, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_FAST };
  if (level <= 5) return { 15, 72, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_STATIC };
  if (level <= 7) return { 16, 96, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_STATIC };
  return            { 17, 128, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE };
}

std::vector<unsigned char> framed_encode(
    const std::vector<unsigned char> &input,
    int level, unsigned block_size) {
  CParams pp = params_for_level(level);
  std::vector<unsigned char> out;
  std::vector<unsigned char> chunk_out(block_size + LIBBSC_HEADER_SIZE);
  size_t pos = 0;
  while (pos < input.size()) {
    int n = (int)std::min((size_t)block_size, input.size() - pos);
    int rc = bsc_compress(input.data() + pos, chunk_out.data(), n,
                          pp.lzp_hash, pp.lzp_min_len, pp.sorter, pp.coder, kFeatures);
    if (rc == LIBBSC_NOT_COMPRESSIBLE)
      rc = bsc_store(input.data() + pos, chunk_out.data(), n, kFeatures);
    if (rc < LIBBSC_NO_ERROR) { std::fprintf(stderr, "bsc_compress=%d\n", rc); std::exit(1); }
    out.insert(out.end(), chunk_out.begin(), chunk_out.begin() + rc);
    pos += (size_t)n;
  }
  return out;
}

std::vector<unsigned char> framed_decode(const std::vector<unsigned char> &compressed) {
  std::vector<unsigned char> out;
  std::vector<unsigned char> compressed_buf;
  std::vector<unsigned char> data_buf;
  size_t pos = 0;
  while (pos < compressed.size()) {
    if (pos + LIBBSC_HEADER_SIZE > compressed.size()) { std::fprintf(stderr, "truncated header\n"); std::exit(1); }
    int blockSize = 0, dataSize = 0;
    int rc = bsc_block_info(compressed.data() + pos, LIBBSC_HEADER_SIZE,
                            &blockSize, &dataSize, kFeatures);
    if (rc != LIBBSC_NO_ERROR) { std::fprintf(stderr, "bsc_block_info=%d\n", rc); std::exit(1); }
    if (pos + (size_t)blockSize > compressed.size()) { std::fprintf(stderr, "truncated block\n"); std::exit(1); }

    if ((size_t)dataSize > data_buf.size()) data_buf.resize((size_t)dataSize);
    rc = bsc_decompress(compressed.data() + pos, blockSize,
                        data_buf.data(), dataSize, kFeatures);
    if (rc != LIBBSC_NO_ERROR) { std::fprintf(stderr, "bsc_decompress=%d\n", rc); std::exit(1); }
    out.insert(out.end(), data_buf.begin(), data_buf.begin() + dataSize);
    pos += (size_t)blockSize;
  }
  return out;
}

bool round_trip(const std::vector<unsigned char> &input, int level, unsigned block_size, const char *label) {
  auto enc = framed_encode(input, level, block_size);
  auto dec = framed_decode(enc);
  if (dec.size() != input.size() || std::memcmp(dec.data(), input.data(), input.size()) != 0) {
    std::fprintf(stderr, "  FAIL %s level=%d block=%u in=%zu enc=%zu dec=%zu\n",
                 label, level, block_size, input.size(), enc.size(), dec.size());
    return false;
  }
  std::fprintf(stderr, "  PASS %s level=%d block=%u in=%zu enc=%zu (%.2f%%)\n",
               label, level, block_size, input.size(), enc.size(),
               100.0 * enc.size() / (input.size() ? input.size() : 1));
  return true;
}

}  // anon

int main() {
  if (bsc_init(kFeatures) != LIBBSC_NO_ERROR) {
    std::fprintf(stderr, "bsc_init failed\n");
    return 1;
  }

  unsigned passed = 0, total = 0;

  // 1. Empty input — block loop should simply not run.
  {
    std::vector<unsigned char> v;
    auto enc = framed_encode(v, 5, 1024 * 1024);
    auto dec = framed_decode(enc);
    ++total;
    if (dec.size() == 0 && enc.size() == 0) { ++passed; std::fprintf(stderr, "  PASS empty\n"); }
    else std::fprintf(stderr, "  FAIL empty enc=%zu dec=%zu\n", enc.size(), dec.size());
  }

  // 2. Tiny text.
  {
    const char *s = "Hello, libbsc! Hello, libbsc! Hello, libbsc!";
    std::vector<unsigned char> v(s, s + std::strlen(s));
    for (int lvl : {1, 5, 9}) { ++total; if (round_trip(v, lvl, 1024 * 1024, "tiny")) ++passed; }
  }

  // 3. 64 KiB pseudo-random.
  {
    std::vector<unsigned char> v(65536);
    unsigned x = 0xC0FFEEu;
    for (size_t i = 0; i < v.size(); i++) {
      x = x * 1664525u + 1013904223u;
      v[i] = (unsigned char)(x ^ (i & 0x1F));
    }
    for (int lvl : {1, 5}) { ++total; if (round_trip(v, lvl, 1024 * 1024, "rand-64K")) ++passed; }
  }

  // 4. 1 MiB highly-compressible (sparse).
  {
    std::vector<unsigned char> v(1024 * 1024, 0);
    for (size_t i = 7; i < v.size(); i += 257) v[i] = (unsigned char)(i & 0xFF);
    for (int lvl : {1, 5, 9}) { ++total; if (round_trip(v, lvl, 1024 * 1024, "sparse-1M")) ++passed; }
  }

  // 5. 4 MiB across multiple blocks (block_size = 1 MiB).
  {
    std::vector<unsigned char> v(4 * 1024 * 1024);
    unsigned x = 12345u;
    for (size_t i = 0; i < v.size(); i++) {
      x = x * 1103515245u + 12345u;
      v[i] = (unsigned char)((x >> 16) & 0xFF);
    }
    ++total;
    if (round_trip(v, 5, 1024 * 1024, "multi-block-4M")) ++passed;
  }

  std::fprintf(stderr, "==== %u / %u ====\n", passed, total);
  return (passed == total) ? 0 : 1;
}
