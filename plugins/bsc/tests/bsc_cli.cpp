// bsc_cli.cpp — minimal CLI for benchmarking. Reads stdin, encodes
// in fixed-size blocks (using the same framing as src/BscCoder.cpp),
// writes the concatenated framed output to stdout. Or decodes.

#include <cstddef>
extern "C" {
#include "../upstream/libbsc/libbsc.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

static const int kFeatures = LIBBSC_FEATURE_FASTMODE;

struct CParams { int lzp_hash, lzp_min_len, sorter, coder; };

static CParams params_for_level(int level) {
  if (level <= 1) return { 0,  0, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_FAST };
  if (level <= 3) return { 14, 64, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_FAST };
  if (level <= 5) return { 15, 72, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_STATIC };
  if (level <= 7) return { 16, 96, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_STATIC };
  return            { 17, 128, LIBBSC_BLOCKSORTER_BWT, LIBBSC_CODER_QLFC_ADAPTIVE };
}

static std::string slurp_stdin() {
  std::string s;
  char buf[64 * 1024];
  while (auto n = std::fread(buf, 1, sizeof(buf), stdin)) s.append(buf, n);
  return s;
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s e|d [level=5] [block=25M]\n", argv[0]); return 2; }
  bsc_init(kFeatures);

  if (argv[1][0] == 'e') {
    int level = (argc > 2) ? std::atoi(argv[2]) : 5;
    if (level < 1) level = 1;
    if (level > 9) level = 9;
    int block_size = (argc > 3) ? std::atoi(argv[3]) : 25 * 1024 * 1024;
    if (block_size < 1024) block_size = 1024;

    CParams pp = params_for_level(level);
    std::string in = slurp_stdin();
    std::vector<unsigned char> out_buf((size_t)block_size + LIBBSC_HEADER_SIZE);
    size_t pos = 0;
    while (pos < in.size()) {
      int n = (int)std::min((size_t)block_size, in.size() - pos);
      int rc = bsc_compress((const unsigned char*)in.data() + pos, out_buf.data(), n,
                            pp.lzp_hash, pp.lzp_min_len, pp.sorter, pp.coder, kFeatures);
      if (rc == LIBBSC_NOT_COMPRESSIBLE)
        rc = bsc_store((const unsigned char*)in.data() + pos, out_buf.data(), n, kFeatures);
      if (rc < LIBBSC_NO_ERROR) { std::fprintf(stderr, "bsc_compress=%d\n", rc); return 1; }
      std::fwrite(out_buf.data(), 1, (size_t)rc, stdout);
      pos += (size_t)n;
    }
    return 0;
  }

  if (argv[1][0] == 'd') {
    std::string in = slurp_stdin();
    size_t pos = 0;
    std::vector<unsigned char> compressed;
    std::vector<unsigned char> data;
    while (pos < in.size()) {
      if (pos + LIBBSC_HEADER_SIZE > in.size()) { std::fprintf(stderr, "trunc header\n"); return 1; }
      int blockSize = 0, dataSize = 0;
      int rc = bsc_block_info((const unsigned char*)in.data() + pos, LIBBSC_HEADER_SIZE, &blockSize, &dataSize, kFeatures);
      if (rc != LIBBSC_NO_ERROR) { std::fprintf(stderr, "block_info=%d\n", rc); return 1; }
      if (pos + (size_t)blockSize > in.size()) { std::fprintf(stderr, "trunc block\n"); return 1; }
      if ((size_t)blockSize > compressed.size()) compressed.resize((size_t)blockSize);
      if ((size_t)dataSize  > data.size()) data.resize((size_t)dataSize);
      std::memcpy(compressed.data(), in.data() + pos, (size_t)blockSize);
      rc = bsc_decompress(compressed.data(), blockSize, data.data(), dataSize, kFeatures);
      if (rc != LIBBSC_NO_ERROR) { std::fprintf(stderr, "decompress=%d\n", rc); return 1; }
      std::fwrite(data.data(), 1, (size_t)dataSize, stdout);
      pos += (size_t)blockSize;
    }
    return 0;
  }
  std::fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
