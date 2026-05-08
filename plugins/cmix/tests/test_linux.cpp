// test_linux.cpp — Linux smoke test for the CMIX wrapper.
//
// CMIX has tons of file-scope globals (paq8::y, bpos, blpos, c4, x4,
// ...) that are not reset between Predictor instances; round-tripping
// in a single process therefore does not work. The 7-Zip plugin DLL
// is fine for the typical workflow of "encode in one 7z.exe
// invocation, decode in another", because each process starts with
// zero-initialised globals.
//
// We test that here too: this binary takes a mode argument and either
// encodes from stdin to stdout, or decodes from stdin to stdout. The
// shell driver below the cpp file pipes between two fresh invocations.
//
// Build:
//   g++ -O2 -std=c++14 tests/test_linux.cpp <cmix sources> -o tests/test_linux
//
// Run:
//   tests/run.sh

#include "../upstream/src/predictor.h"
#include "../upstream/src/coder/encoder.h"
#include "../upstream/src/coder/decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

char *dictionary_path = nullptr;

static const unsigned long long kMinVocabFileSize = 10000ULL;

static void write_header(std::ostream *os, unsigned long long length) {
  for (int i = 4; i >= 0; --i) {
    char c = (char)(length >> (8 * i));
    if (i == 4) c &= 0x7F;
    os->put(c);
  }
  if (length >= kMinVocabFileSize) {
    for (int i = 0; i < 32; ++i) os->put((char)0xFF);
  }
}

static bool read_header(std::istream *is, unsigned long long *length, std::vector<bool> *vocab) {
  unsigned long long len = 0;
  for (int i = 4; i >= 0; --i) {
    int c = is->get();
    if (c == EOF) return false;
    unsigned char uc = (unsigned char)c;
    if (i == 4) uc &= 0x7F;
    len = (len << 8) | uc;
  }
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

static int do_encode() {
  std::string input((std::istreambuf_iterator<char>(std::cin)),
                    std::istreambuf_iterator<char>());
  std::ostringstream encoded;
  unsigned long long length = (unsigned long long)input.size();
  write_header(&encoded, length);
  std::vector<bool> vocab(256, true);
  Predictor p(vocab);
  Encoder e(&encoded, &p);
  for (size_t pos = 0; pos < input.size(); ++pos) {
    unsigned char c = (unsigned char)input[pos];
    for (int j = 7; j >= 0; --j) e.Encode((c >> j) & 1);
  }
  e.Flush();
  std::string out = encoded.str();
  std::cout.write(out.data(), (std::streamsize)out.size());
  std::fprintf(stderr, "[cmix encode] in=%zu enc=%zu\n", input.size(), out.size());
  return 0;
}

static int do_decode() {
  std::string input((std::istreambuf_iterator<char>(std::cin)),
                    std::istreambuf_iterator<char>());
  std::istringstream encoded(input);
  unsigned long long length = 0;
  std::vector<bool> vocab;
  if (!read_header(&encoded, &length, &vocab)) {
    std::fprintf(stderr, "header read failed\n");
    return 1;
  }
  Predictor p(vocab);
  Decoder d(&encoded, &p);
  std::string out;
  out.reserve((size_t)length);
  for (unsigned long long pos = 0; pos < length; ++pos) {
    int byte = 1;
    while (byte < 256) byte += byte + d.Decode();
    out.push_back((char)(byte & 0xFF));
  }
  std::cout.write(out.data(), (std::streamsize)out.size());
  std::fprintf(stderr, "[cmix decode] enc=%zu dec=%zu\n", input.size(), out.size());
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <encode|decode>  (data via stdin/stdout)\n", argv[0]);
    return 2;
  }
  if (std::strcmp(argv[1], "encode") == 0) return do_encode();
  if (std::strcmp(argv[1], "decode") == 0) return do_decode();
  std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
  return 2;
}
