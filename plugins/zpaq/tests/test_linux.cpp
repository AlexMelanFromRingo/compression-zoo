// test_linux.cpp — Linux-side smoke test for the ZPAQ wrapping logic.
//
// We can't run Windows DLLs from this WSL2 instance (no binfmt_misc
// interop registered), so this test exercises the parts of the plugin
// that *are* portable: libzpaq itself and our libzpaq::Reader/Writer
// adapters mirroring the design in src/ZpaqCoder.cpp.
//
// It does NOT cover:
//   - the COM exports (CreateObject / GetMethodProperty / GetNumberOfMethods)
//   - the ICompressCoder vtable layout
// Those need a Windows-side run of tests/test_zpaq.exe (or 7z.exe with the
// installed plugin).
//
// Build (Linux):
//   g++ -O2 -std=c++14 tests/test_linux.cpp upstream/libzpaq.cpp \
//       -DNOJIT -Dunix -o tests/test_linux
// Run:
//   ./tests/test_linux

#include "../upstream/libzpaq.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// libzpaq requires this global to be defined by the host.
namespace libzpaq {
  void error(const char *m) {
    std::fprintf(stderr, "libzpaq error: %s\n", m ? m : "?");
    std::exit(2);
  }
}

namespace {

class VecReader : public libzpaq::Reader {
public:
  const unsigned char *p; size_t pos, n;
  VecReader(const std::vector<unsigned char> &v) : p(v.data()), pos(0), n(v.size()) {}
  int get() override { return (pos < n) ? p[pos++] : -1; }
  int read(char *buf, int k) override {
    int avail = (int)((n - pos) < (size_t)k ? (n - pos) : (size_t)k);
    std::memcpy(buf, p + pos, avail); pos += (size_t)avail; return avail;
  }
};

class VecWriter : public libzpaq::Writer {
public:
  std::vector<unsigned char> buf;
  void put(int c) override { buf.push_back((unsigned char)c); }
  void write(const char *b, int k) override { buf.insert(buf.end(), (const unsigned char*)b, (const unsigned char*)b + k); }
};

bool round_trip(const std::vector<unsigned char> &input, const char *method) {
  VecReader rin(input);
  VecWriter compressed_out;
  libzpaq::compress(&rin, &compressed_out, method);

  std::vector<unsigned char> compressed(compressed_out.buf);
  VecReader cin(compressed);
  VecWriter decompressed_out;
  libzpaq::decompress(&cin, &decompressed_out);

  if (decompressed_out.buf.size() != input.size()) {
    std::fprintf(stderr, "  size mismatch: got %zu, expected %zu\n",
                 decompressed_out.buf.size(), input.size());
    return false;
  }
  if (std::memcmp(decompressed_out.buf.data(), input.data(), input.size()) != 0) {
    std::fprintf(stderr, "  content mismatch\n");
    return false;
  }
  std::fprintf(stderr, "  PASS  method=%s  in=%zu  out=%zu  (%.2f%%)\n",
               method, input.size(), compressed.size(),
               100.0 * compressed.size() / (input.size() ? input.size() : 1));
  return true;
}

}  // anon

int main() {
  unsigned passed = 0, total = 0;

  // Tiny text.
  {
    const char *s = "Hello, ZPAQ! Hello, ZPAQ! Hello, ZPAQ!";
    std::vector<unsigned char> v(s, s + std::strlen(s));
    for (const char *m : {"0", "1", "2", "3", "4", "5"}) {
      ++total;
      std::fprintf(stderr, "[tiny text]\n");
      if (round_trip(v, m)) ++passed;
    }
  }

  // 8 KiB pseudo-random with periodic structure.
  {
    std::vector<unsigned char> v(8192);
    unsigned x = 0xC0FFEEu;
    for (size_t i = 0; i < v.size(); i++) {
      x = x * 1664525u + 1013904223u;
      v[i] = (unsigned char)(x ^ (i & 0x1F));
    }
    for (const char *m : {"1", "2", "3"}) {
      ++total;
      std::fprintf(stderr, "[8 KiB pseudo-random]\n");
      if (round_trip(v, m)) ++passed;
    }
  }

  // 64 KiB highly-compressible (mostly zeros with rare bumps).
  {
    std::vector<unsigned char> v(65536, 0);
    for (size_t i = 7; i < v.size(); i += 257) v[i] = (unsigned char)(i & 0xFF);
    for (const char *m : {"1", "3", "5"}) {
      ++total;
      std::fprintf(stderr, "[64 KiB sparse]\n");
      if (round_trip(v, m)) ++passed;
    }
  }

  std::fprintf(stderr, "==== %u / %u ====\n", passed, total);
  return (passed == total) ? 0 : 1;
}
