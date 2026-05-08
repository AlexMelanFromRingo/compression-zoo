// test_linux.cpp — Linux-side smoke test for the Brotli wrapping logic.
//
// We can't run Windows DLLs from this WSL2 instance (no binfmt_misc
// interop registered), so this test exercises the parts of the plugin
// that *are* portable: brotli's streaming encoder / decoder mirroring
// the buffer-pump in src/BrotliCoder.cpp.
//
// It does NOT cover:
//   - the COM exports (CreateObject / GetMethodProperty / GetNumberOfMethods)
//   - the ICompressCoder vtable layout
// Those need a Windows-side run of tests/test_brotli.exe (or 7z.exe
// with the installed plugin).
//
// Build (Linux):
//   g++ -O2 -std=c++14 tests/test_linux.cpp -lbrotlienc -lbrotlidec \
//       -lbrotlicommon -o tests/test_linux
// (Needs libbrotli-dev installed.)
// Run:
//   ./tests/test_linux

#include <brotli/encode.h>
#include <brotli/decode.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

namespace {

constexpr size_t kBufSize = 64 * 1024;

bool encode(const std::vector<uint8_t> &input, int quality,
            std::vector<uint8_t> *out) {
  BrotliEncoderState *st = BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
  if (!st) return false;
  if (!BrotliEncoderSetParameter(st, BROTLI_PARAM_QUALITY, (uint32_t)quality)) {
    BrotliEncoderDestroyInstance(st); return false;
  }

  const uint8_t *next_in = input.data();
  size_t avail_in = input.size();
  std::vector<uint8_t> outbuf(kBufSize);

  while (!BrotliEncoderIsFinished(st)) {
    size_t avail_out = outbuf.size();
    uint8_t *next_out = outbuf.data();
    BrotliEncoderOperation op = BROTLI_OPERATION_FINISH;
    if (!BrotliEncoderCompressStream(st, op, &avail_in, &next_in,
                                     &avail_out, &next_out, nullptr)) {
      BrotliEncoderDestroyInstance(st); return false;
    }
    out->insert(out->end(), outbuf.data(), outbuf.data() + (outbuf.size() - avail_out));

    while (BrotliEncoderHasMoreOutput(st)) {
      size_t take_size = 0;
      const uint8_t *take = BrotliEncoderTakeOutput(st, &take_size);
      out->insert(out->end(), take, take + take_size);
    }
  }
  BrotliEncoderDestroyInstance(st);
  return true;
}

bool decode(const std::vector<uint8_t> &input, std::vector<uint8_t> *out) {
  BrotliDecoderState *st = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
  if (!st) return false;

  const uint8_t *next_in = input.data();
  size_t avail_in = input.size();
  std::vector<uint8_t> outbuf(kBufSize);
  BrotliDecoderResult result = BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT;

  while (result != BROTLI_DECODER_RESULT_SUCCESS) {
    if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT && avail_in == 0) {
      BrotliDecoderDestroyInstance(st); return false;
    }
    size_t avail_out = outbuf.size();
    uint8_t *next_out = outbuf.data();
    result = BrotliDecoderDecompressStream(st, &avail_in, &next_in,
                                           &avail_out, &next_out, nullptr);
    out->insert(out->end(), outbuf.data(), outbuf.data() + (outbuf.size() - avail_out));
    if (result == BROTLI_DECODER_RESULT_ERROR) {
      BrotliDecoderDestroyInstance(st); return false;
    }
  }
  BrotliDecoderDestroyInstance(st);
  return true;
}

bool round_trip(const std::vector<uint8_t> &input, int q) {
  std::vector<uint8_t> compressed;
  if (!encode(input, q, &compressed)) {
    std::fprintf(stderr, "  ENCODE FAIL  q=%d\n", q);
    return false;
  }
  std::vector<uint8_t> back;
  if (!decode(compressed, &back)) {
    std::fprintf(stderr, "  DECODE FAIL  q=%d\n", q);
    return false;
  }
  if (back.size() != input.size() || std::memcmp(back.data(), input.data(), input.size()) != 0) {
    std::fprintf(stderr, "  MISMATCH     q=%d  got %zu vs expected %zu\n",
                 q, back.size(), input.size());
    return false;
  }
  std::fprintf(stderr, "  PASS  q=%-2d  in=%zu  out=%zu  (%.2f%%)\n",
               q, input.size(), compressed.size(),
               100.0 * compressed.size() / (input.size() ? input.size() : 1));
  return true;
}

}  // anon

int main() {
  unsigned passed = 0, total = 0;

  {
    const char *s = "Hello, Brotli! Hello, Brotli! Hello, Brotli!";
    std::vector<uint8_t> v(s, s + std::strlen(s));
    for (int q : {1, 5, 9, 11}) {
      std::fprintf(stderr, "[tiny text]\n");
      ++total; if (round_trip(v, q)) ++passed;
    }
  }

  {
    std::vector<uint8_t> v(8192);
    unsigned x = 0xC0FFEEu;
    for (size_t i = 0; i < v.size(); i++) {
      x = x * 1664525u + 1013904223u;
      v[i] = (unsigned char)(x ^ (i & 0x1F));
    }
    for (int q : {1, 5, 9}) {
      std::fprintf(stderr, "[8 KiB pseudo-random]\n");
      ++total; if (round_trip(v, q)) ++passed;
    }
  }

  {
    std::vector<uint8_t> v(65536, 0);
    for (size_t i = 7; i < v.size(); i += 257) v[i] = (unsigned char)(i & 0xFF);
    for (int q : {1, 5, 9, 11}) {
      std::fprintf(stderr, "[64 KiB sparse]\n");
      ++total; if (round_trip(v, q)) ++passed;
    }
  }

  // Empty input — brotli should produce a valid empty stream.
  {
    std::vector<uint8_t> v;
    std::fprintf(stderr, "[empty]\n");
    ++total; if (round_trip(v, 5)) ++passed;
  }

  std::fprintf(stderr, "==== %u / %u ====\n", passed, total);
  return (passed == total) ? 0 : 1;
}
