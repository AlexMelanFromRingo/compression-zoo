// zpaq_cli.cpp — minimal CLI for benchmarking. Reads stdin, encodes
// or decodes, writes stdout. Mode and method given on argv[1] and
// argv[2].
//
// Build:
//   g++ -O2 -std=c++14 tests/zpaq_cli.cpp upstream/libzpaq.cpp -DNOJIT -Dunix -o tests/zpaq_cli

#include "../upstream/libzpaq.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace libzpaq {
  void error(const char *m) {
    std::fprintf(stderr, "libzpaq error: %s\n", m ? m : "?");
    std::exit(2);
  }
}

class StdinReader : public libzpaq::Reader {
public:
  int get() override { int c = std::getchar(); return (c == EOF) ? -1 : c; }
};
class StdoutWriter : public libzpaq::Writer {
public:
  void put(int c) override { std::putchar(c); }
};

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s e|d [method]\n", argv[0]);
    return 2;
  }
  StdinReader rin;
  StdoutWriter wout;
  if (argv[1][0] == 'e') {
    const char *m = (argc > 2) ? argv[2] : "5";
    libzpaq::compress(&rin, &wout, m);
  } else if (argv[1][0] == 'd') {
    libzpaq::decompress(&rin, &wout);
  } else {
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
  }
  return 0;
}
