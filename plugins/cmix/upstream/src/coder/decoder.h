#ifndef DECODER_H
#define DECODER_H

// CMIX-MOD: take std::istream* instead of std::istream* so callers
// can pass any istream-derived source (e.g., std::istringstream).
#include <istream>

#include "../predictor.h"

class Decoder {
 public:
  Decoder(std::istream* is, Predictor* p);
  int Decode();

 private:
  int ReadByte();
  unsigned int Discretize(float p);

  std::istream* is_;
  unsigned int x1_, x2_, x_;
  Predictor* p_;
};

#endif
