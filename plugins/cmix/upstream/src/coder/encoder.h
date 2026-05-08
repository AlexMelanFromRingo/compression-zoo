#ifndef ENCODER_H
#define ENCODER_H

// CMIX-MOD: take std::ostream* instead of std::ostream* so callers
// can pass any ostream-derived sink (e.g., std::ostringstream).
#include <ostream>

#include "../predictor.h"

class Encoder {
 public:
  Encoder(std::ostream* os, Predictor* p);
  void Encode(int bit);
  void Flush();

 private:
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  std::ostream* os_;
  unsigned int x1_, x2_;
  Predictor* p_;
};

#endif
