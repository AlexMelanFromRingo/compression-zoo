// BrotliCoder.h — 7-Zip codec wrapper around Google Brotli.

#ifndef COMPRESSION_ZOO_BROTLI_CODER_H
#define COMPRESSION_ZOO_BROTLI_CODER_H

#include "../../sdk/CPP/Common/MyCom.h"
#include "../../sdk/CPP/7zip/ICoder.h"

namespace ZooBrotli {

// 7-Zip "level" 1..9 maps onto Brotli quality 0..11. We pick a curve
// that leaves the slowest brotli levels (10, 11) for -mx9 and bunches
// the practical ones (4..9) at -mx3..-mx7.
Z7_CLASS_IMP_COM_2(
    CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
)
  int _quality;
public:
  CEncoder() : _quality(6) {}
};

// Decoder is parameterless; brotli streams are self-describing.
Z7_CLASS_IMP_COM_1(
    CDecoder
  , ICompressCoder
)
public:
  CDecoder() {}
};

}  // namespace ZooBrotli

#endif
