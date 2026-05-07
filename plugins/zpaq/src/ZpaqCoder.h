// ZpaqCoder.h — 7-Zip codec wrapper around libzpaq.

#ifndef COMPRESSION_ZOO_ZPAQ_CODER_H
#define COMPRESSION_ZOO_ZPAQ_CODER_H

#include "../../sdk/CPP/Common/MyCom.h"
#include "../../sdk/CPP/7zip/ICoder.h"

namespace ZooZpaq {

// Encoder: ICompressCoder + ICompressSetCoderProperties (7-Zip "level" 1..9
// is mapped onto ZPAQ's "0".."5" methods).
Z7_CLASS_IMP_COM_2(
    CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
)
  int _level;
public:
  CEncoder() : _level(2) {}
};

// Decoder: ICompressCoder. ZPAQ stream is self-describing; no SetDecoderProps.
Z7_CLASS_IMP_COM_1(
    CDecoder
  , ICompressCoder
)
public:
  CDecoder() {}
};

}  // namespace ZooZpaq

#endif
