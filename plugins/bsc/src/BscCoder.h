// BscCoder.h — 7-Zip codec wrapper around libbsc.

#ifndef COMPRESSION_ZOO_BSC_CODER_H
#define COMPRESSION_ZOO_BSC_CODER_H

#include "../../sdk/CPP/Common/MyCom.h"
#include "../../sdk/CPP/7zip/ICoder.h"

namespace ZooBsc {

// Encoder: ICompressCoder + ICompressSetCoderProperties.
//   - kLevel       (1..9) maps to (lzpHashSize, lzpMinLen, blockSorter, coder)
//   - kDictionarySize maps to per-block size in bytes
Z7_CLASS_IMP_COM_2(
    CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
)
  int _level;
  unsigned _block_size;
public:
  CEncoder();
};

// Decoder: ICompressCoder. Header self-describes block size / data size.
Z7_CLASS_IMP_COM_1(
    CDecoder
  , ICompressCoder
)
public:
  CDecoder() {}
};

}  // namespace ZooBsc

#endif
