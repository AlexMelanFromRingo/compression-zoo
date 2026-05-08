// CmixCoder.h — 7-Zip codec wrapper around CMIX.

#ifndef COMPRESSION_ZOO_CMIX_CODER_H
#define COMPRESSION_ZOO_CMIX_CODER_H

#include "../../sdk/CPP/Common/MyCom.h"
#include "../../sdk/CPP/7zip/ICoder.h"

namespace ZooCmix {

// Encoder: ICompressCoder + ICompressSetCoderProperties (level is ignored
// because CMIX has no levels — there is one mode and it always tries hard).
Z7_CLASS_IMP_COM_2(
    CEncoder
  , ICompressCoder
  , ICompressSetCoderProperties
)
public:
  CEncoder() {}
};

// Decoder: ICompressCoder. Format is self-describing.
Z7_CLASS_IMP_COM_1(
    CDecoder
  , ICompressCoder
)
public:
  CDecoder() {}
};

}  // namespace ZooCmix

#endif
