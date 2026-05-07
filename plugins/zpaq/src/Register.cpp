// Register.cpp — pulls our codec classes into the static registration list
// so DllExportsCompress.cpp can find them.

#include "StdAfx.h"

#include "../../sdk/CPP/7zip/Common/RegisterCodec.h"

#include "ZpaqCoder.h"

namespace ZooZpaq {

REGISTER_CODEC_CREATE(CreateDec, CDecoder)
REGISTER_CODEC_CREATE(CreateEnc, CEncoder)

REGISTER_CODEC_VAR(Zpaq) {
  CreateDec, CreateEnc, 0x4F71103, "ZPAQ", 1 /* NumStreams */, false /* IsFilter */
};

REGISTER_CODEC(Zpaq)

}  // namespace ZooZpaq
