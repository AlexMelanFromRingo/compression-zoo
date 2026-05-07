// Register.cpp — pulls our codec classes into the static registration list
// so DllExportsCompress.cpp can find them.

#include "StdAfx.h"

#include "../../sdk/CPP/7zip/Common/RegisterCodec.h"

#include "BscCoder.h"

namespace ZooBsc {

REGISTER_CODEC_CREATE(CreateDec, CDecoder)
REGISTER_CODEC_CREATE(CreateEnc, CEncoder)

REGISTER_CODEC_VAR(Bsc) {
  CreateDec, CreateEnc, 0x4F71200, "bsc", 1 /* NumStreams */, false /* IsFilter */
};

REGISTER_CODEC(Bsc)

}  // namespace ZooBsc
