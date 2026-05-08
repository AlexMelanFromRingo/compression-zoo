// Register.cpp — pulls our codec classes into the static registration list
// so DllExportsCompress.cpp can find them.

#include "StdAfx.h"

#include "../../sdk/CPP/7zip/Common/RegisterCodec.h"

#include "CmixCoder.h"

namespace ZooCmix {

REGISTER_CODEC_CREATE(CreateDec, CDecoder)
REGISTER_CODEC_CREATE(CreateEnc, CEncoder)

REGISTER_CODEC_VAR(Cmix) {
  CreateDec, CreateEnc, 0x4F71201, "CMIX", 1 /* NumStreams */, false /* IsFilter */
};

REGISTER_CODEC(Cmix)

}  // namespace ZooCmix
