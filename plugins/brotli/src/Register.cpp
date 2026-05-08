// Register.cpp — pulls the Brotli codec classes into the static
// registration list so DllExportsCompress.cpp picks them up.

#include "StdAfx.h"

#include "../../sdk/CPP/7zip/Common/RegisterCodec.h"

#include "BrotliCoder.h"

namespace ZooBrotli {

REGISTER_CODEC_CREATE(CreateDec, CDecoder)
REGISTER_CODEC_CREATE(CreateEnc, CEncoder)

// 0x4F71102 is the community-allocated Brotli method ID (originally
// from mcmilk/7-Zip-zstd). Reusing it lets archives created by other
// brotli plugins decompress with this DLL.
REGISTER_CODEC_VAR(Brotli) {
  CreateDec, CreateEnc, 0x4F71102, "Brotli", 1 /* NumStreams */, false /* IsFilter */
};

REGISTER_CODEC(Brotli)

}  // namespace ZooBrotli
