# 7-Zip plugin SDK (vendored)

This directory contains the subset of 7-Zip's C/C++ headers (and a few
`.cpp` files) needed to compile a codec plugin DLL.

It is a verbatim copy from the upstream 7-Zip 25.x source tree, so we
can build without a parallel 7-Zip checkout. Updates: re-run the copy
pipeline whenever you bump 7-Zip; nothing in this directory should be
edited locally except in synchronised "vendor uplifts".

## Directory map

```
sdk/
├── C/           — 7-Zip C-side headers (CPU, types, threading, version)
├── CPP/Common/  — common C++ helpers (CMyComPtr, MyVector, etc.)
├── CPP/Windows/ — Windows-side glue (PROPVARIANT helpers)
├── CPP/7zip/    — codec interfaces (ICoder, IStream, IPassword)
└── StdAfx.h     — empty stub satisfying 7-Zip's per-file PCH include
```

## Include paths used by plugins

Each plugin's Makefile/build script adds `-I<repo>/plugins/sdk` and a
few subpath includes, so that `#include "../Common/MyTypes.h"` inside a
vendored 7-Zip header resolves correctly.

## Licence

LGPL-2.1+ / BSD-3-Clause / BSD-2-Clause as per upstream
`7-Zip/License.txt`. See `LICENSE` in this directory.
