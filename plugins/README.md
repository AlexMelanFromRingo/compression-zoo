# 7-Zip codec plugins

Each subdirectory builds a single Windows DLL that 7-Zip loads at startup
from its `Codecs/` folder. The DLL exports the four entry points that
7-Zip looks up by name (`CreateObject`, `GetMethodProperty`,
`GetNumberOfMethods`, `GetHashers`/`GetHasherProperty` are optional) and
implements `ICompressCoder` for the actual compress/decompress work.

## Build prerequisites

- MinGW-w64 (`x86_64-w64-mingw32-g++`) for cross-compiling on Linux/WSL,
  or MSVC + the Windows SDK on Windows.
- A copy of 7-Zip 22.x or newer to test against. The community
  `Codecs/` directory lives at:
  - `C:\Program Files\7-Zip\Codecs\` for the default install
  - `%APPDATA%\7-Zip\Codecs\` for a per-user override

## Installing a built plugin

```
cp build/zpaq.dll "/mnt/c/Program Files/7-Zip/Codecs/"
```

Then in 7-Zip CLI:

```
7z a archive.7z -m0=zpaq -mx5 input
7z x archive.7z
```

The plugin DLLs are independent of each other — install only the ones
you want.
