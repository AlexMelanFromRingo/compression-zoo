// test_zpaq.cpp — self-contained smoke test for zpaq.dll.
//
// Loads zpaq.dll via LoadLibrary, calls GetNumberOfMethods / GetMethodProperty
// to verify it advertises method 0x4F71103 "ZPAQ", then instantiates
// encoder + decoder via CreateObject and round-trips a buffer through
// ICompressCoder::Code in memory.
//
// Build (from plugins/zpaq):
//   x86_64-w64-mingw32-g++ -O2 -std=c++14 -Iplugins/sdk -Iplugins/sdk/CPP \
//     tests/test_zpaq.cpp -o tests/test_zpaq.exe -lole32 -loleaut32 -luuid
//
// Run on Windows (with build/zpaq.dll next to it OR placed in 7-Zip's
// Codecs folder):
//   tests\\test_zpaq.exe build\\zpaq.dll
//   echo %ERRORLEVEL%   (0 = pass)

#define INITGUID
#include <windows.h>
#include <oleauto.h>
#include <objbase.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

// --- Minimal subset of 7-Zip interfaces (binary-compatible) ----------

typedef int8_t   Int8;
typedef uint8_t  Byte;
typedef uint16_t UInt16;
typedef int32_t  Int32;
typedef uint32_t UInt32;
typedef int64_t  Int64;
typedef uint64_t UInt64;

// Match k_7zip_GUID_Data1/2/3_Common and the layout from IDecl.h's
// Z7_DECL_IFACE_7ZIP_SUB macro: Data4 = {0, 0, 0, groupId, 0, subId, 0, 0}.
//   ISequentialInStream  : group=3, sub=1
//   ISequentialOutStream : group=3, sub=2
//   ICompressCoder       : group=4, sub=5
static const GUID kSeqInStream  = { 0x23170F69, 0x40C1, 0x278A, { 0, 0, 0, 3, 0, 1, 0, 0 } };
static const GUID kSeqOutStream = { 0x23170F69, 0x40C1, 0x278A, { 0, 0, 0, 3, 0, 2, 0, 0 } };
static const GUID kIIDCoder     = { 0x23170F69, 0x40C1, 0x278A, { 0, 0, 0, 4, 0, 5, 0, 0 } };

// k_7zip_GUID_Data3_{Decoder,Encoder} from IDecl.h:
static const UInt16 kDecoderData3 = 0x2790;
static const UInt16 kEncoderData3 = 0x2791;

#define ZIP_STDMETHOD STDMETHODCALLTYPE

struct ISequentialInStream : public IUnknown {
    virtual HRESULT ZIP_STDMETHOD Read(void *data, UInt32 size, UInt32 *processedSize) = 0;
};
struct ISequentialOutStream : public IUnknown {
    virtual HRESULT ZIP_STDMETHOD Write(const void *data, UInt32 size, UInt32 *processedSize) = 0;
};
struct ICompressProgressInfo : public IUnknown {
    virtual HRESULT ZIP_STDMETHOD SetRatioInfo(const UInt64 *inSize, const UInt64 *outSize) = 0;
};
struct ICompressCoder : public IUnknown {
    virtual HRESULT ZIP_STDMETHOD Code(
        ISequentialInStream *inStream,
        ISequentialOutStream *outStream,
        const UInt64 *inSize,
        const UInt64 *outSize,
        ICompressProgressInfo *progress) = 0;
};

// --- Test helpers ----------------------------------------------------

class CMemReader : public ISequentialInStream {
public:
    const Byte *data; size_t size; size_t pos; ULONG ref;
    CMemReader(const Byte *d, size_t s) : data(d), size(s), pos(0), ref(1) {}
    HRESULT ZIP_STDMETHOD QueryInterface(REFIID iid, void **out) override {
        if (IsEqualIID(iid, IID_IUnknown) || IsEqualIID(iid, kSeqInStream)) {
            *out = static_cast<ISequentialInStream*>(this); AddRef(); return S_OK;
        }
        *out = NULL; return E_NOINTERFACE;
    }
    ULONG ZIP_STDMETHOD AddRef() override { return ++ref; }
    ULONG ZIP_STDMETHOD Release() override { return --ref; }
    HRESULT ZIP_STDMETHOD Read(void *buf, UInt32 n, UInt32 *got) override {
        size_t avail = size - pos;
        size_t take  = (n < avail) ? n : avail;
        memcpy(buf, data + pos, take); pos += take;
        if (got) *got = (UInt32)take;
        return S_OK;
    }
};

class CMemWriter : public ISequentialOutStream {
public:
    std::vector<Byte> buf; ULONG ref;
    CMemWriter() : ref(1) {}
    HRESULT ZIP_STDMETHOD QueryInterface(REFIID iid, void **out) override {
        if (IsEqualIID(iid, IID_IUnknown) || IsEqualIID(iid, kSeqOutStream)) {
            *out = static_cast<ISequentialOutStream*>(this); AddRef(); return S_OK;
        }
        *out = NULL; return E_NOINTERFACE;
    }
    ULONG ZIP_STDMETHOD AddRef() override { return ++ref; }
    ULONG ZIP_STDMETHOD Release() override { return --ref; }
    HRESULT ZIP_STDMETHOD Write(const void *data, UInt32 n, UInt32 *done) override {
        buf.insert(buf.end(), (const Byte*)data, (const Byte*)data + n);
        if (done) *done = n;
        return S_OK;
    }
};

// --- DLL loading -----------------------------------------------------

typedef HRESULT (STDAPICALLTYPE *PFN_CreateObject)(const GUID*, const GUID*, void**);
typedef HRESULT (STDAPICALLTYPE *PFN_GetNumberOfMethods)(UInt32 *);
typedef HRESULT (STDAPICALLTYPE *PFN_GetMethodProperty)(UInt32, PROPID, PROPVARIANT*);

static GUID make_clsid(UInt16 data3, UInt64 methodId) {
    GUID g = { 0x23170F69, 0x40C1, data3, { 0,0,0,0, 0,0,0,0 } };
    for (int i = 0; i < 8; i++) g.Data4[i] = (Byte)(methodId >> ((7 - i) * 8));
    return g;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s path-to-zpaq.dll\n", argv[0]); return 2; }

    HMODULE m = LoadLibraryA(argv[1]);
    if (!m) { fprintf(stderr, "LoadLibrary failed (%lu)\n", (unsigned long)GetLastError()); return 1; }

    auto pCreateObject       = (PFN_CreateObject) GetProcAddress(m, "CreateObject");
    auto pGetNumberOfMethods = (PFN_GetNumberOfMethods) GetProcAddress(m, "GetNumberOfMethods");
    auto pGetMethodProperty  = (PFN_GetMethodProperty) GetProcAddress(m, "GetMethodProperty");
    if (!pCreateObject || !pGetNumberOfMethods || !pGetMethodProperty) {
        fprintf(stderr, "missing required exports\n"); return 1;
    }

    UInt32 numMethods = 0;
    if (pGetNumberOfMethods(&numMethods) != S_OK || numMethods == 0) {
        fprintf(stderr, "GetNumberOfMethods returned 0\n"); return 1;
    }
    fprintf(stderr, "DLL advertises %u methods\n", (unsigned)numMethods);

    UInt64 methodId = 0;
    char  methodName[64] = {0};
    bool found = false;
    for (UInt32 i = 0; i < numMethods; i++) {
        PROPVARIANT idVal; memset(&idVal, 0, sizeof(idVal));
        PROPVARIANT nameVal; memset(&nameVal, 0, sizeof(nameVal));
        pGetMethodProperty(i, 0 /* kID */, &idVal);
        pGetMethodProperty(i, 1 /* kName */, &nameVal);
        UInt64 id = (idVal.vt == VT_UI8) ? idVal.uhVal.QuadPart : 0;
        const char *nm = "(?)";
        char nameBuf[64] = {0};
        if (nameVal.vt == VT_BSTR && nameVal.bstrVal) {
            UINT len = SysStringLen(nameVal.bstrVal);
            if (len >= sizeof(nameBuf)) len = sizeof(nameBuf) - 1;
            for (UINT k = 0; k < len; k++) nameBuf[k] = (char)nameVal.bstrVal[k];
            nm = nameBuf;
        }
        fprintf(stderr, "  [%u] id=0x%llX  name=%s\n", (unsigned)i, (unsigned long long)id, nm);
        if (id == 0x4F71103ULL) {
            found = true; methodId = id; strncpy(methodName, nm, sizeof(methodName) - 1);
        }
        VariantClear((VARIANTARG*)&idVal);
        VariantClear((VARIANTARG*)&nameVal);
    }
    if (!found) { fprintf(stderr, "method 0x4F71103 not advertised\n"); return 1; }
    fprintf(stderr, "method ZPAQ found: id=0x%llX name=%s\n", (unsigned long long)methodId, methodName);

    // Construct encoder + decoder CLSIDs.
    GUID encClsid = make_clsid(kEncoderData3, methodId);
    GUID decClsid = make_clsid(kDecoderData3, methodId);

    // --- Round-trip a buffer ------------------------------------------------
    std::vector<Byte> input;
    for (int i = 0; i < 4096; i++) input.push_back((Byte)((i * 31 + 7) ^ (i >> 4)));
    // also add some repetition to give zpaq something to compress
    input.insert(input.end(), input.begin(), input.end());
    fprintf(stderr, "input size = %zu bytes\n", input.size());

    // Encode.
    ICompressCoder *enc = NULL;
    if (pCreateObject(&encClsid, &kIIDCoder, (void**)&enc) != S_OK || !enc) {
        fprintf(stderr, "CreateObject(encoder) failed\n"); return 1;
    }
    CMemReader  encIn(input.data(), input.size());
    CMemWriter  encOut;
    HRESULT hr = enc->Code(&encIn, &encOut, NULL, NULL, NULL);
    enc->Release();
    if (hr != S_OK) { fprintf(stderr, "encoder Code() returned 0x%lx\n", (long)hr); return 1; }
    fprintf(stderr, "compressed size = %zu bytes (ratio %.2f%%)\n",
        encOut.buf.size(), 100.0 * encOut.buf.size() / input.size());

    // Decode.
    ICompressCoder *dec = NULL;
    if (pCreateObject(&decClsid, &kIIDCoder, (void**)&dec) != S_OK || !dec) {
        fprintf(stderr, "CreateObject(decoder) failed\n"); return 1;
    }
    CMemReader  decIn(encOut.buf.data(), encOut.buf.size());
    CMemWriter  decOut;
    hr = dec->Code(&decIn, &decOut, NULL, NULL, NULL);
    dec->Release();
    if (hr != S_OK) { fprintf(stderr, "decoder Code() returned 0x%lx\n", (long)hr); return 1; }

    if (decOut.buf.size() != input.size() ||
        memcmp(decOut.buf.data(), input.data(), input.size()) != 0) {
        fprintf(stderr, "ROUND-TRIP FAILED: decoded size=%zu expected=%zu\n",
            decOut.buf.size(), input.size());
        return 1;
    }

    fprintf(stderr, "ROUND-TRIP OK\n");
    return 0;
}
