# Plugin verification checklist

The plugin DLLs were cross-compiled with MinGW-w64 from WSL2 and have
all been round-tripped through Linux harnesses, but the actual COM
interface contract — `CreateObject`, `GetMethodProperty`,
`GetNumberOfMethods` — has not been exercised in a real Windows
process from this development environment (WSL2 does not have its
binfmt_misc Windows-EXE interop registered here).

When you have a Windows console open, run through this checklist to
confirm each plugin is loaded by 7-Zip correctly. All commands assume
your 7-Zip lives at `L:\Programs\7-Zip\` and the DLLs are in
`L:\Programs\7-Zip\Codecs\` (already true for this dev machine).

## 0. List recognised methods

```cmd
"L:\Programs\7-Zip\7z.exe" i
```

You should see `ZPAQ`, `bsc`, and `CMIX` in the *Codecs* section
alongside the existing flzma2, plus the built-in LZMA2 etc. If they're
missing, 7-Zip didn't load the DLL — check `Codecs/` permissions and
that the DLL is 64-bit (PE32+).

## 1. ZPAQ standalone smoke test

A self-contained DLL loader test was cross-compiled to
`L:\Programs\7-Zip\Codecs\test_zpaq.exe`. It opens `zpaq.dll`,
discovers method `0x4F71103`, instantiates encoder + decoder via
`CreateObject`, and round-trips an 8 KiB buffer in memory.

```cmd
"L:\Programs\7-Zip\Codecs\test_zpaq.exe" "L:\Programs\7-Zip\Codecs\zpaq.dll"
```

Expected last line on stderr: `ROUND-TRIP OK`. Exit code 0.

## 2. ZPAQ round-trip through 7-Zip CLI

```cmd
"L:\Programs\7-Zip\7z.exe" a -t7z -m0=zpaq -mx5 test_zpaq.7z some_file.txt
"L:\Programs\7-Zip\7z.exe" t test_zpaq.7z
"L:\Programs\7-Zip\7z.exe" x test_zpaq.7z -odecoded\
fc some_file.txt decoded\some_file.txt
```

## 3. bsc round-trip

```cmd
"L:\Programs\7-Zip\7z.exe" a -t7z -m0=bsc -mx5 test_bsc.7z some_file.txt
"L:\Programs\7-Zip\7z.exe" t test_bsc.7z
"L:\Programs\7-Zip\7z.exe" x test_bsc.7z -odecoded_bsc\
fc some_file.txt decoded_bsc\some_file.txt
```

## 4. CMIX (one-shot only)

CMIX has many file-scope `static` mutables that are not reset between
codec instances; encoding then decoding **in the same 7-Zip process**
won't round-trip. Encode in one invocation, exit, then decode in
another:

```cmd
:: pick a small file — CMIX needs hundreds of MB RAM and seconds per
:: kilobyte even on tiny inputs.
"L:\Programs\7-Zip\7z.exe" a -t7z -m0=cmix test_cmix.7z small_file.txt
:: now exit this 7z.exe (it has already exited above).
"L:\Programs\7-Zip\7z.exe" x test_cmix.7z -odecoded_cmix\
fc small_file.txt decoded_cmix\small_file.txt
```

If you specifically want to run `7z t test_cmix.7z` on the same
archive: that creates a fresh 7z.exe process and is fine.

## What to report if something fails

For any failure, please capture:

  - The exact 7z.exe command line.
  - 7z.exe's stdout + stderr.
  - Output of `"L:\Programs\7-Zip\7z.exe" i` (so we can see if the
    method was registered).
  - The DLL size + `mtime` for the failing plugin
    (`dir L:\Programs\7-Zip\Codecs\zpaq.dll` etc.).
