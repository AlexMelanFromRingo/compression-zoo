# 7-Zip codec method IDs

7-Zip identifies a compression method by an 8-byte big-endian "method
ID". Built-in methods occupy the low end of the space (`0x21` is LZMA2,
`0x040108` is Deflate, etc.); plugin authors traditionally allocate IDs
inside the `0x4F71xxx` block, which Igor Pavlov reserved for community
codecs.

## Built-in 7-Zip methods (subset)

| ID | Method |
|---|---|
| `0x000000` | Copy (no compression) |
| `0x030101` | LZMA |
| `0x030401` | PPMd |
| `0x040108` | Deflate |
| `0x040109` | Deflate64 |
| `0x040202` | BZip2 |
| `0x21`     | LZMA2 |
| `0x06F10701` | AES-256 CBC |

## Community method IDs (`0x4F71xxx`)

These are de-facto allocations from existing third-party plugins.

| ID | Method | Plugin source |
|---|---|---|
| `0x4F71101` | FastLZMA2 | [conor42/fast-lzma2](https://github.com/conor42/fast-lzma2) |
| `0x4F71102` | Brotli | [mcmilk/7-Zip-zstd](https://github.com/mcmilk/7-Zip-zstd) |
| `0x4F71103` | ZPAQ | (this repo) |
| `0x4F71104` | Lzip | mcmilk |
| `0x4F71105` | LZ4 | mcmilk |
| `0x4F71106` | LZ5 | mcmilk |
| `0x4F71107` | Zstandard | mcmilk |

## Allocations in this repo

| ID | Method | Plugin |
|---|---|---|
| `0x4F71103` | ZPAQ | `plugins/zpaq/` (consistent with prior community plugin) |
| `0x4F71200` | libbsc | `plugins/bsc/` (proposed, open) |
| `0x4F71201` | CMIX | `plugins/cmix/` (proposed, open) |

The `0x4F712xx` block is currently unused upstream; we use it to avoid
collisions with the `0x4F711xx` codecs that ship in mcmilk's plugin
distribution.
