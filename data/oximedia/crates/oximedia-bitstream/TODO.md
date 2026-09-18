# oximedia-bitstream TODO

Version: 0.2.0 | Status as of: 2026-07-15

## Status: Stable

## Completed [x]
- [x] Bit-level reader/writer (LSB and MSB modes)
- [x] BitReader with read_bits(), read_bit(), peek_bits()
- [x] BitPacker with write_bits(), write_bit(), flush()
- [x] Exponential Golomb coding (k-order)
- [x] CABAC arithmetic coding foundations
- [x] Big-endian and little-endian bitstream support
- [x] Huffman coding support
- [x] Integer types and bit counting utilities
- [x] Checked arithmetic operations
- [x] Byte-level I/O abstraction
- [x] Integration tests and doc tests
- [x] `shl_default`/`shr_default` — enabled `unbounded_shl`/`unbounded_shr` override in `define_unsigned_integer!` macro (Rust 1.87+ stable); 30 new unit tests in `tests/integer_shifts.rs` (2026-06-24)

## In Progress [ ]
(none)
