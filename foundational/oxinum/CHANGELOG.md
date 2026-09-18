# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - Unreleased

## [0.1.4] - 2026-08-06

### Added

- **oxinum-int** `checked_divrem_int`: non-panicking counterpart to `divrem_int` /
  `Div`/`Rem` for `BigInt`; returns `None` on a zero divisor instead of panicking.
- **oxinum-rational** `checked_div`, `checked_rem`: non-panicking counterparts to
  `Div`/`Rem` for `BigRational`; return `None` on a zero divisor instead of panicking.
- **oxinum-float** `BigFloat::try_to_bigint_trunc` / `try_to_bigint_floor` /
  `try_to_bigint_ceil` / `try_to_bigint_round`: non-panicking counterparts to the
  infallible `to_bigint_*` conversions; return `OxiNumError::Overflow` when the exact
  integer would exceed the new `MAX_EXACT_CONVERSION_BITS` (2^27 bits) budget instead
  of requesting an allocation large enough to abort the process.
- **oxinum-float** `MAX_EXACT_CONVERSION_BITS` constant, re-exported from
  `oxinum_float::native`.
- **oxinum-float** enforced exponent range: `BigFloat::EMAX` / `BigFloat::EMIN`
  (`±2^40`) bound the binary magnitude of every finite non-zero value, and
  `BigFloat::ilogb()` exposes that magnitude (the position of the leading set bit,
  `None` for zero/NaN/±Inf). `BigFloat::try_from_parts` is the new fallible
  constructor: it rejects an out-of-range magnitude with `OxiNumError::Overflow` and
  a zero precision with `OxiNumError::Precision` instead of asserting. This turns the
  unbounded `i64` exponent — the root cause behind the per-operation allocation
  guards below — into a checked invariant established at construction.
- **oxinum-rational** `try_float_to_rational`: non-panicking counterpart to
  `float_to_rational`, guarded by the same `MAX_EXACT_CONVERSION_BITS` budget.
- Regression test suites: `crates/oxinum-float/tests/regression_rem_align.rs`,
  `crates/oxinum-float/tests/regression_unbounded_shift.rs`,
  `crates/oxinum-float/tests/exponent_range.rs` (EMIN/EMAX boundaries, ilogb,
  saturation vs rejection, and arithmetic at the boundary),
  `crates/oxinum-rational/tests/regression_float_to_rational_budget.rs`,
  `crates/oxinum-float/tests/subnormal_rounding.rs` (`to_f64` at the IEEE-754
  underflow and overflow boundaries: the `2^-1075` tie, gradual underflow, the
  subnormal/normal seam at `2^-1022`, the `2^1024 - 2^970` overflow threshold, and
  fixed-seed sweeps cross-validating div/mul/add/round-trip against the hardware),
  and `crates/oxinum/tests/fuzz_parse.rs` (1024-case proptest harness for the
  universal `oxinum::parse` string parser: garbage-input no-panic + valid-input
  round-trip).
- `SECURITY.md`, `CONTRIBUTING.md`, `rustfmt.toml`, `clippy.toml` (MSRV 1.80) at the
  workspace root.

### Changed

- **oxinum-float** `BigFloat::to_bigint_trunc` / `to_bigint_floor` / `to_bigint_ceil` /
  `to_bigint_round` now panic (documented under `# Panics`) when the exact integer
  would exceed `MAX_EXACT_CONVERSION_BITS`, rather than requesting an unbounded
  allocation that aborts the process. Use the new `try_*` variants to handle this
  case without panicking.
- **oxinum-rational** `float_to_rational` now panics (documented) under the same
  budget; use `try_float_to_rational` to avoid the panic.
- **oxinum-float** `BigFloat::to_scientific_string` / `to_engineering_string` now fall
  back to the lossless `to_hex_string` rendering instead of attempting an exact
  decimal conversion when the value's exponent magnitude would make that conversion
  cost more than `MAX_EXACT_CONVERSION_BITS`.
- **oxinum-float** `BigFloat::from_parts` now *saturates* the exponent range instead
  of accepting any `i64` exponent: a magnitude above `2^(EMAX+1)` clamps to
  `ilogb == EMAX` and a non-zero magnitude below `2^EMIN` clamps to `ilogb == EMIN`,
  leaving sign, mantissa bits and precision untouched (documented under
  "Saturation semantics"). The infallible signature — and therefore the total
  `Add`/`Sub`/`Mul`/`Div`/`Rem` operators built on it — is unchanged; use
  `try_from_parts` to detect the condition. Reaching saturation requires a value
  beyond `2^(2^40)`, i.e. ~331 billion decimal digits.
- **oxinum-float** `BigFloat::from_hex_float` now returns `OxiNumError::Overflow` for
  a well-formed literal whose `p` exponent falls outside `[EMIN, EMAX]` (e.g.
  `"0x1p9223372036854775807"`). As the crate's untrusted-string entry point it
  refuses such input rather than saturating it into a silently different number;
  in-range exponents (up to `±2^40`) parse exactly as before.
- **oxinum-float** (`serde` feature) `Deserialize for BigFloat` now enforces the
  exponent range as well as the existing precision/mantissa invariants: a record
  carrying an out-of-range `exponent` (e.g. `i64::MAX`) is rejected with a typed
  error instead of being turned into a value that could drive an exponent-sized
  allocation. Records this crate emits are unaffected and still round-trip
  bit-for-bit, including at both boundaries.
- **oxinum-float** internal constructors on fallible paths (`from_f64`,
  `from_hex_float`, `sqrt`, `ln`, `ln_agm`) now route through `try_from_parts`, so an
  out-of-range intermediate surfaces as a typed error instead of being clamped.
  Deliberately-saturating call sites (`add`/`mul`/`div`/`rem` cores, the `exp`
  argument reduction) carry an inline rationale for why saturation is the correct
  behaviour there.
- Version bump to 0.1.4; all workspace crates aligned to this version (`oxinum-core`,
  `oxinum-int`, `oxinum-float`, `oxinum-rational`, `oxinum-complex`).

### Fixed

- **oxinum-float** `%`/`Rem`: `rem_core` rounded the quotient to `prec` significant
  bits before truncating, which could produce a remainder outside `[0, |b|)` whenever
  the integer part of `|a/b|` needed more than `prec` bits. It now works directly on
  exact mantissas and always returns a remainder in `[0, |b|)`.
- **oxinum-float** `Add`/`Sub`/`AddAssign`/`SubAssign`: aligning two operands by
  shifting the higher-exponent mantissa the full exponent difference could allocate
  memory proportional to that gap — unbounded for a perfectly valid but extreme
  exponent, aborting the process. `align_to_common_exp` now caps the shift and folds
  the far operand into a single sticky bit once it can no longer affect the rounded
  result.
- **oxinum-float** `%`/`Rem` (`rem_core`) had the same unbounded-allocation class as
  the add/sub fix above for a large exponent gap. It now reduces `2^d mod m_b` by
  square-and-multiply (bounded to the modulus width) instead of materializing the
  full shift.
- **oxinum-float** `bs_transcendental::split_arg` (binary-splitting `exp`/`sin`/`cos`
  path, precision >= 512 bits): lifting an argument to an exact rational could request
  a denominator proportional to `|exponent|` for a tiny-but-valid value (e.g.
  `2^-1_000_000_000`) — same unbounded-allocation class. Negligible arguments now
  short-circuit to a 1-2 term Taylor approximation before reaching the exact-rational
  lift; the lift itself is now capped and returns `OxiNumError::Precision` instead of
  allocating past the cap.
- **oxinum-int** / **oxinum-rational**: all 17 production `panic!` sites audited; the
  5 reachable from safe public APIs (`native/div.rs`, `native/ops_int.rs`,
  `native/rational_ops.rs` ×2, `native/convert.rs`) now carry `# Panics` documentation
  pointing to a non-panicking `checked_*` alternative (see Added).
- **oxinum** `parse()`: `"<n>/0"`-style strings with an explicit zero denominator now
  return `OxiNumError::Parse` instead of silently building a mathematically-invalid
  `numerator/0` rational that would panic later on arithmetic such as `.to_f64()`.
- **oxinum-float** `BigFloat::to_f64` flushed every magnitude below `2^-1074` straight
  to zero instead of rounding it onto the subnormal grid. IEEE 754 requires gradual
  underflow: a value in `(2^-1075, 2^-1074)` is above *half* the smallest subnormal
  and owes `2^-1074`, not `0.0`. `to_f64` was rewritten to round **once**, directly
  from the exact stored value onto the `f64` grid (normal and subnormal alike), which
  also makes the overflow threshold the exact IEEE halfway point `2^1024 - 2^970`
  rather than an approximation, and preserves `-0.0` on a negative underflow.
- **oxinum-float** `BigFloat::to_f64` double-rounded. A value that is itself the
  rounded result of an earlier operation was re-rounded blind, and ties-to-even then
  resolved a tie the un-rounded value never sat on — a 1-ULP error in roughly a
  quarter of all quotients landing in `[2^-1023, 2^-1022)`, where the halfway points
  of the 52-bit subnormal grid coincide with the points of the 53-bit normal grid.
  `BigFloat` now records the direction of its last rounding (MPFR's *ternary value*)
  and `to_f64` breaks exact ties with it, the equivalent of `mpfr_subnormalize`. The
  field is internal: it is excluded from equality, ordering, `Debug` and the serde
  wire format. Only the round-to-nearest modes (`HalfEven` / `HalfAway` /
  `HalfToZero`) record a direction — a directed rounding is a deliberate bias rather
  than an attempt at the nearest value, so `ToZero` / `ToInf` / `ToNegInf` /
  `AwayFromZero` keep the ties-to-even conversion behaviour they have always had. `div_ref(..).to_f64()` and `(a * b).to_f64()` now agree with the
  hardware quotient/product bit for bit all the way down to `2^-1074`, so
  `cross_val_mul_200_random_pairs` no longer skips subnormal results.
- **oxinum-float** `div_ref` / `div_ref_with_mode` sized the quotient shift against
  the target precision alone, so a divisor much wider than the dividend produced a
  quotient *narrower* than the target: the rounding pass was skipped in favour of
  zero-padding, and the sticky bit forced for correct rounding was frozen into the
  value. `1` at precision 8 divided by `3` at precision 53 returned
  `0.33333587646484375` — seventeen correct bits in a fifty-three bit result. The
  shift now also covers the operands' width difference.

## [0.1.3] - 2026-06-19

### Changed

- Version bump to 0.1.3; all workspace crates aligned to this version (`oxinum-core`,
  `oxinum-int`, `oxinum-float`, `oxinum-rational`, `oxinum-complex`).

## [0.1.2] - 2026-06-10

### Fixed

- **oxinum-complex** `parity_cross_validation`: `prop_asin_cbig_native_agree` and
  `prop_atanh_cbig_native_agree` proptest cases timed out (>120 s) when run with
  `HEAVY_CASES = 16` at full precision (40 significant digits). The two tests are
  now split into a separate `proptest!` block with `VERY_HEAVY_CASES = 6` and
  `PREC_LIGHT = 20`, bringing each test well under the 120-second per-test budget
  while still exercising cross-family agreement to `1e-9` tolerance.

### Maintenance

- Version bump to 0.1.2; all workspace crates aligned to this version.
- Dependency version pins updated across workspace `[workspace.dependencies]`.

## [0.1.1] - 2026-06-04

### Added

- **oxinum-complex** (new crate): Arbitrary-precision complex numbers for the OxiNum
  workspace. `CBig` pairs two `DBig` components; `native::BigComplex` pairs two
  `BigFloat` components for binary-base complex arithmetic with explicit rounding
  control. Both types provide construction, arithmetic operators (all ownership
  variants), conjugate, norm-squared, transcendental functions (`exp`, `ln`, `sqrt`,
  `pow`), trigonometric functions (`sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`),
  inverse-trig functions (`asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`),
  `Display`/`Debug` formatting, and optional `serde` and `num-traits` features.
  `CBig` is re-exported from the `oxinum` facade as `oxinum::CBig` / `oxinum::Complex`.
- **oxinum-complex** `num-complex` feature: Two-way conversions between `CBig` and
  `num_complex::Complex<f64>` / `Complex<i64>` via `From` impls. Integer conversions
  store parts at unlimited `DBig` precision to prevent silent precision collapse.
- **oxinum-complex** `TryFrom<&CBig> for (f64, f64)`: Fallible projection to `f64`
  pairs; returns `OxiNumError::Overflow` when either component exceeds `f64::MAX`.
- **oxinum-complex** cross-validation test suite: `parity_cross_validation` tests
  verify that `CBig` and `native::BigComplex` agree on known-value results; includes
  SciRS2 `ArbitraryComplex` compatibility tests.
- **oxinum-float** `special` module: Pure-Rust special mathematical functions on
  `DBig` — `gamma`, `ln_gamma`, `digamma`, `erf`, `erfc`, `bessel_j0`, `euler_gamma`
  (Euler–Mascheroni constant to 200 digits), `catalan` (Catalan's constant to 200
  digits). Gamma uses Lanczos (g=7) for x ∈ (0, 20] and Stirling series for x > 20.
- **oxinum-float** `MpFloat` and `MpComplex` adapter types (`mp_float` module):
  `rug::Float`/`rug::Complex`-compatible wrappers over `DBig` for drop-in replacement
  of GMP-backed types in SciRS2's `arbitrary_precision` module.
- **oxinum-float** `native::bs_transcendental` module: Binary-splitting evaluation of
  `exp`, `sin`, and `cos` for `BigFloat` above a 512-bit precision threshold,
  replacing the O(N²) iterative Taylor series.
- **oxinum-int** `native::simd_ops` module: SIMD-accelerated (nightly `core::simd`,
  with scalar fallback on stable) inner kernels for AND, OR, XOR, and within-limb
  shift operations on `BigUint` limb slices. Activated by `oxinum_simd` cfg emitted
  from `build.rs` only on nightly + `simd` feature.
- **oxinum-int** `BitAndAssign`, `BitOrAssign`, `BitXorAssign` for `native::BigUint`
  (both owned and borrowed right-hand-side variants).
- SciRS2 compatibility integration tests across all sub-crates (`scirs2_int_compat`,
  `scirs2_float_compat`, `scirs2_rational_compat`, `scirs2_facade_compat`,
  `scirs2_trait_hierarchy_compat`, `scirs2_arbitrary_complex_compat`).
- Allocation-profiling Criterion benchmarks for `oxinum-int`, `oxinum-float`, and
  `oxinum-rational`; bitwise/shift operation benchmarks for `oxinum-int`.

### Fixed

- **oxinum-float** `atan` and `atan2` precision collapse: intermediate arithmetic was
  carried at the (narrow) input precision rather than the requested guard precision,
  capping output accuracy at approximately 3 significant digits regardless of the
  `precision` argument. All internal literals, reductions, and halving loops now use
  `dbig_at_precision` / `extend_precision` at `precision + 20` guard digits, giving
  accurate results to full requested precision.

## [0.1.0] - 2026-06-01

### Added

- **oxinum-core**: Core traits (`OxiNumTrait`, `OxiSigned`), `OxiNumError`/`OxiNumResult`,
  `RoundingMode` enum, `Sign` re-export from `dashu-base`, serde feature gate.
- **oxinum-int**: Arbitrary-precision integers via `dashu-int` re-exports (`UBig`, `IBig`)
  plus a full native Pure-Rust implementation:
  - `native::BigUint` — little-endian `Vec<u64>` limbs; schoolbook, Karatsuba, and
    Toom-Cook-3 multiplication; Knuth Algorithm D division; binary GCD;
    Newton integer sqrt and nth-root; Lehmer GCD.
  - `native::BigInt` — signed wrapper on `BigUint`; canonical zero invariant.
  - Number theory: Miller-Rabin + BPSW (Jacobi + strong Lucas) primality; Sieve of
    Eratosthenes; modular arithmetic; Montgomery multiplication context.
  - Conversions, radix I/O (2–36), serde, rand, num-traits features.
- **oxinum-float**: Arbitrary-precision floats via `dashu-float` re-exports (`FBig`, `DBig`)
  plus a full native `native::BigFloat`:
  - Binary-base (`b=2`), explicit precision, post-operation rounding.
  - Elementary functions: sqrt, exp, ln, pow.
  - Trigonometric functions: sin, cos, tan, asin, acos, atan, atan2.
  - High-precision constants: π, e, ln 2 (binary-splitting / AGM).
  - serde, num-traits features.
- **oxinum-rational**: Exact rationals via `dashu-ratio` re-exports (`RBig`, `Relaxed`)
  plus a full native `native::BigRational`:
  - Automatic reduction (GCD on construction), canonical zero/sign.
  - Continued-fraction expansion, reconstruction, convergents,
    and semiconvergent best-rational-approximation.
  - Cross-domain conversions (`BigFloat` ↔ `BigRational` ↔ `BigInt`).
  - serde, num-traits features.
- **oxinum** (facade): Prelude, constants module (π, e, ln 2), parse helpers,
  and feature-gated re-exports of all sub-crates.
- `deny.toml` banning GMP/MPFR/rug crates tree-wide.
- `Dockerfile.ffi-audit` + `scripts/ffi-audit.sh` for C/FFI-free verification.
- Benchmark harnesses (Criterion) for mul/div/factorial/primality/transcendentals.
- Property-based tests with proptest across all arithmetic laws.
- 1282 tests passing at 0.1.0, zero clippy warnings, zero rustdoc warnings.

[0.1.4]: https://github.com/cool-japan/oxinum/compare/v0.1.3...v0.1.4
[0.1.4]: https://github.com/cool-japan/oxinum/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/cool-japan/oxinum/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/cool-japan/oxinum/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxinum/releases/tag/v0.1.1
