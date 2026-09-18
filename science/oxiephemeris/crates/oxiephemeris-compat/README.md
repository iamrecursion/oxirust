# oxiephemeris-compat

A Swiss-Ephemeris-*shaped* API surface — `Context`, body/flag constants,
house systems, sidereal modes — built entirely from the SE public
programmer's documentation and verified against 960 real `pyswisseph`
output fixtures.

This crate sits directly on top of
[`oxiephemeris-core`](../oxiephemeris-core),
[`oxiephemeris-de`](../oxiephemeris-de),
[`oxiephemeris-bodies`](../oxiephemeris-bodies), and
[`oxiephemeris-astro`](../oxiephemeris-astro): every quantity `Context`
returns is computed by those crates, and this one only maps the
request/response *shapes* — function names, `SEFLG_*` bits, house-system
characters — onto the ones `swe_calc`/`swe_calc_ut`/`swe_houses`/
`swe_houses_ex`/`swe_julday`/`swe_revjul`/`swe_sidtime`/`swe_get_ayanamsa`
document. It exists for code migrating from Swiss Ephemeris; new code in
this workspace should generally call `oxiephemeris-astro` (or the
[`oxiephemeris-chart`](../oxiephemeris-chart) facade) directly instead.

No Swiss Ephemeris **source** was ever read while writing this crate —
only its public programmer's documentation, which this project's
clean-room policy treats as fair game for API shape (see the crate's own
`# Clean-room provenance` doc comment and the workspace
[`CONTRIBUTING.md`](../../CONTRIBUTING.md)).

```rust
use oxiephemeris_compat::{Context, SiderealMode, SEFLG_SIDEREAL, SEFLG_SPEED, SE_GREG_CAL, SE_SUN};
use oxiephemeris_de::DeFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("data/de440/linux_p1550p2650.440")?;
    let de = DeFile::parse(&bytes)?;
    let mut ctx = Context::with_ephemeris(de);

    // swe_julday(2000, 1, 1, 12.0, SE_GREG_CAL) equivalent; needs no
    // Context instance.
    let jd_ut = Context::julday(2000, 1, 1, 12.0, SE_GREG_CAL)?;

    // swe_calc_ut(jd_ut, SE_SUN, SEFLG_SPEED) equivalent.
    let xx = ctx.calc_ut(jd_ut, SE_SUN, SEFLG_SPEED)?;
    println!("Sun: lon={:.6} deg, speed={:.6} deg/day", xx[0], xx[3]);

    // Sidereal (Lahiri) longitude instead of tropical.
    ctx.set_sid_mode(SiderealMode::Lahiri);
    let sidereal = ctx.calc_ut(jd_ut, SE_SUN, SEFLG_SIDEREAL)?;
    println!("Sun (sidereal, Lahiri): lon={:.6} deg", sidereal[0]);

    // swe_houses(jd_ut, geolat, geolon, hsys) equivalent; needs no DE
    // file at all (the Royal Observatory, Greenwich).
    let houses = ctx.houses(jd_ut, 51.4779, 0.0, 'P')?;
    println!("Ascendant = {:.6} deg", houses.asc());
    Ok(())
}
```

`calc`/`houses` take a TT (SE: "ET") epoch directly; `calc_ut`/`houses_ex`
(used above) take UT and convert internally via
`oxiephemeris-core`'s Espenak–Meeus ΔT polynomial. `revjul` (Julian Date →
calendar) and `sidtime` (Greenwich apparent sidereal time) round out the
mirrored surface — see the per-method docs
(`cargo doc -p oxiephemeris-compat --open`) for exact signatures.

## Differences from Swiss Ephemeris

`Context` is a deliberate departure from SE's C API in one respect: SE
keeps its loaded ephemeris, topocentric observer, and sidereal mode in
process-global state, set once via `swe_set_*` calls and read back
implicitly by every later `swe_calc`. `Context` makes that state an
explicit, instance-local value the caller constructs and threads through
instead, so multiple independent contexts (e.g. one per thread, or one per
ephemeris file) can coexist safely with no `unsafe` and no serialization
requirement between calls.

A few other differences worth knowing before porting SE-calling code:

- **Typed `Result`, not `(retflag, serr)`.** Every fallible entry point
  returns `Result<_, CompatError>`. In particular, where the SE
  documentation describes silently falling back to Porphyry cusps when
  Placidus/Koch are undefined inside the polar circles, this crate returns
  a typed error instead (`CompatError::Houses`) — the caller decides what
  fallback, if any, is acceptable.
- **No asteroids, Chiron, or fixed stars.** Only `SE_SUN..=SE_EARTH`
  (body numbers 0–14) are implemented; see the `bodies` module for the
  exact set.
- **A smaller sidereal-mode catalog.** Five ayanamshas
  (`SiderealMode::{FaganBradley, Lahiri, Krishnamurti, Raman, J2000Zero}`,
  plus a caller-supplied `User` anchor), not SE's full `SE_SIDM_*` catalog
  of dozens of modes.
- **`houses`/`houses_ex` cusp indexing is 0-based** (`cusps[0]` = cusp 1),
  not SE's 1-based `cusps[1..=12]` C convention.

See the crate's own module-level documentation for the complete list,
including the sidereal-longitude convention and the time-scale-model
differences.

## Feature flags

- `std` (off by default, so the crate is `no_std` otherwise) — forwards to
  the `std` feature of the four crates it wraps
  (`oxiephemeris-core`/`-de`/`-bodies`/`-astro`).

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview. Licensed under Apache-2.0 (see workspace
[LICENSE](../../LICENSE)).
