# JPL Horizons osculating-elements fixtures (geocentric Moon)

Raw, unmodified responses of the public JPL Horizons API, fetched
2026-07-06. They are the oracle for `tests/nodes_oracle.rs`
(`horizons_osculating_elements_oracle`).

## Query

One GET request per epoch against
`https://ssd.jpl.nasa.gov/api/horizons.api` with the parameters

```
format=text  COMMAND='301'  OBJ_DATA='NO'  MAKE_EPHEM='YES'
EPHEM_TYPE='ELEMENTS'  CENTER='500@399'  REF_PLANE='ECLIPTIC'
TLIST='<JD>'  OUT_UNITS='AU-D'
```

i.e. geocentric (`500@399`) **geometric osculating elements** of the Moon
(`301`), ecliptic reference plane, AU/day units, one discrete epoch each.
`REF_SYSTEM` was left at its `ICRF` default.

| file                   | TLIST (JD, TDB) | calendar epoch (TDB)      |
|------------------------|-----------------|---------------------------|
| `moon_jd2440587_5.txt` | 2440587.5       | 1970-Jan-01 00:00:00.0000 |
| `moon_jd2447161_5.txt` | 2447161.5       | 1988-Jan-01 00:00:00.0000 |
| `moon_jd2453736_5.txt` | 2453736.5       | 2006-Jan-01 00:00:00.0000 |
| `moon_jd2460676_5.txt` | 2460676.5       | 2025-Jan-01 00:00:00.0000 |

## What the response headers document (why the test is set up the way
it is)

Quoted from the raw files; identical in all four:

* **Ephemeris source**: `Target body name: Moon (301) {source: DE441}`,
  `Center body name: Earth (399) {source: DE441}` — Horizons serves
  **DE441**. The test evaluates **DE440**; over 1970-2025 (deep inside
  the common LLR fit span) the two agree on the geocentric lunar state at
  the meter level (Park et al. 2021, AJ 161, 105, §5), far below every
  gate.
* **Gravitational parameter**:
  `Keplerian GM : 8.9970113929473456E-10 au^3/d^2`. This is exactly the
  DE440/DE441 header constant `GMB = GM_Earth + GM_Moon`; the test
  asserts the DE440 header value matches it to 1e-13 relative before
  comparing eccentricities.
* **Reference frame**: `Reference frame : Ecliptic of J2000.0`, expanded
  in the footer as: X-Y plane = "adopted Earth orbital plane at the
  reference epoch — Note: IAU76 obliquity of 84381.448 arcseconds wrt
  ICRF X-Y plane", X-axis = "ICRF". The matching transformation of the
  DE state (ICRF axes) is therefore the single frame rotation
  `R1(84381.448")` about the ICRF x-axis — **not** the
  frame-bias-corrected IAU 2006 mean ecliptic of J2000 (which differs by
  the frame-bias angles and by the 42 mas IAU76-vs-P03 obliquity
  offset).
* **Time scale**: `Start time : A.D. ... TDB` — the epochs are JD TDB,
  the native independent variable of the DE files, so no time-scale
  conversion is applied on either side of the comparison.
* **No aberrations**: "Geometric osculating elements have NO corrections
  or aberrations applied." — raw geometric state on both sides.
* **Units**: `Output units : AU-D, deg, Julian Day Number (Tp)`, with
  `1 au = 149597870.700 km` (the DE `AU` header constant, used by the
  test for the km -> AU conversion).

The element symbols compared: `EC` (eccentricity), `IN` (inclination
w.r.t. the X-Y plane, deg), `OM` (longitude of ascending node, deg), `W`
(argument of perifocus, deg), `A` (semi-major axis, AU; informational
only). The apsis comparison uses the dog-leg longitude
`OM + W + 180 (mod 360)`, since that is what Horizons' published `OM`/`W`
define (the ecliptic-*projected* apogee longitude differs by up to
~0.12 deg at i = 5 deg).

The test hard-fails if a fixture stops containing the frame/GM/source
sentences above (guard against silently re-fetching with different
settings).

## Re-fetching

```sh
for jd in 2440587.5 2447161.5 2453736.5 2460676.5; do
  curl -s "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='301'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='ELEMENTS'&CENTER='500@399'&REF_PLANE='ECLIPTIC'&TLIST='${jd}'&OUT_UNITS='AU-D'" \
    -o "moon_jd${jd/./_}.txt"
done
```

Clean-room note: these files are JPL Horizons *output*, a permitted
oracle artifact (like the `testpo` files and the other Horizons fixture
sets in this workspace). No Swiss Ephemeris or SOFA/ERFA material is
involved.
