# JPL Horizons topocentric spot-check fixtures (Tokyo site)

Companion set to `../horizons/` (the geocentric ten-body set). Read that
README first: everything stated there about provenance, the public-domain
status of Horizons output, the TT time scale, the DE441 `{source: ...}`
annotation, the EOP-coverage epoch policy and the column semantics of
quantities 1, 2 and 20 applies here unchanged. This file documents only
what is *different*: the topocentric observer, the per-epoch
Earth-orientation parameters, and the manifest fields added for them.

## Provenance

- Endpoint: `https://ssd.jpl.nasa.gov/api/horizons.api` (documented HTTP
  API), retrieved 2026-07-05, banner times 09:39:44–09:39:54 "Pasadena,
  USA" server local time (printed verbatim in each response's
  `Ephemeris / API_USER ...` line).
- Underlying planetary ephemeris served: **DE441** (per the
  `{source: DE441}` tags), compared in the tests against **DE440** — same
  sub-milliarcsecond / sub-km caveat as the geocentric set.
- Horizons EOP file at retrieval: `eop.260703.p260929`, coverage
  `DATA-BASED 1962-JAN-20 TO 2026-JUL-03` (printed in every response).
  All four epochs are far inside the data-based span.

## Observer site

All four requests used the same user-defined geodetic site (airless):

```
CENTER='coord@399'
COORD_TYPE='GEODETIC'
SITE_COORD='139.7414,35.6581,0.04'   (E-lon deg, geodetic lat deg, height km)
```

i.e. **139.7414° E, 35.6581° N, 40 m** (Tokyo). Horizons echoes the site
as `Center geodetic : 139.7414, 35.6581, .040` and `Center radii :
6378.137, 6378.137, 6356.752 km` — the WGS84 ellipsoid (a = 6378137 m,
1/f = 298.257223563; NIMA TR8350.2 Table 3.1), which is exactly the
ellipsoid `oxiephemeris_bodies::topocentric::Observer` uses. Cross-check:
Horizons' `Center cylindric : 139.7414, 5188.23768, 3697.45794`
{E-lon(deg), Dxy(km), Dz(km)} agrees with our geodetic → ITRS conversion
for this site (5 188 237.68 m, 3 697 457.94 m) to all printed digits.

## Exact curl template used

Identical to the geocentric set except for the three site parameters and
`QUANTITIES='1,2,20'` (quantity 31 was not requested here):

```sh
curl -sG "https://ssd.jpl.nasa.gov/api/horizons.api" \
  --data-urlencode "format=text" \
  --data-urlencode "COMMAND='<command-code>'" \
  --data-urlencode "OBJ_DATA='NO'" \
  --data-urlencode "MAKE_EPHEM='YES'" \
  --data-urlencode "EPHEM_TYPE='OBSERVER'" \
  --data-urlencode "CENTER='coord@399'" \
  --data-urlencode "COORD_TYPE='GEODETIC'" \
  --data-urlencode "SITE_COORD='139.7414,35.6581,0.04'" \
  --data-urlencode "START_TIME='<epoch>'" \
  --data-urlencode "STOP_TIME='<epoch + 2 minutes>'" \
  --data-urlencode "STEP_SIZE='1 m'" \
  --data-urlencode "TIME_TYPE='TT'" \
  --data-urlencode "QUANTITIES='1,2,20'" \
  --data-urlencode "ANG_FORMAT='DEG'" \
  --data-urlencode "EXTRA_PREC='YES'" \
  --data-urlencode "APPARENT='AIRLESS'" \
  --data-urlencode "CSV_FORMAT='YES'" \
  -o "<fixture>.txt"
```

As in the geocentric set, a 2-minute / 3-row window was requested for
visual sanity and **the first CSV row (the requested epoch) is the one
recorded in `manifest.json`**. All four requests returned HTTP 200 with
`$$SOE` present on the first attempt. The topocentric CSV rows carry two
extra marker columns after the date (solar-presence and lunar-presence
codes, e.g. `*,m`) that the geocentric set had blank; they are ignored.

## Bodies and epochs

| Fixture         | COMMAND | Body                 | Epoch (TT)           | Note |
|-----------------|---------|----------------------|----------------------|------|
| `moon_2020.txt` | 301     | Moon body center     | 2020-01-15 00:00     | Moon above the horizon (marker `m`), parallax ≈ 1° |
| `moon_1987.txt` | 301     | Moon body center     | 1987-04-10 00:00     | Moon below the horizon — parallax geometry differs |
| `mars_2003.txt` | 4       | Mars **system barycenter** | 2003-08-27 00:00 | Historic close opposition, delta ≈ 0.3728 AU |
| `venus_2015.txt`| 299     | Venus body center    | 2015-04-01 00:00     | |

Command codes follow the geocentric set's convention (4 = Mars system
barycenter, matching what the DE files natively store and what
`Target::Mars` evaluates).

## Per-epoch Earth-orientation parameters (IERS `finals2000A.all`)

Horizons' apparent frame is EOP-corrected; reproducing its topocentric
states requires UT1 and polar motion. The values below were read from
`data/iers/finals2000A.all` (IERS combined series, standard Bulletin A
fixed-column format: `PM-x` cols 19–27 arcsec, `PM-y` cols 38–46 arcsec,
`UT1−UTC` cols 59–68 s; all four rows are IERS **final** values, flag
`I`). They are hardcoded in `tests/topocentric_spotcheck.rs` with the
same source lines quoted in comments, and the test cross-checks them
against this manifest.

| Fixture | MJD (0h UTC) | UT1−UTC (s) | x_p (″) | y_p (″) |
|---------|--------------|-------------|---------|---------|
| `moon_2020.txt` | 58863 | −0.1799645 | 0.059576 | 0.293792 |
| `moon_1987.txt` | 46895 | −0.2910912 | 0.086991 | 0.210887 |
| `mars_2003.txt` | 52878 | −0.3492927 | 0.260148 | 0.412496 |
| `venus_2015.txt`| 57113 | −0.5750634 | 0.013822 | 0.396474 |

Source lines, verbatim (trailing padding trimmed):

```
20 115 58863.00 I  0.059576 0.000031  0.293792 0.000018  I-0.1799645 0.0000022  0.7463 0.0017  I
87 410 46895.00 I  0.086991 0.000262  0.210887 0.000501  I-0.2910912 0.0000149  1.7858 0.0102  I
 3 827 52878.00 I  0.260148 0.000064  0.412496 0.000080  I-0.3492927 0.0000067 -0.0703 0.0055  I
15 4 1 57113.00 I  0.013822 0.000026  0.396474 0.000048  I-0.5750634 0.0000072  1.3823 0.0050  I
```

**Epoch-vs-tabulation note.** The fixtures' epochs are 00:00 **TT**,
which is 55.184–69.184 s *before* 0h UTC of the same calendar date
(TT − UTC = 32.184 s + the cumulative leap seconds). The EOP rows above
are tabulated at 0h UTC of that same date, i.e. ≈ 1 minute after each
epoch. No interpolation is applied: over one minute UT1−UTC changes by
≲ 2 µs (≲ 1 mm of station displacement) and the pole by ≲ 0.2 µas —
many orders of magnitude below the test gates. Horizons interpolates its
own EOP file (`eop.260703.p260929`, an independent solution from the same
observations); solution-to-solution differences are ≲ 0.1 ms in UT1 and
≲ 0.3 mas in the pole, i.e. ≲ 5 cm of station position.

## Manifest fields

Same schema style as `../horizons/manifest.json`, minus the quantity-31
fields (not requested), plus:

- `site_east_lon_deg`, `site_lat_deg`, `site_height_m`: the geodetic site
  of the request (identical for all four entries)
- `eop_mjd`: the MJD of the `finals2000A.all` row used
- `dut1_s`, `xp_arcsec`, `yp_arcsec`: the EOP values from that row
- `finals_line`: the row itself, verbatim (trimmed), so the numbers stay
  auditable without the (uncommitted) `data/iers/finals2000A.all`

## Sanity checks performed

- Both RA pairs in `[0, 360)`, both Dec pairs in `[−90, 90]`, no blank /
  `n.a.` CSV fields in any first row.
- `delta_au` physically plausible per body: Moon 0.00244 / 0.00267 AU
  (inside the perigee–apogee envelope as seen from a station), Mars
  0.37276 AU (the 2003 close opposition), Venus 1.20733 AU.
- Minute-to-minute rows vary smoothly in every file.
- The two extra epochs of each file are left untouched in the raw
  responses for transparency (only the first row is in the manifest).
- The Moon fixtures deliberately differ in horizon geometry (`m` marker
  above vs below the horizon) so the topocentric parallax is exercised
  at two distinct zenith angles; the parallax-consistency test in
  `topocentric_spotcheck.rs` checks both against the sine-rule identity
  `sin p = ρ sin z / d_topo`.
