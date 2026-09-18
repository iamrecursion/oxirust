# JPL Horizons sidereal-time (quantity 7) oracle fixtures

## Provenance

These fixtures are raw, unmodified text responses from NASA JPL's public
**Horizons** ephemeris system, retrieved via its documented HTTP API:

- Endpoint: `https://ssd.jpl.nasa.gov/api/horizons.api`
- API docs: <https://ssd-api.jpl.nasa.gov/doc/horizons.html>
- Retrieved: 2026-07-05, banner times 08:56:09–08:56:15 "Pasadena, USA"
  server-local (printed verbatim in each response's
  `Ephemeris / API_USER … Pasadena, USA` line).
  **Exception:** `1987-04-10.txt` was retrieved later the same day
  (banner time 09:09:57) with the identical curl template, after the
  1975 epoch turned out to sit in the pre-VLBI UT1 noise floor — see
  "Why the early gated epoch is 1987, not 1975" below.
- JPL Horizons output is produced by a U.S. Government agency (NASA/JPL)
  and is in the public domain in the United States; no license
  restriction applies to storing/redistributing this raw text.
- Server EOP file at retrieval time (printed in every header):
  `EOP file : eop.260703.p260929`,
  `EOP coverage : DATA-BASED 1962-JAN-20 TO 2026-JUL-03`. All three
  epochs below lie **inside** the data-based EOP span, which is required
  for the constant −52.93 mas equinox-tie comparison to be valid (see
  `../horizons/README.md`, "EOP-coverage epoch policy").

## What this directory contains

Four raw Horizons API text responses, byte-for-byte as returned:

| File             | Epoch (UTC)         | UT1−UTC used by the test | Gated?        |
|------------------|---------------------|--------------------------|---------------|
| `1975-06-20.txt` | 1975-06-20 00:00:00 | +0.2239539 s             | no (see below)|
| `1987-04-10.txt` | 1987-04-10 00:00:00 | −0.2910912 s             | yes           |
| `1998-11-10.txt` | 1998-11-10 00:00:00 | −0.2191202 s             | yes           |
| `2019-03-05.txt` | 2019-03-05 00:00:00 | −0.0913344 s             | yes           |

Each response contains a `$$SOE … $$EOE` CSV block with three rows
(1-minute steps over a 2-minute window, same policy as the
`../horizons/` fixtures: the extra rows make malformed output obvious
and show the quantity varying smoothly). **The first row — the one at
the requested epoch — is the one the test compares against.** The
column of interest is `L_Ap_Sid_Time` (Horizons quantity 7), *local
apparent sidereal time*, format `HH MM SS.ffff` (hours-minutes-seconds
of time), whose meaning Horizons' own footer states as: "The angle
measured westward in the body true-equator of-date plane from the
meridian containing the body-fixed observer to the meridian containing
the true Earth equinox".

## Exact curl template used

One HTTP GET per epoch (`curl -sG … --data-urlencode` so the literal
single quotes are sent as part of each value):

```sh
curl -sG "https://ssd.jpl.nasa.gov/api/horizons.api" \
  --data-urlencode "format=text" \
  --data-urlencode "COMMAND='10'" \
  --data-urlencode "OBJ_DATA='NO'" \
  --data-urlencode "MAKE_EPHEM='YES'" \
  --data-urlencode "EPHEM_TYPE='OBSERVER'" \
  --data-urlencode "CENTER='coord@399'" \
  --data-urlencode "COORD_TYPE='GEODETIC'" \
  --data-urlencode "SITE_COORD='0,0,0'" \
  --data-urlencode "START_TIME='<epoch date> 00:00'" \
  --data-urlencode "STOP_TIME='<epoch date> 00:02'" \
  --data-urlencode "STEP_SIZE='1 m'" \
  --data-urlencode "TIME_TYPE='UT'" \
  --data-urlencode "QUANTITIES='7'" \
  --data-urlencode "APPARENT='AIRLESS'" \
  --data-urlencode "EXTRA_PREC='YES'" \
  --data-urlencode "CSV_FORMAT='YES'" \
  -o "<epoch date>.txt"
```

Choices, and why they are safe:

- **Site**: `CENTER='coord@399'` + `COORD_TYPE='GEODETIC'` +
  `SITE_COORD='0,0,0'` — a topocentric site at geodetic longitude 0,
  latitude 0, altitude 0 km (each header echoes
  `Center geodetic : 0.0, 0.0, 0.0`). Quantity 7 depends only on the
  site and the clock. At longitude 0 **LAST = GAST**; and at latitude 0
  the polar-motion longitude correction (USNO Circular 179 eq. 2.16,
  `(x_p sin λ + y_p cos λ) tan φ / 3600`) vanishes identically —
  doubly so, since it is also zero at λ = 0 for the `sin λ` part and
  the `tan φ` factor kills the rest.
- **`COMMAND='10'`** (Sun): an arbitrary bright body; quantity 7 is a
  property of the observing site, not of the target.
- **`TIME_TYPE='UT'`**: for the 1962+ epochs used here Horizons' UT
  time scale **is UTC** — each response's own footer states: "Times
  PRIOR to 1962 are UT1 … Times AFTER 1962 are in UTC". So the request
  epochs are exact UTC instants.
- **Epochs at 00:00:00 UTC**: the `finals2000A.all` EOP series tabulates
  UT1−UTC at 0h UTC of each MJD, so no interpolation is needed.

## UT1−UTC provenance (hardcoded into the test, not read at run time)

Values read from the workspace EOP snapshot `data/iers/finals2000A.all`
(IERS combined series "finals2000A", standard format: I2,I2,I2 date,
F9.2 MJD, …, with UT1−UTC (sec of time) in columns 59–68; snapshot
downloaded 2026-07-05). The exact source lines, quoted with the MJD and
the UT1−UTC field (line numbers as of that snapshot):

```
line   900: 75 620 42583.00 I  0.143347 0.029711  0.261640 0.029269  I 0.2239539 0.0007530 ...
line  5212: 87 410 46895.00 I  0.086991 0.000262  0.210887 0.000501  I-0.2910912 0.0000149 ...
line  9444: 981110 51127.00 I  0.164879 0.000054  0.381264 0.000082  I-0.2191202 0.0000094 ...
line 16864: 19 3 5 58547.00 I  0.043918 0.000029  0.350848 0.000026  I-0.0913344 0.0000057 ...
```

i.e. UT1−UTC = +0.2239539 s at MJD 42583 (1975-06-20), −0.2910912 s at
MJD 46895 (1987-04-10), −0.2191202 s at MJD 51127 (1998-11-10),
−0.0913344 s at MJD 58547 (2019-03-05). These are hardcoded in
`tests/sidereal.rs` (with the same citation) so the test is
deterministic and does not silently track a refreshed EOP file.
Horizons uses its own EOP file (see above); for the three gated (VLBI
era) epochs the residual UT1 disagreement between EOP realizations is
far below the 1 ms test gate (measured tied residuals ≤ 0.06 ms).

## Why the early gated epoch is 1987, not 1975

The fixture set was commissioned with 1975/1998/2019. The 1998 and 2019
epochs closed to −0.04 ms and −0.05 ms after the constant −52.93 mas
equinox tie — validating the whole GAST chain (ERA, GMST, equation of
the equinoxes, tie, time scales) at the 50 µs level — but 1975 showed a
tied residual of +1.34 ms. Investigation:

- Unlike the TT-driven RA spotchecks of `../horizons/` (which validated
  the −52.93 mas frame tie to ≤ 3 mas at epochs back to 1965), LAST
  depends directly on **UT1**. A 1.34 ms sidereal residual is ≈ 1.34 ms
  of UT1.
- At MJD 42583 the modern IERS series *themselves* disagree:
  `finals2000A.all` gives UT1−UTC = +0.2239539 s while EOP 20 C04
  (`eopc04.1962-now`, Paris Observatory) gives +0.2236377 s with a
  quoted formal error of **1.9 ms** — pre-VLBI UT1 comes from optical
  astrometry and is genuinely uncertain at the millisecond level.
- The UT1 implied by Horizons' 1975 LAST (inverting our GAST, which the
  1998/2019 epochs pin down to 50 µs) is ≈ +0.22262 s — about 1.3 ms
  from finals2000A and 1.0 ms from C04. Horizons' `eop.*` file descends
  from JPL's own historical Earth-orientation solution, a third
  independent realization.
- Conclusion: at 1975 the comparison measures EOP-series disagreement,
  not the sidereal-time model; a 1 ms gate is not meaningful there. The
  gate was **not** loosened; instead the earliest *gated* epoch was
  re-chosen inside the VLBI era at a date where `finals2000A.all` and
  EOP 20 C04 agree to 0.0074 ms (1987-04-10, MJD 46895: −0.2910912 s vs
  −0.2910986 s, C04 formal error 0.04 ms). The 1975 fixture is kept and
  still evaluated/printed by the test as an *informational* row so the
  pre-VLBI caveat stays visible instead of being quietly discarded.

## Frame note (why a constant −52.93 mas tie is applied)

Horizons' apparent quantities are expressed on the EOP-corrected
IAU76/80 frame; its RA origin is offset by a constant −52.93 mas from
the IAU 2006/2000A true-of-date frame realized by OxiEphemeris (full
derivation from IERS TN36 eq. 5.21 frame-bias angles in
`tests/horizons_spotcheck.rs`, validated there against ten Horizons
RA fixtures spanning 1965–2026; Horizons' own footers document the
same value rounded, "−53 mas"). Apparent sidereal time is the apparent
RA of the meridian, so the identical tie applies:
`LAST(Horizons) = GAST(IAU 2006/2000A) + (−0.05293″)`, i.e.
−0.05293″/15 = −3.529 ms of time. The test always prints both the raw
and the tied residuals and gates only the tied one (< 1 ms).
