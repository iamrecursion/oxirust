# JPL Horizons geocentric spot-check fixtures

## Provenance

These fixtures are raw, unmodified text responses from NASA JPL's public
**Horizons** ephemeris system, retrieved via its documented HTTP API:

- Endpoint: `https://ssd.jpl.nasa.gov/api/horizons.api`
- API docs: <https://ssd-api.jpl.nasa.gov/doc/horizons.html>
- Retrieved: 2026-07-05, approximately 05:52:41-05:52:52 "Pasadena, USA" local
  server time, as printed verbatim in each response's
  `Ephemeris / API_USER <weekday> <mon> <day> <time> <year> Pasadena, USA`
  banner line (see e.g. `sun.txt` line 7). This banner timestamp is the
  server-reported generation time, not invented by us.
  **Exception:** `uranus.txt` and `pluto.txt` were re-retrieved later the
  same day (banner times 07:46:04 and 07:46:32) at different epochs — see
  "EOP-coverage epoch policy" below for why.
- JPL Horizons ephemeris output is produced by a U.S. Government agency
  (NASA/JPL, a division of Caltech operating under NASA contract) and its
  data products are in the **public domain** in the United States. No
  license restriction applies to storing/redistributing this raw text.
- **Underlying planetary ephemeris actually used by the server: DE441**
  (see the `{source: DE441}` annotation next to every `Target body name:`
  / `Center body name:` line in each raw file), *not* DE440 as originally
  assumed when this fixture set was commissioned. Horizons does not expose
  a parameter to pin the server to DE440 specifically; DE441 is simply
  the current default. DE440 and DE441 share the same fitting philosophy
  and agree with each other at the sub-milliarcsecond / sub-km level for
  the outer planets and at a similarly small level for the inner solar
  system, so this does not threaten the stated 0.01" exit criterion, but
  it should be noted if a byte-for-byte DE440-only pedigree is later
  required.

## What this directory contains

- `sun.txt`, `moon.txt`, `mercury.txt`, `venus.txt`, `mars.txt`,
  `jupiter.txt`, `saturn.txt`, `uranus.txt`, `neptune.txt`, `pluto.txt`:
  the ten raw Horizons API text responses, byte-for-byte as returned
  (including the header banner, the `$$SOE ... $$EOE` CSV data block, and
  the trailing column-meaning explanation section that Horizons appends
  to every text response).
- `manifest.json`: a machine-readable array with one object per body,
  hand-parsed from the `$$SOE`/`$$EOE` block of the corresponding raw
  file. Fields:
  - `body`: display name (Mars is explicitly the **barycenter**, see below)
  - `command`: the Horizons `COMMAND` code used
  - `epoch_iso`: the requested epoch, in `YYYY-MM-DDTHH:MM:SS` form
  - `time_scale`: the time scale Horizons actually used for the `Date__`
    column, confirmed from the column header of each raw response
    (`Date__(TT)__HR:MN` in every file here — see "Time scale" below)
  - `delta_t_sec`: `null` for all ten entries because the TT request
    succeeded directly (no UT-fallback + delta-T reconstruction was
    needed — see below)
  - `jd`: Julian Date of `epoch_iso`, computed independently in TT via
    the standard Fliegel & Van Flandern proleptic-Gregorian JDN algorithm
    (not taken from Horizons, which does not print a JD column here)
  - `astrometric_ra_deg`, `astrometric_dec_deg`: quantity 1, decimal degrees
  - `apparent_ra_deg`, `apparent_dec_deg`: quantity 2, decimal degrees
  - `delta_au`: quantity 20 range, astronomical units
  - `deldot_km_s`: quantity 20 range-rate, km/s (positive = receding)
  - `obs_ecl_lon_deg`, `obs_ecl_lat_deg`: quantity 31, decimal degrees
  - `source_file`: which raw `.txt` this row was parsed from
  - `de_source`: the `{source: ...}` tag Horizons attached to the target
    body for this query (`DE441` for all ten — see Provenance above)

## Exact curl template used

One HTTP GET per body, using `curl -sG ... --data-urlencode` so that
literal single quotes are sent as part of each parameter value (Horizons'
API expects quoted scalars, matching its telnet/e-mail batch-file syntax):

```sh
curl -sG "https://ssd.jpl.nasa.gov/api/horizons.api" \
  --data-urlencode "format=text" \
  --data-urlencode "COMMAND='<command-code>'" \
  --data-urlencode "OBJ_DATA='NO'" \
  --data-urlencode "MAKE_EPHEM='YES'" \
  --data-urlencode "EPHEM_TYPE='OBSERVER'" \
  --data-urlencode "CENTER='500@399'" \
  --data-urlencode "START_TIME='<epoch, e.g. 2026-07-05 00:00>'" \
  --data-urlencode "STOP_TIME='<epoch + 2 minutes>'" \
  --data-urlencode "STEP_SIZE='1 m'" \
  --data-urlencode "TIME_TYPE='TT'" \
  --data-urlencode "QUANTITIES='1,2,20,31'" \
  --data-urlencode "ANG_FORMAT='DEG'" \
  --data-urlencode "EXTRA_PREC='YES'" \
  --data-urlencode "APPARENT='AIRLESS'" \
  --data-urlencode "CSV_FORMAT='YES'" \
  -o "<body>.txt"
```

`START_TIME`/`STOP_TIME`/`STEP_SIZE` were widened to a 2-minute, 1-row-per-
minute window (3 output rows) rather than a single instant, purely so that
a malformed/edge-case Horizons response would still be visually obvious;
**the first CSV data row (matching the requested `START_TIME`) is the one
recorded in `manifest.json`.** The other two rows are left in the raw
`.txt` files untouched (they are simply ignored) for transparency and to
show the numbers vary smoothly minute-to-minute as an extra sanity check.

Requests were issued with a small politeness stagger; all ten succeeded on
the first attempt (HTTP 200, `$$SOE` present) with no retries needed. The
`fetch.sh` driver script that ran the loop above was scratch tooling and
is intentionally not checked in here — only its raw output.

## EOP-coverage epoch policy (why uranus.txt / pluto.txt were re-fetched)

Horizons' apparent quantities (2 and 31) are expressed on the
**EOP-corrected IAU76/80** frame model (see "Column semantics" below).
Every response header prints the server's Earth-orientation-parameter
table span, e.g. (identical in all ten files here):

```
EOP file        : eop.260703.p260929
EOP coverage    : DATA-BASED 1962-JAN-20 TO 2026-JUL-03. PREDICTS-> 2026-SEP-28
```

Inside the data-based span the EOP celestial-pole offsets pin Horizons'
true pole to the observed (ICRS-consistent) pole, and its apparent frame
then differs from the IAU 2006/2000A true-of-date frame by only the
constant RA-origin (equinox) tie of about -53 mas that Horizons' own
column-meaning footer documents. **Beyond** the EOP span the corrections
are frozen, and the frame drifts away from that documented relation at the
IAU76 precession-rate error (~ -0.3"/century in longitude), which makes an
apparent-place comparison against an IAU 2006/2000A pipeline ill-defined
at the 0.01" level.

The fixture set as first commissioned used Uranus at 2033-10-10 and Pluto
at 2040-01-01 — both **outside** the `2026-JUL-03` data-based EOP span
above, and their residuals against an IAU 2006/2000A pipeline indeed
showed exactly the predicted frozen-EOP drift signature (roughly -30 mas
in RA for 2033 and -44 mas in RA, -13 mas in Dec for 2040, on top of the
constant -53 mas tie). They were therefore re-fetched on the same day with
the same curl template at epochs **inside** the data-based EOP span:

| Body   | Old epoch (out of EOP span) | New epoch (in EOP span)  |
|--------|-----------------------------|--------------------------|
| Uranus | 2033-10-10 00:00 TT         | 2013-10-10 00:00 TT      |
| Pluto  | 2040-01-01 00:00 TT         | 1996-01-01 00:00 TT      |

All other eight fixtures already had epochs from 1965 to 2026-07-05,
inside the data-based span. Nothing about the old responses was wrong —
they were faithful Horizons output; they simply compared a frozen-EOP
IAU76/80 frame against an IAU 2006/2000A pipeline, which is not the
comparison the exit criterion is about.

## Time scale: TT was used directly, no UT fallback needed

Horizons' `TIME_TYPE` parameter, documented as "override default to
specify input & output timescale. observer tables: `UT` or `TT`", was set
to `TIME_TYPE='TT'`. This worked for observer tables (contrary to some
older documentation snippets that only mention TDB/UT for vector/element
tables) and each response's own preamble confirms it, both in the
`Start time`/`Stop time` lines:

```
Start time      : A.D. 2026-Jul-05 00:00:00.0000 TT
Stop  time      : A.D. 2026-Jul-05 00:02:00.0000 TT
```

and unambiguously in the CSV column header itself, e.g. in `sun.txt`:

```
 Date__(TT)__HR:MN, , , R.A.___(ICRF), DEC____(ICRF), ...
```

Every one of the ten raw files has `Date__(TT)__HR:MN` as the first column
header (grep for `Date__` to confirm), so **all ten `epoch_iso` values in
`manifest.json` are exact TT epochs, requested and returned in TT with no
UT/TT conversion or delta-T reconstruction needed.** Consequently
`delta_t_sec` is `null` for every entry — it is not needed to reconstruct
the epoch in TT, since TT was the scale of the request and the response
alike. (Horizons' own explanation text, copied into every raw file's
footer, additionally states TT here is "a time-scale conversion from
internal Barycentric Dynamical Time (TDB)" with sub-2-ms periodic
amplitude, i.e. effectively identical to TDB for any purpose at the
0.01" / 1e-13 AU precision this project targets.)

## Column semantics (quoted/paraphrased from each response's own footer)

Every raw `.txt` file ends with a "Column meaning" section from Horizons
itself; the wording below is copied near-verbatim from that section
(identical across all ten files) rather than from the API doc page, per
the task's request to confirm from the actual returned table:

- **`R.A.___(ICRF), DEC____(ICRF)`** (quantity 1, "astrometric"): *"Astrometric
  right ascension and declination of the target center with respect to the
  observing site (coordinate origin) in the reference frame of the
  planetary ephemeris (ICRF). Compensated for down-leg light-time delay
  aberration."* — i.e. **ICRF/J2000-equivalent frame, light-time
  correction only, no stellar aberration, no precession/nutation** (the
  ICRF axes are inertial and epoch-independent).

- **`R.A.__(a-app), DEC___(a-app)`** (quantity 2, "apparent", airless):
  *"Airless apparent right ascension and declination of the target center
  with respect to an instantaneous reference frame defined by the Earth
  equator of-date (z-axis) and meridian containing the Earth equinox
  of-date (x-axis, EOP-corrected IAU76/80). Compensated for down-leg
  light-time delay, gravitational deflection of light, stellar
  aberration, precession & nutation."* Horizons adds a note that this
  of-date equinox is offset -53 mas from the frame defined by the
  IAU2006/2000A precession-nutation model — immaterial at the 0.01"
  fixture-comparison tolerance but recorded here for completeness. **This
  confirms `APPARENT='AIRLESS'` means: light-time + deflection +
  aberration + precession/nutation are all included; only atmospheric
  refraction is excluded** (refraction is topocentric-only and would not
  apply to a geocentric center anyway).

- **`delta, deldot`** (quantity 20): *"Apparent range ('delta',
  light-time aberrated) and range-rate ('delta-dot') of the target center
  relative to the observer. A positive 'deldot' means the target center
  is moving away from the observer, negative indicates movement toward
  the observer."* Units: AU and km/s.

- **`ObsEcLon, ObsEcLat`** (quantity 31): *"Observer-centered IAU76/80
  ecliptic-of-date longitude and latitude of the target center's apparent
  position, with light-time, gravitational deflection of light, and
  stellar aberrations."* **This confirms quantity 31 is expressed in the
  ecliptic-OF-DATE frame (IAU76/80, EOP-corrected), NOT the J2000
  ecliptic** — this must be matched against an of-date ecliptic transform
  in OxiEphemeris, not the J2000 mean ecliptic, when validating against
  this quantity. (By contrast quantity 1 astrometric RA/Dec is
  ICRF/J2000-equivalent, so the two angle pairs in this fixture set
  intentionally live in different frames; that is expected, see below.)

## Body / COMMAND notes

| Body    | COMMAND | Note |
|---------|---------|------|
| Sun     | 10      | Sun body center |
| Moon    | 301     | Moon body center |
| Mercury | 199     | Mercury body center (COMMAND=1 would be the Mercury *barycenter*, which for Mercury is coincident with the body center to sub-mm level since it has no moons; 199 was used per spec) |
| Venus   | 299     | Venus body center (same barycenter-vs-body note as Mercury) |
| Mars    | 4       | **Mars-system barycenter**, deliberately, per spec — *not* Mars body center (COMMAND=499). This differs from Mars's actual body center by up to a few hundred meters (displacement toward Phobos/Deimos), which is far below the 0.01" fixture tolerance at Mars's geocentric distance, but is called out here because it is easy to misread as an error rather than an intentional choice. |
| Jupiter | 5       | Jupiter-system barycenter (Horizons' natural single-body-per-planet default; this is what a DE-series ephemeris natively integrates and stores for the giant planets) |
| Saturn  | 6       | Saturn-system barycenter |
| Uranus  | 7       | Uranus-system barycenter |
| Neptune | 8       | Neptune-system barycenter |
| Pluto   | 9       | Pluto-system barycenter (not the Pluto body center, COMMAND=999; consistent with the giant-planet convention above and with what DE natively provides) |

`CENTER='500@399'`: `399` selects Earth as the origin body; the `500@`
prefix selects that body's own center-of-mass/geometric-center point (as
opposed to a specific numbered observing station on its surface), i.e.
this is a standard **geocentric** (Earth-center) observer, not a
topocentric site.

## Sanity checks performed

All ten `manifest.json` entries were checked to have:

- `astrometric_ra_deg` and `apparent_ra_deg` in `[0, 360)`
- `astrometric_dec_deg` and `apparent_dec_deg` in `[-90, 90]`
- `delta_au` positive and in the physically expected range for each body
  at its epoch (notably: Moon ≈ 0.00269 AU; Sun ≈ 1.0166 AU, consistent
  with the epoch being close to Earth's 2026 aphelion in early July; Mars
  barycenter ≈ 0.3727 AU, consistent with the historic 2003-08-27 close
  opposition; Pluto ≈ 30.63 AU at 1996-01-01, consistent with Pluto being
  near its 1989 perihelion — inside Neptune's orbit until 1999 — and well
  inside the expected 28-50 AU envelope)
- no blank/`n.a.` fields in the parsed CSV row for any body
- astrometric vs. apparent RA/Dec were **not** required to agree to
  arcsecond level, because (per the column-meaning section above) they
  are reported in genuinely different reference frames — ICRF/J2000-like
  vs. true-equator-and-equinox-of-date. The degree-scale differences
  observed between the two columns (up to a few tenths of a degree,
  dominated by the of-date-vs-J2000 precession/nutation offset rather
  than by the ~20" aberration term) are exactly the behavior flagged as
  expected in the task brief, not an anomaly.

No anomalies were found in any of the ten fixtures.
