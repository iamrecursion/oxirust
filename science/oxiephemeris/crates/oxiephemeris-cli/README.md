# oxiephemeris-cli

`oxieph`: the command-line front end for OxiEphemeris — Julian Date/
calendar conversion, body positions, house cusps, and full natal,
synastry, transit, progression, and composite charts.

This crate is binary-only (no library target beyond its own internal
modules). `chart`, `synastry`, `transit`, `progress`, and `composite` are
thin renderers over [`oxiephemeris-chart`](../oxiephemeris-chart)'s
compute functions, so their numeric output is bit-identical to the Python
and WebAssembly bindings built on that same facade; `convert`, `pos`, and
`houses` call `oxiephemeris-core`/`-de`/`-bodies`/`-astro` more directly.
Date/time parsing is hand-rolled (the `iso8601` module) — no `chrono`
dependency.

```sh
cargo install --path crates/oxiephemeris-cli   # installs the `oxieph` binary
oxieph --help                                  # nine subcommands
```

## Commands that need no ephemeris file

`convert` (Julian Date <-> calendar date) and `houses` (cusps and chart
angles) are pure calendar/spherical-trigonometry — no DE kernel required:

```sh
$ oxieph convert --date 2024-03-15T06:30:00
input = 2024-03-15T06:30:00 (Gregorian)
jd = 2460384.770833333

$ oxieph houses --system placidus --lat 35.6895 --lon 139.6917 1999-03-30T20:15:00
system = Placidus
jd_tt = 2451268.344492870
...
ascendant = 1.921085907 deg
mc = 271.113750714 deg
vertex = 180.825069614 deg
east_point = 1.322998091 deg
```

`--system` has **no default** on `houses` — it is a required flag
(`Usage: oxieph houses [OPTIONS] --system <SYSTEM> --lat <LAT> --lon <LON>
<DATE>`, with `--system` listed outside `[OPTIONS]`) — unlike every
chart-family command below, where `--system` defaults to `placidus`.

## Commands that need a DE kernel

`pos`, `chart`, `synastry`, `transit`, `progress`, and `composite` all
resolve a JPL DE classic-binary file (DE440/DE441) in the same order:
`--de PATH`, then `$OXIEPH_DE`, then `data/de440/linux_p1550p2650.440`
relative to the current directory (never an absolute path baked into the
binary).

```sh
$ oxieph pos sun 2000-01-01T12:00:00 --scale tt --de data/de440/linux_p1550p2650.440
body = Sun
jd_tt = 2451545.000000000
frame = ecliptic
center = geocentric
lon = 280.368165282 deg  (speed 1.019433643 deg/day)
lat = 0.000227407 deg  (speed -0.000006486 deg/day)
distance_au = 0.983327631989  (speed -0.000007354369 au/day)
light_time_days = 0.005679227
```

`chart` computes a full natal chart — bodies, houses, angles, nodes, Lots,
element/modality distribution, essential dignities, and the aspect table
— in one call (the reference chart used throughout this workspace: the
Unix epoch at the Royal Observatory, Greenwich):

```sh
$ oxieph chart 1970-01-01T00:00:00 --lat 51.4779 --lon 0.0 \
    --de data/de440/linux_p1550p2650.440
jd_tt = 2440587.500465196
sidereal = none (tropical)
sect = nocturnal

-- bodies --
Sun      10°09'22" Capricorn    house  4   dec = -23.0567 deg   dist_au =  0.983310846
Moon     10°41'56" Libra        house  1   dec =  -6.3401 deg   dist_au =  0.002620266
...

-- dignities (traditional) --
Mars     score  +3   triplicity
...

-- aspects --
Sun      square         Moon     offset =   -0.5428 deg (separating)
...
```

`synastry`/`transit`/`progress`/`composite` follow the same pattern for
two-chart comparisons:

```sh
$ oxieph synastry --date-a 1970-01-01T00:00:00 --lat-a 51.4779 --lon-a 0.0 \
    --date-b 2000-01-01T12:00:00 --lat-b 48.8566 --lon-b 2.3522 \
    --de data/de440/linux_p1550p2650.440
jd_tt_a = 2440587.500465196
jd_tt_b = 2451545.000742870

-- chart A --
Sun      10°09'22" Capricorn     speed =  +1.0193 deg/day
...

-- cross-aspects (A -> B) --
A Sun      conjunction    B Sun      offset =   -0.2126 deg (separating)
...
```

`transit` takes one natal chart plus a `--transit <DATE>` instant and
reports transiting-body aspects back to the natal chart; `progress` takes
one natal chart plus a `--target <DATE>` to progress to ("a day for a
year") and reports the progressed chart's aspects to the natal chart;
`composite` takes two charts, like `synastry`, but reports their midpoint
chart instead of cross-aspects.

## Structured and Linked Data output

Every chart-family command (`chart`, `synastry`, `transit`, `progress`,
`composite`) accepts `--format text|json|turtle|ntriples` (`text` is the
default; the legacy `--json` flag still works when `--format` is left at
its default):

```sh
$ oxieph chart 1970-01-01T00:00:00 --lat 51.4779 --lon 0.0 \
    --de data/de440/linux_p1550p2650.440 --format json
{"kind":"natal","jd_tt":2440587.500465196,"epoch_utc":"1970-01-01T00:00:00.000000Z","rulership":"traditional","sect":"nocturnal","bodies":[{"body":"Sun","lon_deg":280.15628416439495,...
```

`--base-iri`/`--chart-iri` (RDF formats only) control the minted subject
IRIs. `oxieph vocab` emits the `oxa:` ontology and `oxc:`/`oxs:` SKOS
concept schemes standalone (`--part all|ontology|concepts`, `--format`
defaulting to `turtle`) — the same vocabulary
[`oxiephemeris-lod`](../oxiephemeris-lod) serves over HTTP:

```sh
$ oxieph vocab --part ontology --format turtle
@prefix oxa: <https://cooljapan.tech/ns/oxiephemeris/astro#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
...
```

`pos`/`houses`/`chart` also accept `--sidereal <ayanamsha>` (e.g.
`lahiri`) to shift reported longitudes to a sidereal zodiac; `pos` alone
accepts `--frame {ecliptic,equatorial,j2000}` and
`--center {geo,helio,bary}`. Run `oxieph <command> --help` for the
complete, current flag reference for any subcommand.

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview. Licensed under Apache-2.0 (see workspace
[LICENSE](../../LICENSE)).
