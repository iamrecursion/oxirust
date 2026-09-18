# oxiephemeris (Python)

Pure-Rust astrology / ephemeris engine with Python bindings.

Compute full **natal charts**, **synastry**, **transits**, **secondary
progressions**, and **midpoint composites** — signs, houses (7 systems),
aspects, essential dignities, Arabic Parts, declinations — plus their
**Linked Open Data** (RDF Turtle / N-Triples with SKOS and PROV-O
provenance).

The astronomy comes from JPL DE ephemerides (DE440/DE441), passed in as a
`bytes` buffer, so nothing touches the filesystem.

```python
import oxiephemeris as oxi

with open("de440/linux_p1550p2650.440", "rb") as f:
    de = f.read()

# The Unix epoch at the Royal Observatory, Greenwich.
chart = oxi.natal(de, "1970-01-01T00:00:00Z", lat=51.4779, lon=0.0)
sun = next(b for b in chart["bodies"] if b["body"] == "Sun")
print(sun["sign"], round(sun["sign_degrees"], 2), "house", sun["house"])
# -> Capricorn 10.16 house 4

turtle = oxi.natal_rdf(de, "1970-01-01T00:00:00Z", lat=51.4779, lon=0.0)
print(turtle.splitlines()[-3:])
```

## API

- `natal(de, date, lat, lon, *, alt=0, dut1=0, cal="gregorian",
  system="placidus", sidereal=None, rulership="traditional") -> dict`
- `natal_rdf(...) -> str` (Turtle; `ntriples=True` for N-Triples)
- `synastry(de, a, b, *, system, dut1, cal) -> dict` /
  `synastry_rdf(...) -> str` — `a`/`b` are objects with `date`, `lat`, `lon`
- `transit(de, natal, transit, ...) -> dict`
- `progress(de, natal, target, ...) -> dict`
- `composite(de, a, b, ...) -> dict` / `composite_rdf(...) -> str`
- `julday(year, month, day, hours=0, calendar="gregorian") -> float`
- `revjul(jd, calendar="gregorian") -> dict`

All results are the same stable schema the CLI and the WASM build emit.

Apache-2.0.
