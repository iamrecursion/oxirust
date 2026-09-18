# @cooljapan/oxiephemeris

A full Western-astrology chart engine as **WebAssembly** — compute natal
charts, synastry, transits, secondary progressions, and midpoint
composites, plus their **Linked Open Data** (RDF Turtle / N-Triples with
SKOS and PROV-O), **entirely in the browser**. No server, no upload: the
JPL DE ephemeris file you supply never leaves the page.

Published to npm as `@cooljapan/oxiephemeris`.

## Install

```sh
npm install @cooljapan/oxiephemeris
```

```js
import init, { natal_json, natal_turtle } from '@cooljapan/oxiephemeris';
await init();
```

## Build from source

Local/demo build (unscoped, `target web` — what `www/index.html` actually
uses via a relative import):

```sh
wasm-pack build crates/oxiephemeris-wasm --target web --out-dir pkg
```

Published-package-shaped build (matches CI):

```sh
wasm-pack build crates/oxiephemeris-wasm --scope cooljapan --target bundler --out-dir pkg-bundler
```

## Use (browser / ES modules)

```js
import init, { natal_json, natal_turtle } from './pkg/oxiephemeris_wasm.js';
await init();

// DE440/DE441 bytes, e.g. from <input type="file"> → arrayBuffer():
const de = new Uint8Array(buffer);
// The Unix epoch at the Royal Observatory, Greenwich.
const req = JSON.stringify({ date: "1970-01-01T00:00:00Z", lat: 51.4779, lon: 0.0 });

const chart = JSON.parse(natal_json(de, req));
const sun = chart.bodies.find(b => b.body === "Sun");
// -> Capricorn, 10.16°, house 4

const turtle = natal_turtle(de, req);   // RDF Turtle with SKOS + PROV-O
```

`crates/oxiephemeris-wasm/www/index.html` is a self-contained demo page.

## Functions

Each takes the DE bytes and a JSON request string, and returns a JSON (or
Turtle) string:

- `natal_json(de, req)` / `natal_turtle(de, req, base_iri?)`
- `synastry_json(de, req)` / `synastry_turtle(de, req, base_iri?)`
- `transit_json(de, req)`
- `progress_json(de, req)`
- `composite_json(de, req)` / `composite_turtle(de, req, base_iri?)`

The request/response schemas are identical to the Python binding and the
CLI, so results are bit-for-bit the same across all three.

Apache-2.0.
