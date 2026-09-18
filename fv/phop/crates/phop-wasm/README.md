# phop-wasm — discover **and prove** scientific laws, entirely in the browser

WebAssembly bindings for [phop](https://github.com/cool-japan/phop). They turn a web page into a
complete, **proof-carrying scientific-discovery environment** that never contacts a server:

> **Paste data → a closed-form law is discovered → canonicalized by a CAS → its range and roots are
> certified → and its properties are proved by an SMT solver. All of it runs in your browser tab.**

No backend. No install. Nothing is uploaded, and nothing is stored — close the tab and it is gone.

![phop in-browser demo](./www/demo.svg)

*Snapshot of the bundled demo on `y = eˣ`. The values shown are the actual runtime output; the two
`Proven` verdicts are produced by the **OxiZ SMT solver running inside the browser's WASM engine**.*

## Why this is hard to do anywhere else

A symbolic-regression result is only trustworthy if you can *check* it. Doing that check **client-side,
offline, with no install** requires the entire stack — the search engine, a computer-algebra system,
an interval verifier, and a full SMT solver — to compile to WebAssembly. The mainstream tools can't:

| Capability (in-browser, offline) | phop (Pure Rust → WASM) | PySR (Julia) | SymPy (Python) | Mathematica |
|---|:--:|:--:|:--:|:--:|
| Discover a closed-form law | ✅ | ❌ | ❌ | ❌ |
| CAS analysis + canonicalization | ✅ | ❌ | ❌ | ❌ |
| **Certified** range / root enclosure | ✅ | ❌ | ❌ | ❌ |
| **SMT-proved** properties | ✅ | ❌ | ❌ | ❌ |
| Zero server · zero install · zero trace | ✅ | ❌ | ❌ | ❌ |

Julia and CPython have no browser runtime; Mathematica is proprietary and server-bound. The
COOLJAPAN pure-Rust ecosystem — [`oxieml`](https://crates.io/crates/oxieml) (EML-IR CAS),
[`oxiz`](https://crates.io/crates/oxiz) (a Z3-class SMT solver), and `scirs2-symbolic` (e-graphs) —
*all* compile to `wasm32`, so the whole pipeline ships as one ~0.9 MB (gzipped) module. To our
knowledge no other toolchain assembles a **proof-carrying discovery environment that runs purely in
the browser**.

## The verified pipeline (one call, all client-side)

```text
data ─▶ discover (EML-tree + rich-leaf search)
     ─▶ analyze    (derivative / antiderivative / Maclaurin · oxieml CAS)
     ─▶ canonical  (equality-saturation e-graph · scirs2-symbolic)
     ─▶ certify    (interval range enclosure + interval-Newton root · sound)
     ─▶ prove      (SMT: "no root in box", "≡ this target law" · OxiZ)
```

Real output, captured from the compiled WASM module running the `y = eˣ` example:

```
capabilities : analyze ✓  certify ✓  canonical ✓  smt ✓
law          : eml(x0, 1)          (= eˣ⁰)        R² = 1.000000
d/dx₀ (CAS)  : e^{x_0}
canonical    : e^{x_0}
certified    : f(x) ∈ [1, 20.0855]               (= [e⁰, e³], a *guaranteed* enclosure)
certified root: RootCertificate { … status: NoRoot }
SMT no-root  : Proven                             ← OxiZ, in the browser
SMT ≡ true   : Proven                             ← equivalence to the true law, proved in the browser
```

## Run the demo

```bash
# 1. Build the WASM package (carries discovery + CAS + e-graph + SMT)
./crates/phop-wasm/build-wasm.sh

# 2. Serve the static files (any static host works; this touches no application server)
cd crates/phop-wasm && python3 -m http.server 8000

# 3. Open the demo
#    http://localhost:8000/www/
```

The page ships built-in examples (exponential growth, a product law, Kepler's `T = a³ᐟ²`), a CSV box,
and toggles for each verification tier. The capability chips light up to show exactly which tiers the
loaded WASM build supports.

## JavaScript API

```js
import init, { discover_and_verify, capabilities, set_panic_hook } from "./pkg/phop_wasm.js";

await init();
set_panic_hook();

const data = JSON.stringify({ x: [[0],[0.5],[1]], y: [1, 1.6487, 2.7183] });
const cfg  = JSON.stringify({
  method: "enumerate",        // "enumerate" | "auto" (EML + rich-leaf) | "rich" (products/powers)
  top_k: 1,
  analyze: true,              // CAS derivative / antiderivative
  canonical: true,            // e-graph canonical form        (needs the `egraph` feature)
  certify: true,              // certified range + root         (always available)
  prove_no_root: true,        // SMT: no root over the data box (needs the `smt` feature)
  target_model: "<json>",     // optional: SMT-prove equivalence to a serialized EML law
  // units: [[0,1,0,0,0,0,0], ...]   // optional: Buckingham-π reduction before discovery
});

const result = JSON.parse(discover_and_verify(data, cfg));
console.log(result.solutions[0]);   // { latex, pretty, mse, r2, certified_range, proven_no_root, ... }
```

- `capabilities()` returns `{ analyze, certify, canonical, smt }` so the UI can show only the tiers
  this build supports.
- `discover_json(data, cfg)` remains for the lightweight discover-only path.

### Cargo features

`egraph` (e-graph canonicalization) and `smt` (OxiZ proofs) are opt-in so a minimal build stays tiny;
the demo build enables both. The `certify` and `analyze` tiers need no feature.

Published on npm as **`@cooljapan/phop-wasm`**. Part of the [phop](https://github.com/cool-japan/phop)
project.

## License

Apache-2.0
