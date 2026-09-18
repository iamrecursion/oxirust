# phop-wasm — TODO

> WebAssembly front end for in-browser demos — the deployment surface Python/Julia SR can't
> reach (design §1.2, novelty L5). Exposes a **Layer-A subset** of `phop-core` via wasm-bindgen.

**Crate type:** `cdylib`. **Build:** `wasm-pack` → `pkg/`. **Deps:** `phop-core`
(default features only; **no GPU, no rayon threads** unless wasm-threads configured),
`wasm-bindgen`, `serde-wasm-bindgen`, `console_error_panic_hook`, optional `getrandom`/`js`.

> Implemented bindings are recorded in CHANGELOG.md (0.1.0); only the open items below remain (2026-06-25).

Legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Build/packaging
- [ ] Keep payload small: `opt-level = "z"`/`s`, `lto`, `wasm-opt`; report bundle size.

## API (JS-facing)
- [ ] Optional streaming/progress callback so the browser can animate epochs.

## Demo (ties to M8 launch + docs)
- [ ] Minimal static page: paste/upload CSV → watch a Pareto front appear live.
- [ ] "Planck discovered in the browser" interactive demo (mirrors the YouTube timelapse).
- [ ] Host on GitHub Pages; link from README.
