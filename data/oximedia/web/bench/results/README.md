# bench/results/

`../run.sh` writes its output here as `local-latest.json`, matching the
schema documented in `../README.md` (`{schema_version, env, results}`).

**No results are committed here yet.** This directory holds *snapshots* —
a results JSON is only meaningful together with the machine and browser
that produced it (see `env` in the file itself: user agent, hardware
concurrency, device pixel ratio, wasm resource sizes, timestamp). When a
snapshot is worth keeping for comparison over time, commit it under a
descriptive name (e.g. `2026-07-12-m1-macbook-chrome150.json`) — do not
overwrite `local-latest.json` in a commit; that name is reserved for the
gitignored-in-spirit-but-not-in-practice scratch output of the most recent
local run (it is *not* excluded in `.gitignore`, but nothing in this
codebase commits it automatically — that is a deliberate human decision
each time).

Until then: run `../run.sh` locally to produce your own numbers.
