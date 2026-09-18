# Releasing OxiXML

**Publishing is a maintainer-only action.** Nothing in this repository — no script, no CI job —
may run `cargo publish` (without `--dry-run`), `git push`, or `cargo yank`. The commands below are
documentation for the maintainer to execute deliberately.

## Scope

The workspace has **46 member crates**. **45** are published to crates.io; `oxixml-testsuite`
carries `publish = false` (it is a development-only conformance harness that drives externally
provisioned W3C/oxigraph suites and ships no reusable API).

All crates share one version, defined once in `[workspace.package]` of the root `Cargo.toml`, and
depend on each other through `[workspace.dependencies]` entries that carry both a `version` and a
`path`. Cargo rewrites those to plain version requirements at package time, so **every sibling
dependency must already be live on crates.io before the depending crate can be published.**

## Pre-flight gates

Run all of these green before starting. `OXIXML_XML_SUITES_DIR` / `OXIXML_TESTSUITE_DIR` point at
the externally provisioned conformance corpora and are only needed for the `#[ignore]`d heavy
tests.

```bash
cargo check --workspace --all-features
cargo check --workspace
cargo clippy --workspace --all-targets --all-features
cargo test  --workspace --all-features
cargo deny  check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo fmt --all --check
```

### Purity invariant

A **default-feature** build of every crate must resolve to zero third-party dependencies. Verify
with:

```bash
cargo tree -p oxixml --edges normal,build --prefix none | grep -v '^oxixml' | sort -u
```

The output must be empty. `serde` and `tokio` are permitted only behind off-by-default features
(`oxixml-xml`'s `serialize` / `serde-types` / `async-tokio`, and the corresponding
`oxixml-quickxml-compat` re-exports) and as dev-dependencies.

## Publish order

Topological order of the dependency graph. Publish strictly top to bottom; after each crate, wait
for the crates.io index to catch up (`cargo publish` blocks on this automatically, but a manual
pause is safer when the index is lagging).

```bash
# Tier 1 — leaves (no intra-workspace dependencies)
cargo publish -p oxixml-support
cargo publish -p oxixml-unicode
cargo publish -p oxixml-xsd

# Tier 2
cargo publish -p oxixml-encoding
cargo publish -p oxixml-json
cargo publish -p oxixml-langtag
cargo publish -p oxixml-qname
cargo publish -p oxixml-regex
cargo publish -p oxixml-text

# Tier 3
cargo publish -p oxixml-iri
cargo publish -p oxixml-xml

# Tier 4
cargo publish -p oxixml-catalog
cargo publish -p oxixml-dom
cargo publish -p oxixml-dtd
cargo publish -p oxixml-fo
cargo publish -p oxixml-infoset
cargo publish -p oxixml-model
cargo publish -p oxixml-quickxml-compat
cargo publish -p oxixml-sax

# Tier 5
cargo publish -p oxixml-c14n
cargo publish -p oxixml-enc
cargo publish -p oxixml-fn
cargo publish -p oxixml-hdt
cargo publish -p oxixml-html
cargo publish -p oxixml-relaxng
cargo publish -p oxixml-sparql-results
cargo publish -p oxixml-sparql-syntax
cargo publish -p oxixml-svg
cargo publish -p oxixml-turtle
cargo publish -p oxixml-vocab
cargo publish -p oxixml-xinclude

# Tier 6
cargo publish -p oxixml-canon
cargo publish -p oxixml-dsig
cargo publish -p oxixml-jsonld
cargo publish -p oxixml-rdfa
cargo publish -p oxixml-rdfxml
cargo publish -p oxixml-trix
cargo publish -p oxixml-xpath

# Tier 7
cargo publish -p oxixml-io
cargo publish -p oxixml-schema
cargo publish -p oxixml-schematron
cargo publish -p oxixml-xquery
cargo publish -p oxixml-xslt

# Tier 8 — facade and binary
cargo publish -p oxixml
cargo publish -p oxixml-cli

# NOT PUBLISHED: oxixml-testsuite (publish = false)
```

The tiers above are the exact rounds of a Kahn topological sort over the intra-workspace
dependency graph (normal + build + dev edges), and can be regenerated with:

```bash
cargo metadata --format-version 1 --all-features
```

`oxixml-cli` sits in tier 8 because it depends on the individual component crates directly rather
than on the `oxixml` facade; it and `oxixml` are mutually independent.

Crates inside a single tier have no dependency on one another and may be published in any order
(or concurrently).

## Dry runs

```bash
cargo publish --dry-run -p <crate>
```

Before the first real release, a dry run of any crate with at least one intra-workspace
dependency **will fail** with `no matching package named 'oxixml-…' found` — the sibling is not on
crates.io yet. That failure is expected and is not a defect; it disappears tier by tier as the
real publish proceeds. Only the tier-1 leaf crates can be dry-run meaningfully ahead of time.

Baseline for the 0.1.0 release (`cargo publish --dry-run -p <crate> --allow-dirty`, run over the
44 crates publishable at that version — `oxixml-svg` was added in 0.1.1, making 45): the three
tier-1 crates — `oxixml-support`, `oxixml-unicode`, `oxixml-xsd` — package and verify cleanly; the
other 41 stop at
`failed to prepare local package for uploading / no matching package named 'oxixml-…' found`.
No crate failed for any other reason.

## Yanking 0.1.0 — DONE

0.1.1 broke API and behaviour compatibility with 0.1.0 on purpose and shipped no compatibility
shims — the list is the `### Breaking` section at the head of the 0.1.1 entry in
[`CHANGELOG.md`](CHANGELOG.md). 0.1.0 has been **yanked** across all 44 crates that had a 0.1.0
release (`oxixml-svg` didn't exist yet at 0.1.0 — added in 0.1.1 — so it was never published at
that version and has nothing to yank). Verified live via the crates.io API on 2026-08-10:

```bash
curl -s -A "<identifying UA>" https://crates.io/api/v1/crates/<crate>/0.1.0
# every one of the 44 returns "yanked": true
```

For reference, this is how a yank is done (already executed, kept here for the next major-breaking
release):

```bash
# Maintainer action, one crate at a time.
cargo yank <crate>@0.1.0
```

A yank does not delete the version and does not break an existing `Cargo.lock`; it stops new
resolutions from choosing it. Yank in the reverse of the publish order (facade and binary first,
leaves last) so that no intermediate state advertises a crate whose dependencies are already
yanked.

## After the release

- Tag and push the release commit (user action).
- Confirm every crate rendered on docs.rs. Each crate sets
  `[package.metadata.docs.rs] all-features = true` plus `--cfg docsrs`, so feature-gated items
  carry their availability badges.
