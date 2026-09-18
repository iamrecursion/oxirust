# oximedia-imf TODO

## Current Status
- 37+ modules implementing SMPTE ST 2067 Interoperable Master Format
- Core: CPL (Composition Playlist), PKL (Packing List), AssetMap, OPL (Output Profile List)
- Parsing: cpl_parser, pkl_document, opl_document, xml_util (quick-xml based)
- Validation: validator, cpl_validator, package_validator with conformance levels
- Essence: MXF descriptors, essence constraints, track files, audio layout
- Advanced: CPL merge, supplemental packages, versioning, content versions
- Timeline: imf_timeline, composition_sequence, cpl_segment
- Metadata: marker_list, marker_resource, subtitle_resource, sidecar, IMSC1 subtitles
- Delivery: delivery, output_profile_list, application_profile, imf_report
- Hash: SHA-1 and MD5 verification via sha1 and md-5 crates
- Dependencies: oximedia-core, quick-xml, uuid, chrono, serde, sha1, md-5, hex

## Enhancements
- [x] Add SHA-256 and SHA-512 hash support to `essence_hash.rs` alongside existing SHA-1/MD5
- [x] Extend `cpl_validator.rs` with SMPTE ST 2067-2:2020 (latest revision) constraint checks (verified 2026-05-16; src/cpl_validator.rs:317 ST 2067-2:2020 §6.1 edit rates, §6.4 segment/UUID constraints)
- [x] Add incremental hash computation to `essence_hash.rs` for large MXF files (streaming digest)
- [x] Implement CPL diff in `cpl_merge.rs` to show what changed between two compositions
- [x] Extend `application_profile.rs` with Netflix IMF App 2.1 and Disney DECE profiles (done — already implemented; note was stale)
- [x] Add timeline gap detection and overlap reporting in `imf_timeline.rs` (verified 2026-05-16; src/imf_timeline.rs:52 overlaps fn, validate_timeline gaps/overlaps:198)
- [x] Implement `subtitle_resource.rs` support for TTML and WebVTT subtitle formats (done — already implemented; note was stale)
- [x] Extend versioning.rs with automatic version increment and change log generation (completed 2026-06-01)
  - **Goal:** Add `next_version` and `change_log` methods to `VersionChain`.
  - **Design:** `src/versioning.rs:66` `VersionChain` has `add_version`/`latest`/`full_chain` but no `next_version`/`change_log`. Add `next_version(&self, kind: VersionKind) -> PackageVersion` that increments the appropriate semver component based on `kind`. Add `change_log(&self) -> Vec<VersionChange>` that diffs consecutive `PackageVersion` annotation fields.
  - **Files:** `src/versioning.rs`, `TODO.md`.
  - **Tests:** `next_version(Patch)` increments patch; `change_log` over a 3-version chain returns correct diffs.
  - **Risk:** Define `VersionChange` struct matching the annotation fields available.

## New Features
- [x] Add an `imf_builder.rs` high-level API for creating IMF packages from scratch with fluent interface
- [x] Implement an `imf_inspector.rs` module for detailed package inspection with human-readable report (verified 2026-05-16; src/imf_inspector.rs:711 lines)
- [x] Add an `essence_probe.rs` module for probing MXF essence files without full parse (quick metadata) (verified 2026-05-16; src/essence_probe.rs:784 lines)
- [x] Implement a `qc_report.rs` module for automated quality control reporting (EBU QC checks) (verified 2026-05-16; src/qc_report.rs:632 lines)
- [x] Add a `compliance_matrix.rs` module mapping application profiles to required constraints (verified 2026-05-16; src/compliance_matrix.rs:934 lines)
- [x] Implement an `imf_diff.rs` module for comparing two IMF packages (structural and content diff) (verified 2026-05-16; src/imf_diff.rs:570 lines)
- [x] Add a `partial_restore.rs` module for extracting specific segments/tracks from an IMP (verified 2026-05-16; src/partial_restore.rs:285 lines)
- [x] Implement a `metadata_extractor.rs` module for extracting all metadata to JSON/XML export (verified 2026-05-16; src/metadata_extractor.rs:249 lines)
- [x] Add an `imf_archive.rs` module for creating archival packages with long-term preservation metadata (verified 2026-05-16; src/imf_archive.rs:325 lines)

## Performance
- [x] Add parallel hash verification in `package_validator.rs` using rayon for multi-asset packages
- [x] Implement lazy XML parsing in cpl_parser.rs (parse only requested sections) (completed 2026-06-01)
  - **Goal:** Add section-targeted parse entrypoints that skip unrequested XML subtrees.
  - **Design:** `src/cpl_parser.rs` currently parses the entire CPL XML eagerly. Add `parse_cpl_header`, `parse_reel_list`, `parse_segment_list` entrypoints that use `quick-xml` event skipping to skip unwanted subtrees. `quick-xml` is already a dep.
  - **Files:** `src/cpl_parser.rs`, `TODO.md`.
  - **Tests:** lazy header parse == eager parse on the same CPL; reel-list parse skips segment subtree.
  - **Risk:** `quick-xml` event-skip correctness — test round-trip.
- [x] Cache parsed CPL/PKL structures to avoid re-parsing during repeated validation (completed 2026-06-01)
  - **Goal:** Add a parse cache keyed by UUID/path with mtime-based invalidation.
  - **Design:** Add `CplCache` struct with `HashMap<Uuid, Arc<Cpl>>` + mtime invalidation (`std::fs::metadata`); add `PklCache` similarly. Expose `CplParser::cached(path) -> Result<Arc<Cpl>>`. No new dep — only `std::fs`.
  - **Files:** `src/cpl_cache.rs` (new), `src/pkl_cache.rs` (new), `src/cpl_parser.rs`, `TODO.md`.
  - **Tests:** cache hit avoids re-parse (check a hit-count counter); mtime change invalidates the cache; thread-safe (Arc<RwLock> or Mutex).
  - **Risk:** mtime precision (may be 1s on some filesystems); use file-size as secondary key.
- [x] Add streaming XML writing in xml_util.rs for large OPL/CPL generation (completed 2026-06-01)
  - **Goal:** Emit large OPL/CPL XML incrementally without buffering the whole document.
  - **Design:** `src/xml_util.rs` currently builds full strings. Add `XmlStreamWriter<W: Write>` using `quick-xml::Writer<W>` (already a dep) for streaming OPL/CPL element emission. Expose `write_cpl_streaming<W: Write>` and `write_opl_streaming<W: Write>`.
  - **Files:** `src/xml_util.rs`, `TODO.md`.
  - **Tests:** streaming writer output is valid XML parseable back to the same structure as the eager path; streaming to a `Vec<u8>` matches the string-building path byte-for-byte on a reference CPL.
  - **Risk:** quick-xml Writer API shape — read actual API before implementing.

## Testing
- [ ] Add conformance tests with reference IMF packages from the SMPTE IMF plugfest test suite
- [x] Test `cpl_merge.rs` with conflicting edit rates and overlapping timeline segments (Wave 29 / Slice 2, completed 2026-06-06)
  - [x] **Bug fix (A):** `merge_cpls` now selects the merged edit rate by `ConflictStrategy` — `KeepSupplemental` adopts the supplemental rate, while `KeepBase`/`Fail`/`Concatenate` retain the base rate (previously the merged CPL always used the base rate, ignoring the strategy). The Error-severity `edit_rate` mismatch conflict is still surfaced regardless of strategy. (`src/cpl_merge.rs`)
  - [x] **Bug fix (B):** `MergeResult.supplemental_segments` now reports the TOTAL supplemental segment count (matching the sibling `base_segments` total), instead of the non-overlapping-only count; removed the now-dead `supp_only_count` binding and updated the field doc-comment. (`src/cpl_merge.rs`)
  - [x] Added `tests/edit_rate_strategy.rs` (5 tests): per-strategy edit-rate selection (KeepBase/KeepSupplemental/Fail/Concatenate), matching-rate no-conflict case, and total supplemental count including an overlapping segment id.
- [x] Add round-trip tests: create CPL -> serialize to XML -> parse XML -> compare structure
- [ ] Test `package_validator.rs` with intentionally corrupted hash values
- [ ] Add `supplemental_package.rs` tests with multi-level supplemental chain
- [ ] Test `imsc1.rs` subtitle parsing with multi-language and styled IMSC1 documents

## Documentation
- [ ] Add a reference table mapping each module to its SMPTE standard section
- [ ] Document the CPL/PKL/AssetMap relationship with a structural diagram
- [ ] Add examples for creating and validating common IMF delivery scenarios (broadcast, OTT, archive)
