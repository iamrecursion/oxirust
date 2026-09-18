# oximedia-metadata TODO

## Current Status
- 42 modules implementing comprehensive metadata standards support
- Formats: ID3v2 (v2.3/v2.4), Vorbis Comments, APEv2, iTunes/MP4, XMP, EXIF, IPTC/IIM, QuickTime, Matroska
- Core: Metadata container, MetadataValue (Text, TextList, Binary, Integer, Float, Picture, Boolean, DateTime)
- Picture handling: 21 PictureType variants, MIME type, dimensions, color depth
- Conversion: MetadataConverter for cross-format field mapping, CommonFields abstraction
- Parsing/writing: parse()/write() dispatch to format-specific implementations
- Specialized: av1_metadata, exif_parse, iptc_iim, linked_data, opengraph, schema_org, provenance, rights_metadata
- Management: metadata_diff, metadata_merge, metadata_export, metadata_history, metadata_sanitize, metadata_template
- Search/index: search, schema_registry, tag_normalize, sidecar, field_validator
- Advanced: bulk_update, embedding, media_metadata, schema, opus_tags, musicbrainz, geotag
- Dependencies: oximedia-core, quick-xml, encoding_rs, serde, bytes

## Enhancements
- [x] Add ID3v2.4 UTF-8 text encoding preference in `id3v2.rs` (currently defaults to UTF-16) (verified 2026-05-16; src/id3v2.rs:42 prefers UTF-8 for v2.4, falls back to UTF-16 for v2.3)
- [x] Extend `vorbis.rs` with multi-value tag support (repeated keys per Vorbis Comment spec) (verified 2026-05-16; src/vorbis.rs:25 VorbisCommentMultiValue, test_vorbis_comments_multivalue:349)
- [x] Add `itunes.rs` support for all iTunes atom types including tempo, compilation, gapless flags (verified 2026-05-16; src/itunes.rs:411 test_parse_compilation, compilation/gapless/tempo atoms)
- [x] Implement `xmp.rs` structured property support (arrays, alternatives, bags per XMP spec) (verified 2026-05-16; src/xmp.rs:32 XmpArrayKind, XmpArray struct:76)
- [x] Extend `exif.rs` with Makernote parsing for Canon, Nikon, Sony camera-specific tags (verified 2026-05-16; src/exif.rs:19 Makernote Support Canon/Nikon IFD)
- [x] Add `matroska.rs` support for nested Matroska tag elements (Targets/SimpleTags hierarchy) (verified 2026-05-16; src/matroska.rs:127 MatroskaTargets, SimpleTags nested:173)
- [x] Implement `metadata_sanitize.rs` with configurable sanitization rules (strip GPS, strip personal data)
- [x] Extend `metadata_diff.rs` with three-way merge for resolving concurrent metadata edits
- [x] Decode XML character entities in `xmp.rs` element text (predefined `&amp;`/`&lt;`/`&gt;`/`&quot;`/`&apos;` + decimal/hex numeric refs) so values like `Rock &amp; Roll` are no longer truncated at `&`; quick_xml splits text around `Event::GeneralRef`, so fragments are accumulated per element via `TextBuf` (zero-copy borrow fast-path preserved) and trimmed/committed on element End (src/xmp.rs:46 append_entity, :88 TextBuf, :332 GeneralRef handling, :368 End commit; test_xmp_escaped_entity_roundtrip:902)

## New Features
- [x] Add an `opus_tags.rs` module for Opus-specific metadata (R128 gain, output gain) per RFC 7845
- [x] Implement a `chapter.rs` module for chapter metadata (Matroska chapters, MP4 chapters, ID3 CHAP)
- [x] Add a `lyrics.rs` module for synchronized lyrics (ID3 SYLT, LRC format) and unsynchronized lyrics
- [x] Implement a `musicbrainz.rs` module for MusicBrainz tag mappings and MBID validation
- [x] Add a `replaygain.rs` module for ReplayGain metadata (track gain, album gain, peak)
- [x] Implement a `c2pa.rs` module for Content Credentials / C2PA provenance metadata (verified 2026-05-16; src/c2pa.rs:1100 lines C2PA manifest store)
- [x] Add a `podcast.rs` module for podcast-specific metadata (iTunes podcast tags, RSS mapping) (verified 2026-05-16; src/podcast.rs:866 lines PodcastNamespace 2.0)
- [x] Implement a `geotag.rs` module for GPS coordinate extraction, display, and reverse geocoding
- [x] Add a `metadata_streaming.rs` module for parsing metadata from streaming data (partial buffers) (verified 2026-05-16; src/metadata_streaming.rs:750 lines StreamingMetadataParser)

## Performance
- [x] Add lazy parsing in `id3v2.rs` (parse frame headers only, defer body parsing until accessed) (verified: id3v2.rs:586:LazyId3v2, :606:struct LazyId3v2, :615:impl LazyId3v2)
- [x] Implement zero-copy XMP parsing in `xmp.rs` using borrowed strings from the input buffer (2026-06-24; new `parse_borrowed() -> XmpView<'a>` with `&'a str` keys + `XmpValue::{Text,TextList}(Cow<'a,str>)` borrowing the input — `Cow::Owned` only when entity-unescaping/fragment-reassembly is needed; QName slices re-anchored to the input lifetime via safe offset+`str::get` since quick_xml `name()` is `&self`-bound; owned `parse()` now delegates to it and `.into_owned_metadata()`s; src/xmp.rs:261 XmpValue, :326 XmpView, :408 anchor_name, :492 parse_borrowed; 13 new tests incl. pointer-range borrow proof, entity→Owned, trimmed-still-borrowed, multibyte UTF-8, last-wins, malformed/invalid-UTF-8 clean errors)
- [x] Add parallel metadata extraction in `media_metadata.rs` for multi-format probing (verified: media_metadata.rs:799:ParallelMetadataExtractor, :808:par_iter, extract_all)
- [x] Cache encoding_rs decoders in `id3v2.rs` to avoid re-initialization per text frame (Wave 24; id3v2.rs:44 decode_static reuses one Decoder + pre-sizes the String via max_utf8_buffer_length, zero Cow alloc; routes Latin-1/UTF-8/UTF-16BE and BOM-stripped UTF-16 bodies through it while preserving Encoding::decode BOM semantics)
- [x] Optimize `tag_normalize.rs` with pre-compiled regex patterns for common tag normalization rules (Wave 24; tag_normalize.rs:21 static DEFAULT_MAPPINGS: OnceLock + :24 default_mappings() build the ~63 default mappings once; :167 with_defaults() clones the shared tables instead of rebuilding every call — no regex needed, lookups are exact hash-map hits)
- [x] Add batch metadata write in `bulk_update.rs` to reduce I/O operations for multi-file updates (Wave 14, bulk_update.rs — write_batch, BulkWriteMode)

## Testing
- [ ] Add round-trip tests for all 9 metadata formats: parse -> modify -> write -> parse -> compare
- [ ] Test `id3v2.rs` with ID3v2.3 and v2.4 tags from real-world MP3 files (various encoders)
- [ ] Add `exif.rs` tests with EXIF data from multiple camera manufacturers
- [ ] Test `converter.rs` cross-format conversion preserving all CommonFields
- [ ] Add `xmp.rs` tests with complex XMP documents containing nested arrays and alternatives
- [ ] Test `metadata_merge.rs` conflict resolution with overlapping fields from different sources
- [ ] Add `encoding_rs` integration tests for Shift_JIS, EUC-KR, and Latin-1 text frames

## Documentation
- [ ] Add a cross-format field mapping table (ID3v2 TIT2 = Vorbis TITLE = iTunes nam = etc.)
- [ ] Document the MetadataValue type coercion rules when converting between formats
- [ ] Add examples for common metadata workflows (tag music files, strip EXIF from photos, batch rename)
