# oximedia-mir TODO

## 0.1.8 Wave 5 — 2026-05-29

- [x] Register 22 orphan modules (audio_similarity, chroma_cache, chromagram, cover_art_features, dj_features, genre_classify_new, harmonic_percussive, harmonic_spectral, instrument_classifier, lsh_similarity, lyrics_align, melody_extract, melody_extractor, multitrack, onset_peak, section_segmenter, similarity_search, subgenre, tempo_stability, thumbnail, watermark, watermark_detect) + chromagram cfix (Goertzel refactor, A4 440Hz pipeline test, dead_code attribute removed from subgenre.rs) — 1081 tests, 0 warnings

## Current Status
- 74 source files/directories covering tempo, beat, key, chord, melody, structure, genre, mood, spectral, rhythm, harmonic analysis
- Additional modules: fingerprint, playlist, cover_detect, chorus_detect, vocal_detect, instrument_detection, source_separation, fade_detect, tuning_detect
- Feature-gated per analysis type (tempo, beat, key, chord, melody, structure, genre, mood, spectral, rhythm, harmonic)
- MirAnalyzer provides unified analysis pipeline returning AnalysisResult with all enabled features

## Enhancements
- [x] Improve `to_mono()` in MirAnalyzer to actually detect and convert stereo -- currently just clones input
- [x] Add confidence thresholds to MirConfig so low-confidence results can be automatically filtered
- [x] Implement streaming/incremental analysis in MirAnalyzer for real-time use (process chunk-by-chunk)
- [x] Add tempo stability metric to TempoResult (distinguish steady BPM from rubato/accelerando) (verified 2026-05-16; src/tempo_stability.rs:37 TempoStabilityReport, TempoStabilityAnalyzer:70)
- [x] Improve chord_recognition to handle 7th, diminished, augmented, and suspended chords beyond major/minor
- [x] Add time-varying key detection for songs with key changes (modulations)
- [x] Extend genre_classify with sub-genre classification (e.g., not just "rock" but "progressive rock") (verified 2026-05-16; src/subgenre.rs:28 SubGenre enum, SubGenreClassifier:216)
- [x] Add multi-track analysis support -- analyze stems separately then combine MIR results (verified 2026-05-16; src/multitrack.rs:95 MultiTrackAnalyzer, StemAnalysis:68, combined_tempo:625)

## New Features
- [x] Implement audio-to-MIDI conversion using pitch_track and onset_strength data
- [x] Add rhythm complexity metric (syncopation, polyrhythm detection) to rhythm module
- [x] Implement audio thumbnailing -- extract the most representative 15-30s clip using structure analysis (verified 2026-05-16; src/thumbnail.rs:72 AudioThumbnail, ThumbnailConfig:34, ThumbnailResult:17)
- [x] Add music similarity search using fingerprint module with locality-sensitive hashing (verified 2026-05-16; src/lsh_similarity.rs:90 LshSimilarityIndex, band-based LSH)
- [x] Implement real-time DJ features: beat-matching suggestions, compatible key detection (Camelot wheel) (verified 2026-05-16; src/dj_features.rs:30 CamelotCode, CamelotWheel:162, BeatMatcher)
- [x] Add lyrics timing alignment support (given lyrics text, align to audio using vocal_detect + onset_strength) (completed 2026-06-05; src/lyrics_align.rs: align_lyrics_dtw DTW forced alignment + 5 new tests; greedy align_lyrics preserved)
- [x] Implement audio watermark detection using spectral analysis (verified 2026-05-16; src/watermark_detect.rs:93 WatermarkDetector, detect fn:103)

## Performance
- [x] Replace rustfft with OxiFFT (COOLJAPAN policy) in spectral analysis
- [x] Remove ndarray dependency -- use Vec<f32> with manual stride operations
- [x] Parallelize independent analysis branches in MirAnalyzer::analyze() using rayon (tempo/key/spectral are independent)
- [x] Add early-exit in TempoDetector when confidence exceeds threshold to avoid full autocorrelation scan (implemented 2026-06-01: bounded_acf_with_early_exit in tempo/utils.rs; TempoDetector::detect fast-path at conf≥0.8)
- [x] Cache chromagram computation so chord_recognition and key_detection share the same chroma features (verified 2026-05-16; src/chroma_cache.rs:157 ChromaCache lazy compute, CachedChromagram:13)

## Testing
- [x] Add test with known-BPM audio (e.g., synthesized 120 BPM click track) verifying TempoDetector accuracy (2026-06-01: test_early_exit_clean_tempo + test_acf_120bpm_click_track in tempo/detect.rs)
- [x] Test key detection against known musical keys (C major scale, A minor arpeggio) (2026-06-01: test_key_detection in tempo/detect.rs)
- [x] Add regression test for chord recognition with isolated major/minor triads (2026-06-01: test_chord_recognition in tempo/detect.rs)
- [x] Test structure analysis with synthetic audio that has clear verse/chorus boundaries (volume/timbre changes) (2026-06-01: test_structure_boundaries in tempo/detect.rs)
- [ ] Validate MelodyExtractor output for monophonic sine sweep input

## Documentation
- [ ] Document the feature flags and which modules each feature gate controls
- [ ] Add usage examples for fingerprint-based music identification workflow
- [ ] Document the chromagram computation pipeline shared between key and chord analysis
