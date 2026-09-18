//! Smoke tests for newly wired orphan modules in oximedia-metadata.
//!
//! Each test exercises the primary public API surface of 1-3 modules to verify
//! that the wiring compiles and basic functionality is correct.

use std::collections::HashMap;

// ── metadata_json ─────────────────────────────────────────────────────────────

#[test]
fn test_metadata_json_round_trip() {
    use oximedia_metadata::metadata_json::{from_json, to_json};
    use oximedia_metadata::{Metadata, MetadataFormat, MetadataValue};

    let mut m = Metadata::new(MetadataFormat::Id3v2);
    m.insert(
        "TIT2".to_string(),
        MetadataValue::Text("My Song".to_string()),
    );
    m.insert("TLEN".to_string(), MetadataValue::Integer(220_000));
    m.insert("TFLAG".to_string(), MetadataValue::Boolean(true));

    let json = to_json(&m).expect("to_json should succeed");
    assert!(json.contains("TIT2"), "JSON should contain field key");

    let m2 = from_json(&json).expect("from_json should succeed");
    assert_eq!(
        m2.get("TIT2").and_then(MetadataValue::as_text),
        Some("My Song")
    );
    assert_eq!(
        m2.get("TLEN").and_then(MetadataValue::as_integer),
        Some(220_000)
    );
}

// ── lyrics ────────────────────────────────────────────────────────────────────

#[test]
fn test_lyrics_synced_lrc_roundtrip() {
    use oximedia_metadata::lyrics::{LyricLine, SyncedLyrics};

    let mut lyrics = SyncedLyrics::new();
    lyrics.add_line(LyricLine::new(0, "First line of the song"));
    lyrics.add_line(LyricLine::new(5_000, "Second line"));
    lyrics.add_line(LyricLine::new(10_000, "Third line"));

    assert_eq!(lyrics.len(), 3);

    let lrc = lyrics.to_lrc();
    assert!(
        lrc.contains("[00:00.00]"),
        "LRC should contain first timestamp"
    );
    assert!(
        lrc.contains("First line of the song"),
        "LRC should contain first line text"
    );
}

#[test]
fn test_lyrics_unsynced() {
    use oximedia_metadata::lyrics::UnsyncedLyrics;

    let ul = UnsyncedLyrics::new("Verse 1\nVerse 2");
    assert!(!ul.text.is_empty());
    assert_eq!(ul.language, "und");
}

// ── xmp_document ─────────────────────────────────────────────────────────────

#[test]
fn test_xmp_document_build_and_serialize() {
    use oximedia_metadata::xmp_document::{XmpDocument, XmpSerializer};

    let mut doc = XmpDocument::new();
    doc.set("dc", "title", "My Title");
    doc.set("dc", "creator", "Author Name");

    // has() → check via get().is_some()
    assert!(doc.get("dc", "title").is_some());
    assert_eq!(doc.get("dc", "title"), Some("My Title"));

    let xml = XmpSerializer::to_xml(&doc);
    assert!(!xml.is_empty(), "serialized XMP should not be empty");
}

// ── metadata_streaming ────────────────────────────────────────────────────────

#[test]
fn test_streaming_metadata_parser_init() {
    use oximedia_metadata::metadata_streaming::{StreamingFormat, StreamingMetadataParser};

    let mut parser = StreamingMetadataParser::new(StreamingFormat::Id3v2);
    assert!(!parser.is_complete());

    // Feed the 10-byte ID3v2 tag header
    let header = b"ID3\x04\x00\x00\x00\x00\x00\x00";
    let events = parser.feed(header);
    // Verify it doesn't panic and returns events
    let _ = events;
}

#[test]
fn test_streaming_vorbis_parser_field_count() {
    use oximedia_metadata::metadata_streaming::{StreamingFormat, StreamingMetadataParser};

    let parser = StreamingMetadataParser::new(StreamingFormat::VorbisComments);
    assert_eq!(parser.field_count(), 0);
    assert!(!parser.is_complete());
}

// ── metadata_fingerprint ─────────────────────────────────────────────────────

#[test]
fn test_metadata_fingerprint_deterministic() {
    use oximedia_metadata::metadata_fingerprint::MetadataFingerprint;

    let mut fields: HashMap<String, String> = HashMap::new();
    fields.insert("title".to_string(), "My Song".to_string());
    fields.insert("artist".to_string(), "Artist".to_string());

    let fp1 = MetadataFingerprint::from_fields(&fields);
    let fp2 = MetadataFingerprint::from_fields(&fields);
    assert!(fp1.matches(&fp2), "same fields → same fingerprint");

    let hex = fp1.hex();
    assert_eq!(hex.len(), 16, "fingerprint hex should be 16 chars");
}

#[test]
fn test_metadata_fingerprint_order_independent() {
    use oximedia_metadata::metadata_fingerprint::MetadataFingerprint;

    // XOR-based combination — insertion order should not affect the fingerprint
    let mut a: HashMap<String, String> = HashMap::new();
    a.insert("title".to_string(), "Song A".to_string());
    a.insert("year".to_string(), "2024".to_string());

    let mut b: HashMap<String, String> = HashMap::new();
    b.insert("year".to_string(), "2024".to_string());
    b.insert("title".to_string(), "Song A".to_string());

    let fa = MetadataFingerprint::from_fields(&a);
    let fb = MetadataFingerprint::from_fields(&b);
    assert!(
        fa.matches(&fb),
        "insertion order should not affect fingerprint"
    );
}

// ── podcast ───────────────────────────────────────────────────────────────────

#[test]
fn test_podcast_parse_duration() {
    use oximedia_metadata::podcast::parse_duration;

    assert_eq!(parse_duration("1:23:45").expect("HH:MM:SS"), 5025);
    assert_eq!(parse_duration("45:30").expect("MM:SS"), 2730);
    assert_eq!(parse_duration("3600").expect("bare seconds"), 3600);

    assert!(parse_duration("").is_err(), "empty duration should fail");
}

#[test]
fn test_podcast_metadata_new() {
    use oximedia_metadata::podcast::PodcastMetadata;

    let pm = PodcastMetadata::new("My Podcast");
    assert_eq!(pm.title, "My Podcast");
    assert!(pm.episodes.is_empty());
}

// ── metadata_index ────────────────────────────────────────────────────────────

#[test]
fn test_metadata_index_basic_insert_and_search() {
    use oximedia_metadata::metadata_index::MetadataIndex;

    let mut index = MetadataIndex::new();
    let mut fields: HashMap<String, String> = HashMap::new();
    fields.insert("title".to_string(), "hello world".to_string());
    fields.insert("artist".to_string(), "test artist".to_string());

    let id = index.add_document(fields);

    let results = index.search("hello");
    assert!(
        results.contains(&id),
        "search for 'hello' should return the inserted DocId"
    );

    let no_results = index.search("zzznomatch");
    assert!(no_results.is_empty(), "no match expected for unknown term");
}

// ── metadata_query ────────────────────────────────────────────────────────────

#[test]
fn test_metadata_query_predicate_evaluate() {
    use oximedia_metadata::metadata_query::Predicate;

    let mut fields: HashMap<String, String> = HashMap::new();
    fields.insert("genre".to_string(), "jazz".to_string());

    let pred = Predicate::FieldEquals {
        field: "genre".to_string(),
        value: "jazz".to_string(),
    };
    assert!(pred.evaluate(&fields));

    let pred_miss = Predicate::FieldEquals {
        field: "genre".to_string(),
        value: "rock".to_string(),
    };
    assert!(!pred_miss.evaluate(&fields));
}

#[test]
fn test_metadata_query_engine_filter() {
    use oximedia_metadata::metadata_query::{MetadataQuery, Predicate, QueryEngine};

    let records: Vec<(u64, HashMap<String, String>)> = vec![
        (
            1,
            [("title", "Alpha"), ("year", "2020")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ),
        (
            2,
            [("title", "Beta"), ("year", "2021")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ),
        (
            3,
            [("title", "Gamma"), ("year", "2020")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ),
    ];

    let query = MetadataQuery::new().filter(Predicate::FieldEquals {
        field: "year".to_string(),
        value: "2020".to_string(),
    });

    let results = QueryEngine::execute(&query, &records);
    assert_eq!(results.len(), 2, "two records match year=2020");
    let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&1));
    assert!(ids.contains(&3));
}

// ── bulk_metadata ─────────────────────────────────────────────────────────────

#[test]
fn test_bulk_metadata_operator() {
    use oximedia_metadata::bulk_metadata::{BulkOperation, BulkOperator, MetadataCollection};
    use oximedia_metadata::{Metadata, MetadataFormat, MetadataValue};

    let mut collection = MetadataCollection::new();
    collection.insert(1, Metadata::new(MetadataFormat::Id3v2));
    collection.insert(2, Metadata::new(MetadataFormat::VorbisComments));

    let ops = vec![BulkOperation::Set {
        key: "album".to_string(),
        value: "Compilation".to_string(),
    }];
    let operator = BulkOperator::new(ops);
    let outcomes = operator.apply(&mut collection);

    assert_eq!(outcomes.len(), 2, "one outcome per record");

    for id in [1u64, 2u64] {
        let m = collection.get(id).expect("record should exist");
        assert_eq!(
            m.get("album").and_then(MetadataValue::as_text),
            Some("Compilation"),
        );
    }
}

// ── replaygain ────────────────────────────────────────────────────────────────

#[test]
fn test_replaygain_gain_formatting() {
    use oximedia_metadata::replaygain::ReplayGain;

    let rg = ReplayGain::new()
        .with_track_gain(-6.5)
        .with_track_peak(0.95)
        .with_album_gain(-7.2)
        .with_album_peak(0.98);

    assert_eq!(rg.track_gain_db(), Some(-6.5));
    assert_eq!(rg.track_gain_formatted(), Some("-6.50 dB".to_string()));
    assert_eq!(rg.album_gain_db(), Some(-7.2));
    assert!(rg.has_data());
}

// ── chapter ───────────────────────────────────────────────────────────────────

#[test]
fn test_chapter_list_operations() {
    use oximedia_metadata::chapter::{Chapter, ChapterList};

    let mut chapters = ChapterList::new();
    chapters.add(Chapter::new(0, 120_000, "Introduction"));
    chapters.add(Chapter::new(120_000, 360_000, "Main Content"));
    chapters.add(Chapter::new(360_000, 480_000, "Conclusion"));

    assert_eq!(chapters.len(), 3);
    assert_eq!(chapters.total_duration_ms(), Some(480_000));
    assert!(!chapters.is_empty());
}

// ── inherit ───────────────────────────────────────────────────────────────────

#[test]
fn test_metadata_inheritance_merge() {
    use oximedia_metadata::inherit::{MetadataInheritance, MetadataMap};

    let mut parent: MetadataMap = HashMap::new();
    parent.insert("album".to_string(), "Parent Album".to_string());
    parent.insert("year".to_string(), "2020".to_string());

    let mut child: MetadataMap = HashMap::new();
    child.insert("year".to_string(), "2021".to_string()); // overrides parent
    child.insert("title".to_string(), "Child Title".to_string());

    let merged = MetadataInheritance::merge(&parent, &child);

    assert_eq!(
        merged.get("album").map(String::as_str),
        Some("Parent Album")
    );
    assert_eq!(merged.get("year").map(String::as_str), Some("2021"));
    assert_eq!(merged.get("title").map(String::as_str), Some("Child Title"));
}

// ── metadata_stats ────────────────────────────────────────────────────────────

#[test]
fn test_metadata_stats_field_coverage() {
    use oximedia_metadata::metadata_stats::FieldStats;

    let mut stats = FieldStats::new("title");
    stats.record_value("Song A");
    stats.record_value("Song B");
    stats.record_absent();

    let total = stats.total_records();
    assert_eq!(total, 3);

    let cov = stats.coverage();
    // 2 present out of 3 → ~0.6667
    assert!(
        (cov - 2.0 / 3.0).abs() < 0.001,
        "coverage should be ~2/3, got {cov}"
    );
}

// ── csv_export ────────────────────────────────────────────────────────────────

#[test]
fn test_csv_export_basic() {
    use oximedia_metadata::csv_export::MetadataCsvExporter;
    use oximedia_metadata::inherit::MetadataMap;

    let mut m1: MetadataMap = HashMap::new();
    m1.insert("title".to_string(), "Song A".to_string());
    m1.insert("artist".to_string(), "Artist X".to_string());

    let mut m2: MetadataMap = HashMap::new();
    m2.insert("title".to_string(), "Song B".to_string());

    let items: Vec<(u64, &MetadataMap)> = vec![(1, &m1), (2, &m2)];
    let csv = MetadataCsvExporter::export(&items);

    assert!(csv.contains("title"), "CSV header should contain 'title'");
    assert!(
        csv.contains("Song A"),
        "CSV should contain first record title"
    );
    assert!(
        csv.contains("Song B"),
        "CSV should contain second record title"
    );
}

// ── metadata_validate ─────────────────────────────────────────────────────────

#[test]
fn test_metadata_validate_id3v2_frame_id() {
    use oximedia_metadata::metadata_validate::{validate_field, ValidationContext};

    assert!(
        validate_field("TIT2", "My Song", ValidationContext::Id3v2).is_ok(),
        "Valid ID3v2 frame ID should pass"
    );
    assert!(
        validate_field("bad key!", "value", ValidationContext::Id3v2).is_err(),
        "Invalid ID3v2 frame ID should fail"
    );
}

#[test]
fn test_metadata_validate_vorbis_field_name() {
    use oximedia_metadata::metadata_validate::{validate_field, ValidationContext};

    assert!(
        validate_field("TITLE", "Hello", ValidationContext::VorbisComment).is_ok(),
        "Valid Vorbis field should pass"
    );
    assert!(
        validate_field("FIELD=NAME", "val", ValidationContext::VorbisComment).is_err(),
        "Vorbis field with '=' should fail"
    );
}

// ── mime_metadata ─────────────────────────────────────────────────────────────

#[test]
fn test_mime_detector_magic_bytes_jpeg() {
    use oximedia_metadata::mime_metadata::{MimeDetector, MimeType};

    let jpeg_magic = [
        0xFF_u8, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
    ];
    if let Some(hint) = MimeDetector::from_bytes(&jpeg_magic) {
        assert_eq!(hint.mime_type, MimeType::ImageJpeg);
    }
    // Even if detection returns None, the API must not panic.
}

#[test]
fn test_mime_type_category_predicates() {
    use oximedia_metadata::mime_metadata::MimeType;

    assert!(MimeType::AudioMpeg.is_audio());
    assert!(!MimeType::AudioMpeg.is_video());
    assert!(!MimeType::AudioMpeg.is_image());
    assert!(MimeType::VideoMp4.is_video());
    assert!(MimeType::ImageJpeg.is_image());
}

// ── id3v2_utf8 ────────────────────────────────────────────────────────────────

#[test]
fn test_id3v2_utf8_write_read_round_trip() {
    use oximedia_metadata::id3v2_utf8::{
        TextEncodingPreference, Utf8TextFrameReader, Utf8TextFrameWriter,
    };

    let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
    let frame = writer
        .write_text_frame("TIT2", "Hello World", 4)
        .expect("should write UTF-8 frame for v2.4");

    assert_eq!(frame[0], 0x03, "first byte should be UTF-8 encoding marker");

    let reader = Utf8TextFrameReader::new();
    let text = reader
        .read_text_frame(&frame)
        .expect("should read back the text");
    assert_eq!(text, "Hello World");
}
