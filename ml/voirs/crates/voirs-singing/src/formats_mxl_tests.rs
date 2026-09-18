//! Tests for compressed MusicXML (`.mxl`) reading. Archives are built
//! in-test with `oxiarc_archive::zip::ZipWriter`.

use super::*;
use crate::formats::{FormatParser, MusicXmlParser};
use oxiarc_archive::zip::{CompressionMethod as ZipMethod, ZipCompressionLevel, ZipWriter};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

const SCORE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <work><work-title>Zipped Song</work-title></work>
  <identification><creator type="composer">Zip Composer</creator></identification>
  <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths><mode>major</mode></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
      </attributes>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration></note>
      <note><pitch><step>G</step><alter>1</alter><octave>4</octave></pitch><duration>8</duration></note>
    </measure>
  </part>
</score-partwise>"#;

fn container(rootfile: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<container>
  <rootfiles>
    <rootfile full-path="{rootfile}" media-type="application/vnd.recordare.musicxml+xml"/>
  </rootfiles>
</container>"#
    )
}

/// Build a zip archive from `(name, data, level)` entries.
fn zip(
    entries: &[(&str, &[u8], ZipCompressionLevel)],
) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data, level) in entries {
        writer.add_file_with_options(name, data, *level)?;
    }
    writer.finish()?;
    Ok(writer.into_inner()?.into_inner())
}

/// Parse `score_xml` both as plain MusicXML and from `archive`, and require
/// identical scores.
async fn assert_same_score(archive: &[u8]) -> TestResult {
    let parser = MusicXmlParser::new();
    let plain = parser.parse_string(SCORE_XML).await?;
    let zipped = parser.parse_mxl_bytes(archive)?;
    assert_eq!(
        serde_json::to_value(&zipped)?,
        serde_json::to_value(&plain)?
    );
    assert_eq!(zipped.title, "Zipped Song");
    assert_eq!(zipped.notes.len(), 2);
    Ok(())
}

fn expect_error(result: crate::Result<String>, needle: &str) -> TestResult {
    match result {
        Err(crate::Error::Format(message)) if message.contains(needle) => Ok(()),
        other => Err(format!("expected an error containing {needle:?}, got {other:?}").into()),
    }
}

#[tokio::test]
async fn test_mxl_stored_with_container_matches_plain_score() -> TestResult {
    let container = container("score/song.musicxml");
    let archive = zip(&[
        (
            "META-INF/container.xml",
            container.as_bytes(),
            ZipCompressionLevel::Store,
        ),
        (
            "score/song.musicxml",
            SCORE_XML.as_bytes(),
            ZipCompressionLevel::Store,
        ),
    ])?;
    assert_same_score(&archive).await
}

#[tokio::test]
async fn test_mxl_deflated_with_container_matches_plain_score() -> TestResult {
    let container = container("song.xml");
    let archive = zip(&[
        (
            "mimetype",
            b"application/vnd.recordare.musicxml",
            ZipCompressionLevel::Store,
        ),
        (
            "META-INF/container.xml",
            container.as_bytes(),
            ZipCompressionLevel::Best,
        ),
        (
            "song.xml",
            SCORE_XML.as_bytes(),
            ZipCompressionLevel::Normal,
        ),
    ])?;
    assert_same_score(&archive).await
}

/// Without a container, the first `.musicxml`/`.xml` outside `META-INF/` wins.
#[tokio::test]
async fn test_mxl_without_container_uses_first_score_entry() -> TestResult {
    let archive = zip(&[
        (
            "META-INF/other.xml",
            b"<not-a-score/>",
            ZipCompressionLevel::Normal,
        ),
        ("readme.txt", b"hello", ZipCompressionLevel::Normal),
        (
            "Song.MusicXML",
            SCORE_XML.as_bytes(),
            ZipCompressionLevel::Normal,
        ),
        ("second.xml", b"<not-a-score/>", ZipCompressionLevel::Normal),
    ])?;
    assert_same_score(&archive).await
}

#[test]
fn test_mxl_missing_rootfile_is_rejected() -> TestResult {
    let container = container("missing.musicxml");
    let archive = zip(&[
        (
            "META-INF/container.xml",
            container.as_bytes(),
            ZipCompressionLevel::Normal,
        ),
        (
            "song.musicxml",
            SCORE_XML.as_bytes(),
            ZipCompressionLevel::Normal,
        ),
    ])?;
    expect_error(extract_score_xml(&archive), "missing.musicxml")?;

    let no_score = zip(&[("readme.txt", b"hello", ZipCompressionLevel::Normal)])?;
    expect_error(extract_score_xml(&no_score), "no container.xml")
}

#[test]
fn test_mxl_oversized_entry_is_rejected() -> TestResult {
    let size = usize::try_from(MAX_MXL_ENTRY_BYTES)? + 1;
    let big = vec![b'a'; size];
    let archive = zip(&[("song.musicxml", &big, ZipCompressionLevel::Store)])?;
    expect_error(extract_score_xml(&archive), "byte limit")
}

/// A highly compressible entry beyond the ratio cap (a zip bomb) is refused
/// before it is inflated.
#[test]
fn test_mxl_compression_ratio_bomb_is_rejected() -> TestResult {
    let zeros = vec![0_u8; 4 * 1024 * 1024];
    let archive = zip(&[("song.musicxml", &zeros, ZipCompressionLevel::Best)])?;
    expect_error(extract_score_xml(&archive), "compression ratio")
}

/// An entry whose header under-declares its size never inflates past the
/// declared length.
#[test]
fn test_mxl_lying_size_header_is_rejected() -> TestResult {
    let real = vec![b'x'; 10_000];
    let compressed = oxiarc_deflate::deflate(&real, 6)?;
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer.add_file_raw(
        "song.musicxml",
        ZipMethod::Deflate,
        Crc32::compute(&real),
        100,
        None,
        &compressed,
    )?;
    writer.finish()?;
    let archive = writer.into_inner()?.into_inner();
    expect_error(extract_score_xml(&archive), "does not inflate")
}

#[test]
fn test_mxl_path_tricks_are_rejected() -> TestResult {
    let archive = zip(&[(
        "../evil.musicxml",
        SCORE_XML.as_bytes(),
        ZipCompressionLevel::Store,
    )])?;
    expect_error(extract_score_xml(&archive), "unsafe entry path")?;

    let container = container("../outside.musicxml");
    let archive = zip(&[
        (
            "META-INF/container.xml",
            container.as_bytes(),
            ZipCompressionLevel::Store,
        ),
        (
            "song.musicxml",
            SCORE_XML.as_bytes(),
            ZipCompressionLevel::Store,
        ),
    ])?;
    expect_error(extract_score_xml(&archive), "not a safe relative path")?;

    assert!(!is_safe_entry_path("/abs.xml"));
    assert!(!is_safe_entry_path(r"C:\score.xml"));
    assert!(!is_safe_entry_path(r"a\..\b.xml"));
    assert!(is_safe_entry_path("score/song.musicxml"));
    Ok(())
}

/// Non-archive input is an error (oxiarc may read it as an empty archive,
/// which then has no score), never a panic or an empty score.
#[test]
fn test_mxl_garbage_is_rejected() -> TestResult {
    match extract_score_xml(b"not a zip archive") {
        Err(crate::Error::Format(_)) => Ok(()),
        other => Err(format!("expected a format error, got {other:?}").into()),
    }
}

#[test]
fn test_rootfile_path_parsing() {
    assert_eq!(
        rootfile_path(&container("a/b.musicxml")).as_deref(),
        Some("a/b.musicxml")
    );
    assert_eq!(
        rootfile_path("<rootfiles><rootfile media-type='x' full-path='c.xml'/></rootfiles>")
            .as_deref(),
        Some("c.xml")
    );
    assert_eq!(rootfile_path("<rootfiles></rootfiles>"), None);
}

/// End to end through the public file API: a `.mxl` on disk parses to the
/// same score as the plain MusicXML.
#[tokio::test]
async fn test_mxl_file_parses_through_music_xml_parser() -> TestResult {
    let container = container("song.musicxml");
    let archive = zip(&[
        (
            "META-INF/container.xml",
            container.as_bytes(),
            ZipCompressionLevel::Normal,
        ),
        (
            "song.musicxml",
            SCORE_XML.as_bytes(),
            ZipCompressionLevel::Normal,
        ),
    ])?;
    let path = std::env::temp_dir().join(format!("voirs_singing_song_{}.mxl", std::process::id()));
    std::fs::write(&path, &archive)?;
    let path_str = path.to_str().ok_or("temp path is not UTF-8")?;

    let parser = MusicXmlParser::new();
    let result = parser.parse_file(path_str).await;
    let _ = std::fs::remove_file(&path);
    let zipped = result?;
    let plain = parser.parse_string(SCORE_XML).await?;
    assert_eq!(
        serde_json::to_value(&zipped)?,
        serde_json::to_value(&plain)?
    );
    Ok(())
}
