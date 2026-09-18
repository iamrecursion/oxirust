//! Smoke tests for newly-wired orphan modules in oximedia-captions.

#[test]
fn test_caption_localization_track() {
    use oximedia_captions::caption_localization::LocalizedTrack;
    let track = LocalizedTrack::new("en-US".to_string(), "English".to_string());
    assert_eq!(track.language_tag, "en-US");
}

#[test]
fn test_caption_versioning_history() {
    use oximedia_captions::caption_versioning::VersionHistory;
    let history = VersionHistory::new();
    assert_eq!(history.len(), 0, "New VersionHistory should be empty");
}

#[test]
fn test_karaoke_timing_track() {
    use oximedia_captions::karaoke_timing::KaraokeTrack;
    let track = KaraokeTrack::new();
    assert_eq!(track.len(), 0);
}

#[test]
fn test_ocr_subtitle_bounding_box() {
    use oximedia_captions::ocr_subtitle::BoundingBox;
    let bbox = BoundingBox {
        x: 10,
        y: 20,
        width: 100,
        height: 30,
    };
    assert_eq!(bbox.width, 100);
}

#[test]
fn test_sdh_generator_tag() {
    use oximedia_captions::sdh_generator::{SdhPosition, SdhTag, SoundDescription};
    let tag = SdhTag::new(
        SoundDescription::Music("jazz".to_string()),
        SdhPosition::Before,
    );
    let s = tag.sound.to_bracketed_string();
    assert!(!s.is_empty());
}

#[test]
fn test_smpte_2052_profile_uri() {
    use oximedia_captions::smpte_2052::SmpteProfile;
    let uri = SmpteProfile::Base.uri();
    assert!(!uri.is_empty());
}

#[test]
fn test_teletext_page_editor_colour() {
    use oximedia_captions::teletext_page_editor::TeletextColour;
    let (r, g, b) = TeletextColour::Red.srgb();
    assert_eq!(r, 255);
    assert_eq!(g, 0);
    assert_eq!(b, 0);
}
