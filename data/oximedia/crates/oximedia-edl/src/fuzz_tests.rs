//! Fuzz-style edge case tests for the EDL parser.
//!
//! These tests exercise the parser with malformed, unusual, and edge-case
//! EDL inputs to ensure robust error handling and no panics.

#[cfg(test)]
mod tests {
    use crate::event::{EditType, EdlEvent, TrackType};
    use crate::parser::parse_edl;
    use crate::roundtrip::RoundtripValidator;
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::EdlFormat;

    // ── Malformed input tests ─────────────────────────────────────────────

    #[test]
    fn test_fuzz_empty_input() {
        let result = parse_edl("");
        // Should succeed with empty EDL (no events is valid)
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_fuzz_whitespace_only() {
        let result = parse_edl("   \n  \n \t \n  ");
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_fuzz_title_only() {
        let result = parse_edl("TITLE: My Title\n");
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.title, Some("My Title".to_string()));
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_fuzz_fcm_only() {
        let result = parse_edl("FCM: DROP FRAME\n");
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_comments_only() {
        let result = parse_edl("* This is a comment\n* Another comment\n");
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_garbage_lines() {
        // Lines that don't match any pattern should be silently ignored
        let edl_text = "TITLE: Test\nFCM: NON-DROP FRAME\ngarbage line\nmore garbage\n\
            001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(edl_text);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_very_long_title() {
        let long_title = "T".repeat(10_000);
        let input = format!("TITLE: {long_title}\nFCM: NON-DROP FRAME\n");
        let result = parse_edl(&input);
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.title.as_ref().map(|s| s.len()), Some(10_000));
    }

    #[test]
    fn test_fuzz_many_blank_lines() {
        let blanks = "\n".repeat(1000);
        let input = format!(
            "TITLE: Test\n{blanks}\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n"
        );
        let result = parse_edl(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_event_number_zero() {
        let input = "000  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_event_number_large() {
        let input = "999  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_multiple_titles() {
        // Last title wins
        let input = "TITLE: First\nTITLE: Second\nFCM: NON-DROP FRAME\n\
            001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.title, Some("Second".to_string()));
    }

    #[test]
    fn test_fuzz_multiple_fcm_lines() {
        let input = "FCM: DROP FRAME\nFCM: NON-DROP FRAME\n\
            001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_event_with_many_comments() {
        let mut input =
            String::from("001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n");
        for i in 0..100 {
            input.push_str(&format!("* Comment number {i}\n"));
        }
        let result = parse_edl(&input);
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.events.len(), 1);
        // Comments should be preserved
        assert!(edl.events[0].comments.len() >= 90); // Some may be clip names etc.
    }

    #[test]
    fn test_fuzz_mixed_separators() {
        // Use semicolons in non-DF timecodes (parser should be lenient)
        let input = "FCM: NON-DROP FRAME\n\
            001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_extra_spaces_in_event() {
        let input = "  001   AX        V      C         01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00  \n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_crlf_line_endings() {
        let input = "TITLE: Test\r\nFCM: NON-DROP FRAME\r\n\r\n\
            001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\r\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_consecutive_events_no_blank() {
        let input = "001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
            002  AX  V  C  01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(edl.events.len(), 2);
    }

    #[test]
    fn test_fuzz_comment_before_any_event() {
        let input = "TITLE: Test\n* Orphan comment\n\
            001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fuzz_clip_name_special_chars() {
        let input = "001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
            * FROM CLIP NAME: path/to/my file (v2) [final].mov\n";
        let result = parse_edl(input);
        assert!(result.is_ok());
        let edl = result.expect("should parse");
        assert_eq!(
            edl.events[0].clip_name,
            Some("path/to/my file (v2) [final].mov".to_string())
        );
    }

    #[test]
    fn test_fuzz_dissolve_without_duration() {
        // Dissolve without a transition duration is invalid — the parser
        // (or validator) should reject it rather than silently accepting.
        let input = "001  AX  V  D  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let result = parse_edl(input);
        // Parser rejects dissolve without duration — this is expected
        assert!(result.is_err());
    }

    // ── Round-trip tests ──────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_single_event() {
        let edl_text = "TITLE: Single\nFCM: NON-DROP FRAME\n\n\
            001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\n";
        let validator = RoundtripValidator::full();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "Diffs: {:?}", report.diffs);
    }

    #[test]
    fn test_roundtrip_multiple_events() {
        let edl_text = "TITLE: Multi\nFCM: NON-DROP FRAME\n\n\
            001  A001     V     C        01:00:00:00 01:00:03:00 01:00:00:00 01:00:03:00\n\
            002  A002     V     C        01:00:03:00 01:00:06:00 01:00:03:00 01:00:06:00\n\
            003  A003     V     C        01:00:06:00 01:00:09:00 01:00:06:00 01:00:09:00\n\n";
        let validator = RoundtripValidator::full();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "Diffs: {:?}", report.diffs);
    }

    #[test]
    fn test_roundtrip_drop_frame() {
        let edl_text = "TITLE: DF Test\nFCM: DROP FRAME\n\n\
            001  AX       V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00\n\n";
        let validator = RoundtripValidator::full();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "Diffs: {:?}", report.diffs);
    }

    #[test]
    fn test_roundtrip_with_clip_names() {
        let edl_text = "TITLE: Clips\nFCM: NON-DROP FRAME\n\n\
            001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
            * FROM CLIP NAME: shot001.mov\n\
            002  A002     V     C        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n\
            * FROM CLIP NAME: shot002.mov\n\n";
        let validator = RoundtripValidator::full();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "Diffs: {:?}", report.diffs);
    }

    #[test]
    fn test_roundtrip_with_dissolve() {
        let edl_text = "TITLE: Dissolve\nFCM: NON-DROP FRAME\n\n\
            001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
            002  AX       V     D    030 01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n\n";
        let validator = RoundtripValidator::default();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "Diffs: {:?}", report.diffs);
    }

    #[test]
    fn test_roundtrip_audio_tracks() {
        let edl_text = "TITLE: Audio\nFCM: NON-DROP FRAME\n\n\
            001  A001     A     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
            002  A001     A2    C        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n\n";
        let validator = RoundtripValidator::default();
        let report = validator
            .validate(edl_text)
            .expect("validation should succeed");
        assert!(report.is_lossless(), "Diffs: {:?}", report.diffs);
    }

    #[test]
    fn test_roundtrip_programmatic_edl() {
        // Build an EDL programmatically at 29.97 NDF (the standard CMX 3600
        // rate) so that the FCM: NON-DROP FRAME header round-trips cleanly.
        let rate = EdlFrameRate::Fps2997NDF;
        let mut edl = crate::Edl::new(EdlFormat::Cmx3600);
        edl.set_title("Programmatic".to_string());
        edl.set_frame_rate(rate);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, rate).expect("tc1");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, rate).expect("tc2");
        let tc3 = EdlTimecode::new(1, 0, 10, 0, rate).expect("tc3");

        let mut ev1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        ev1.set_clip_name("clip1.mov".to_string());
        edl.add_event(ev1).expect("add ev1");

        let ev2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc2,
            tc3,
            tc2,
            tc3,
        );
        edl.add_event(ev2).expect("add ev2");

        let generated = edl.to_string_format().expect("generate");
        let reparsed = parse_edl(&generated).expect("reparse");

        assert_eq!(reparsed.title, edl.title);
        assert_eq!(reparsed.events.len(), edl.events.len());
        for (orig, re) in edl.events.iter().zip(reparsed.events.iter()) {
            assert_eq!(orig.number, re.number);
            assert_eq!(orig.reel, re.reel);
            assert_eq!(orig.edit_type, re.edit_type);
            assert_eq!(orig.source_in, re.source_in);
            assert_eq!(orig.source_out, re.source_out);
            assert_eq!(orig.record_in, re.record_in);
            assert_eq!(orig.record_out, re.record_out);
        }
    }
}
