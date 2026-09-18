//! Smoke tests for newly-wired orphan modules in oximedia-imf.

// ─── compliance_matrix ───────────────────────────────────────────────────────
#[test]
fn test_compliance_matrix_app2_rules_non_empty() {
    use oximedia_imf::compliance_matrix::{ApplicationProfile, ComplianceMatrix};

    let rules = ComplianceMatrix::required_constraints(ApplicationProfile::App2);
    assert!(
        !rules.is_empty(),
        "App2 profile should have compliance rules"
    );
}

#[test]
fn test_compliance_matrix_different_profiles_distinct() {
    use oximedia_imf::compliance_matrix::{ApplicationProfile, ComplianceMatrix};

    let app2_rules = ComplianceMatrix::required_constraints(ApplicationProfile::App2);
    let app2e_rules = ComplianceMatrix::required_constraints(ApplicationProfile::App2E);
    // Both profiles should have rules
    assert!(!app2_rules.is_empty(), "App2 should have rules");
    assert!(!app2e_rules.is_empty(), "App2E should have rules");
}

// ─── essence_probe ───────────────────────────────────────────────────────────
#[test]
fn test_essence_probe_nonexistent_file() {
    use oximedia_imf::essence_probe::EssenceProbe;

    let result = EssenceProbe::probe("/nonexistent/video.mxf");
    assert!(
        result.is_err(),
        "probing non-existent file should fail gracefully"
    );
}

#[test]
fn test_essence_probe_temp_file() {
    use oximedia_imf::essence_probe::EssenceProbe;
    use std::io::Write;

    let mut path = std::env::temp_dir();
    path.push("oximedia_imf_probe_smoke.mxf");
    {
        let mut f = std::fs::File::create(&path).expect("create temp MXF");
        // MXF key prefix bytes + padding
        f.write_all(&[0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01])
            .expect("write prefix");
        f.write_all(&[0x0d, 0x01, 0x02, 0x01, 0x01, 0x02, 0x01, 0x00])
            .expect("write suffix");
        f.write_all(&vec![0u8; 64]).expect("write padding");
    }
    // Should either succeed (partial parse) or fail gracefully
    let _result = EssenceProbe::probe(&path);
    let _ = std::fs::remove_file(&path);
}

// ─── imf_archive ─────────────────────────────────────────────────────────────
#[test]
fn test_imf_archive_new_aip_xml() {
    use oximedia_imf::imf_archive::{ImfArchive, OaisPackageType};

    let archive = ImfArchive::new("/tmp/test_imp", OaisPackageType::Aip)
        .with_originator("Test Studio")
        .with_rights("MIT");
    // Should produce valid XML
    let xml = archive.to_xml();
    assert!(!xml.is_empty(), "archive XML should not be empty");
    assert!(
        xml.contains("AIP") || xml.contains("Aip") || xml.contains("aip"),
        "XML should reference package type"
    );
}

// ─── imf_builder ─────────────────────────────────────────────────────────────
#[test]
fn test_imf_builder_basic_construction() {
    use oximedia_imf::imf_builder::{EditRate, ImfPackageBuilder};

    let builder = ImfPackageBuilder::new("Test Feature")
        .add_video_track("/essence/video.mxf", EditRate::fps_24())
        .add_audio_track("/essence/audio.mxf", 2, 48000);

    let pkg = builder.build();
    assert!(pkg.is_ok(), "builder should produce a valid package spec");
}

// ─── imf_diff ────────────────────────────────────────────────────────────────
#[test]
fn test_imf_diff_nonexistent_paths_error() {
    use oximedia_imf::imf_diff::ImfDiff;

    let result = ImfDiff::compare("/nonexistent/pkg_v1", "/nonexistent/pkg_v2");
    assert!(
        result.is_err(),
        "diffing non-existent directories should fail"
    );
}

#[test]
fn test_imf_diff_two_temp_dirs() {
    use oximedia_imf::imf_diff::ImfDiff;
    use std::fs;

    let mut d1 = std::env::temp_dir();
    d1.push("oximedia_imf_diff_a");
    let mut d2 = std::env::temp_dir();
    d2.push("oximedia_imf_diff_b");

    fs::create_dir_all(&d1).expect("create dir a");
    fs::create_dir_all(&d2).expect("create dir b");

    let result = ImfDiff::compare(&d1, &d2);
    assert!(result.is_ok(), "diff of two empty dirs should succeed");
    let report = result.unwrap();
    assert_eq!(
        report.added().len(),
        0,
        "no added files in identical empty dirs"
    );
    assert_eq!(
        report.removed().len(),
        0,
        "no removed files in identical empty dirs"
    );

    fs::remove_dir_all(&d1).ok();
    fs::remove_dir_all(&d2).ok();
}

// ─── imf_inspector ───────────────────────────────────────────────────────────
#[test]
fn test_imf_inspector_empty_package() {
    use oximedia_imf::imf_inspector::{ImfInspector, InspectablePackage};

    let pkg = InspectablePackage {
        package_id: "test-pkg-1".to_string(),
        title: "Test".to_string(),
        edit_rate_num: 24,
        edit_rate_den: 1,
        total_duration_eu: 0,
        video_tracks: vec![],
        audio_tracks: vec![],
        subtitle_tracks: vec![],
        pkl_asset_count: 0,
        pkl_total_size_bytes: 0,
        pkl_hash_algorithm: "SHA-1".to_string(),
        cpl_count: 1,
        has_supplemental: false,
        application_profile_urn: None,
    };
    let report = ImfInspector::inspect(&pkg);
    assert_eq!(report.video_track_count, 0);
    assert_eq!(report.audio_track_count, 0);
}

// ─── integrity_check ─────────────────────────────────────────────────────────
#[test]
fn test_integrity_check_nonexistent_path() {
    use oximedia_imf::integrity_check::ImfIntegrityChecker;

    let issues = ImfIntegrityChecker::verify("/nonexistent/imp");
    assert!(
        !issues.is_empty(),
        "non-existent path should produce integrity issues"
    );
}

#[test]
fn test_integrity_check_empty_dir() {
    use oximedia_imf::integrity_check::ImfIntegrityChecker;
    use std::fs;

    let mut path = std::env::temp_dir();
    path.push("oximedia_imf_integrity_smoke");
    fs::create_dir_all(&path).expect("create temp dir");

    // Verify doesn't panic on an empty directory
    let _ = ImfIntegrityChecker::verify(path.to_str().expect("valid path"));
    fs::remove_dir_all(&path).ok();
}

// ─── metadata_extractor ──────────────────────────────────────────────────────
#[test]
fn test_metadata_extractor_nonexistent() {
    use oximedia_imf::metadata_extractor::{MetadataExtractor, OutputFormat};

    let result = MetadataExtractor::extract("/nonexistent/imp", OutputFormat::Json);
    assert!(result.is_err(), "non-existent path should fail");
}

#[test]
fn test_metadata_extractor_empty_dir_json() {
    use oximedia_imf::metadata_extractor::{MetadataExtractor, OutputFormat};
    use std::fs;

    let mut path = std::env::temp_dir();
    path.push("oximedia_imf_meta_smoke");
    fs::create_dir_all(&path).expect("create dir");

    let result = MetadataExtractor::extract(path.to_str().expect("valid path"), OutputFormat::Json);
    if let Ok(json) = result {
        assert!(
            json.contains('{') || json.contains('['),
            "should produce JSON"
        );
    }
    fs::remove_dir_all(&path).ok();
}

// ─── partial_restore ─────────────────────────────────────────────────────────
#[test]
fn test_partial_restore_nonexistent_package() {
    use oximedia_imf::partial_restore::{PartialRestore, RestoreFilter};

    let filter = RestoreFilter::time_range(0, 5760);
    let result = PartialRestore::extract("/nonexistent/imp", filter);
    assert!(
        result.is_err(),
        "restoring non-existent package should fail"
    );
}

// ─── pkl_gen ─────────────────────────────────────────────────────────────────
#[test]
fn test_pkl_gen_xml_output() {
    use oximedia_imf::pkl_gen::PklGenerator;

    let mut gen = PklGenerator::new();
    gen.add_asset("video.mxf", 1_048_576, "deadbeef01234567");
    gen.add_asset("audio.mxf", 262_144, "0123456789abcdef");
    let xml = gen.build_xml();
    assert!(
        xml.contains("<PackingList>"),
        "XML should contain PackingList element"
    );
    assert!(xml.contains("video.mxf"), "XML should list video asset");
    assert!(xml.contains("audio.mxf"), "XML should list audio asset");
}

// ─── qc_report ───────────────────────────────────────────────────────────────
#[test]
fn test_qc_report_nonexistent_package() {
    use oximedia_imf::qc_report::QcReporter;

    let report = QcReporter::check("/nonexistent/imp");
    let summary = report.summary();
    assert!(
        !summary.is_empty(),
        "summary should describe the check outcome"
    );
}
