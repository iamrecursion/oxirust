//! Round-trip and quarantine workflow integration tests.
//!
//! Task 13: ingest → compute checksums → verify → confirm match.
//! Task 14: corrupt file → detect → quarantine simulation → restore.
//!
//! These tests exercise multiple modules together to verify end-to-end
//! correctness of the archive verification pipeline.

#[cfg(test)]
mod roundtrip {
    use crate::archive_verify::{
        ArchiveManifest, ArchiveVerifier, ManifestEntry, VerificationLevel,
    };
    use crate::mmap_checksum::{compute_checksums_mmap, verify_file_checksum, MmapChecksumConfig};
    use crate::parallel_checksum::ParallelChecksumConfig;
    use crate::sidecar::{SidecarConfig, SidecarGenerator, SidecarManifest};
    use std::io::Write;
    use std::path::PathBuf;

    // ----- helpers --------------------------------------------------------

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&d).expect("create temp dir");
        d
    }

    fn write_file(dir: &std::path::Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(content).expect("write file");
        path
    }

    // ----- Task 13: ingest → checksum → verify → confirm ------------------

    /// Full round-trip: write files, compute checksums, build manifest,
    /// verify manifest, confirm all files pass.
    #[test]
    fn test_roundtrip_ingest_checksum_verify_confirm() {
        let dir = temp_dir("oximedia_rt_roundtrip");

        let files = vec![
            (
                "alpha.bin",
                b"alpha file content for round-trip test" as &[u8],
            ),
            ("beta.bin", b"beta  file content for round-trip test"),
            (
                "gamma.bin",
                b"gamma file content for round-trip test with extra bytes",
            ),
        ];

        // 1. INGEST: write files to disk
        let paths: Vec<PathBuf> = files
            .iter()
            .map(|(name, content)| write_file(&dir, name, content))
            .collect();

        // 2. COMPUTE CHECKSUMS using parallel_checksum
        let cfg = ParallelChecksumConfig {
            enable_sha256: true,
            enable_blake3: true,
            enable_crc32: true,
            enable_md5: false,
            buffer_size: 1024 * 1024,
        };

        let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
        let checksum_results = crate::parallel_checksum::compute_parallel_batch(&path_refs, &cfg);

        // 3. BUILD MANIFEST
        let entries: Vec<ManifestEntry> = files
            .iter()
            .zip(checksum_results.iter())
            .map(|((name, content), result)| {
                let checksums = result.as_ref().expect("checksum should succeed");
                ManifestEntry {
                    path: (*name).to_string(),
                    size_bytes: content.len() as u64,
                    sha256: checksums.checksums.sha256.clone().unwrap_or_default(),
                    compressed_size: content.len() as u64,
                    modified_at: 0,
                }
            })
            .collect();

        let manifest = ArchiveManifest::build(entries);
        assert_eq!(manifest.entries.len(), 3);
        assert_eq!(
            manifest.total_size_bytes,
            files.iter().map(|(_, c)| c.len() as u64).sum::<u64>()
        );

        // 4. VERIFY using ArchiveVerifier (Checksum level)
        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report = verifier.verify_manifest(&manifest, &dir);

        // 5. CONFIRM: all files pass, no errors
        assert_eq!(report.total_entries, 3);
        assert_eq!(
            report.verified_ok, 3,
            "all files should pass: {:?}",
            report.errors
        );
        assert!(
            report.errors.is_empty(),
            "unexpected errors: {:?}",
            report.errors
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Round-trip with parallel verification (rayon).
    #[test]
    fn test_roundtrip_parallel_verify() {
        let dir = temp_dir("oximedia_rt_parallel");

        let files: Vec<(&str, Vec<u8>)> = (0..10)
            .map(|i| {
                let content = format!("parallel test file content number {i}").into_bytes();
                (
                    Box::leak(format!("file_{i:02}.bin").into_boxed_str()) as &str,
                    content,
                )
            })
            .collect();

        let paths: Vec<PathBuf> = files
            .iter()
            .map(|(name, content)| write_file(&dir, name, content))
            .collect();

        let cfg = ParallelChecksumConfig {
            enable_sha256: true,
            enable_blake3: false,
            enable_crc32: false,
            enable_md5: false,
            buffer_size: 1024 * 1024,
        };

        let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
        let checksums = crate::parallel_checksum::compute_parallel_batch(&path_refs, &cfg);

        let entries: Vec<ManifestEntry> = files
            .iter()
            .zip(checksums.iter())
            .map(|((name, content), result)| {
                let cs = result.as_ref().expect("ok");
                ManifestEntry {
                    path: (*name).to_string(),
                    size_bytes: content.len() as u64,
                    sha256: cs.checksums.sha256.clone().unwrap_or_default(),
                    compressed_size: content.len() as u64,
                    modified_at: 0,
                }
            })
            .collect();

        let manifest = ArchiveManifest::build(entries);

        // Parallel verifier
        let verifier = ArchiveVerifier::with_parallelism(VerificationLevel::Checksum, 4);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 10);
        assert!(report.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Round-trip using mmap checksums.
    #[test]
    fn test_roundtrip_mmap_checksum_verify() {
        let dir = temp_dir("oximedia_rt_mmap");

        let content = b"mmap round-trip test content that is larger to exercise the pipeline";
        let path = write_file(&dir, "mmap_test.bin", content);

        // Compute via mmap
        let cfg = MmapChecksumConfig {
            enable_sha256: true,
            enable_blake3: true,
            enable_crc32: true,
            enable_md5: false,
            mmap_threshold: 32, // force mmap even for small file
        };
        let result = compute_checksums_mmap(&path, &cfg).expect("mmap checksum");
        assert!(result.sha256.is_some());
        assert!(result.blake3.is_some());

        // Build manifest and verify
        let entry = ManifestEntry {
            path: "mmap_test.bin".to_string(),
            size_bytes: content.len() as u64,
            sha256: result.sha256.clone().unwrap_or_default(),
            compressed_size: content.len() as u64,
            modified_at: 0,
        };
        let manifest = ArchiveManifest::build(vec![entry]);

        let verifier = ArchiveVerifier::new(VerificationLevel::Full);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 1);
        assert!(report.errors.is_empty());

        // Also verify using verify_file_checksum
        let ok = verify_file_checksum(
            &path,
            result.sha256.as_deref(),
            result.blake3.as_deref(),
            result.crc32.as_deref(),
        )
        .expect("verify_file_checksum");
        assert!(ok, "file verification should pass");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Round-trip with sidecar generation.
    #[test]
    fn test_roundtrip_with_sidecar_generation() {
        let dir = temp_dir("oximedia_rt_sidecar");

        let content = b"sidecar round-trip content for verification";
        let path = write_file(&dir, "media.mxf", content);

        // Compute checksums
        let cfg = ParallelChecksumConfig {
            enable_sha256: true,
            enable_blake3: true,
            enable_crc32: false,
            enable_md5: false,
            buffer_size: 1024 * 1024,
        };
        let cs = crate::parallel_checksum::compute_parallel_file(&path, &cfg)
            .expect("parallel checksum");

        // Build sidecar manifest
        let sidecar_gen = SidecarGenerator::new(SidecarConfig {
            generate_sha256: true,
            generate_md5: false,
            generate_blake3: true,
            generate_json: true,
            generate_text: false,
            inline: false,
            sidecar_dir: Some(dir.join("sidecars")),
        });

        let mut manifest = SidecarManifest::new("rt-archive");
        let mut entry = crate::sidecar::ChecksumEntry::new("media.mxf", content.len() as u64);
        entry.sha256 = cs.checksums.sha256.clone();
        entry.blake3 = cs.checksums.blake3.clone();
        manifest.add(entry);

        let written = sidecar_gen
            .write_sidecars(&manifest, &dir)
            .expect("write sidecars");
        assert!(!written.is_empty());

        // Reload JSON sidecar and verify it matches
        let json_path = written
            .iter()
            .find(|p| p.extension().map_or(false, |e| e == "json"));
        if let Some(json_path) = json_path {
            let json_content = std::fs::read_to_string(json_path).expect("read json");
            let loaded = SidecarManifest::from_json(&json_content).expect("parse json");
            assert_eq!(loaded.entries[0].sha256, cs.checksums.sha256);
            assert_eq!(loaded.entries[0].blake3, cs.checksums.blake3);
        }

        // Now verify the original file against the stored checksums
        let ok = verify_file_checksum(
            &path,
            cs.checksums.sha256.as_deref(),
            cs.checksums.blake3.as_deref(),
            None,
        )
        .expect("verify");
        assert!(ok, "file must pass checksum verification");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Full round-trip with manifest JSON serialization.
    #[test]
    fn test_roundtrip_manifest_json_persistence() {
        let dir = temp_dir("oximedia_rt_json_persist");

        let content = b"persistent manifest content for json roundtrip test";
        write_file(&dir, "persist.bin", content);

        let cfg = ParallelChecksumConfig {
            enable_sha256: true,
            enable_blake3: false,
            enable_crc32: false,
            enable_md5: false,
            buffer_size: 1024 * 1024,
        };
        let path = dir.join("persist.bin");
        let cs = crate::parallel_checksum::compute_parallel_file(&path, &cfg).expect("cs");

        let entry = ManifestEntry {
            path: "persist.bin".to_string(),
            size_bytes: content.len() as u64,
            sha256: cs.checksums.sha256.clone().unwrap_or_default(),
            compressed_size: content.len() as u64,
            modified_at: 0,
        };

        // Serialize manifest to JSON, then reload and re-verify
        let original_manifest = ArchiveManifest::build(vec![entry]);
        let json = original_manifest.to_json();

        let reloaded = ArchiveManifest::from_json(&json).expect("reload manifest");
        assert_eq!(
            reloaded.total_size_bytes,
            original_manifest.total_size_bytes
        );
        assert_eq!(
            reloaded.archive_checksum,
            original_manifest.archive_checksum
        );

        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report = verifier.verify_manifest(&reloaded, &dir);
        assert_eq!(report.verified_ok, 1);
        assert!(report.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }
}

// ---------------------------------------------------------------------------
// Task 14: Quarantine workflow
// ---------------------------------------------------------------------------

#[cfg(test)]
mod quarantine_workflow {
    use crate::archive_verify::{
        ArchiveManifest, ArchiveVerifier, ManifestEntry, VerificationError, VerificationLevel,
    };
    use crate::mmap_checksum::{compute_checksums_mmap, MmapChecksumConfig};
    use crate::quarantine_policy::{
        AdmissionDecision, EvictionStrategy, QuarantineInventory, QuarantinePolicy,
    };
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&d).expect("create temp dir");
        d
    }

    fn write_file(dir: &std::path::Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(content).expect("write file");
        path
    }

    // ---- Corrupt, detect, quarantine, restore (in-process simulation) -----

    /// Quarantine workflow:
    /// 1. Write a valid file, record its checksum.
    /// 2. Corrupt the file on disk.
    /// 3. Verify → detect corruption.
    /// 4. Move file to quarantine directory.
    /// 5. Restore backup copy to original location.
    /// 6. Re-verify → confirm restored file passes.
    #[test]
    fn test_quarantine_corrupt_detect_move_restore() {
        let dir = temp_dir("oximedia_qw_corrupt_restore");
        let quarantine_dir = dir.join("quarantine");
        let backup_dir = dir.join("backup");
        std::fs::create_dir_all(&quarantine_dir).ok();
        std::fs::create_dir_all(&backup_dir).ok();

        let original_content = b"original valid media content for quarantine test";
        let corrupted_content = b"XXXXX corrupted media content XXXXXXXXXXXXXXXXXX";
        assert_eq!(original_content.len(), corrupted_content.len()); // same size

        let path = write_file(&dir, "media.mxf", original_content);
        let backup_path = write_file(&backup_dir, "media.mxf", original_content);

        // 1. Compute checksums for the valid file
        let cfg = MmapChecksumConfig {
            enable_sha256: true,
            enable_blake3: false,
            enable_crc32: false,
            enable_md5: false,
            mmap_threshold: 64 * 1024,
        };
        let valid_cs = compute_checksums_mmap(&path, &cfg).expect("valid checksum");
        let expected_sha256 = valid_cs.sha256.clone().unwrap_or_default();

        // Build manifest with valid checksum
        let entry = ManifestEntry {
            path: "media.mxf".to_string(),
            size_bytes: original_content.len() as u64,
            sha256: expected_sha256.clone(),
            compressed_size: original_content.len() as u64,
            modified_at: 0,
        };
        let manifest = ArchiveManifest::build(vec![entry]);

        // 2. Corrupt the file
        std::fs::write(&path, corrupted_content).expect("corrupt file");

        // 3. Verify → should detect corruption (checksum mismatch)
        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report = verifier.verify_manifest(&manifest, &dir);
        assert_eq!(
            report.verified_ok, 0,
            "corrupted file should fail verification"
        );
        assert!(!report.errors.is_empty(), "should have errors");
        assert!(
            matches!(
                &report.errors[0],
                VerificationError::ChecksumMismatch { .. }
            ),
            "expected ChecksumMismatch, got: {:?}",
            report.errors[0]
        );

        // 4. Move corrupted file to quarantine
        let quarantine_path = quarantine_dir.join("media.mxf.quarantined");
        std::fs::rename(&path, &quarantine_path).expect("move to quarantine");
        assert!(!path.exists(), "original should be gone");
        assert!(quarantine_path.exists(), "quarantine copy should exist");

        // 5. Restore backup to original location
        std::fs::copy(&backup_path, &path).expect("restore from backup");
        assert!(path.exists(), "restored file should exist");

        // 6. Re-verify → should pass now
        let report2 = verifier.verify_manifest(&manifest, &dir);
        assert_eq!(report2.verified_ok, 1, "restored file should pass");
        assert!(report2.errors.is_empty(), "no errors after restore");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Test that the quarantine policy correctly rejects files above the per-file limit.
    #[test]
    fn test_quarantine_policy_rejects_oversized_file() {
        let policy = QuarantinePolicy {
            max_single_file_bytes: Some(1024),
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(2048, 0, 0);
        assert!(decision.is_rejected(), "oversized file should be rejected");
    }

    /// Test that the quarantine policy triggers eviction when quota is exceeded.
    #[test]
    fn test_quarantine_policy_eviction_on_quota_exceeded() {
        let policy = QuarantinePolicy {
            max_total_bytes: Some(10_000),
            eviction_strategy: EvictionStrategy::OldestFirst,
            ..QuarantinePolicy::unlimited()
        };
        let decision = policy.check_admission(5000, 8000, 0);
        assert!(decision.is_admitted(), "should admit after eviction");
        assert!(
            matches!(decision, AdmissionDecision::AdmitAfterEviction(_)),
            "should require eviction"
        );
    }

    /// Test inventory cleanup candidates.
    #[test]
    fn test_quarantine_inventory_cleanup() {
        let mut inv = QuarantineInventory::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        inv.add(1, now - 100 * 86400, 1000, PathBuf::from("/q/old_file.bin"));
        inv.add(2, now - 10 * 86400, 2000, PathBuf::from("/q/recent.bin"));
        inv.add(3, now - 50 * 86400, 500, PathBuf::from("/q/middle.bin"));

        let policy = QuarantinePolicy {
            auto_cleanup_after_days: Some(30),
            ..QuarantinePolicy::default()
        };
        let mut candidates = inv.cleanup_candidates(&policy, now);
        candidates.sort();
        assert_eq!(
            candidates,
            vec![1, 3],
            "should select files older than 30 days"
        );
    }

    /// Simulate full quarantine workflow with inventory and eviction.
    #[test]
    fn test_quarantine_workflow_full_simulation() {
        let dir = temp_dir("oximedia_qw_full_sim");
        let quarantine_dir = dir.join("quarantine");
        std::fs::create_dir_all(&quarantine_dir).ok();

        // Write several files
        let files = vec![
            ("file_a.bin", b"content of file a" as &[u8]),
            ("file_b.bin", b"content of file b"),
            ("file_c.bin", b"content of file c"),
        ];

        let mut inv = QuarantineInventory::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(1_700_000_000);

        // Populate inventory with simulated quarantine records
        for (i, (name, content)) in files.iter().enumerate() {
            let path = write_file(&dir, name, content);
            let age_secs = (i as u64 + 1) * 40 * 86400; // 40, 80, 120 days old
            inv.add(i as u64 + 1, now - age_secs, content.len() as u64, path);
        }

        assert_eq!(inv.count(), 3);
        assert_eq!(
            inv.total_bytes(),
            files.iter().map(|(_, c)| c.len() as u64).sum::<u64>()
        );

        // Policy: cleanup after 30 days
        let policy = QuarantinePolicy {
            auto_cleanup_after_days: Some(30),
            max_total_bytes: Some(1_000_000),
            ..QuarantinePolicy::default()
        };

        let candidates = inv.cleanup_candidates(&policy, now);
        // All files are > 30 days old
        assert_eq!(candidates.len(), 3);

        // Evict oldest first
        let records = inv.as_tuples();
        let eviction_req = crate::quarantine_policy::EvictionRequest {
            reason: "quota exceeded".into(),
            strategy: EvictionStrategy::OldestFirst,
            bytes_to_free: 0,
            files_to_evict: 1,
        };
        let to_evict = policy.select_for_eviction(&records, &eviction_req);
        assert_eq!(to_evict.len(), 1);
        // Should evict ID 1 (oldest = 40 days old) — actually sorted by timestamp
        // ID 1 = 40 days, ID 2 = 80 days, ID 3 = 120 days
        // oldest timestamp = now - 120*86400 = ID 3
        // After sort by timestamp ascending: ID 3 (120d), ID 2 (80d), ID 1 (40d)
        assert!(
            to_evict.contains(&3),
            "should evict oldest (ID 3 = 120 days old)"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Test detecting corruption via checksum mismatch (no quarantine dir needed).
    #[test]
    fn test_detect_corruption_by_checksum() {
        let dir = temp_dir("oximedia_qw_detect");

        let original = b"original uncorrupted file content";
        let path = write_file(&dir, "check.bin", original);

        // Compute checksum
        let cfg = MmapChecksumConfig {
            enable_sha256: true,
            enable_blake3: false,
            enable_crc32: false,
            enable_md5: false,
            mmap_threshold: 64 * 1024,
        };
        let cs = compute_checksums_mmap(&path, &cfg).expect("cs");

        // Corrupt file (same size to avoid SizeMismatch masking the ChecksumMismatch)
        let corrupted = b"xoriginal uncorrupted file conten";
        assert_eq!(original.len(), corrupted.len());
        std::fs::write(&path, corrupted).expect("corrupt");

        // Verify → expect ChecksumMismatch
        let entry = ManifestEntry {
            path: "check.bin".to_string(),
            size_bytes: original.len() as u64,
            sha256: cs.sha256.clone().unwrap_or_default(),
            compressed_size: original.len() as u64,
            modified_at: 0,
        };
        let manifest = ArchiveManifest::build(vec![entry]);
        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 0);
        assert!(
            matches!(
                &report.errors[0],
                VerificationError::ChecksumMismatch { .. }
            ),
            "expected ChecksumMismatch"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Test that restore brings back the original file and re-verification passes.
    #[test]
    fn test_quarantine_restore_re_verify() {
        let dir = temp_dir("oximedia_qw_restore_verify");
        let quarantine = dir.join("quarantine");
        let backup = dir.join("backup");
        std::fs::create_dir_all(&quarantine).ok();
        std::fs::create_dir_all(&backup).ok();

        let content = b"file to be corrupted then restored";
        let path = write_file(&dir, "restore_test.bin", content);
        write_file(&backup, "restore_test.bin", content);

        // Compute checksum
        let cfg = MmapChecksumConfig {
            enable_sha256: true,
            enable_blake3: false,
            enable_crc32: false,
            enable_md5: false,
            mmap_threshold: 64 * 1024,
        };
        let cs = compute_checksums_mmap(&path, &cfg).expect("cs");

        let entry = ManifestEntry {
            path: "restore_test.bin".to_string(),
            size_bytes: content.len() as u64,
            sha256: cs.sha256.clone().unwrap_or_default(),
            compressed_size: content.len() as u64,
            modified_at: 0,
        };
        let manifest = ArchiveManifest::build(vec![entry]);

        // Corrupt
        let corrupt = vec![0xFFu8; content.len()];
        std::fs::write(&path, &corrupt).expect("corrupt");

        // Verify (should fail)
        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report1 = verifier.verify_manifest(&manifest, &dir);
        assert_eq!(report1.verified_ok, 0, "should fail after corruption");

        // Quarantine (move corrupted)
        let qpath = quarantine.join("restore_test.bin.corrupted");
        std::fs::rename(&path, &qpath).expect("quarantine");

        // Restore from backup
        std::fs::copy(&backup.join("restore_test.bin"), &path).expect("restore");

        // Re-verify (should pass)
        let report2 = verifier.verify_manifest(&manifest, &dir);
        assert_eq!(report2.verified_ok, 1, "should pass after restore");
        assert!(report2.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }
}
