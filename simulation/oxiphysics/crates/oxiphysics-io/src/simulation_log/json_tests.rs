// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the JSON output of [`SimSnapshot`] and [`SimLogger`]:
//! no precision is lost, the text is always valid JSON, strings are escaped,
//! and a snapshot can be read back.

use super::*;
use crate::Error;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn flat_bits(vectors: &[[f64; 3]]) -> Vec<u64> {
    vectors.iter().flatten().map(|v| v.to_bits()).collect()
}

fn assert_same_snapshot(a: &SimSnapshot, b: &SimSnapshot) {
    assert_eq!(a.version, b.version);
    assert_eq!(a.sim_time.to_bits(), b.sim_time.to_bits());
    assert_eq!(flat_bits(&a.positions), flat_bits(&b.positions));
    assert_eq!(flat_bits(&a.velocities), flat_bits(&b.velocities));
    assert_eq!(bits(&a.masses), bits(&b.masses));
    assert_eq!(a.metadata, b.metadata);
}

fn parse_message(result: crate::Result<SimSnapshot>) -> String {
    match result {
        Err(Error::Parse(message)) => message,
        Err(other) => panic!("expected a parse error, got {other}"),
        Ok(_) => panic!("expected a parse error, got a snapshot"),
    }
}

#[test]
fn small_masses_and_times_survive_a_round_trip_bit_for_bit() {
    // One atomic mass unit in kilograms and a picosecond-scale time: both were
    // written as 0.000000000 by the fixed {:.9} format.
    let snapshot = SimSnapshot::new(
        1e-12,
        vec![[1e-10, -2.5e-11, 3.0], [0.1, 0.2, 0.3]],
        vec![[1e3, -1e-3, 0.0], [f64::MIN_POSITIVE, 5e-324, 1e300]],
        vec![1.66e-27, 9.109_383_7e-31],
    );
    let json = snapshot.to_json();
    assert!(json.contains("\"sim_time\":1e-12"), "{json}");
    assert!(
        json.contains("\"masses\":[1.66e-27,9.1093837e-31]"),
        "{json}"
    );
    let restored = SimSnapshot::from_json(&json).expect("own output parses");
    assert_same_snapshot(&snapshot, &restored);
}

#[test]
fn negative_zero_and_extreme_values_round_trip() {
    let snapshot = SimSnapshot::new(
        -0.0,
        vec![
            [-0.0, 0.0, f64::MAX],
            [f64::MIN, 1.0 + f64::EPSILON, 1.0 - f64::EPSILON],
        ],
        vec![[0.0; 3], [1e-320, -1e-320, 123_456_789.012_345]],
        vec![f64::MAX, f64::MIN_POSITIVE],
    );
    let restored = SimSnapshot::from_json(&snapshot.to_json()).expect("parses");
    assert_same_snapshot(&snapshot, &restored);
    assert!(restored.sim_time.is_sign_negative());
}

#[test]
fn non_finite_values_are_written_as_strings_and_read_back() {
    let snapshot = SimSnapshot::new(
        f64::NAN,
        vec![[f64::INFINITY, f64::NEG_INFINITY, 1.0]],
        vec![[f64::NAN, 0.0, 0.0]],
        vec![f64::INFINITY],
    );
    let json = snapshot.to_json();
    // Still a valid JSON document (a bare NaN or inf token would not be).
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(value["sim_time"], serde_json::json!("NaN"));
    assert_eq!(value["positions"][0][0], serde_json::json!("Infinity"));
    assert_eq!(value["positions"][0][1], serde_json::json!("-Infinity"));
    assert_eq!(value["masses"][0], serde_json::json!("Infinity"));

    let restored = SimSnapshot::from_json(&json).expect("parses");
    assert!(restored.sim_time.is_nan());
    assert_eq!(restored.positions[0][0], f64::INFINITY);
    assert_eq!(restored.positions[0][1], f64::NEG_INFINITY);
    assert!(restored.velocities[0][0].is_nan());
    assert_eq!(restored.masses[0], f64::INFINITY);
}

#[test]
fn metadata_text_is_escaped_and_round_trips() {
    let mut snapshot = SimSnapshot::new(0.5, vec![], vec![], vec![]);
    snapshot
        .metadata
        .insert("path".to_string(), "C:\\data\\run \"1\"".to_string());
    snapshot.metadata.insert(
        "multi\nline".to_string(),
        "tab\tbell\u{7}nul\u{1}".to_string(),
    );
    snapshot
        .metadata
        .insert("unicode".to_string(), "μ 日本語 🚀".to_string());
    snapshot
        .metadata
        .insert("quote\"key".to_string(), "back\\slash".to_string());
    let json = snapshot.to_json();
    assert!(
        json.chars().all(|c| u32::from(c) >= 0x20),
        "raw control character in {json:?}"
    );
    // A strict parser accepts it and sees the original strings.
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(value["metadata"]["path"], "C:\\data\\run \"1\"");
    assert_eq!(value["metadata"]["multi\nline"], "tab\tbell\u{7}nul\u{1}");
    let restored = SimSnapshot::from_json(&json).expect("parses");
    assert_same_snapshot(&snapshot, &restored);
}

#[test]
fn json_text_has_the_documented_keys_and_sorted_metadata() {
    let mut snapshot = SimSnapshot::new(
        2.0,
        vec![[1.0, 2.0, 3.0]],
        vec![[0.5, 0.25, 0.125]],
        vec![4.0],
    );
    snapshot.metadata.insert("b".to_string(), "2".to_string());
    snapshot.metadata.insert("a".to_string(), "1".to_string());
    assert_eq!(
        snapshot.to_json(),
        "{\"version\":\"1.0\",\"sim_time\":2.0,\"n_bodies\":1,\"positions\":[[1.0,2.0,3.0]],\
         \"velocities\":[[0.5,0.25,0.125]],\"masses\":[4.0],\"metadata\":{\"a\":\"1\",\"b\":\"2\"}}"
    );
}

#[test]
fn empty_snapshot_round_trips() {
    let snapshot = SimSnapshot::new(0.0, vec![], vec![], vec![]);
    let restored = SimSnapshot::from_json(&snapshot.to_json()).expect("parses");
    assert_same_snapshot(&snapshot, &restored);
    assert_eq!(restored.n_bodies(), 0);
}

#[test]
fn from_json_accepts_integers_unknown_keys_and_missing_optional_keys() {
    let json = "{\"version\":\"1.7\",\"sim_time\":3,\"positions\":[[1,2,3]],\
                \"velocities\":[[0,0,0]],\"masses\":[2],\"extra\":{\"anything\":[1,2]}}";
    let snapshot = SimSnapshot::from_json(json).expect("parses");
    assert_eq!(snapshot.version, SnapshotVersion { major: 1, minor: 7 });
    assert_eq!(snapshot.sim_time, 3.0);
    assert_eq!(snapshot.positions, vec![[1.0, 2.0, 3.0]]);
    assert_eq!(snapshot.masses, vec![2.0]);
    assert!(snapshot.metadata.is_empty());
}

#[test]
fn from_json_rejects_malformed_documents_with_a_reason() {
    assert!(parse_message(SimSnapshot::from_json("not json")).starts_with("snapshot JSON:"));
    assert!(parse_message(SimSnapshot::from_json("[1,2]")).contains("must be an object"));
    assert!(parse_message(SimSnapshot::from_json("{}")).contains("missing key 'version'"));

    let valid = |replace: &str, with: &str| {
        let base = "{\"version\":\"1.0\",\"sim_time\":1.0,\"n_bodies\":1,\
                    \"positions\":[[1.0,2.0,3.0]],\"velocities\":[[0.0,0.0,0.0]],\
                    \"masses\":[1.0],\"metadata\":{}}";
        assert!(base.contains(replace), "test setup: {replace}");
        base.replace(replace, with)
    };
    let message = parse_message(SimSnapshot::from_json(&valid("[1.0,2.0,3.0]", "[1.0,2.0]")));
    assert!(
        message.contains("'positions[0]' must be an array of three numbers"),
        "{message}"
    );
    let message = parse_message(SimSnapshot::from_json(&valid(
        "\"masses\":[1.0]",
        "\"masses\":[\"heavy\"]",
    )));
    assert!(message.contains("masses[0]"), "{message}");
    let message = parse_message(SimSnapshot::from_json(&valid(
        "\"n_bodies\":1",
        "\"n_bodies\":2",
    )));
    assert!(
        message.contains("'n_bodies' is 2 but there are 1 positions"),
        "{message}"
    );
    let message = parse_message(SimSnapshot::from_json(&valid(
        "\"metadata\":{}",
        "\"metadata\":{\"k\":3}",
    )));
    assert!(
        message.contains("metadata value for 'k' must be a string"),
        "{message}"
    );
    let message = parse_message(SimSnapshot::from_json(&valid(
        "\"sim_time\":1.0",
        "\"sim_time\":\"soon\"",
    )));
    assert!(message.contains("'sim_time' must be a number"), "{message}");
}

#[test]
fn from_json_checks_the_snapshot_version() {
    let with_version = |version: &str| {
        format!(
            "{{\"version\":\"{version}\",\"sim_time\":0.0,\"positions\":[],\"velocities\":[],\"masses\":[]}}"
        )
    };
    assert!(SimSnapshot::from_json(&with_version("1.0")).is_ok());
    assert!(SimSnapshot::from_json(&with_version("1.99")).is_ok());
    let message = parse_message(SimSnapshot::from_json(&with_version("2.0")));
    assert!(
        message.contains("unsupported snapshot version 2.0"),
        "{message}"
    );
    for bad in ["1", "one.zero", "1.", ".5", "1.0.0", "-1.0"] {
        let message = parse_message(SimSnapshot::from_json(&with_version(bad)));
        assert!(message.contains("malformed version"), "{bad}: {message}");
    }
}

#[test]
fn logger_json_keeps_small_values_and_escapes_custom_keys() {
    let mut entry = SimLogEntry::new(7, 1e-12, 2.5e-9, 1.5e-27, 3);
    entry.insert("we\"ird\\key", 6.02e23);
    entry.insert("nan", f64::NAN);
    let mut logger = SimLogger::new();
    logger.append(entry);
    let json = logger.to_json();
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let first = &value[0];
    assert_eq!(first["timestep"], 7);
    assert_eq!(
        first["sim_time"].as_f64().map(f64::to_bits),
        Some(1e-12_f64.to_bits())
    );
    assert_eq!(
        first["wall_time"].as_f64().map(f64::to_bits),
        Some(2.5e-9_f64.to_bits())
    );
    assert_eq!(
        first["energy"].as_f64().map(f64::to_bits),
        Some(1.5e-27_f64.to_bits())
    );
    assert_eq!(first["we\"ird\\key"].as_f64(), Some(6.02e23));
    assert_eq!(first["nan"], serde_json::json!("NaN"));
    assert_eq!(first["n_bodies"], 3);
}
