//! Round-trip tests for the JSON export surface of `oximedia-shots`.
//!
//! `Shot` and `Scene` both derive `Serialize + Deserialize + PartialEq`, so a
//! shot list exported via [`ShotListExporter::export_json`] must deserialize
//! back into an identical `Vec<Shot>`.  This pins the export format against
//! silent serialization regressions.
//!
//! Note on EDL: the EDL exporter is *export-only* (there is no EDL parser /
//! re-importer in this crate), so the "export then re-import" round-trip is
//! satisfied here via the JSON path, which is the lossless machine-readable
//! interchange format.  The CSV / FCP-XML / Resolve-marker exporters are
//! likewise one-way and intentionally not round-tripped.

use oximedia_core::types::{Rational, Timestamp};
use oximedia_shots::export::ShotListExporter;
use oximedia_shots::types::{
    CameraAngle, CameraMovement, CompositionAnalysis, CoverageType, MovementType, Scene, Shot,
    ShotType, TransitionType,
};

/// 30 fps timebase used throughout.
fn tb() -> Rational {
    Rational::new(1, 30)
}

/// Build three fully-populated, mutually-distinct shots covering a spread of
/// every enum field plus a non-empty movements list and non-trivial
/// composition scores — so the round-trip exercises the whole struct.
fn populated_shots() -> Vec<Shot> {
    let shot0 = Shot {
        id: 0,
        start: Timestamp::new(0, tb()),
        end: Timestamp::new(45, tb()),
        shot_type: ShotType::ExtremeCloseUp,
        angle: CameraAngle::Low,
        movements: vec![CameraMovement {
            movement_type: MovementType::PanRight,
            start: 0.0,
            end: 0.5,
            confidence: 0.875,
            speed: 0.25,
        }],
        composition: CompositionAnalysis {
            rule_of_thirds: 0.5,
            symmetry: 0.25,
            balance: 0.75,
            leading_lines: 0.125,
            depth: 0.625,
        },
        coverage: CoverageType::Single,
        confidence: 0.9375,
        transition: TransitionType::Cut,
    };

    let shot1 = Shot {
        id: 1,
        start: Timestamp::new(45, tb()),
        end: Timestamp::new(150, tb()),
        shot_type: ShotType::MediumShot,
        angle: CameraAngle::EyeLevel,
        movements: vec![
            CameraMovement {
                movement_type: MovementType::ZoomIn,
                start: 0.0,
                end: 0.25,
                confidence: 0.5,
                speed: 0.5,
            },
            CameraMovement {
                movement_type: MovementType::TiltDown,
                start: 0.25,
                end: 1.0,
                confidence: 0.75,
                speed: 0.125,
            },
        ],
        composition: CompositionAnalysis {
            rule_of_thirds: 0.8125,
            symmetry: 0.5,
            balance: 0.5,
            leading_lines: 0.5,
            depth: 0.375,
        },
        coverage: CoverageType::TwoShot,
        confidence: 0.5,
        transition: TransitionType::Dissolve,
    };

    let shot2 = Shot {
        id: 2,
        start: Timestamp::new(150, tb()),
        end: Timestamp::new(330, tb()),
        shot_type: ShotType::ExtremeLongShot,
        angle: CameraAngle::BirdsEye,
        movements: Vec::new(),
        composition: CompositionAnalysis {
            rule_of_thirds: 0.0,
            symmetry: 1.0,
            balance: 0.25,
            leading_lines: 0.75,
            depth: 1.0,
        },
        coverage: CoverageType::Master,
        confidence: 0.25,
        transition: TransitionType::FadeToBlack,
    };

    vec![shot0, shot1, shot2]
}

/// Test 13 (shots) — export 3 fully-populated shots to JSON and confirm the
/// deserialized `Vec<Shot>` is bit-for-bit equal to the originals.
#[test]
fn shots_json_roundtrip_is_identity() {
    let exporter = ShotListExporter::new();
    let shots = populated_shots();

    let json = exporter
        .export_json(&shots)
        .expect("export_json must succeed for valid shots");

    let reimported: Vec<Shot> = serde_json::from_str(&json)
        .expect("exported shot JSON must deserialize back into Vec<Shot>");

    assert_eq!(
        reimported, shots,
        "JSON round-trip must reproduce the exact same shot list"
    );
}

/// Test 13 (scenes) — the scenes JSON export round-trips to an identical
/// `Vec<Scene>` as well.
#[test]
fn scenes_json_roundtrip_is_identity() {
    let exporter = ShotListExporter::new();

    let scene0 = Scene {
        id: 0,
        start: Timestamp::new(0, tb()),
        end: Timestamp::new(150, tb()),
        shots: vec![0, 1],
        scene_type: String::from("Interior - Day"),
        confidence: 0.8125,
    };
    let scene1 = Scene {
        id: 1,
        start: Timestamp::new(150, tb()),
        end: Timestamp::new(330, tb()),
        shots: vec![2],
        scene_type: String::from("Exterior - Night"),
        confidence: 0.5,
    };
    let scenes = vec![scene0, scene1];

    let json = exporter
        .export_scenes_json(&scenes)
        .expect("export_scenes_json must succeed");

    let reimported: Vec<Scene> = serde_json::from_str(&json)
        .expect("exported scene JSON must deserialize back into Vec<Scene>");

    assert_eq!(
        reimported, scenes,
        "scene JSON round-trip must reproduce the exact same scene list"
    );
}
