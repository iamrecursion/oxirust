// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The crate-wide [`Error`] as the public fallible APIs of the collision crate
//! produce it.
//!
//! The enum used to be defined and classified but never returned by anything:
//! callers had no API that could fail with it. This test exercises, from
//! outside the crate, the fallible entry points that now report their failures
//! through it.

use oxiphysics_collision::ray_casting::Heightfield;
use oxiphysics_collision::sap::GridBroadphase;
use oxiphysics_collision::{
    BvhAabb, DynamicBvh, Error, ErrorKind, GjkSphere, ProxyId, RecastBuilder, RecastConfig,
    RecastError, RecoveryHint, Severity, try_gjk_intersect,
};
use oxiphysics_core::math::Vec3;

fn unit_box(x: f64) -> BvhAabb {
    BvhAabb::new(Vec3::new(x - 0.5, -0.5, -0.5), Vec3::new(x + 0.5, 0.5, 0.5))
}

#[test]
fn a_stale_proxy_handle_is_an_invalid_handle_error() {
    let mut tree = DynamicBvh::new();
    let first = tree.insert(unit_box(0.0), 1);
    tree.remove(first);
    let second = tree.insert(unit_box(4.0), 2);
    // The second proxy reuses the slot of the first one; the stale handle must
    // not be able to touch it.
    assert_eq!(second.index, first.index);

    let removal = tree.try_remove(first);
    match removal {
        Err(Error::InvalidHandle { index }) => assert_eq!(index, first.index as usize),
        other => panic!("expected an invalid handle, got {other:?}"),
    }
    let update = tree.try_update(first, unit_box(9.0));
    assert!(
        matches!(update, Err(Error::InvalidHandle { .. })),
        "{update:?}"
    );
    assert_eq!(tree.n_leaves(), 1);
    assert_eq!(tree.query_aabb(&unit_box(4.0)), vec![2]);

    // The live handle still works, and a handle that was never issued does not.
    assert!(matches!(tree.try_update(second, unit_box(40.0)), Ok(true)));
    let never_issued = ProxyId {
        index: 500,
        generation: 0,
    };
    assert!(tree.try_remove(never_issued).is_err());
    assert!(tree.try_remove(second).is_ok());
}

#[test]
fn the_invalid_handle_error_is_triaged_as_a_skippable_bounds_error() {
    let mut tree = DynamicBvh::new();
    let id = tree.insert(unit_box(0.0), 1);
    tree.remove(id);
    let error = tree.try_remove(id).expect_err("the handle is stale");
    assert_eq!(error.kind(), ErrorKind::Bounds);
    assert_eq!(error.severity(), Severity::Error);
    assert_eq!(error.recovery_hint(), RecoveryHint::SkipPair);
    assert!(error.is_bounds() && error.is_severe());
}

#[test]
fn the_boolean_gjk_test_reports_decided_runs_as_ok() {
    let a = GjkSphere {
        center: [0.0; 3],
        radius: 1.0,
    };
    let near = GjkSphere {
        center: [1.5, 0.0, 0.0],
        radius: 1.0,
    };
    let far = GjkSphere {
        center: [4.0, 0.0, 0.0],
        radius: 1.0,
    };
    assert!(matches!(try_gjk_intersect(&a, &near), Ok(true)));
    assert!(matches!(try_gjk_intersect(&a, &far), Ok(false)));
}

#[test]
fn an_unusable_navmesh_configuration_names_the_offending_parameter() {
    let config = RecastConfig {
        cell_size: -0.3,
        ..RecastConfig::default()
    };
    match config.check_parameters() {
        Err(Error::InvalidParameter { name, value }) => {
            assert_eq!(name, "cell_size");
            assert_eq!(value, -0.3);
        }
        other => panic!("expected an invalid parameter, got {other:?}"),
    }
    assert!(RecastConfig::default().check_parameters().is_ok());

    // The pipeline reports the same configuration error through RecastError.
    let builder = RecastBuilder::new(config);
    assert!(matches!(
        builder.build(&[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        Err(RecastError::InvalidConfig(_))
    ));
}

#[test]
fn pipeline_errors_convert_into_the_crate_error() {
    let empty: Error = RecastError::EmptyInput.into();
    assert!(matches!(empty, Error::DegenerateShape { .. }));
    assert_eq!(empty.kind(), ErrorKind::Numerical);
    let failed: Error = RecastError::ContourBuildFailed("open contour".into()).into();
    assert_eq!(failed.to_string(), "contour build failed: open contour");
}

#[test]
fn the_error_enums_are_matched_with_a_wildcard_arm_outside_the_crate() {
    // Error and its companions are `#[non_exhaustive]`: new failure modes are not
    // a breaking change, so a downstream `match` always carries a wildcard arm.
    fn describe(error: &Error) -> &'static str {
        match error {
            Error::InvalidHandle { .. } => "handle",
            Error::InvalidParameter { .. } => "parameter",
            Error::GjkNoConverge { .. } => "convergence",
            _ => "other",
        }
    }
    assert_eq!(describe(&Error::InvalidHandle { index: 1 }), "handle");
    assert_eq!(
        describe(&Error::InvalidParameter {
            name: "radius",
            value: -1.0
        }),
        "parameter"
    );
    assert_eq!(describe(&Error::General("x".into())), "other");

    fn severity_label(severity: Severity) -> &'static str {
        match severity {
            Severity::Fatal => "fatal",
            _ => "recoverable",
        }
    }
    assert_eq!(severity_label(Severity::Fatal), "fatal");
    assert_eq!(severity_label(Severity::Warning), "recoverable");
}

/// The parameter an invalid-parameter error names, and the value it rejects.
fn rejected_parameter<T>(outcome: Result<T, Error>) -> (&'static str, f64) {
    match outcome {
        Err(Error::InvalidParameter { name, value }) => (name, value),
        Err(other) => panic!("expected an invalid parameter, got {other:?}"),
        Ok(_) => panic!("the parameter was accepted"),
    }
}

#[test]
fn a_grid_broadphase_rejects_cell_sizes_that_cannot_index_space() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (name, value) = rejected_parameter(GridBroadphase::try_new(bad));
        assert_eq!(name, "cell_size");
        assert!(value == bad || (value.is_nan() && bad.is_nan()), "{value}");
    }
    let grid = GridBroadphase::try_new(2.0).expect("a positive finite cell size");
    assert_eq!(grid.cell_size, 2.0);
    // A usable grid behaves like one made with `new`.
    let mut grid = grid;
    grid.insert(1, [0.0; 3], [1.0; 3]);
    grid.insert(2, [0.5; 3], [1.5; 3]);
    assert_eq!(grid.query_potential_pairs(), vec![(1, 2)]);
}

#[test]
fn a_heightfield_rejects_grids_it_cannot_sample() {
    // Fewer than two columns or rows: no cell to interpolate in.
    assert_eq!(
        rejected_parameter(Heightfield::try_new(vec![0.0; 4], 1, 4, 1.0)),
        ("cols", 1.0)
    );
    assert_eq!(
        rejected_parameter(Heightfield::try_new(Vec::new(), 4, 0, 1.0)),
        ("rows", 0.0)
    );
    // The number of heights must match the grid.
    assert_eq!(
        rejected_parameter(Heightfield::try_new(vec![0.0; 5], 2, 3, 1.0)),
        ("heights.len()", 5.0)
    );
    assert_eq!(
        rejected_parameter(Heightfield::try_new(vec![0.0; 4], usize::MAX, 2, 1.0)),
        ("heights.len()", 4.0)
    );
    // The cell size must be finite and positive.
    for bad in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let (name, _) = rejected_parameter(Heightfield::try_new(vec![0.0; 4], 2, 2, bad));
        assert_eq!(name, "cell_size");
    }

    // A valid grid samples like one made with `new`.
    let field = Heightfield::try_new(vec![0.0, 1.0, 2.0, 3.0], 2, 2, 1.0).expect("a 2 x 2 grid");
    assert_eq!((field.cols, field.rows), (2, 2));
    assert!((field.height_at(0.5, 0.5) - 1.5).abs() < 1e-12);
}
