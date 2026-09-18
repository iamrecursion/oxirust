// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the configuration check and of the bridge from [`RecastError`] to
//! the crate-wide [`crate::Error`].

use super::*;
use crate::{Error, ErrorKind};

/// The offending parameter of a configuration the check rejects.
fn rejected(config: &RecastConfig) -> (&'static str, f64) {
    match config.check_parameters() {
        Err(Error::InvalidParameter { name, value }) => (name, value),
        other => panic!("expected an invalid parameter, got {other:?}"),
    }
}

#[test]
fn the_default_configuration_is_usable() {
    let config = RecastConfig::default();
    assert!(config.validate().is_ok());
    assert!(config.check_parameters().is_ok());
}

#[test]
fn positive_parameters_reject_zero_negative_and_non_finite_values() {
    for (case, name) in [
        (
            RecastConfig {
                cell_size: 0.0,
                ..RecastConfig::default()
            },
            "cell_size",
        ),
        (
            RecastConfig {
                cell_height: -0.2,
                ..RecastConfig::default()
            },
            "cell_height",
        ),
        (
            RecastConfig {
                agent_height: f64::INFINITY,
                ..RecastConfig::default()
            },
            "agent_height",
        ),
    ] {
        let (rejected_name, value) = rejected(&case);
        assert_eq!(rejected_name, name);
        assert!(value <= 0.0 || value.is_infinite(), "{name}: {value}");
        assert!(matches!(
            case.validate(),
            Err(RecastError::InvalidConfig(_))
        ));
    }
    let nan = RecastConfig {
        cell_size: f64::NAN,
        ..RecastConfig::default()
    };
    let (name, value) = rejected(&nan);
    assert_eq!(name, "cell_size");
    assert!(value.is_nan());
}

#[test]
fn non_negative_parameters_reject_negative_values() {
    for (case, name, expected) in [
        (
            RecastConfig {
                agent_radius: -0.5,
                ..RecastConfig::default()
            },
            "agent_radius",
            -0.5,
        ),
        (
            RecastConfig {
                agent_max_climb: -1.0,
                ..RecastConfig::default()
            },
            "agent_max_climb",
            -1.0,
        ),
        (
            RecastConfig {
                edge_max_len: -3.0,
                ..RecastConfig::default()
            },
            "edge_max_len",
            -3.0,
        ),
        (
            RecastConfig {
                edge_max_error: -0.1,
                ..RecastConfig::default()
            },
            "edge_max_error",
            -0.1,
        ),
    ] {
        assert_eq!(rejected(&case), (name, expected));
    }
    // Zero is allowed for these.
    let zero = RecastConfig {
        agent_radius: 0.0,
        agent_max_climb: 0.0,
        edge_max_len: 0.0,
        edge_max_error: 0.0,
        ..RecastConfig::default()
    };
    assert!(zero.check_parameters().is_ok());
}

#[test]
fn slope_polygon_size_and_region_limits_are_checked() {
    let steep = RecastConfig {
        agent_max_slope: 90.0,
        ..RecastConfig::default()
    };
    assert_eq!(rejected(&steep), ("agent_max_slope", 90.0));

    for verts in [2_usize, 7] {
        let config = RecastConfig {
            max_verts_per_poly: verts,
            ..RecastConfig::default()
        };
        assert_eq!(rejected(&config), ("max_verts_per_poly", verts as f64));
    }

    let huge_region = RecastConfig {
        region_min_size: usize::MAX,
        ..RecastConfig::default()
    };
    assert_eq!(
        rejected(&huge_region),
        ("region_min_size", usize::MAX as f64)
    );
    let huge_merge = RecastConfig {
        region_merge_size: usize::MAX,
        ..RecastConfig::default()
    };
    assert_eq!(rejected(&huge_merge).0, "region_merge_size");
}

#[test]
fn derived_voxel_counts_are_named_by_their_expression() {
    let tall = RecastConfig {
        agent_height: 1.0e9,
        cell_height: 0.2,
        ..RecastConfig::default()
    };
    let (name, value) = rejected(&tall);
    assert_eq!(name, "agent_height / cell_height");
    assert!((value - 5.0e9).abs() < 1.0, "{value}");

    let wide = RecastConfig {
        agent_radius: 1.0e9,
        cell_size: 0.3,
        ..RecastConfig::default()
    };
    assert_eq!(rejected(&wide).0, "agent_radius / cell_size");
}

#[test]
fn the_first_unusable_parameter_wins_and_both_checks_agree() {
    // Both cell_size and agent_radius are unusable: cell_size comes first.
    let config = RecastConfig {
        cell_size: 0.0,
        agent_radius: -1.0,
        ..RecastConfig::default()
    };
    assert_eq!(rejected(&config).0, "cell_size");
    match config.validate() {
        Err(RecastError::InvalidConfig(message)) => {
            assert!(
                message.starts_with("cell_size must be finite and > 0"),
                "{message}"
            );
        }
        other => panic!("expected an invalid configuration, got {other:?}"),
    }
}

#[test]
fn recast_errors_convert_into_the_crate_error() {
    let empty: Error = RecastError::EmptyInput.into();
    match &empty {
        Error::DegenerateShape { detail } => assert!(detail.contains("empty"), "{detail}"),
        other => panic!("expected a degenerate shape, got {other:?}"),
    }
    assert_eq!(empty.kind(), ErrorKind::Numerical);

    let invalid: Error = RecastError::InvalidConfig("cell_size is zero".into()).into();
    match &invalid {
        Error::General(message) => {
            assert_eq!(message, "invalid recast config: cell_size is zero");
        }
        other => panic!("expected a general error, got {other:?}"),
    }

    for (stage, prefix) in [
        (
            RecastError::RasterizationFailed("grid too large".into()),
            "rasterization failed: grid too large",
        ),
        (
            RecastError::RegionBuildFailed("no spans".into()),
            "region build failed: no spans",
        ),
        (
            RecastError::ContourBuildFailed("open contour".into()),
            "contour build failed: open contour",
        ),
        (
            RecastError::PolyMeshBuildFailed("too many vertices".into()),
            "poly mesh build failed: too many vertices",
        ),
    ] {
        match Error::from(stage) {
            Error::General(message) => assert_eq!(message, prefix),
            other => panic!("expected a general error, got {other:?}"),
        }
    }
}

#[test]
fn the_builder_errors_propagate_through_the_question_mark() {
    fn build_empty() -> crate::Result<PolyMesh> {
        let mesh = RecastBuilder::with_default_config().build(&[])?;
        Ok(mesh)
    }
    assert!(matches!(build_empty(), Err(Error::DegenerateShape { .. })));

    fn build_with_bad_config() -> crate::Result<PolyMesh> {
        let builder = RecastBuilder::new(RecastConfig {
            cell_size: 0.0,
            ..RecastConfig::default()
        });
        let triangle = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]];
        Ok(builder.build(&triangle)?)
    }
    match build_with_bad_config() {
        Err(Error::General(message)) => assert!(message.contains("cell_size"), "{message}"),
        Err(other) => panic!("expected the configuration error, got {other:?}"),
        Ok(_) => panic!("a zero cell size must be rejected"),
    }
}
