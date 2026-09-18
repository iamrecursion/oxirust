// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0
//! Regression tests for `ExplosiveFragmentation`.
//!
//! The mean fragment mass used to be `0.01 t^2.5 / sqrt(rho)`, about
//! `3e-10 kg` for a 6 mm wall and 4.7 billion fragments for a 1.5 kg case.  It
//! is now Mott's `2 mu` with `mu = B^2 t^(5/3) d_i^(2/3) (1 + t / d_i)^2`.  The
//! Gurney constant used to be a hard-coded TNT value that ignored the
//! detonation velocity; it now follows `sqrt(2E) = D / 2.97`.

use oxiphysics_rigid::Error;
use oxiphysics_rigid::fracture::ExplosiveFragmentation;
use std::f64::consts::PI;

/// One pound in kilograms and one inch in metres, spelled out independently of
/// the library.
const LB_KG: f64 = 0.453_592_37;
const INCH_M: f64 = 0.0254;

/// Mott constant of TNT in Table 1 of Moreira et al. (2022), read as
/// `lb^(1/2) in^(-7/6)`.
const B_TNT_IMPERIAL: f64 = 0.0779;

fn rel_close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs())
}

/// Mean fragment mass (kg) from `2 mu`, evaluated entirely in pounds and
/// inches with `b` in `lb^(1/2) in^(-7/6)`.
fn imperial_mean_mass(b: f64, thickness_m: f64, diameter_m: f64) -> f64 {
    let (t, d) = (thickness_m / INCH_M, diameter_m / INCH_M);
    let mu_lb = b * b * t.powf(5.0 / 3.0) * d.powf(2.0 / 3.0) * (1.0 + t / d).powi(2);
    2.0 * mu_lb * LB_KG
}

/// SI value of `mott_b` for the preset: `0.0779 lb^(1/2) in^(-7/6)`.
fn tnt_b_si() -> f64 {
    B_TNT_IMPERIAL * LB_KG.sqrt() * INCH_M.powf(-7.0 / 6.0)
}

/// The preset with another wall thickness.
fn with_thickness(thickness: f64) -> ExplosiveFragmentation {
    ExplosiveFragmentation {
        wall_thickness: thickness,
        ..ExplosiveFragmentation::small_munition()
    }
}

// -- Mott fragment mass ------------------------------------------------------

#[test]
fn fragment_count_times_mean_mass_recovers_the_casing_mass() {
    let munition = ExplosiveFragmentation::small_munition();
    let mean = munition.mott_mean_mass();
    let count = munition.fragment_count();
    assert!(count > 1000, "{count} fragments");
    let total = count as f64 * mean;
    // Rounding the count to an integer is the only source of error.
    assert!(
        (total - munition.casing_mass).abs() <= 0.5 * mean * (1.0 + 1.0e-12),
        "{count} fragments of {mean} kg add up to {total} kg"
    );
    assert!((total - munition.casing_mass).abs() < 1.0e-3 * munition.casing_mass);
}

#[test]
fn fragment_count_times_mean_mass_holds_for_other_casings() {
    for (casing_mass, thickness, diameter, b) in [
        (6.9, 0.0165, 0.122, 3.0),
        (0.3, 0.002, 0.02, 2.7),
        (40.0, 0.02, 0.3, 3.4),
    ] {
        let model =
            ExplosiveFragmentation::new(2.0, casing_mass, 7800.0, thickness, diameter, 7000.0, b)
                .expect("valid parameters");
        let mean = model.mott_mean_mass();
        let total = model.fragment_count() as f64 * mean;
        assert!(
            (total - casing_mass).abs() <= 0.5 * mean * (1.0 + 1.0e-12),
            "casing {casing_mass} kg: fragments add up to {total} kg"
        );
    }
}

#[test]
fn mean_mass_rises_with_the_wall_thickness() {
    let mut previous_mean = 0.0;
    let mut previous_count = usize::MAX;
    for millimetres in 2..=14 {
        let munition = with_thickness(f64::from(millimetres) * 1.0e-3);
        let mean = munition.mott_mean_mass();
        let count = munition.fragment_count();
        assert!(
            mean > previous_mean,
            "{millimetres} mm: mean mass {mean} kg must exceed {previous_mean} kg"
        );
        assert!(
            count < previous_count,
            "{millimetres} mm: a thicker wall fragments into fewer pieces ({count} vs {previous_count})"
        );
        previous_mean = mean;
        previous_count = count;
    }
}

#[test]
fn mean_mass_matches_an_independent_evaluation_in_pounds_and_inches() {
    let munition = ExplosiveFragmentation::small_munition();
    let expected = imperial_mean_mass(B_TNT_IMPERIAL, 0.006, 0.044);
    assert!(
        rel_close(munition.mott_mean_mass(), expected, 1.0e-12),
        "{} vs {expected}",
        munition.mott_mean_mass()
    );
    // Another casing, with a constant given directly in SI.
    let model = ExplosiveFragmentation::new(1.0, 5.0, 7800.0, 0.01, 0.1, 7500.0, tnt_b_si())
        .expect("valid parameters");
    let expected = imperial_mean_mass(B_TNT_IMPERIAL, 0.01, 0.1);
    assert!(rel_close(model.mott_mean_mass(), expected, 1.0e-12));
}

#[test]
fn the_preset_stores_the_tnt_constant_in_si() {
    let munition = ExplosiveFragmentation::small_munition();
    assert!(
        rel_close(munition.mott_b, tnt_b_si(), 1.0e-12),
        "mott_b = {}",
        munition.mott_b
    );
    assert!((munition.mott_b - 3.8098).abs() < 1.0e-4);
}

#[test]
fn the_preset_gives_thousands_of_gram_sized_fragments() {
    // The fabricated formula gave 3e-10 kg and 4.7e9 fragments; reading the
    // tabulated constant as SI would give 4e-7 kg and 4e6 fragments.
    let munition = ExplosiveFragmentation::small_munition();
    let mean = munition.mott_mean_mass();
    let count = munition.fragment_count();
    assert!((5.0e-4..2.0e-3).contains(&mean), "mean mass {mean} kg");
    assert!((1000..3000).contains(&count), "{count} fragments");
    assert_eq!(count, 1621);
}

#[test]
fn mean_mass_scales_with_the_square_of_the_constant() {
    let base = ExplosiveFragmentation::small_munition();
    let doubled = ExplosiveFragmentation {
        mott_b: 2.0 * base.mott_b,
        ..base.clone()
    };
    assert!(rel_close(
        doubled.mott_mean_mass(),
        4.0 * base.mott_mean_mass(),
        1.0e-12
    ));
}

#[test]
fn geometric_scaling_follows_the_five_thirds_and_two_thirds_exponents() {
    let base = ExplosiveFragmentation::small_munition();
    let t = base.wall_thickness;
    let d = base.inner_diameter;
    let mean = base.mott_mean_mass();

    // Scaling every length by k leaves t / d alone, so only t^(5/3) d^(2/3)
    // changes: a factor k^(7/3).
    let scaled = ExplosiveFragmentation {
        wall_thickness: 2.0 * t,
        inner_diameter: 2.0 * d,
        ..base.clone()
    };
    assert!(rel_close(
        scaled.mott_mean_mass(),
        2.0_f64.powf(7.0 / 3.0) * mean,
        1.0e-12
    ));

    // A thicker wall at a fixed bore.
    let thicker = ExplosiveFragmentation {
        wall_thickness: 2.0 * t,
        ..base.clone()
    };
    let expected = 2.0_f64.powf(5.0 / 3.0) * ((1.0 + 2.0 * t / d) / (1.0 + t / d)).powi(2) * mean;
    assert!(rel_close(thicker.mott_mean_mass(), expected, 1.0e-12));

    // A wider bore at a fixed wall.
    let wider = ExplosiveFragmentation {
        inner_diameter: 2.0 * d,
        ..base
    };
    let expected = 2.0_f64.powf(2.0 / 3.0) * ((1.0 + t / (2.0 * d)) / (1.0 + t / d)).powi(2) * mean;
    assert!(rel_close(wider.mott_mean_mass(), expected, 1.0e-12));
}

#[test]
fn casing_density_does_not_enter_the_mott_mass() {
    let base = ExplosiveFragmentation::small_munition();
    let light = ExplosiveFragmentation {
        casing_density: 2700.0,
        ..base.clone()
    };
    assert_eq!(light.mott_mean_mass(), base.mott_mean_mass());
}

// -- Gurney velocity ---------------------------------------------------------

#[test]
fn gurney_velocity_follows_the_kamlet_finger_relation() {
    let munition = ExplosiveFragmentation::small_munition();
    // sqrt(2E) = D / 2.97 and M / C = 1.5 / 0.5.
    let expected = 6900.0 / 2.97 / (3.0_f64 + 0.5).sqrt();
    assert!(
        rel_close(munition.gurney_velocity(), expected, 1.0e-12),
        "{} vs {expected}",
        munition.gurney_velocity()
    );
    assert!((munition.gurney_velocity() - 1241.8).abs() < 0.1);
}

#[test]
fn gurney_velocity_is_proportional_to_the_detonation_velocity() {
    let base = ExplosiveFragmentation::small_munition();
    let reference = base.gurney_velocity();
    for factor in [0.5, 1.0, 1.25, 2.0] {
        let faster = ExplosiveFragmentation {
            det_velocity: factor * base.det_velocity,
            ..base.clone()
        };
        assert!(
            rel_close(faster.gurney_velocity(), factor * reference, 1.0e-12),
            "factor {factor}"
        );
    }
    let still = ExplosiveFragmentation {
        det_velocity: 0.0,
        ..base
    };
    assert_eq!(still.gurney_velocity(), 0.0);
}

#[test]
fn gurney_velocity_falls_with_the_metal_to_charge_ratio() {
    let base = ExplosiveFragmentation::small_munition();
    let mut previous = f64::INFINITY;
    for casing_mass in [0.5, 1.0, 1.5, 3.0, 6.0] {
        let velocity = ExplosiveFragmentation {
            casing_mass,
            ..base.clone()
        }
        .gurney_velocity();
        assert!(
            velocity < previous,
            "casing {casing_mass} kg: {velocity} m/s"
        );
        // Cylindrical Gurney equation with sqrt(2E) = D / 2.97.
        let expected = base.det_velocity / 2.97 / (casing_mass / base.explosive_mass + 0.5).sqrt();
        assert!(rel_close(velocity, expected, 1.0e-12));
        previous = velocity;
    }
}

// -- Geometry and units ------------------------------------------------------

#[test]
fn the_preset_geometry_is_self_consistent() {
    let munition = ExplosiveFragmentation::small_munition();
    let length = munition.casing_length();
    assert!((length - 0.204_04).abs() < 1.0e-4, "length {length} m");
    // The bore holds the charge at the density of cast TNT.
    let bore_volume = PI / 4.0 * munition.inner_diameter.powi(2) * length;
    let loading_density = munition.explosive_mass / bore_volume;
    assert!(
        (1550.0..1650.0).contains(&loading_density),
        "loading density {loading_density} kg/m^3"
    );
    // The wall volume times the density gives the casing mass back.
    let wall_volume = PI / 4.0
        * ((munition.inner_diameter + 2.0 * munition.wall_thickness).powi(2)
            - munition.inner_diameter.powi(2))
        * length;
    assert!(rel_close(
        wall_volume * munition.casing_density,
        munition.casing_mass,
        1.0e-12
    ));
}

#[test]
fn unit_conversion_of_the_mott_constant() {
    // lb^(1/2) in^(-7/6) -> kg^(1/2) m^(-7/6).
    let imperial = ExplosiveFragmentation::mott_constant_from_units(1.0, LB_KG, INCH_M);
    assert!(rel_close(imperial, 48.905_781, 1.0e-7), "{imperial}");
    // Grams and millimetres: sqrt(1e-3) * (1e-3)^(-7/6) = 100 exactly.
    let metric = ExplosiveFragmentation::mott_constant_from_units(0.05, 1.0e-3, 1.0e-3);
    assert!(rel_close(metric, 5.0, 1.0e-12), "{metric}");
    // SI is the identity.
    assert_eq!(
        ExplosiveFragmentation::mott_constant_from_units(3.25, 1.0, 1.0),
        3.25
    );
}

// -- Degenerate input --------------------------------------------------------

#[test]
fn degenerate_parameters_give_no_fragments_and_no_nan() {
    let base = ExplosiveFragmentation::small_munition();
    let broken = [
        ExplosiveFragmentation {
            wall_thickness: 0.0,
            ..base.clone()
        },
        ExplosiveFragmentation {
            wall_thickness: -0.006,
            ..base.clone()
        },
        ExplosiveFragmentation {
            inner_diameter: 0.0,
            ..base.clone()
        },
        ExplosiveFragmentation {
            inner_diameter: f64::INFINITY,
            ..base.clone()
        },
        ExplosiveFragmentation {
            mott_b: 0.0,
            ..base.clone()
        },
        ExplosiveFragmentation {
            mott_b: f64::NAN,
            ..base.clone()
        },
        ExplosiveFragmentation {
            casing_mass: -1.0,
            ..base.clone()
        },
    ];
    for (index, model) in broken.iter().enumerate() {
        assert!(
            !model.mott_mean_mass().is_nan(),
            "case {index}: mean mass is NaN"
        );
        assert_eq!(model.fragment_count(), 0, "case {index}");
        assert!(model.gurney_velocity().is_finite(), "case {index}");
        assert!(model.casualty_radius().is_finite(), "case {index}");
        assert!(model.casing_length().is_finite(), "case {index}");
    }
    assert_eq!(broken[0].mott_mean_mass(), 0.0);
    assert_eq!(broken[2].casing_length(), 0.0);
}

#[test]
fn the_constructor_validates_every_parameter() {
    let ok = [0.5, 1.5, 7800.0, 0.006, 0.044, 6900.0, 3.8];
    let build = |p: [f64; 7]| ExplosiveFragmentation::new(p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
    let model = build(ok).expect("valid parameters");
    assert_eq!(model.explosive_mass, 0.5);
    assert_eq!(model.casing_mass, 1.5);
    assert_eq!(model.casing_density, 7800.0);
    assert_eq!(model.wall_thickness, 0.006);
    assert_eq!(model.inner_diameter, 0.044);
    assert_eq!(model.det_velocity, 6900.0);
    assert_eq!(model.mott_b, 3.8);

    for slot in 0..7 {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut parameters = ok;
            parameters[slot] = bad;
            assert!(
                matches!(build(parameters), Err(Error::General(_))),
                "parameter {slot} = {bad} must be rejected"
            );
        }
    }
}
