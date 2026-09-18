//! Independent oracle for [`oxiephemeris_astro::declination::declination`].
//!
//! The library computes the declination from the scalar identity
//! `sin(delta) = cos(eps) sin(beta) + sin(eps) cos(beta) sin(lambda)`.
//! This oracle re-derives it the other way: it builds the ecliptic unit
//! vector, applies the frame rotation `R1(-eps)` from
//! `oxiephemeris_bodies::math` (the same primitive the frames pipeline
//! uses), and reads the declination off the rotated z-component. Two
//! different code paths must agree.

use oxiephemeris_astro::declination::declination;
use oxiephemeris_bodies::math::r1;

type TestResult = Result<(), String>;

/// Oracle declination via explicit ecliptic → equatorial rotation.
fn oracle_declination(lon: f64, lat: f64, eps: f64) -> f64 {
    // Ecliptic unit vector (r = 1).
    let ecl = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    // Frame rotation from the ecliptic to the equator is R1(-eps).
    let eq = r1(-eps).apply(ecl);
    // Declination is the elevation of the equatorial unit vector.
    let z = eq[2].clamp(-1.0, 1.0);
    z.asin()
}

const EPS_2000: f64 = 0.409_092_6; // ~23.4366 deg

#[test]
fn matches_rotation_oracle_on_a_grid() -> TestResult {
    let mut lon_deg: f64 = 0.0;
    while lon_deg < 360.0 {
        let mut lat_deg: f64 = -8.0;
        while lat_deg <= 8.0 {
            for eps_deg in [23.0_f64, 23.4366, 24.0] {
                let lon = lon_deg.to_radians();
                let lat = lat_deg.to_radians();
                let eps = eps_deg.to_radians();
                let got = declination(lon, lat, eps);
                let want = oracle_declination(lon, lat, eps);
                if (got - want).abs() > 1e-12 {
                    return Err(format!(
                        "lon={lon_deg} lat={lat_deg} eps={eps_deg}: {got} vs {want}"
                    ));
                }
            }
            lat_deg += 1.0;
        }
        lon_deg += 3.0;
    }
    Ok(())
}

#[test]
fn extreme_at_solstice_uses_2000_obliquity() -> TestResult {
    let north = declination(90.0_f64.to_radians(), 0.0, EPS_2000);
    if (north - EPS_2000).abs() > 1e-12 {
        return Err(format!("0 Cancer declination {north} != obliquity"));
    }
    Ok(())
}
