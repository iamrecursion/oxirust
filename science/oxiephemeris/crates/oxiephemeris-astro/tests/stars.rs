//! Catalog-subset integrity + an end-to-end apparent place of Sirius.

use std::path::PathBuf;

use oxiephemeris_astro::stars::{all, by_hip, by_name};
use oxiephemeris_bodies::star::apparent_star;
use oxiephemeris_bodies::{Center, Frame, NutationModel, Options};
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_core::time::JulianDate;
use oxiephemeris_de::DeFile;

#[test]
fn table_integrity() {
    let mut count = 0;
    for star in all() {
        count += 1;
        assert!(star.vmag <= 3.0, "HIP {}: vmag {}", star.hip, star.vmag);
        assert!(
            (0.0..360.0).contains(&star.catalog.ra_deg),
            "HIP {} ra",
            star.hip
        );
        assert!(
            (-90.0..=90.0).contains(&star.catalog.dec_deg),
            "HIP {} dec",
            star.hip
        );
        assert!(star.catalog.parallax_mas.is_finite());
    }
    assert_eq!(count, 177, "generated subset size");
}

#[test]
fn named_stars_resolve_and_look_right() {
    // Every named entry must exist in the generated table.
    for name in [
        "Sirius",
        "Canopus",
        "Arcturus",
        "Vega",
        "Capella",
        "Rigel",
        "Procyon",
        "Achernar",
        "Betelgeuse",
        "Altair",
        "Aldebaran",
        "Antares",
        "Spica",
        "Pollux",
        "Fomalhaut",
        "Deneb",
        "Regulus",
        "Polaris",
    ] {
        let Some(star) = by_name(name) else {
            panic!("{name} missing from the table")
        };
        assert_eq!(star.name, Some(name));
    }
    // Case-insensitivity.
    let Some(sirius) = by_name("sIrIuS") else {
        panic!("case-insensitive lookup failed")
    };
    assert_eq!(sirius.hip, 32_349);
    // Identity spot checks against well-known values: any name/HIP
    // transposition would trip these.
    assert!(sirius.vmag < -1.0, "Sirius V = {}", sirius.vmag);
    assert!(
        (sirius.catalog.parallax_mas - 379.2).abs() < 5.0,
        "Sirius parallax {}",
        sirius.catalog.parallax_mas
    );
    let Some(vega) = by_name("Vega") else {
        panic!("Vega missing")
    };
    assert!(vega.vmag.abs() < 0.2, "Vega V = {}", vega.vmag);
    let Some(polaris) = by_hip(11_767) else {
        panic!("Polaris missing")
    };
    assert!(
        (polaris.catalog.parallax_mas - 7.56).abs() < 0.2,
        "Polaris parallax {}",
        polaris.catalog.parallax_mas
    );
}

#[test]
fn sirius_apparent_place_2026() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440");
    if !path.exists() {
        eprintln!("skipping: DE440 not present");
        return;
    }
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => panic!("read: {e}"),
    };
    let de = match DeFile::parse(&bytes) {
        Ok(d) => d,
        Err(e) => panic!("parse: {e}"),
    };
    let Some(sirius) = by_name("Sirius") else {
        panic!("Sirius missing")
    };
    let opts = Options::new(
        Center::Geocentric,
        Frame::TrueOfDate,
        true,
        true,
        true,
        false,
        NutationModel::Iau2000a,
    );
    let p = match apparent_star(
        &de,
        &sirius.catalog,
        JulianDate::from_f64(2_461_226.5),
        opts,
    ) {
        Ok(p) => p,
        Err(e) => panic!("apparent_star: {e}"),
    };
    // Sirius apparent place mid-2026: RA ~ 101.6°, Dec ~ -16.77°
    // (J2000: 101.287°/-16.716°; +26 yr of precession moves it ~+0.36°
    // in RA and ~-0.05° in Dec, proper motion ~-0.04° Dec/century).
    let (ra_deg, dec_deg) = (p.lon_rad * RAD2DEG, p.lat_rad * RAD2DEG);
    assert!((101.2..102.1).contains(&ra_deg), "Sirius RA {ra_deg}");
    assert!((-16.95..-16.60).contains(&dec_deg), "Sirius Dec {dec_deg}");
}
