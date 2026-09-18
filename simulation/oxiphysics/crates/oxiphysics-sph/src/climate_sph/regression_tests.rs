// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

// Regression tests of the ocean carbonate chemistry (`CarbonateConstants`,
// `CarbonCycleOcean::ocean_pco2`, `ocean_ph`). `ocean_pco2` used to ignore the
// alkalinity, assume a fixed pH of 8.1 and a made-up `K₁K₂` factor, which gave
// ≈ 0 µatm for any DIC (and, together with the old K₀, 1e50 mol/m²/yr fluxes).
//
// The reference values below were generated with PyCO2SYS 1.8.3.4
// (`pyco2.sys(par1=TA, par2=DIC, par1_type=1, par2_type=2, salinity=S,
// temperature=T, opt_k_carbonic=10, opt_k_bisulfate=1, opt_total_borate=1,
// opt_pH_scale=1, opt_k_fluoride=1, total_phosphate=0, total_silicate=0)`):
// Lueker et al. (2000) K₁/K₂, Dickson (1990) K_B and K_HSO4, Uppström (1974)
// total boron, Dickson & Riley (1979) K_HF, total pH scale, no nutrients.
// They live in their own file so that `climate_sph.rs` stays small.

use super::*;

/// `(T [°C], S, K₀, K₁, K₂, K_B, K_w)` on the total scale (PyCO2SYS).
const CONSTANTS: [(f64, f64, f64, f64, f64, f64, f64); 6] = [
    (
        25.0,
        35.0,
        2.839188180402e-02,
        1.421828137139e-06,
        1.081554747221e-09,
        2.526572990247e-09,
        6.019824161803e-14,
    ),
    (
        15.0,
        35.0,
        3.745922299903e-02,
        1.148238197682e-06,
        7.430176512679e-10,
        1.920801661345e-09,
        2.368884006591e-14,
    ),
    (
        10.0,
        34.0,
        4.413251901157e-02,
        1.007248908526e-06,
        5.977726745780e-10,
        1.642784438460e-09,
        1.416936133310e-14,
    ),
    (
        0.0,
        35.0,
        6.287012427630e-02,
        7.671335788752e-07,
        4.089193832773e-10,
        1.221675443237e-09,
        4.934549042835e-15,
    ),
    (
        30.0,
        35.0,
        2.517134600843e-02,
        1.559265749056e-06,
        1.296766967891e-09,
        2.883876164754e-09,
        9.309908499039e-14,
    ),
    (
        20.0,
        5.0,
        3.811722788461e-02,
        7.947468643073e-07,
        3.579259987997e-10,
        1.064221542108e-09,
        1.496676261479e-14,
    ),
];

/// `(TA [µmol/kg], DIC [µmol/kg], T [°C], S, pH_total, fCO₂ [µatm])` (PyCO2SYS).
const SOLUTIONS: [(f64, f64, f64, f64, f64, f64); 10] = [
    (2300.0, 2000.0, 25.0, 35.0, 8.045886181, 395.6920414),
    (2300.0, 2100.0, 15.0, 35.0, 8.007081407, 443.0790815),
    (2350.0, 1900.0, 10.0, 34.0, 8.513563495, 109.3481934),
    (2300.0, 2000.0, 0.0, 35.0, 8.441384242, 134.2777219),
    (2200.0, 2000.0, 30.0, 35.0, 7.797835485, 743.5204799),
    (2400.0, 1800.0, 20.0, 5.0, 9.089256318, 33.5828643),
    (2300.0, 2300.0, 25.0, 35.0, 7.326192749, 2545.5729720),
    (2300.0, 1700.0, 25.0, 35.0, 8.467154167, 108.8536304),
    (2000.0, 2200.0, 25.0, 35.0, 6.798277811, 7751.5143836),
    (500.0, 2000.0, 25.0, 35.0, 5.374454614, 52693.7471948),
];

fn relative_difference(value: f64, reference: f64) -> f64 {
    ((value - reference) / reference).abs()
}

#[test]
fn equilibrium_constants_match_the_co2sys_reference() {
    for (t, s, k0, k1, k2, kb, kw) in CONSTANTS {
        let ocean = CarbonCycleOcean::new(400.0, t, s, 2300.0, 2000.0);
        let c = ocean.carbonate_constants();
        assert_eq!(c, CarbonateConstants::at(t, s));
        assert!(
            relative_difference(ocean.co2_solubility(), k0) < 1.0e-9,
            "K0 at {t} °C"
        );
        assert!(
            relative_difference(c.k1, k1) < 1.0e-9,
            "K1 at {t} °C, S = {s}: {}",
            c.k1
        );
        assert!(
            relative_difference(c.k2, k2) < 1.0e-9,
            "K2 at {t} °C, S = {s}: {}",
            c.k2
        );
        assert!(
            relative_difference(c.kb, kb) < 1.0e-9,
            "KB at {t} °C, S = {s}: {}",
            c.kb
        );
        assert!(
            relative_difference(c.kw, kw) < 1.0e-9,
            "Kw at {t} °C, S = {s}: {}",
            c.kw
        );
        assert!(
            relative_difference(c.boron_total, 0.0004157 * s / 35.0) < 1.0e-12,
            "B_T at S = {s}"
        );
    }
    // Spot checks of the well known values at 25 °C, S = 35 (total scale):
    // pK1 = 5.847, pK2 = 8.966, pK_B = 8.598, pK_w = 13.22.
    let c = CarbonateConstants::at(25.0, 35.0);
    assert!((-c.k1.log10() - 5.847).abs() < 1.0e-3);
    assert!((-c.k2.log10() - 8.966).abs() < 1.0e-3);
    assert!((-c.kb.log10() - 8.598).abs() < 1.0e-3);
    assert!((-c.kw.log10() - 13.22).abs() < 1.0e-2);
}

#[test]
fn ph_and_pco2_agree_with_co2sys_for_many_states() {
    for (ta, dic, t, s, ph_ref, fco2_ref) in SOLUTIONS {
        let ocean = CarbonCycleOcean::new(400.0, t, s, ta, dic);
        let ph = ocean.ocean_ph();
        let pco2 = ocean.ocean_pco2();
        // The two most acidic states are far outside the ocean (the omitted
        // hydrogen fluoride term then reaches ~1e-4 of the alkalinity).
        let (ph_tol, pco2_tol) = if ph_ref < 6.0 {
            (1.0e-3, 2.0e-3)
        } else {
            (2.0e-5, 2.0e-5)
        };
        assert!(
            (ph - ph_ref).abs() < ph_tol,
            "TA {ta}, DIC {dic}, {t} °C, S {s}: pH {ph}, reference {ph_ref}"
        );
        assert!(
            relative_difference(pco2, fco2_ref) < pco2_tol,
            "TA {ta}, DIC {dic}, {t} °C, S {s}: pCO2 {pco2}, reference {fco2_ref}"
        );
        // [H+] is consistent with the pH.
        assert!(
            relative_difference(ocean.hydrogen_ion_concentration(), 10_f64.powf(-ph)) < 1.0e-14
        );
    }
}

#[test]
fn surface_ocean_pco2_is_in_the_observed_range() {
    // TA = 2300, DIC = 2000 µmol/kg at 25 °C, S = 35: typical warm surface
    // water, ≈ 396 µatm (the former formula gave ≈ 0).
    let ocean = CarbonCycleOcean::new(400.0, 25.0, 35.0, 2300.0, 2000.0);
    let pco2 = ocean.ocean_pco2();
    assert!((300.0..=450.0).contains(&pco2), "pCO2 = {pco2}");
    assert!((pco2 - 395.69).abs() < 0.05, "pCO2 = {pco2}");
    assert!((ocean.ocean_ph() - 8.0459).abs() < 1.0e-3);
}

#[test]
fn solved_hydrogen_ion_closes_the_alkalinity_balance() {
    for (ta, dic, t, s, _, _) in SOLUTIONS {
        let ocean = CarbonCycleOcean::new(400.0, t, s, ta, dic);
        let constants = ocean.carbonate_constants();
        let h = ocean.hydrogen_ion_concentration();
        let residual = constants.alkalinity(h, dic * 1.0e-6) - ta * 1.0e-6;
        assert!(
            residual.abs() < 1.0e-12 * (1.0 + ta * 1.0e-6),
            "TA {ta}, DIC {dic}: alkalinity residual {residual:e} mol/kg"
        );
    }
}

#[test]
fn alkalinity_increases_strictly_with_ph() {
    // TA(H) decreases with [H+], i.e. increases with pH: the bisection on
    // pH ∈ [0, 14] always brackets the root.
    for (t, s) in [(0.0, 35.0), (15.0, 35.0), (30.0, 5.0)] {
        let c = CarbonateConstants::at(t, s);
        let mut previous = f64::NEG_INFINITY;
        for k in 0..=140 {
            let h = 10_f64.powf(-(k as f64) * 0.1);
            let alkalinity = c.alkalinity(h, 2.0e-3);
            assert!(
                alkalinity > previous,
                "{t} °C, S {s}: not increasing at pH {}",
                k as f64 * 0.1
            );
            previous = alkalinity;
        }
    }
}

#[test]
fn pco2_responds_to_alkalinity_dic_and_temperature_like_seawater() {
    let base = CarbonCycleOcean::new(400.0, 15.0, 35.0, 2300.0, 2000.0);
    let pco2 = base.ocean_pco2();

    // More dissolved carbon, same alkalinity: pCO2 rises.
    let mut more_dic = base.clone();
    more_dic.dic = 2050.0;
    assert!(more_dic.ocean_pco2() > pco2);
    // More alkalinity, same carbon: the buffer takes up CO2, pCO2 falls.
    let mut more_alk = base.clone();
    more_alk.alkalinity = 2350.0;
    assert!(more_alk.ocean_pco2() < pco2);
    // Warmer water at fixed TA and DIC: pCO2 rises (≈ 4 %/K).
    let mut warmer = base.clone();
    warmer.sst = 16.0;
    let ratio = warmer.ocean_pco2() / pco2;
    assert!(
        (1.03..1.05).contains(&ratio),
        "d ln pCO2/dT = {}",
        ratio.ln()
    );

    // Revelle (buffer) factor (dpCO2/pCO2)/(dDIC/DIC) at fixed TA: 9.6 here
    // (PyCO2SYS: 9.5965), not the factor 2 of the former pCO2 ∝ DIC².
    let reference = CarbonCycleOcean::new(400.0, 25.0, 35.0, 2300.0, 2000.0);
    let delta = 1.0;
    let mut up = reference.clone();
    up.dic += delta;
    let mut down = reference.clone();
    down.dic -= delta;
    let revelle = (up.ocean_pco2().ln() - down.ocean_pco2().ln()) / (up.dic.ln() - down.dic.ln());
    assert!((revelle - 9.5965).abs() < 0.01, "Revelle factor {revelle}");
}

#[test]
fn zero_dic_has_no_pco2_and_unphysical_inputs_give_nan() {
    let mut ocean = CarbonCycleOcean::new(400.0, 15.0, 35.0, 2300.0, 0.0);
    assert_eq!(ocean.ocean_pco2(), 0.0);
    // Negative or zero alkalinity is a valid (acidic) state with a huge pCO2.
    ocean.dic = 2000.0;
    ocean.alkalinity = 0.0;
    let acidic = ocean.ocean_pco2();
    assert!(acidic.is_finite() && acidic > 10_000.0, "pCO2 = {acidic}");
    assert!(ocean.ocean_ph() < 5.0);

    for (alkalinity, dic, sst, salinity) in [
        (f64::NAN, 2000.0, 15.0, 35.0),
        (2300.0, f64::NAN, 15.0, 35.0),
        (2300.0, -5.0, 15.0, 35.0),
        (2300.0, 2000.0, 15.0, -1.0),
        (2300.0, 2000.0, -300.0, 35.0),
        (f64::INFINITY, 2000.0, 15.0, 35.0),
    ] {
        let ocean = CarbonCycleOcean::new(400.0, sst, salinity, alkalinity, dic);
        assert!(
            ocean.ocean_pco2().is_nan(),
            "TA {alkalinity}, DIC {dic}, {sst} °C, S {salinity}"
        );
        assert!(ocean.ocean_ph().is_nan());
    }
}

// ---------------------------------------------------------------------------
// Air-sea flux with the real chemistry
// ---------------------------------------------------------------------------

/// Atmospheric CO2 above the ocean pCO2 drives uptake, below it outgassing.
#[test]
fn air_sea_flux_follows_the_sign_of_the_pco2_difference() {
    // 15 °C: ocean pCO2 ≈ 330 µatm for TA 2300 / DIC 2000.
    let mut ocean = CarbonCycleOcean::new(800.0, 15.0, 35.0, 2300.0, 2000.0);
    let pco2_ocean = ocean.ocean_pco2();
    assert!(pco2_ocean < 400.0, "ocean pCO2 {pco2_ocean}");
    let uptake = ocean.air_sea_flux(8.0);
    assert!(
        uptake > 0.0,
        "atmosphere 800 µatm above ocean {pco2_ocean}: flux {uptake}"
    );

    // Supersaturated water (DIC 2200): outgassing into a 400 µatm atmosphere.
    ocean.pco2_atm = 400.0;
    ocean.dic = 2200.0;
    let pco2_super = ocean.ocean_pco2();
    assert!(pco2_super > 400.0, "ocean pCO2 {pco2_super}");
    let outgassing = ocean.air_sea_flux(8.0);
    assert!(outgassing < 0.0, "flux {outgassing}");

    // Equilibrium: no flux.
    ocean.pco2_atm = ocean.ocean_pco2();
    assert!(ocean.air_sea_flux(8.0).abs() < 1.0e-12);
}

/// The flux is the gas-exchange arithmetic times the chemistry's pCO2
/// difference, at a realistic magnitude (a few mol/m²/yr for ΔpCO2 ≈ 40 µatm).
#[test]
fn air_sea_flux_has_a_realistic_magnitude() {
    let mut ocean = CarbonCycleOcean::new(0.0, 15.0, 35.0, 2300.0, 2000.0);
    ocean.pco2_atm = ocean.ocean_pco2() + 40.0;
    let flux = ocean.air_sea_flux(8.0);
    assert!((flux - 2.163).abs() < 0.005, "flux {flux}");
}
