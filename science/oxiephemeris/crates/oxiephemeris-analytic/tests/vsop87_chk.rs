//! Oracle test against the BDL's own check values (`vsop87.chk` of CDS
//! VI/81): full-series positions AND velocities for every VSOP87E body
//! at ten epochs spanning 1099–2000 CE.
//!
//! The committed tables are truncated, so the residual measured here
//! *is* the truncation error (plus f64 roundoff): the gates document
//! it. The file is fetched by the data scripts and not committed; the
//! test skips when it is absent.

use std::path::PathBuf;

use oxiephemeris_analytic::{state_ecliptic_au, VsopBody};

/// One check case: body, JD (TDB), position (au), velocity (au/day).
struct Case {
    body: VsopBody,
    jd: f64,
    pos: [f64; 3],
    vel: [f64; 3],
}

fn body_of(name: &str) -> Option<VsopBody> {
    Some(match name {
        "SUN" => VsopBody::Sun,
        "MERCURY" => VsopBody::Mercury,
        "VENUS" => VsopBody::Venus,
        "EARTH" => VsopBody::Earth,
        "MARS" => VsopBody::Mars,
        "JUPITER" => VsopBody::Jupiter,
        "SATURN" => VsopBody::Saturn,
        "URANUS" => VsopBody::Uranus,
        "NEPTUNE" => VsopBody::Neptune,
        _ => return None,
    })
}

/// Extracts the three values of a `x .. y .. z ..` line
/// (tokens 1, 4, 7 after whitespace splitting).
fn triple(line: &str) -> Result<[f64; 3], String> {
    let tok: Vec<&str> = line.split_whitespace().collect();
    if tok.len() < 8 {
        return Err(format!("short value line: {line:?}"));
    }
    let mut out = [0.0; 3];
    for (i, &t) in [tok[1], tok[4], tok[7]].iter().enumerate() {
        out[i] = t.parse().map_err(|e| format!("bad float {t:?}: {e}"))?;
    }
    Ok(out)
}

fn parse_chk(text: &str) -> Result<Vec<Case>, String> {
    let mut cases = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let tok: Vec<&str> = line.split_whitespace().collect();
        if tok.first() != Some(&"VSOP87E") {
            continue;
        }
        let Some(body) = body_of(tok.get(1).unwrap_or(&"")) else {
            return Err(format!("unknown body in {line:?}"));
        };
        let jd: f64 = tok
            .get(2)
            .and_then(|t| t.strip_prefix("JD"))
            .ok_or_else(|| format!("missing JD in {line:?}"))?
            .parse()
            .map_err(|e| format!("bad JD in {line:?}: {e}"))?;
        let pos = triple(lines.next().ok_or("missing position line")?)?;
        let vel = triple(lines.next().ok_or("missing velocity line")?)?;
        cases.push(Case { body, jd, pos, vel });
    }
    Ok(cases)
}

#[test]
fn vsop87e_check_values() -> Result<(), String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/vsop87/vsop87.chk");
    let Ok(text) = std::fs::read_to_string(&path) else {
        println!("skipping: {} not present", path.display());
        return Ok(());
    };
    let cases = parse_chk(&text)?;
    assert_eq!(cases.len(), 90, "expected 9 bodies x 10 epochs");

    // Truncation-error gates, au / au·day⁻¹, ~3x the measured maxima
    // (worst case over all bodies/epochs: 3.64e-7 au and 9.34e-9
    // au/day, at the 1099 CE end where the dropped Poisson terms are
    // largest — see the printout).
    let mut worst_pos = 0.0f64;
    let mut worst_vel = 0.0f64;
    for case in &cases {
        let state = state_ecliptic_au(case.body, (case.jd, 0.0));
        for i in 0..3 {
            let dp = (state[i] - case.pos[i]).abs();
            let dv = (state[i + 3] - case.vel[i]).abs();
            worst_pos = worst_pos.max(dp);
            worst_vel = worst_vel.max(dv);
            assert!(
                dp < 1e-6,
                "{:?} JD{}: coord {i} off by {dp} au",
                case.body,
                case.jd
            );
            assert!(
                dv < 2e-8,
                "{:?} JD{}: rate {i} off by {dv} au/day",
                case.body,
                case.jd
            );
        }
    }
    println!("vsop87.chk: 90 cases, worst position {worst_pos:.3e} au, worst rate {worst_vel:.3e} au/day");
    Ok(())
}
