//! `OxiEphemeris` workspace tasks (`cargo xtask` pattern): one-time and
//! repeatable code generation. No external dependencies; std only.
//!
//! Subcommands:
//!
//! - `gen-nutation` — regenerate the committed full IAU `2000A_R06`
//!   nutation coefficient tables in
//!   `crates/oxiephemeris-bodies/src/frames/` from the IERS Conventions
//!   (2010) electronic tables `data/iers/tab5.3a.txt` / `tab5.3b.txt`.
//! - `gen-stars` — regenerate the committed Hipparcos bright-star table
//!   in `crates/oxiephemeris-astro/src/stars_table.rs` from
//!   `data/hipparcos/hip_main.dat`.
//! - `gen-vsop87` — regenerate the committed truncated VSOP87E planetary
//!   tables in `crates/oxiephemeris-analytic/src/vsop87_tables/` from
//!   `data/vsop87/VSOP87E.*`.

mod gen_elp;
mod gen_nutation;
mod gen_stars;
mod gen_vsop87;
mod iers_table;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gen-nutation") => match gen_nutation::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("gen-nutation: {err}");
                ExitCode::FAILURE
            }
        },
        Some("gen-stars") => match gen_stars::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("gen-stars: {err}");
                ExitCode::FAILURE
            }
        },
        Some("gen-vsop87") => match gen_vsop87::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("gen-vsop87: {err}");
                ExitCode::FAILURE
            }
        },
        Some("gen-elp") => match gen_elp::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("gen-elp: {err}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            eprintln!("xtask: unknown subcommand {other:?}");
            eprintln!("usage: cargo run -p xtask -- <gen-nutation|gen-stars|gen-vsop87|gen-elp>");
            ExitCode::FAILURE
        }
        None => {
            eprintln!("usage: cargo run -p xtask -- <gen-nutation|gen-stars|gen-vsop87|gen-elp>");
            ExitCode::FAILURE
        }
    }
}
