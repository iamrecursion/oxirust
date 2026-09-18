//! `gen-nutation`: regenerates the committed full IAU `2000A_R06`
//! nutation coefficient tables under
//! `crates/oxiephemeris-bodies/src/frames/` from the IERS Conventions
//! (2010) electronic tables `data/iers/tab5.3a.txt` (nutation in
//! longitude) and `data/iers/tab5.3b.txt` (nutation in obliquity).
//!
//! Output layout (chosen so every generated file stays well below the
//! 2000-line project limit after `cargo fmt`):
//!
//! | file                        | contents                            |
//! |-----------------------------|-------------------------------------|
//! | `nutation_2000a_psi_ls.rs`  | tab5.3a luni-solar terms, j = 0, 1  |
//! | `nutation_2000a_psi_pl.rs`  | tab5.3a planetary terms, j = 0      |
//! | `nutation_2000a_eps_ls.rs`  | tab5.3b luni-solar terms, j = 0, 1  |
//! | `nutation_2000a_eps_pl.rs`  | tab5.3b planetary terms, j = 0      |
//!
//! The `j = 1` blocks of both tables contain only luni-solar terms; the
//! generator fails loudly if that layout assumption is ever violated.
//!
//! Amplitudes are emitted digit-for-digit (with underscore digit
//! grouping, which does not change the parsed value), so the `f64`
//! constants in the generated code are bit-identical to what a runtime
//! parse of the same table text produces — the oracle test
//! `nutation_oracle.rs` relies on this.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::iers_table::{parse_table, Table, Term};

/// Workspace root, located relative to this crate's manifest directory
/// (`xtask/` lives directly under the root).
fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask manifest directory has no parent".to_owned())
}

/// `-17206424.18` → `-17_206_424.18`: underscore groups of three in the
/// integer part, all decimal digits preserved verbatim (so the parsed
/// `f64` value is unchanged).
fn rust_f64_literal(raw: &str) -> String {
    let (sign, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", raw),
    };
    let (int_part, frac_part) = unsigned.split_once('.').unwrap_or((unsigned, "0"));
    let mut grouped = String::with_capacity(int_part.len() + int_part.len() / 3);
    for (i, ch) in int_part.chars().enumerate() {
        if i > 0 && (int_part.len() - i) % 3 == 0 {
            grouped.push('_');
        }
        grouped.push(ch);
    }
    format!("{sign}{grouped}.{frac_part}")
}

/// Common generated-file header (module doc + type import).
fn file_header(desc: &str, source_file: &str, elem_type: &str) -> String {
    format!(
        "//! {desc}\n\
         //!\n\
         //! **GENERATED FILE — DO NOT EDIT BY HAND.**\n\
         //! Regenerate with `cargo run -p xtask -- gen-nutation`, then\n\
         //! `cargo fmt --all`.\n\
         //!\n\
         //! Source (clean room): IERS Conventions (2010), TN36, electronic\n\
         //! table `data/iers/{source_file}` from the IERS Conventions Centre\n\
         //! (<https://iers-conventions.obspm.fr/content/chapter5/additional_info/>),\n\
         //! IAU `2000A_R06` expression, series form TN36 eq. (5.35).\n\
         //! Amplitudes are copied digit-for-digit from the table: `j = 0`\n\
         //! amplitudes in microarcseconds (µas), `j = 1` amplitudes in µas\n\
         //! per Julian century TT. In each tuple the first amplitude\n\
         //! multiplies `sin(ARG)` and the second multiplies `cos(ARG)`.\n\
         \n\
         use super::nutation::{elem_type};\n"
    )
}

/// Appends one table: a `pub(crate) const NAME_LEN: usize = N;` term
/// count (usable in const contexts, where the `static` array itself is
/// not) followed by `pub(crate) static NAME: [ElemType; NAME_LEN] =
/// [...];` (`static` rather than `const` to keep the multi-kilobyte
/// blobs out of every use site — `clippy::large_const_arrays`), emitting
/// the first `n_mult` multipliers of each term.
fn push_array(
    out: &mut String,
    doc: &str,
    name: &str,
    elem_type: &str,
    n_mult: usize,
    terms: &[&Term],
) -> Result<(), String> {
    let fmt_err = |e: std::fmt::Error| e.to_string();
    out.push('\n');
    writeln!(out, "/// Number of terms in [`{name}`].").map_err(fmt_err)?;
    writeln!(out, "pub(crate) const {name}_LEN: usize = {};", terms.len()).map_err(fmt_err)?;
    out.push('\n');
    for line in doc.lines() {
        writeln!(out, "/// {line}").map_err(fmt_err)?;
    }
    // The allow is needed because some amplitude values (e.g. 6.28 µas)
    // coincidentally resemble mathematical constants.
    out.push_str("#[allow(clippy::approx_constant)]\n");
    writeln!(
        out,
        "pub(crate) static {name}: [{elem_type}; {name}_LEN] = ["
    )
    .map_err(fmt_err)?;
    for term in terms {
        let mults = term.mult[..n_mult]
            .iter()
            .map(i8::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            out,
            "    ([{mults}], {}, {}),",
            rust_f64_literal(&term.sin_amp),
            rust_f64_literal(&term.cos_amp)
        )
        .map_err(fmt_err)?;
    }
    out.push_str("];\n");
    Ok(())
}

/// Splits a block into (luni-solar, planetary) preserving table order.
fn split_ls_pl(terms: &[Term]) -> (Vec<&Term>, Vec<&Term>) {
    terms.iter().partition(|t| t.is_luni_solar())
}

/// One generated file: file name plus rendered contents.
struct GeneratedFile {
    name: &'static str,
    contents: String,
}

/// Renders a luni-solar table file (`j = 0` and `j = 1` arrays).
fn render_ls_file(
    name: &'static str,
    desc: &str,
    source_file: &str,
    j0: (&str, &[&Term]),
    j1: (&str, &[&Term]),
) -> Result<GeneratedFile, String> {
    let mut contents = file_header(desc, source_file, "FullLsTerm");
    let (j0_name, j0_terms) = j0;
    let (j1_name, j1_terms) = j1;
    push_array(
        &mut contents,
        &format!("Luni-solar `j = 0` terms `({j0_name})`, µas."),
        &format!("{}_LS_J0", name_prefix(name)),
        "FullLsTerm",
        5,
        j0_terms,
    )?;
    push_array(
        &mut contents,
        &format!("Luni-solar `j = 1` terms `({j1_name})`, µas/cy."),
        &format!("{}_LS_J1", name_prefix(name)),
        "FullLsTerm",
        5,
        j1_terms,
    )?;
    Ok(GeneratedFile { name, contents })
}

/// Renders a planetary table file (`j = 0` array only).
fn render_pl_file(
    name: &'static str,
    desc: &str,
    source_file: &str,
    j0: (&str, &[&Term]),
) -> Result<GeneratedFile, String> {
    let mut contents = file_header(desc, source_file, "FullPlTerm");
    let (j0_name, j0_terms) = j0;
    push_array(
        &mut contents,
        &format!("Planetary `j = 0` terms `({j0_name})`, µas."),
        &format!("{}_PL_J0", name_prefix(name)),
        "FullPlTerm",
        14,
        j0_terms,
    )?;
    Ok(GeneratedFile { name, contents })
}

/// `"nutation_2000a_psi_ls.rs"` → `"PSI"` / `"..._eps_pl.rs"` → `"EPS"`.
fn name_prefix(file_name: &str) -> &'static str {
    if file_name.contains("_psi_") {
        "PSI"
    } else {
        "EPS"
    }
}

/// Renders the four generated table files from the two parsed tables.
fn render_files(tab_a: &Table, tab_b: &Table) -> Result<Vec<GeneratedFile>, String> {
    let (psi_ls_j0, psi_pl_j0) = split_ls_pl(&tab_a.j0);
    let (psi_ls_j1, psi_pl_j1) = split_ls_pl(&tab_a.j1);
    let (eps_ls_j0, eps_pl_j0) = split_ls_pl(&tab_b.j0);
    let (eps_ls_j1, eps_pl_j1) = split_ls_pl(&tab_b.j1);
    if !psi_pl_j1.is_empty() || !eps_pl_j1.is_empty() {
        return Err(format!(
            "layout assumption violated: {} (tab5.3a) / {} (tab5.3b) planetary terms in the j = 1 blocks",
            psi_pl_j1.len(),
            eps_pl_j1.len()
        ));
    }
    Ok(vec![
        render_ls_file(
            "nutation_2000a_psi_ls.rs",
            "Full IAU `2000A_R06` nutation in longitude: luni-solar terms.",
            "tab5.3a.txt",
            ("A_i, A''_i", &psi_ls_j0),
            ("A'_i, A'''_i", &psi_ls_j1),
        )?,
        render_pl_file(
            "nutation_2000a_psi_pl.rs",
            "Full IAU `2000A_R06` nutation in longitude: planetary terms.",
            "tab5.3a.txt",
            ("A_i, A''_i", &psi_pl_j0),
        )?,
        render_ls_file(
            "nutation_2000a_eps_ls.rs",
            "Full IAU `2000A_R06` nutation in obliquity: luni-solar terms.",
            "tab5.3b.txt",
            ("B''_i, B_i", &eps_ls_j0),
            ("B'''_i, B'_i", &eps_ls_j1),
        )?,
        render_pl_file(
            "nutation_2000a_eps_pl.rs",
            "Full IAU `2000A_R06` nutation in obliquity: planetary terms.",
            "tab5.3b.txt",
            ("B''_i, B_i", &eps_pl_j0),
        )?,
    ])
}

/// Entry point for `cargo run -p xtask -- gen-nutation`.
pub(crate) fn run() -> Result<(), String> {
    let root = workspace_root()?;
    let iers_dir = root.join("data").join("iers");
    let tab_a = parse_table(&iers_dir.join("tab5.3a.txt"))?;
    let tab_b = parse_table(&iers_dir.join("tab5.3b.txt"))?;
    println!(
        "parsed tab5.3a: {} + {} terms; tab5.3b: {} + {} terms",
        tab_a.j0.len(),
        tab_a.j1.len(),
        tab_b.j0.len(),
        tab_b.j1.len()
    );

    let out_dir = root
        .join("crates")
        .join("oxiephemeris-bodies")
        .join("src")
        .join("frames");
    let files = render_files(&tab_a, &tab_b)?;
    for file in &files {
        let path = out_dir.join(file.name);
        let max_width = file
            .contents
            .lines()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0);
        std::fs::write(&path, &file.contents)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        println!(
            "wrote {} ({} lines, max width {max_width})",
            path.display(),
            file.contents.lines().count()
        );
    }
    Ok(())
}
