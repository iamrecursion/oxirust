//! Per-subcommand `--help` smoke tests.
//!
//! Every top-level [`Commands`](../../src/commands.rs) variant should produce
//! non-empty `--help` output with exit code `0`. This catches subcommand-wiring
//! regressions when new commands are added or renamed (e.g. clap `name = "..."`
//! attribute typos).
//!
//! The list below mirrors the order of variants in
//! `oximedia-cli/src/commands.rs::Commands` (90 variants as of 0.1.7).
//!
//! Naming convention:
//! - Test fn name: `help_<snake_case_subcommand>` (mapped from kebab-case).
//! - CLI arg: actual kebab-case name (matches clap `#[command(name = "...")]`
//!   when present, else clap's default kebab-casing of the variant name).

use assert_cmd::Command;

/// Run `oximedia <subcmd> --help` and assert success + non-trivial output.
fn run_help(subcmd: &str) {
    let assert = Command::cargo_bin("oximedia")
        .expect("cargo bin oximedia must be built")
        .arg(subcmd)
        .arg("--help")
        .assert();
    let out = assert.success().get_output().stdout.clone();
    let s = String::from_utf8_lossy(&out);
    assert!(
        s.len() > 50,
        "{subcmd} --help unexpectedly short: {} bytes",
        s.len()
    );
    assert!(
        s.contains("Usage") || s.contains("USAGE"),
        "{subcmd} --help missing usage section; got:\n{s}"
    );
}

// ---------------------------------------------------------------------------
// Simple (no-subcommand) variants
// ---------------------------------------------------------------------------

#[test]
fn help_probe() {
    run_help("probe");
}

#[test]
fn help_info() {
    run_help("info");
}

#[test]
fn help_version() {
    run_help("version");
}

#[test]
fn help_transcode() {
    run_help("transcode");
}

#[test]
fn help_extract() {
    run_help("extract");
}

#[test]
fn help_batch() {
    run_help("batch");
}

#[test]
fn help_concat() {
    run_help("concat");
}

#[test]
fn help_thumbnail() {
    run_help("thumbnail");
}

#[test]
fn help_sprite() {
    run_help("sprite");
}

#[test]
fn help_metadata() {
    run_help("metadata");
}

#[test]
fn help_benchmark() {
    run_help("benchmark");
}

#[test]
fn help_validate() {
    run_help("validate");
}

#[test]
fn help_analyze() {
    run_help("analyze");
}

#[test]
fn help_denoise() {
    run_help("denoise");
}

#[test]
fn help_stabilize() {
    run_help("stabilize");
}

#[test]
fn help_package() {
    run_help("package");
}

#[test]
fn help_forensics() {
    run_help("forensics");
}

#[test]
fn help_ffcompat() {
    run_help("ffcompat");
}

#[test]
fn help_ff_alias() {
    // `ff` is an explicit alias for `ffcompat`.
    run_help("ff");
}

#[test]
fn help_tui() {
    run_help("tui");
}

#[test]
fn help_completions() {
    run_help("completions");
}

#[test]
fn help_man_page() {
    run_help("man-page");
}

#[test]
fn help_doctor() {
    run_help("doctor");
}

#[test]
fn help_convert_alias() {
    // `convert` is an explicit alias for `transcode`.
    run_help("convert");
}

// ---------------------------------------------------------------------------
// Subcommand-wrapping variants
// ---------------------------------------------------------------------------

#[test]
fn help_scene() {
    run_help("scene");
}

#[test]
fn help_scopes() {
    run_help("scopes");
}

#[test]
fn help_audio() {
    run_help("audio");
}

#[test]
fn help_subtitle() {
    run_help("subtitle");
}

#[test]
fn help_filter() {
    run_help("filter");
}

#[test]
fn help_lut() {
    run_help("lut");
}

#[test]
fn help_edl() {
    run_help("edl");
}

#[test]
fn help_monitor() {
    run_help("monitor");
}

#[test]
fn help_restore() {
    run_help("restore");
}

#[test]
fn help_captions() {
    run_help("captions");
}

#[test]
fn help_stream() {
    run_help("stream");
}

#[test]
fn help_search() {
    run_help("search");
}

#[test]
fn help_timecode() {
    run_help("timecode");
}

#[test]
fn help_repair() {
    run_help("repair");
}

#[test]
fn help_color() {
    run_help("color");
}

#[test]
fn help_playlist() {
    run_help("playlist");
}

#[test]
fn help_conform() {
    run_help("conform");
}

#[test]
fn help_archive() {
    run_help("archive");
}

#[test]
fn help_watermark() {
    run_help("watermark");
}

#[test]
fn help_image() {
    run_help("image");
}

#[test]
fn help_graphics() {
    run_help("graphics");
}

#[test]
fn help_multicam() {
    run_help("multicam");
}

#[test]
fn help_timeline() {
    run_help("timeline");
}

#[test]
fn help_vfx() {
    run_help("vfx");
}

#[test]
fn help_optimize() {
    run_help("optimize");
}

#[test]
fn help_preset() {
    run_help("preset");
}

#[test]
fn help_mixer() {
    run_help("mixer");
}

#[test]
fn help_audiopost() {
    run_help("audiopost");
}

#[test]
fn help_distributed() {
    run_help("distributed");
}

#[test]
fn help_farm() {
    run_help("farm");
}

#[test]
fn help_ndi() {
    run_help("ndi");
}

#[test]
fn help_videoip() {
    run_help("videoip");
}

#[test]
fn help_gaming() {
    run_help("gaming");
}

#[test]
fn help_mam() {
    run_help("mam");
}

#[test]
fn help_cloud() {
    run_help("cloud");
}

#[test]
fn help_plugin() {
    run_help("plugin");
}

#[test]
fn help_mir() {
    run_help("mir");
}

#[test]
fn help_qc() {
    run_help("qc");
}

#[test]
fn help_imf() {
    run_help("imf");
}

#[test]
fn help_aaf() {
    run_help("aaf");
}

#[test]
fn help_playout() {
    run_help("playout");
}

#[test]
fn help_switcher() {
    run_help("switcher");
}

#[test]
fn help_workflow() {
    run_help("workflow");
}

#[test]
fn help_collab() {
    run_help("collab");
}

#[test]
fn help_proxy() {
    run_help("proxy");
}

#[test]
fn help_clips() {
    run_help("clips");
}

#[test]
fn help_review() {
    run_help("review");
}

#[test]
fn help_drm() {
    run_help("drm");
}

#[test]
fn help_dedup() {
    run_help("dedup");
}

#[test]
fn help_archive_pro() {
    run_help("archive-pro");
}

#[test]
fn help_dolby_vision() {
    run_help("dolby-vision");
}

#[test]
fn help_timesync() {
    run_help("timesync");
}

#[test]
fn help_align() {
    run_help("align");
}

#[test]
fn help_routing() {
    run_help("routing");
}

#[test]
fn help_calibrate() {
    run_help("calibrate");
}

#[test]
fn help_virtual() {
    run_help("virtual");
}

#[test]
fn help_profiler() {
    run_help("profiler");
}

#[test]
fn help_recommend() {
    run_help("recommend");
}

#[test]
fn help_scaling() {
    run_help("scaling");
}

#[test]
fn help_renderfarm() {
    run_help("renderfarm");
}

#[test]
fn help_access() {
    run_help("access");
}

#[test]
fn help_rights() {
    run_help("rights");
}

#[test]
fn help_auto() {
    run_help("auto");
}

#[test]
fn help_loudness() {
    run_help("loudness");
}

#[test]
fn help_quality() {
    run_help("quality");
}

#[test]
fn help_normalize() {
    run_help("normalize");
}

#[test]
fn help_batch_engine() {
    run_help("batch-engine");
}

#[test]
fn help_ml() {
    // `ml` itself is unconditionally wired into the dispatcher even when the
    // `ml` feature is off — only the *body* errors out. `--help` must still
    // work because clap intercepts it before the handler runs.
    run_help("ml");
}
