//! Integration tests for `oxixml hdt`.

mod support;

use support::{TempFile, run, run_bytes, run_no_stdin};

const NTRIPLES_FIXTURE: &str =
    "<http://example.com/s> <http://example.com/p> <http://example.com/o> .\n";

#[test]
fn top_level_help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["hdt", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml hdt"));
    assert!(stdout.contains("build"));
    assert!(stdout.contains("dump"));
    assert!(stdout.contains("stats"));
    assert!(stderr.is_empty());
}

#[test]
fn each_sub_subcommand_help_is_rendered() {
    for sub in ["build", "dump", "stats"] {
        let (status, stdout, stderr) = run_no_stdin(&["hdt", sub, "--help"]);
        assert_eq!(status, 0, "{sub}: {stderr}");
        assert!(
            stdout.contains(&format!("oxixml hdt {sub}")),
            "{sub}: {stdout}"
        );
    }
}

#[test]
fn happy_path_build_dump_and_stats_round_trip() {
    let source = TempFile::new("hdt-roundtrip", "nt", NTRIPLES_FIXTURE);
    let hdt_path = format!("{}.hdt", source.path());

    let (status, stdout, stderr) =
        run_no_stdin(&["hdt", "build", "--output", &hdt_path, source.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.is_empty());
    let bytes = std::fs::read(&hdt_path).expect("read the built HDT file");
    assert!(bytes.starts_with(b"$HDT"), "not an HDT file");

    let (status, stdout, stderr) = run_no_stdin(&["hdt", "dump", &hdt_path]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("http://example.com/s"), "{stdout}");
    assert!(stdout.contains("http://example.com/p"), "{stdout}");
    assert!(stdout.contains("http://example.com/o"), "{stdout}");

    let (status, stdout, stderr) = run_no_stdin(&["hdt", "stats", &hdt_path]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("triples: 1"), "{stdout}");

    let _ = std::fs::remove_file(&hdt_path);
}

#[test]
fn build_writes_to_stdout_without_an_output_flag() {
    let (status, stdout, stderr) = run_bytes(&["hdt", "build", "-"], NTRIPLES_FIXTURE.as_bytes());
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.starts_with(b"$HDT"));
}

#[test]
fn dump_reads_the_hdt_file_from_stdin() {
    let (build_status, build_stdout, build_stderr) =
        run_bytes(&["hdt", "build", "-"], NTRIPLES_FIXTURE.as_bytes());
    assert_eq!(build_status, 0, "{build_stderr}");

    let (status, stdout, stderr) = run(&["hdt", "dump", "-"], &build_stdout);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("http://example.com/o"), "{stdout}");
}

#[test]
fn an_unknown_sub_subcommand_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["hdt", "bogus"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("bogus"));
}

#[test]
fn build_missing_input_path_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["hdt", "build"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("input path"));
}

#[test]
fn build_rejects_a_named_graph() {
    let (status, stdout, stderr) = run(
        &["hdt", "build", "--from", "nquads", "-"],
        b"<http://example.com/s> <http://example.com/p> <http://example.com/o> <http://example.com/g> .\n",
    );
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("named graph"));
}

#[test]
fn dump_of_a_malformed_file_is_a_processing_error() {
    let (status, stdout, stderr) = run(&["hdt", "dump", "-"], b"not an hdt file");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}

#[test]
fn catalog_is_not_applicable_to_hdt_dump() {
    let (status, _stdout, stderr) = run_no_stdin(&["hdt", "dump", "--catalog", "c.xml", "a.hdt"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--catalog"));
}
