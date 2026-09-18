use assert_cmd::Command;

fn cmd() -> Command {
    Command::cargo_bin("oximedia-cv2").expect("binary not found")
}

#[test]
fn version_exits_zero() {
    cmd().arg("--version").assert().success();
}

#[test]
fn help_exits_zero() {
    cmd().arg("--help").assert().success();
}

#[test]
fn no_args_exits_nonzero() {
    cmd().assert().failure();
}
