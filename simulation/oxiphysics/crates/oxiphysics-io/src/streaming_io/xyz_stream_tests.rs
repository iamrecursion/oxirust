// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the streaming XYZ reader: every line is read once,
//! malformed input is reported with its line number instead of being turned
//! into zeros, and the reader can be repositioned.

use super::functions::{
    next_frame_xyz, open_streaming_writer, open_streaming_xyz, streaming_statistics,
    write_frame_streaming,
};
use super::types::StreamingXYZReader;
use crate::Error;
use std::path::PathBuf;

/// A temporary file that is removed when the test ends.
struct TempXyz(PathBuf);

impl TempXyz {
    fn new(name: &str, contents: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "oxiphysics_xyz_stream_{}_{name}.xyz",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("write temp xyz file");
        Self(path)
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("temp path is valid UTF-8")
    }
}

impl Drop for TempXyz {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `n_frames` frames of `n_atoms` atoms; atom `a` of frame `f` is at
/// `(a, f, 0.5)` with element `C` for even and `H` for odd atoms.
fn trajectory_text(n_frames: usize, n_atoms: usize) -> String {
    let mut text = String::new();
    for frame in 0..n_frames {
        text.push_str(&format!("{n_atoms}\nframe {frame}\n"));
        for atom in 0..n_atoms {
            let element = if atom % 2 == 0 { "C" } else { "H" };
            text.push_str(&format!("{element} {atom} {frame} 0.5\n"));
        }
    }
    text
}

/// Number of lines the reader has consumed from the file so far.
fn lines_read(reader: &StreamingXYZReader) -> usize {
    reader.reader.as_ref().map_or(0, |stream| stream.lines_read)
}

fn parse_message(error: Error) -> String {
    match error {
        Error::Parse(message) => message,
        other => panic!("expected a parse error, got {other}"),
    }
}

#[test]
fn a_thousand_frames_are_read_sequentially_each_line_exactly_once() {
    let (n_frames, n_atoms) = (1000, 3);
    let file = TempXyz::new("thousand", &trajectory_text(n_frames, n_atoms));
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    for frame in 0..n_frames {
        let (positions, symbols) = next_frame_xyz(&mut reader)
            .expect("read")
            .expect("frame present");
        assert_eq!(positions.len(), n_atoms);
        assert_eq!(symbols, vec!["C", "H", "C"]);
        assert_eq!(positions[2], [2.0, frame as f64, 0.5]);
        // The reader has consumed exactly the lines of the frames read so far,
        // not the whole file again.
        assert_eq!(
            lines_read(&reader),
            (frame + 1) * (n_atoms + 2),
            "frame {frame}"
        );
        assert_eq!(reader.current_frame, frame + 1);
    }
    assert!(next_frame_xyz(&mut reader).expect("end of file").is_none());
    assert_eq!(lines_read(&reader), n_frames * (n_atoms + 2));
    // Asking again keeps returning the end of the file.
    assert!(next_frame_xyz(&mut reader).expect("end of file").is_none());
    assert_eq!(reader.n_atoms, n_atoms);
}

#[test]
fn a_malformed_coordinate_is_an_error_with_its_line_number() {
    let text = "2\nfirst\nC 0 0 0\nH 1 0 0\n2\nsecond\nC 0 0 0\nH 1 oops 0\n";
    let file = TempXyz::new("bad_coordinate", text);
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    assert!(next_frame_xyz(&mut reader).expect("frame 0").is_some());
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("bad coordinate"));
    assert!(message.starts_with("line 8:"), "{message}");
    assert!(message.contains("'oops'"), "{message}");
    // The frame counter did not advance.
    assert_eq!(reader.current_frame, 1);
    // Reading again fails the same way instead of continuing inside the frame.
    let again = parse_message(next_frame_xyz(&mut reader).expect_err("still bad"));
    assert_eq!(again, message);
}

#[test]
fn atom_lines_with_too_few_fields_are_errors_not_dropped_atoms() {
    let file = TempXyz::new("short_line", "2\ncomment\nC 0 0 0\nH 1 2\n");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("short line"));
    assert!(message.starts_with("line 4:"), "{message}");
}

#[test]
fn a_frame_cut_off_by_the_end_of_the_file_is_an_error() {
    let file = TempXyz::new("truncated", "3\ncomment\nC 0 0 0\nH 1 0 0\n");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("truncated"));
    assert!(message.contains("expected 3 atoms, found 2"), "{message}");

    let file = TempXyz::new("no_comment", "3\n");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("no comment line"));
    assert!(message.contains("no comment line"), "{message}");
}

#[test]
fn an_atom_count_that_is_not_a_number_is_an_error() {
    for (name, text) in [
        ("count_word", "three\ncomment\nC 0 0 0\n"),
        ("count_negative", "-2\ncomment\nC 0 0 0\n"),
        ("count_float", "2.5\ncomment\nC 0 0 0\n"),
    ] {
        let file = TempXyz::new(name, text);
        let mut reader = open_streaming_xyz(file.path()).expect("open");
        let message = parse_message(next_frame_xyz(&mut reader).expect_err(name));
        assert!(message.starts_with("line 1:"), "{name}: {message}");
        assert!(message.contains("number of atoms"), "{name}: {message}");
    }
}

#[test]
fn an_absurd_atom_count_is_an_error_not_an_allocation() {
    // 2^63 atoms announced, three lines present: must fail at the end of the
    // file without reserving memory for the announced count.
    let file = TempXyz::new("absurd", "9223372036854775808\ncomment\nC 0 0 0\n");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("absurd count"));
    assert!(
        message.contains("expected 9223372036854775808 atoms, found 1"),
        "{message}"
    );
}

#[test]
fn blank_lines_between_frames_and_at_the_end_are_not_errors() {
    let text = "1\nfirst\nC 0 0 0\n\n\n1\n\nH 1 1 1\n\n";
    let file = TempXyz::new("blank_lines", text);
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let (_, symbols) = next_frame_xyz(&mut reader)
        .expect("frame 0")
        .expect("present");
    assert_eq!(symbols, vec!["C"]);
    // The comment line of the second frame is blank: that is a valid comment.
    let (positions, symbols) = next_frame_xyz(&mut reader)
        .expect("frame 1")
        .expect("present");
    assert_eq!(symbols, vec!["H"]);
    assert_eq!(positions, vec![[1.0, 1.0, 1.0]]);
    assert!(next_frame_xyz(&mut reader).expect("end").is_none());
}

#[test]
fn extra_columns_and_a_missing_final_newline_are_accepted() {
    let file = TempXyz::new(
        "extended",
        "2\nLattice=\"1 0 0 0 1 0 0 0 1\"\nC 0.0 1.0 2.0 0.1 0.2 0.3\nH -1.5e-3 2 3",
    );
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let (positions, symbols) = next_frame_xyz(&mut reader)
        .expect("frame")
        .expect("present");
    assert_eq!(symbols, vec!["C", "H"]);
    assert_eq!(positions, vec![[0.0, 1.0, 2.0], [-1.5e-3, 2.0, 3.0]]);
    assert!(next_frame_xyz(&mut reader).expect("end").is_none());
}

#[test]
fn an_empty_frame_and_an_empty_file_are_valid() {
    let file = TempXyz::new("zero_atoms", "0\nnothing here\n");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let (positions, symbols) = next_frame_xyz(&mut reader)
        .expect("frame")
        .expect("present");
    assert!(positions.is_empty() && symbols.is_empty());
    assert!(next_frame_xyz(&mut reader).expect("end").is_none());

    let file = TempXyz::new("empty", "");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    assert!(next_frame_xyz(&mut reader).expect("end").is_none());
}

#[test]
fn invalid_utf8_and_overlong_lines_are_errors() {
    let path = std::env::temp_dir().join(format!(
        "oxiphysics_xyz_stream_{}_binary.xyz",
        std::process::id()
    ));
    std::fs::write(&path, b"1\ncomment\nC \xff\xfe 0 0\n").expect("write");
    let mut reader = open_streaming_xyz(path.to_str().expect("utf-8 path")).expect("open");
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("bad utf-8"));
    assert!(
        message.starts_with("line 3:") && message.contains("UTF-8"),
        "{message}"
    );
    let _ = std::fs::remove_file(&path);

    let mut text = String::from("1\ncomment\nC 0 0 ");
    text.push_str(&"9".repeat((1 << 20) + 16));
    text.push('\n');
    let file = TempXyz::new("overlong", &text);
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("overlong line"));
    assert!(
        message.starts_with("line 3:") && message.contains("longer than"),
        "{message}"
    );
}

#[test]
fn assigning_current_frame_repositions_the_reader() {
    let file = TempXyz::new("seek", &trajectory_text(6, 2));
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    for _ in 0..4 {
        next_frame_xyz(&mut reader).expect("read").expect("present");
    }
    assert_eq!(reader.current_frame, 4);

    // Backwards: rewinds the file.
    reader.current_frame = 1;
    let (positions, _) = next_frame_xyz(&mut reader).expect("read").expect("present");
    assert_eq!(positions[0], [0.0, 1.0, 0.5]);
    assert_eq!(reader.current_frame, 2);

    // Forwards: skips frames.
    reader.current_frame = 5;
    let (positions, _) = next_frame_xyz(&mut reader).expect("read").expect("present");
    assert_eq!(positions[0], [0.0, 5.0, 0.5]);
    assert!(next_frame_xyz(&mut reader).expect("end").is_none());

    // Beyond the end: nothing there.
    reader.current_frame = 40;
    assert!(next_frame_xyz(&mut reader).expect("end").is_none());

    // And back to the start.
    reader.current_frame = 0;
    let (positions, _) = next_frame_xyz(&mut reader).expect("read").expect("present");
    assert_eq!(positions[1], [1.0, 0.0, 0.5]);
}

#[test]
fn an_error_while_skipping_does_not_leave_the_reader_inside_a_frame() {
    // Frame 0 is fine, frame 1 is cut off.
    let file = TempXyz::new("skip_error", "1\nc\nC 0 0 0\n2\nc\nC 1 1 1\n");
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    reader.current_frame = 3;
    // Skipping towards frame 3 runs into the truncated frame 1.
    assert!(next_frame_xyz(&mut reader).is_err());
    // Asking for frame 1 must report that frame as truncated, not silently
    // continue from wherever the failed skip stopped.
    reader.current_frame = 1;
    let message = parse_message(next_frame_xyz(&mut reader).expect_err("truncated frame 1"));
    assert!(message.contains("expected 2 atoms, found 1"), "{message}");
    reader.current_frame = 0;
    let (_, symbols) = next_frame_xyz(&mut reader)
        .expect("rewound")
        .expect("frame 0");
    assert_eq!(symbols, vec!["C"]);
}

#[test]
fn a_reader_built_without_an_open_file_opens_it_on_first_use() {
    let file = TempXyz::new("lazy", &trajectory_text(2, 2));
    let mut reader = StreamingXYZReader {
        path: file.path().to_string(),
        current_frame: 1,
        n_atoms: 0,
        buffer: Vec::new(),
        reader: None,
    };
    let (positions, _) = next_frame_xyz(&mut reader).expect("read").expect("present");
    assert_eq!(positions[0], [0.0, 1.0, 0.5]);

    let mut missing = StreamingXYZReader {
        path: std::env::temp_dir()
            .join("oxiphysics_xyz_stream_does_not_exist.xyz")
            .to_str()
            .expect("utf-8 path")
            .to_string(),
        current_frame: 0,
        n_atoms: 0,
        buffer: Vec::new(),
        reader: None,
    };
    assert!(matches!(next_frame_xyz(&mut missing), Err(Error::Io(_))));
}

#[test]
fn streaming_statistics_rewinds_and_reports_errors() {
    let file = TempXyz::new("stats", &trajectory_text(4, 2));
    let mut reader = open_streaming_xyz(file.path()).expect("open");
    next_frame_xyz(&mut reader).expect("read").expect("present");
    next_frame_xyz(&mut reader).expect("read").expect("present");
    let (count, mean, std_dev) = streaming_statistics(&mut reader).expect("statistics");
    assert_eq!(count, 4);
    // Atoms at x = 0, 1 for each frame, y = 0..3, z = 0.5.
    assert!((mean[0] - 0.5).abs() < 1e-12);
    assert!((mean[1] - 1.5).abs() < 1e-12);
    assert!((mean[2] - 0.5).abs() < 1e-12);
    assert!((std_dev[0] - 0.5).abs() < 1e-12);
    assert!((std_dev[1] - 1.25_f64.sqrt()).abs() < 1e-12);
    assert!(std_dev[2].abs() < 1e-6);

    let broken = TempXyz::new("stats_broken", "1\nc\nC 0 0 0\n1\nc\nC 0 x 0\n");
    let mut reader = open_streaming_xyz(broken.path()).expect("open");
    let message = parse_message(streaming_statistics(&mut reader).expect_err("bad frame"));
    assert!(message.starts_with("line 6:"), "{message}");
}

#[test]
fn frames_written_by_the_streaming_writer_are_read_back() {
    let path = std::env::temp_dir().join(format!(
        "oxiphysics_xyz_stream_{}_roundtrip.xyz",
        std::process::id()
    ));
    let path_str = path.to_str().expect("utf-8 path");
    let mut writer = open_streaming_writer(path_str).expect("writer");
    for frame in 0..50 {
        let positions: Vec<[f64; 3]> = (0..4).map(|a| [a as f64, frame as f64, -0.25]).collect();
        write_frame_streaming(&mut writer, &positions, &["O", "H", "H"]).expect("write");
    }
    let mut reader = open_streaming_xyz(path_str).expect("reader");
    let mut frames = 0;
    while let Some((positions, symbols)) = next_frame_xyz(&mut reader).expect("read") {
        assert_eq!(positions.len(), 4);
        assert_eq!(symbols, vec!["O", "H", "H", "X"]);
        assert_eq!(positions[3], [3.0, frames as f64, -0.25]);
        frames += 1;
    }
    assert_eq!(frames, 50);
    let _ = std::fs::remove_file(&path);
}
