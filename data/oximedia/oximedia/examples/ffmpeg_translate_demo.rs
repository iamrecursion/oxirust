//! Parses a real FFmpeg-style command line and prints the translated
//! OxiMedia transcode job(s), demonstrating the `compat-ffmpeg` FFmpeg CLI
//! compatibility layer.
//!
//! Run with:
//!
//! ```text
//! cargo run -p oximedia --example ffmpeg_translate_demo --features compat-ffmpeg
//! ```
//!
//! Accepts an optional FFmpeg command line via CLI arguments (everything
//! after `--`); falls back to a representative built-in example command when
//! none are given.

use oximedia::compat_ffmpeg::{parse_and_translate, TranscodeJob};

/// A representative FFmpeg command line: scale to 1080p, re-encode video to
/// a CRF-based quality target, transcode audio, and burn in fast-start.
const DEFAULT_ARGS: &[&str] = &[
    "-i",
    "input.mov",
    "-vf",
    "scale=1920:1080",
    "-c:v",
    "libx264",
    "-crf",
    "23",
    "-preset",
    "medium",
    "-c:a",
    "aac",
    "-b:a",
    "192k",
    "-movflags",
    "+faststart",
    "output.mp4",
];

fn print_job(index: usize, job: &TranscodeJob) {
    println!("Job #{index}");
    println!("  input:        {}", job.input_path);
    println!("  output:       {}", job.output_path);
    println!("  video codec:  {:?}", job.video_codec);
    println!("  audio codec:  {:?}", job.audio_codec);
    println!("  video bitrate:{:?}", job.video_bitrate);
    println!("  audio bitrate:{:?}", job.audio_bitrate);
    println!("  crf:          {:?}", job.crf);
    println!("  preset:       {:?}", job.preset);
    println!("  video filters:{:?}", job.video_filters);
    println!("  overwrite:    {}", job.overwrite);
}

fn main() {
    // Everything after `--` on the command line overrides the built-in
    // demo arguments (e.g. `cargo run --example ffmpeg_translate_demo
    // --features compat-ffmpeg -- -i in.mkv -c:v libaom-av1 out.webm`).
    let cli_args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<String> = if cli_args.is_empty() {
        DEFAULT_ARGS.iter().map(|s| s.to_string()).collect()
    } else {
        cli_args
    };

    println!("Translating FFmpeg command line:");
    println!("  ffmpeg {}", args.join(" "));
    println!();

    let result = parse_and_translate(&args);

    if result.jobs.is_empty() {
        println!("No transcode jobs were produced.");
    } else {
        for (index, job) in result.jobs.iter().enumerate() {
            print_job(index, job);
            println!();
        }
    }

    if result.diagnostics.is_empty() {
        println!("No diagnostics.");
    } else {
        println!("Diagnostics:");
        for diagnostic in &result.diagnostics {
            println!("  {diagnostic}");
        }
    }

    if result.has_errors() {
        eprintln!("Translation completed with errors.");
        std::process::exit(1);
    }
}
