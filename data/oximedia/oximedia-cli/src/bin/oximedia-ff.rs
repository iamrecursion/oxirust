//! `oximedia-ff` — FFmpeg drop-in entry point for OxiMedia.
//!
//! This binary accepts raw FFmpeg-style arguments and passes them through the
//! OxiMedia FFmpeg-compat translation layer, then **executes** each resulting
//! transcode job using the native OxiMedia transcode subsystem. It can be
//! symlinked or aliased as `ffmpeg` to act as a transparent drop-in replacement
//! for patent-free workflows.
//!
//! ## Usage
//!
//! ```sh
//! oximedia-ff -i input.mkv -c:v libaom-av1 -crf 28 -c:a libopus output.webm
//! oximedia-ff -i input.mp4 -c:v libx264 output.webm   # libx264 auto-substituted with av1
//! oximedia-ff -y -i src.mkv -vf scale=1280:720 -b:v 2M out.webm
//! oximedia-ff --dry-run -i src.mkv -c:v av1 out.webm  # print plan only
//! ```

use colored::Colorize;
use oximedia_cli::progress::ProgressFormat;
use oximedia_cli::transcode::{self, TranscodeOptions};
use oximedia_compat_ffmpeg::{
    parse_and_translate, Diagnostic, DiagnosticKind, MapSpec, ParsedFilter, TranscodeJob,
    TranslateResult,
};
use std::path::PathBuf;
use tracing::warn;

#[tokio::main]
async fn main() {
    // Install the Pure-Rust `rustls-rustcrypto` crypto provider as the
    // process-wide default before any TLS connection can be opened. See
    // `oximedia_net::tls_provider` for details. Idempotent.
    oximedia_net::install_default_crypto_provider();

    // Initialise a minimal tracing subscriber so warn!/info! calls are visible.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init();

    // Skip argv[0] (the binary name itself).
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_help();
        std::process::exit(1);
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    if args.iter().any(|a| a == "--version" || a == "-version") {
        println!(
            "oximedia-ff {} (OxiMedia FFmpeg-compat layer)",
            env!("CARGO_PKG_VERSION")
        );
        std::process::exit(0);
    }

    match run(&args).await {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{} {}", "oximedia-ff: fatal:".red().bold(), e);
            std::process::exit(1);
        }
    }
}

async fn run(args: &[String]) -> anyhow::Result<()> {
    // Strip --dry-run / --plan / --explain / --json before passing to the FFmpeg compat parser.
    let dry_run = args
        .iter()
        .any(|a| a == "--dry-run" || a == "--plan" || a == "-dry-run");

    let explain_mode = args.iter().any(|a| a == "--explain");
    let json_mode = args.iter().any(|a| a == "--json");

    // Pre-process: extract `-o <file>` or `-o=<file>` as a positional output alias,
    // then strip those tokens so the FFmpeg-compat parser sees only known forms.
    let (prefiltered, dash_o_output) = extract_dash_o(args)?;

    let mut filtered: Vec<String> = prefiltered
        .iter()
        .filter(|a| {
            *a != "--dry-run"
                && *a != "--plan"
                && *a != "-dry-run"
                && *a != "--explain"
                && *a != "--json"
        })
        .cloned()
        .collect();

    // Append the -o value as a bare positional output (FFmpeg convention: last bare arg).
    if let Some(out) = dash_o_output {
        filtered.push(out);
    }

    let result = parse_and_translate(&filtered);

    // --json: emit the structured TranslateResult as JSON on stdout and exit.
    // Honoured both with and without --dry-run; the JSON form never executes a transcode.
    if json_mode {
        let json = render_translate_json(&result)?;
        println!("{}", json);
        if result.has_errors() {
            std::process::exit(1);
        }
        return Ok(());
    }

    // Print diagnostics to stderr in FFmpeg style.
    for diag in &result.diagnostics {
        let formatted = diag.format_ffmpeg_style("oximedia-ff");
        eprintln!("{}", formatted);
    }

    if result.has_errors() {
        anyhow::bail!("aborting due to errors — see diagnostics above");
    }

    if result.jobs.is_empty() {
        print_help();
        return Ok(());
    }

    // --explain: print the argument → field translation table and exit.
    if explain_mode {
        print_explain_table(&result.jobs);
        return Ok(());
    }

    // Print brief job plan.
    for (idx, job) in result.jobs.iter().enumerate() {
        eprintln!(
            "{} Job {}: {} {} {}",
            "oximedia-ff:".green().bold(),
            idx + 1,
            job.input_path.cyan(),
            "→".bold(),
            job.output_path.cyan()
        );

        if let Some(vc) = &job.video_codec {
            eprintln!("  video: {}", vc.green());
        }
        if let Some(ac) = &job.audio_codec {
            eprintln!("  audio: {}", ac.green());
        }
        if let Some(crf) = job.crf {
            eprintln!("  crf: {:.1}", crf);
        }
        if let Some(vb) = &job.video_bitrate {
            eprintln!("  video bitrate: {}", vb);
        }
        if let Some(ab) = &job.audio_bitrate {
            eprintln!("  audio bitrate: {}", ab);
        }
        if !job.video_filters.is_empty() {
            eprintln!("  video filters: {} applied", job.video_filters.len());
        }
        if !job.audio_filters.is_empty() {
            eprintln!("  audio filters: {} applied", job.audio_filters.len());
        }
        if let Some(seek) = &job.seek {
            eprintln!("  seek: {}", seek);
        }
        if let Some(dur) = &job.duration {
            eprintln!("  duration: {}", dur);
        }
        if job.overwrite {
            eprintln!("  {}", "overwrite: yes".dimmed());
        }
        for (k, v) in &job.metadata {
            eprintln!("  metadata: {}={}", k, v);
        }

        if dry_run {
            eprintln!("  {}", "[dry-run: skipping execution]".yellow().italic());
        }
    }

    if dry_run {
        eprintln!(
            "\n{} Dry-run mode — no files were written.",
            "oximedia-ff: note:".cyan()
        );
        return Ok(());
    }

    // Execute each job.
    for (idx, job) in result.jobs.iter().enumerate() {
        eprintln!(
            "\n{} Transcoding ({}/{}) …",
            "oximedia-ff:".green().bold(),
            idx + 1,
            result.jobs.len()
        );

        execute_job(job)
            .await
            .map_err(|e| anyhow::anyhow!("job {} failed: {}", idx + 1, e))?;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Argument pre-processing
// ─────────────────────────────────────────────────────────────────────────────

/// Scan `args` for `-o <value>` or `-o=<value>` and return:
/// - a vec of the remaining args (with the `-o` pair removed), and
/// - the extracted output path, if any.
///
/// The extracted path will later be appended as a bare positional output
/// argument so that `parse_and_translate` sees it as the FFmpeg output file.
fn extract_dash_o(args: &[String]) -> anyhow::Result<(Vec<String>, Option<String>)> {
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    let mut output_override: Option<String> = None;
    let mut iter = args.iter();

    while let Some(tok) = iter.next() {
        if tok == "-o" {
            // `-o <file>` form — value is the next token.
            match iter.next() {
                Some(val) if !val.is_empty() => {
                    output_override = Some(val.clone());
                }
                Some(_) | None => {
                    anyhow::bail!("option '-o' requires an output path argument");
                }
            }
        } else if let Some(val) = tok.strip_prefix("-o=") {
            // `-o=<file>` form — value is embedded.
            if val.is_empty() {
                anyhow::bail!("option '-o=' requires a non-empty output path");
            }
            output_override = Some(val.to_string());
        } else {
            out.push(tok.clone());
        }
    }

    Ok((out, output_override))
}

// ─────────────────────────────────────────────────────────────────────────────
// Job execution
// ─────────────────────────────────────────────────────────────────────────────

async fn execute_job(job: &TranscodeJob) -> anyhow::Result<()> {
    if !job.overwrite && std::path::Path::new(&job.output_path).exists() {
        anyhow::bail!(
            "Output file '{}' already exists. Pass -y to overwrite.",
            job.output_path
        );
    }

    let vf_string = build_filter_string(&job.video_filters);
    let af_string = build_filter_string(&job.audio_filters);

    let scale_from_filters = extract_scale_filter(&job.video_filters);

    let video_codec = match job.video_codec.as_deref() {
        Some("copy") | None if job.no_video => None,
        Some(vc) => Some(vc.to_string()),
        None => None,
    };

    let audio_codec = match job.audio_codec.as_deref() {
        Some("copy") | None if job.no_audio => None,
        Some(ac) => Some(ac.to_string()),
        None => None,
    };

    let crf = job.crf.map(|c| c.round() as u32);

    // See `ffcompat_cmd::execute_job` for the rationale: a `-af loudnorm=...`
    // filter is FFmpeg's classic EBU R128 normalization request, so map it
    // onto `--normalize-audio`'s real pipeline wiring instead of leaving it
    // a no-op.
    let normalize_audio = job
        .audio_filters
        .iter()
        .any(|f| matches!(f, ParsedFilter::LoudNorm { .. }));

    let options = TranscodeOptions {
        input: PathBuf::from(&job.input_path),
        output: PathBuf::from(&job.output_path),
        preset_name: None,
        video_codec,
        audio_codec,
        video_bitrate: job.video_bitrate.clone(),
        audio_bitrate: job.audio_bitrate.clone(),
        scale: scale_from_filters,
        video_filter: vf_string,
        audio_filter: af_string,
        start_time: job.seek.clone(),
        duration: job.duration.clone(),
        framerate: None,
        preset: "medium".to_string(),
        two_pass: false,
        crf,
        // 0 = "auto": the stream-copy pipeline has no threading knob, and a
        // non-zero value only triggers `transcode`'s --threads warning. Keep
        // it silent unless the user explicitly asked for threads.
        threads: 0,
        overwrite: job.overwrite,
        map: map_specs_to_selectors(&job.map),
        normalize_audio,
        progress_format: ProgressFormat::Plain,
    };

    transcode::transcode(options).await
}

/// Convert parsed FFmpeg `-map` specs into `--map` selector strings.
///
/// Filter-graph label maps (`-map "[out]"`) address filter output pads, not
/// input streams, so they have no meaning in the packet-level stream-copy
/// pipeline and are skipped with a warning.
fn map_specs_to_selectors(specs: &[MapSpec]) -> Vec<String> {
    specs
        .iter()
        .filter_map(|spec| {
            if let Some(ref label) = spec.label {
                eprintln!(
                    "{} -map '{label}' addresses a filter-graph pad; \
                     not supported in stream-copy mode, ignoring",
                    "Warning:".yellow().bold()
                );
                return None;
            }
            let sign = if spec.negative { "-" } else { "" };
            let selector = spec
                .stream_selector
                .as_deref()
                .map(|s| format!(":{s}"))
                .unwrap_or_default();
            Some(format!("{sign}{}{selector}", spec.input_index))
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_filter_string(filters: &[ParsedFilter]) -> Option<String> {
    let parts: Vec<String> = filters
        .iter()
        .filter_map(|f| match f {
            ParsedFilter::Scale { w, h } => Some(format!("scale={}:{}", w, h)),
            ParsedFilter::Fps { rate } => Some(format!("fps={}", rate)),
            ParsedFilter::HFlip => Some("hflip".to_string()),
            ParsedFilter::VFlip => Some("vflip".to_string()),
            ParsedFilter::Deinterlace => Some("yadif".to_string()),
            ParsedFilter::Rotate { angle } => Some(format!("rotate={}", angle)),
            ParsedFilter::Crop { w, h, x, y } => Some(format!("crop={}:{}:{}:{}", w, h, x, y)),
            ParsedFilter::ColorCorrect {
                brightness,
                contrast,
                saturation,
            } => Some(format!(
                "eq=brightness={}:contrast={}:saturation={}",
                brightness, contrast, saturation
            )),
            ParsedFilter::Lut3d { file } => Some(format!("lut3d=file={}", file)),
            ParsedFilter::SubtitleBurnIn { file } => Some(format!("subtitles=filename={}", file)),
            ParsedFilter::LoudNorm {
                integrated,
                true_peak,
                lra,
            } => Some(format!(
                "loudnorm=I={}:TP={}:LRA={}",
                integrated, true_peak, lra
            )),
            ParsedFilter::Volume { factor } => Some(format!("volume={}", factor)),
            ParsedFilter::Resample { sample_rate } => Some(format!("aresample={}", sample_rate)),
            ParsedFilter::Compressor { threshold, ratio } => Some(format!(
                "acompressor=threshold={}:ratio={}",
                threshold, ratio
            )),
            ParsedFilter::Passthrough => None,
            ParsedFilter::Unknown { name, args } => {
                warn!(
                    "Skipping unsupported filter '{}' (args: '{}') during execution.",
                    name, args
                );
                eprintln!(
                    "{} Skipping unsupported filter '{}' during execution.",
                    "oximedia-ff: warning:".yellow().bold(),
                    name
                );
                None
            }
        })
        .collect();

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

fn extract_scale_filter(filters: &[ParsedFilter]) -> Option<String> {
    filters.iter().find_map(|f| {
        if let ParsedFilter::Scale { w, h } = f {
            Some(format!("{}:{}", w, h))
        } else {
            None
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON serialization (--json mode)
// ─────────────────────────────────────────────────────────────────────────────

/// Render a [`TranslateResult`] as a stable, human-and-machine-readable JSON
/// document for `--json` mode.
///
/// The schema is intentionally narrow and stable so the golden test suite in
/// `tests/ff_golden.rs` can pin expected outputs without tracking every field
/// of [`TranscodeJob`]. The shape is:
///
/// ```json
/// {
///   "diagnostics": [{"kind": "PatentCodecSubstituted", "message": "..."}, ...],
///   "jobs": [
///     {
///       "input": "in.mp4",
///       "output": "out.mkv",
///       "video_codec": "av1",
///       "audio_codec": "opus",
///       "video_bitrate": "2M",
///       "audio_bitrate": "128k",
///       "crf": 23.0,
///       "preset": "fast",
///       "tune": null,
///       "profile": null,
///       "video_filters": 1,
///       "audio_filters": 0,
///       "seek": null,
///       "duration": null,
///       "format": null,
///       "overwrite": true,
///       "no_video": false,
///       "no_audio": false,
///       "metadata": {"title": "..."},
///       "map": 0,
///       "map_metadata": 0,
///       "muxer_actions": ["FastStart"],
///       "hwaccel": "Cuda",
///       "pass": null,
///       "gop_size": null
///     }
///   ]
/// }
/// ```
fn render_translate_json(result: &TranslateResult) -> anyhow::Result<String> {
    let diagnostics: Vec<serde_json::Value> =
        result.diagnostics.iter().map(diagnostic_to_json).collect();

    let jobs: Vec<serde_json::Value> = result.jobs.iter().map(job_to_json).collect();

    let doc = serde_json::json!({
        "diagnostics": diagnostics,
        "jobs": jobs,
    });

    Ok(serde_json::to_string_pretty(&doc)?)
}

fn diagnostic_to_json(diag: &Diagnostic) -> serde_json::Value {
    let kind = match &diag.kind {
        DiagnosticKind::PatentCodecSubstituted { .. } => "PatentCodecSubstituted",
        DiagnosticKind::UnknownOptionIgnored { .. } => "UnknownOptionIgnored",
        DiagnosticKind::FilterNotSupported { .. } => "FilterNotSupported",
        DiagnosticKind::UnsupportedFeature { .. } => "UnsupportedFeature",
        DiagnosticKind::Info { .. } => "Info",
        DiagnosticKind::Error { .. } => "Error",
        DiagnosticKind::Warning { .. } => "Warning",
    };

    let message = diag.format_ffmpeg_style("oximedia-ff");

    serde_json::json!({
        "kind": kind,
        "message": message,
        "suggestion": diag.suggestion,
    })
}

fn job_to_json(job: &TranscodeJob) -> serde_json::Value {
    let muxer_actions: Vec<String> = job
        .muxer_options
        .iter()
        .map(|opt| muxer_action_label(&opt.oxi_action).to_string())
        .collect();

    let hwaccel = job.hwaccel.as_ref().map(|cfg| format!("{:?}", cfg.backend));

    serde_json::json!({
        "input": job.input_path,
        "output": job.output_path,
        "video_codec": job.video_codec,
        "audio_codec": job.audio_codec,
        "video_bitrate": job.video_bitrate,
        "audio_bitrate": job.audio_bitrate,
        "crf": job.crf,
        "preset": job.preset,
        "tune": job.tune,
        "profile": job.profile,
        "video_filters": job.video_filters.len(),
        "audio_filters": job.audio_filters.len(),
        "seek": job.seek,
        "duration": job.duration,
        "format": job.format,
        "overwrite": job.overwrite,
        "no_video": job.no_video,
        "no_audio": job.no_audio,
        "metadata": job.metadata,
        "map": job.map.len(),
        "map_metadata": job.map_metadata.len(),
        "muxer_actions": muxer_actions,
        "hwaccel": hwaccel,
        "pass": job.pass,
        "gop_size": job.gop_size,
    })
}

fn muxer_action_label(action: &oximedia_compat_ffmpeg::MuxerAction) -> &'static str {
    use oximedia_compat_ffmpeg::MuxerAction;
    match action {
        MuxerAction::FastStart => "FastStart",
        MuxerAction::FragmentedMp4 => "FragmentedMp4",
        MuxerAction::DashCompat => "DashCompat",
        MuxerAction::DisableAudio => "DisableAudio",
        MuxerAction::GlobalHeader => "GlobalHeader",
        MuxerAction::GeneratePts => "GeneratePts",
        MuxerAction::DiscardCorrupt => "DiscardCorrupt",
        MuxerAction::Shortest => "Shortest",
        MuxerAction::Unknown { .. } => "Unknown",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explain mode
// ─────────────────────────────────────────────────────────────────────────────

/// Print a human-readable translation table for `--explain` mode.
///
/// Shows each FFmpeg argument and what OxiMedia field/value it mapped to.
/// Exits without executing any transcode.
fn print_explain_table(jobs: &[TranscodeJob]) {
    println!(
        "{} Translation table (--explain mode):",
        "oximedia-ff:".cyan().bold()
    );
    println!();

    for (idx, job) in jobs.iter().enumerate() {
        println!("{} Job {} of {}", "──".dimmed(), idx + 1, jobs.len());
        println!("  {:22} = {}", "-i".yellow(), job.input_path);
        println!("  {:22} = {}", "<output>".yellow(), job.output_path);

        if let Some(vc) = &job.video_codec {
            println!("  {:22} = {}", "-c:v".yellow(), vc);
        }
        if let Some(ac) = &job.audio_codec {
            println!("  {:22} = {}", "-c:a".yellow(), ac);
        }
        if let Some(vb) = &job.video_bitrate {
            println!("  {:22} = {}", "-b:v".yellow(), vb);
        }
        if let Some(ab) = &job.audio_bitrate {
            println!("  {:22} = {}", "-b:a".yellow(), ab);
        }
        if let Some(crf) = job.crf {
            println!("  {:22} = {:.1}", "-crf".yellow(), crf);
        }
        if !job.video_filters.is_empty() {
            println!(
                "  {:22} = {} filter(s)",
                "-vf".yellow(),
                job.video_filters.len()
            );
        }
        if !job.audio_filters.is_empty() {
            println!(
                "  {:22} = {} filter(s)",
                "-af".yellow(),
                job.audio_filters.len()
            );
        }
        if let Some(seek) = &job.seek {
            println!("  {:22} = {}", "-ss".yellow(), seek);
        }
        if let Some(dur) = &job.duration {
            println!("  {:22} = {}", "-t".yellow(), dur);
        }
        if let Some(fmt) = &job.format {
            println!("  {:22} = {}", "-f".yellow(), fmt);
        }
        if let Some(preset) = &job.preset {
            println!("  {:22} = {}", "-preset".yellow(), preset);
        }
        if let Some(tune) = &job.tune {
            println!("  {:22} = {}", "-tune".yellow(), tune);
        }
        if let Some(profile) = &job.profile {
            println!("  {:22} = {}", "-profile:v".yellow(), profile);
        }
        if let Some(pass) = job.pass {
            println!("  {:22} = {}", "-pass".yellow(), pass);
        }
        if job.overwrite {
            println!("  {:22} = yes", "-y".yellow());
        }
        if job.no_video {
            println!("  {:22} = yes", "-vn".yellow());
        }
        if job.no_audio {
            println!("  {:22} = yes", "-an".yellow());
        }
        if !job.map.is_empty() {
            println!("  {:22} = {} selector(s)", "-map".yellow(), job.map.len());
        }
        for (k, v) in &job.metadata {
            println!("  {:22} = {}={}", "-metadata".yellow(), k, v);
        }
        if !job.map_metadata.is_empty() {
            println!(
                "  {:22} = {} directive(s)",
                "-map_metadata".yellow(),
                job.map_metadata.len()
            );
        }
        if let Some(hw) = &job.hwaccel {
            println!(
                "  {:22} = {} ({})",
                "-hwaccel".yellow(),
                hw.backend,
                hw.description
            );
        }
        if !job.muxer_options.is_empty() {
            println!(
                "  {:22} = {} option(s)",
                "muxer opts".yellow(),
                job.muxer_options.len()
            );
        }
        println!();
    }

    eprintln!(
        "{} Use --dry-run to skip execution without the full translation table.",
        "oximedia-ff: note:".cyan()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Help text
// ─────────────────────────────────────────────────────────────────────────────

fn print_help() {
    println!(
        "{}",
        "oximedia-ff  —  FFmpeg-compatible OxiMedia front-end".bold()
    );
    println!();
    println!("Usage: oximedia-ff [options] -i <input> [options] <output>");
    println!();
    println!("Supported codecs (patent-free only — patent codecs are auto-substituted):");
    println!("  Video  (direct): av1 / libaom-av1, vp9 / libvpx-vp9, vp8 / libvpx");
    println!("  Video  (subst.): libx264/h264 → av1, libx265/hevc → av1, mpeg4 → av1");
    println!("  Audio  (direct): opus / libopus, vorbis / libvorbis, flac, pcm_*");
    println!("  Audio  (subst.): aac → opus, mp3/libmp3lame → opus, ac3 → flac");
    println!();
    println!("Options:");
    println!("  -y                  Overwrite output without asking");
    println!("  -n                  Never overwrite output");
    println!("  --dry-run / --plan  Print plan without executing");
    println!("  --explain           Print arg→field translation table and exit");
    println!(
        "  --json              Print structured JSON of the translation and exit (no execution)"
    );
    println!("  -i <path>           Input file");
    println!("  -o <path>           Output file (alias for positional output)");
    println!("  -c:v / -vcodec      Video codec");
    println!("  -c:a / -acodec      Audio codec");
    println!("  -c:s / -scodec      Subtitle codec");
    println!("  -b:v <rate>         Video bitrate (e.g. 2M, 500k)");
    println!("  -b:a <rate>         Audio bitrate (e.g. 128k)");
    println!("  -crf <n>            Quality (CRF) value");
    println!("  -vf <filter>        Video filter graph");
    println!("  -af <filter>        Audio filter graph");
    println!("  -filter_complex <g> Complex filter graph");
    println!("  -r <fps>            Frame rate");
    println!("  -ar <hz>            Audio sample rate");
    println!("  -ac <n>             Audio channel count");
    println!("  -s <WxH>            Video resolution");
    println!("  -ss <time>          Seek position");
    println!("  -t <duration>       Duration");
    println!("  -map <spec>         Stream mapping");
    println!("  -vn / -an / -sn     Disable video / audio / subtitle");
    println!("  -shortest           Stop when shortest stream ends");
    println!("  -metadata key=val   Set output metadata");
    println!("  -f <format>         Force container format");
    println!("  -threads <n>        Thread count");
    println!("  -hwaccel <method>   Hardware acceleration (parsed, not executed)");
    println!("  -loglevel <level>   Log level");
    println!();
    println!("Supported video filters (-vf / -filter_complex):");
    println!("  scale=W:H, crop=w:h:x:y, fps=N, hflip, vflip, rotate=angle");
    println!("  yadif/bwdif (deinterlace), eq=brightness:contrast:saturation");
    println!("  lut3d=file=x.cube, subtitles=file=x.srt");
    println!();
    println!("Supported audio filters (-af):");
    println!("  loudnorm=I=L:TP=T:LRA=R, volume=N/NdB");
    println!("  aresample=N, acompressor=threshold=T:ratio=R");
}
