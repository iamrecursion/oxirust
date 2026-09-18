//! Music Information Retrieval CLI commands.
//!
//! Provides commands for tempo detection, key detection, structural
//! segmentation, chord analysis, and full MIR reports.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

use oximedia_mir::{MirAnalyzer, MirConfig};

/// MIR subcommands.
#[derive(Subcommand, Debug)]
pub enum MirCommand {
    /// Detect tempo/BPM of audio
    Tempo {
        /// Input audio file
        input: PathBuf,

        /// Show detailed tempo analysis (alternative tempos, stability)
        #[arg(long)]
        detailed: bool,
    },

    /// Detect musical key
    Key {
        /// Input audio file
        input: PathBuf,

        /// Key detection algorithm (default: "krumhansl")
        #[arg(long)]
        algorithm: Option<String>,
    },

    /// Segment audio into structural sections
    Segment {
        /// Input audio file
        input: PathBuf,

        /// Output file for segment data (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Minimum segment duration in seconds
        #[arg(long)]
        min_duration: Option<f64>,
    },

    /// Detect chord progression
    Chords {
        /// Input audio file
        input: PathBuf,

        /// Hop size for chord analysis (samples)
        #[arg(long)]
        hop_size: Option<u32>,
    },

    /// Full MIR analysis report
    Analyze {
        /// Input audio file
        input: PathBuf,

        /// Output file for report
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },
}

/// Handle MIR subcommand dispatch.
pub async fn handle_mir_command(command: MirCommand, json_output: bool) -> Result<()> {
    match command {
        MirCommand::Tempo { input, detailed } => handle_tempo(&input, detailed, json_output).await,
        MirCommand::Key { input, algorithm } => {
            handle_key(&input, algorithm.as_deref(), json_output).await
        }
        MirCommand::Segment {
            input,
            output,
            min_duration,
        } => handle_segment(&input, output.as_ref(), min_duration, json_output).await,
        MirCommand::Chords { input, hop_size } => {
            handle_chords(&input, hop_size, json_output).await
        }
        MirCommand::Analyze {
            input,
            output,
            format,
        } => {
            handle_analyze(
                &input,
                output.as_ref(),
                if json_output { "json" } else { &format },
            )
            .await
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decode the real audio content of `input` into mono f32 samples.
///
/// This performs an **actual decode** of the file (never a synthetic stand-in
/// signal): the shared [`crate::decode_helper::decode_wav_f32`] pipeline
/// demuxes and PCM-decodes WAV/RIFF audio, and the interleaved result is
/// downmixed to a single mono channel for MIR analysis.
///
/// Returns `(mono_samples, sample_rate_hz)`.
///
/// # Errors
///
/// Returns an honest error — never a fabricated tone — when:
/// - the file does not exist,
/// - the container/codec is not decodable here (only WAV/RIFF PCM is wired
///   into the CLI audio-decode path; other formats must be converted first),
/// - the decoded stream contains no samples (e.g. a video-only file).
async fn load_audio_samples(input: &PathBuf) -> Result<(Vec<f32>, f32)> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    let decoded = match crate::decode_helper::decode_wav_f32(input.as_path()).await {
        Ok(audio) => audio,
        Err(crate::decode_helper::DecodeError::UnsupportedFormat(msg)) => {
            return Err(anyhow::anyhow!(
                "Cannot decode '{}' for MIR analysis: {msg}. \
                 MIR decodes WAV/RIFF PCM audio; convert the input to WAV first \
                 (for example: `oximedia audio --input {} --output out.wav`).",
                input.display(),
                input.display()
            ));
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to decode audio from '{}': {e}",
                input.display()
            ));
        }
    };

    let mono = downmix_to_mono(&decoded.samples, decoded.channels);
    if mono.is_empty() {
        return Err(anyhow::anyhow!(
            "No audio samples were decoded from '{}' (empty or non-audio stream).",
            input.display()
        ));
    }

    Ok((mono, decoded.sample_rate as f32))
}

/// Average interleaved multi-channel PCM down to a single mono channel.
///
/// Mono input is returned unchanged. For `n`-channel interleaved input each
/// output sample is the arithmetic mean of the `n` channel samples in a frame.
fn downmix_to_mono(interleaved: &[f32], channels: u32) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let ch = channels as usize;
    interleaved
        .chunks(ch)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async fn handle_tempo(input: &PathBuf, detailed: bool, json_output: bool) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input).await?;

    let config = MirConfig {
        enable_beat_tracking: true,
        enable_key_detection: false,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Tempo analysis failed: {e}"))?;

    let tempo = result
        .tempo
        .ok_or_else(|| anyhow::anyhow!("Tempo detection returned no result"))?;

    if json_output {
        let mut value = serde_json::json!({
            "input": input.display().to_string(),
            "bpm": tempo.bpm,
            "confidence": tempo.confidence,
            "stability": tempo.stability,
        });
        if detailed {
            let alts: Vec<serde_json::Value> = tempo
                .alternatives
                .iter()
                .map(|(bpm, conf)| {
                    serde_json::json!({
                        "bpm": bpm,
                        "confidence": conf,
                    })
                })
                .collect();
            value["alternatives"] = serde_json::json!(alts);
        }
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Tempo Detection".green().bold());
        println!("{}", "=".repeat(50));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {:.1} BPM", "Tempo:", tempo.bpm);
        println!("{:20} {:.1}%", "Confidence:", tempo.confidence * 100.0);
        println!("{:20} {:.1}%", "Stability:", tempo.stability * 100.0);

        if detailed && !tempo.alternatives.is_empty() {
            println!();
            println!("{}", "Alternative Tempos:".cyan().bold());
            for (bpm, conf) in &tempo.alternatives {
                println!("  {:.1} BPM (confidence: {:.1}%)", bpm, conf * 100.0);
            }
        }
    }

    Ok(())
}

async fn handle_key(input: &PathBuf, algorithm: Option<&str>, json_output: bool) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input).await?;

    let algo = algorithm.unwrap_or("krumhansl");

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: true,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Key detection failed: {e}"))?;

    let key = result
        .key
        .ok_or_else(|| anyhow::anyhow!("Key detection returned no result"))?;

    if json_output {
        let value = serde_json::json!({
            "input": input.display().to_string(),
            "algorithm": algo,
            "key": key.key,
            "root": key.root,
            "is_major": key.is_major,
            "confidence": key.confidence,
        });
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Key Detection".green().bold());
        println!("{}", "=".repeat(50));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Algorithm:", algo);
        println!("{:20} {}", "Key:", key.key.bold());
        println!(
            "{:20} {}",
            "Mode:",
            if key.is_major { "Major" } else { "Minor" }
        );
        println!("{:20} {:.1}%", "Confidence:", key.confidence * 100.0);
    }

    Ok(())
}

async fn handle_segment(
    input: &PathBuf,
    output: Option<&PathBuf>,
    min_duration: Option<f64>,
    json_output: bool,
) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input).await?;

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: false,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: true,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Segmentation failed: {e}"))?;

    let structure = result
        .structure
        .ok_or_else(|| anyhow::anyhow!("Structure analysis returned no result"))?;

    let min_dur = min_duration.unwrap_or(0.0) as f32;
    let segments: Vec<_> = structure
        .segments
        .iter()
        .filter(|s| (s.end - s.start) >= min_dur)
        .collect();

    if json_output {
        let seg_json: Vec<serde_json::Value> = segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "start": s.start,
                    "end": s.end,
                    "label": s.label,
                    "confidence": s.confidence,
                    "duration": s.end - s.start,
                })
            })
            .collect();

        let value = serde_json::json!({
            "input": input.display().to_string(),
            "min_duration": min_dur,
            "segment_count": segments.len(),
            "complexity": structure.complexity,
            "segments": seg_json,
        });
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;

        if let Some(out_path) = output {
            std::fs::write(out_path, &json_str)
                .context(format!("Failed to write output to {}", out_path.display()))?;
            println!("Segment data written to {}", out_path.display());
        } else {
            println!("{json_str}");
        }
    } else {
        println!("{}", "Audio Segmentation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Segments found:", segments.len());
        println!(
            "{:20} {:.2}",
            "Structural complexity:", structure.complexity
        );

        if !segments.is_empty() {
            println!();
            println!(
                "  {:<12} {:<10} {:<10} {:<12} {}",
                "Label".bold(),
                "Start".bold(),
                "End".bold(),
                "Duration".bold(),
                "Confidence".bold()
            );
            println!("  {}", "-".repeat(56));
            for seg in &segments {
                println!(
                    "  {:<12} {:<10.2} {:<10.2} {:<12.2} {:.1}%",
                    seg.label,
                    seg.start,
                    seg.end,
                    seg.end - seg.start,
                    seg.confidence * 100.0,
                );
            }
        }

        if let Some(out_path) = output {
            let seg_json: Vec<serde_json::Value> = segments
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "start": s.start,
                        "end": s.end,
                        "label": s.label,
                        "confidence": s.confidence,
                    })
                })
                .collect();
            let json_str =
                serde_json::to_string_pretty(&seg_json).context("Failed to serialize segments")?;
            std::fs::write(out_path, &json_str)
                .context(format!("Failed to write output to {}", out_path.display()))?;
            println!();
            println!("Segment data written to {}", out_path.display());
        }
    }

    Ok(())
}

async fn handle_chords(input: &PathBuf, hop_size: Option<u32>, json_output: bool) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input).await?;

    let hop = hop_size.unwrap_or(512);

    let config = MirConfig {
        hop_size: hop as usize,
        enable_beat_tracking: false,
        enable_key_detection: false,
        enable_chord_recognition: true,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Chord detection failed: {e}"))?;

    let chord_result = result
        .chord
        .ok_or_else(|| anyhow::anyhow!("Chord recognition returned no result"))?;

    if json_output {
        let chords_json: Vec<serde_json::Value> = chord_result
            .chords
            .iter()
            .map(|c| {
                serde_json::json!({
                    "start": c.start,
                    "end": c.end,
                    "label": c.label,
                    "confidence": c.confidence,
                })
            })
            .collect();

        let value = serde_json::json!({
            "input": input.display().to_string(),
            "hop_size": hop,
            "chord_count": chord_result.chords.len(),
            "complexity": chord_result.complexity,
            "progressions": chord_result.progressions,
            "chords": chords_json,
        });
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Chord Detection".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Hop size:", hop);
        println!("{:20} {}", "Chords found:", chord_result.chords.len());
        println!(
            "{:20} {:.2}",
            "Harmonic complexity:", chord_result.complexity
        );

        if !chord_result.progressions.is_empty() {
            println!();
            println!("{}", "Chord Progressions:".cyan().bold());
            for prog in &chord_result.progressions {
                println!("  {prog}");
            }
        }

        if !chord_result.chords.is_empty() {
            println!();
            println!(
                "  {:<10} {:<10} {:<12} {}",
                "Start".bold(),
                "End".bold(),
                "Chord".bold(),
                "Confidence".bold()
            );
            println!("  {}", "-".repeat(48));
            // Show first 20 chords to avoid flooding the terminal
            let display_count = chord_result.chords.len().min(20);
            for chord in &chord_result.chords[..display_count] {
                println!(
                    "  {:<10.2} {:<10.2} {:<12} {:.1}%",
                    chord.start,
                    chord.end,
                    chord.label,
                    chord.confidence * 100.0,
                );
            }
            if chord_result.chords.len() > display_count {
                println!(
                    "  ... and {} more chords",
                    chord_result.chords.len() - display_count
                );
            }
        }
    }

    Ok(())
}

async fn handle_analyze(input: &PathBuf, output: Option<&PathBuf>, format: &str) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input).await?;

    // Enable all features for full analysis
    let config = MirConfig::default();
    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("MIR analysis failed: {e}"))?;

    match format {
        "json" => {
            let value = serde_json::json!({
                "input": input.display().to_string(),
                "duration": result.duration,
                "sample_rate": result.sample_rate,
                "tempo": result.tempo.as_ref().map(|t| serde_json::json!({
                    "bpm": t.bpm,
                    "confidence": t.confidence,
                    "stability": t.stability,
                })),
                "key": result.key.as_ref().map(|k| serde_json::json!({
                    "key": k.key,
                    "root": k.root,
                    "is_major": k.is_major,
                    "confidence": k.confidence,
                })),
                "chord": result.chord.as_ref().map(|c| serde_json::json!({
                    "chord_count": c.chords.len(),
                    "complexity": c.complexity,
                    "progressions": c.progressions,
                })),
                "structure": result.structure.as_ref().map(|s| serde_json::json!({
                    "segment_count": s.segments.len(),
                    "complexity": s.complexity,
                    "segments": s.segments.iter().map(|seg| serde_json::json!({
                        "label": seg.label,
                        "start": seg.start,
                        "end": seg.end,
                    })).collect::<Vec<_>>(),
                })),
                "genre": result.genre.as_ref().map(|g| serde_json::json!({
                    "top_genre": g.top_genre_name,
                    "confidence": g.top_genre_confidence,
                })),
                "mood": result.mood.as_ref().map(|m| serde_json::json!({
                    "valence": m.valence,
                    "arousal": m.arousal,
                })),
            });

            let json_str =
                serde_json::to_string_pretty(&value).context("Failed to serialize report")?;

            if let Some(out_path) = output {
                std::fs::write(out_path, &json_str)
                    .context(format!("Failed to write report to {}", out_path.display()))?;
                println!("Report written to {}", out_path.display());
            } else {
                println!("{json_str}");
            }
        }
        _ => {
            println!("{}", "MIR Analysis Report".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:20} {}", "Input:", input.display());
            println!("{:20} {:.2}s", "Duration:", result.duration);
            println!("{:20} {} Hz", "Sample rate:", result.sample_rate);
            println!();

            if let Some(ref tempo) = result.tempo {
                println!("{}", "Tempo".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {:.1} BPM", "BPM:", tempo.bpm);
                println!("  {:<18} {:.1}%", "Confidence:", tempo.confidence * 100.0);
                println!("  {:<18} {:.1}%", "Stability:", tempo.stability * 100.0);
                println!();
            }

            if let Some(ref key) = result.key {
                println!("{}", "Key".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {}", "Detected key:", key.key);
                println!(
                    "  {:<18} {}",
                    "Mode:",
                    if key.is_major { "Major" } else { "Minor" }
                );
                println!("  {:<18} {:.1}%", "Confidence:", key.confidence * 100.0);
                println!();
            }

            if let Some(ref chord) = result.chord {
                println!("{}", "Chords".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {}", "Chord count:", chord.chords.len());
                println!("  {:<18} {:.2}", "Complexity:", chord.complexity);
                if !chord.progressions.is_empty() {
                    println!(
                        "  {:<18} {}",
                        "Progressions:",
                        chord.progressions.join(", ")
                    );
                }
                println!();
            }

            if let Some(ref structure) = result.structure {
                println!("{}", "Structure".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {}", "Segments:", structure.segments.len());
                println!("  {:<18} {:.2}", "Complexity:", structure.complexity);
                for seg in &structure.segments {
                    println!("  {:<18} {:.2}s - {:.2}s", seg.label, seg.start, seg.end);
                }
                println!();
            }

            if let Some(ref genre) = result.genre {
                println!("{}", "Genre".cyan().bold());
                println!("{}", "-".repeat(40));
                let (top, conf) = genre.top_genre();
                println!("  {:<18} {} ({:.1}%)", "Top genre:", top, conf * 100.0);
                println!();
            }

            if let Some(ref mood) = result.mood {
                println!("{}", "Mood".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {:.2}", "Valence:", mood.valence);
                println!("  {:<18} {:.2}", "Arousal:", mood.arousal);
            }

            if let Some(out_path) = output {
                println!();
                println!(
                    "{}",
                    format!("(Use --format json to save to {out_path:?})").dimmed()
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal 16-bit mono PCM WAV whose amplitude is pulsed at
    /// `beat_hz` beats/second — a real click track. Different `beat_hz` values
    /// produce genuinely different onset timing, so a real beat tracker must
    /// report different tempi (a fabricated fixed tone could not).
    fn write_click_wav(name: &str, beat_hz: f32, duration_secs: f32) -> PathBuf {
        let sample_rate: u32 = 44_100;
        let num_samples = (sample_rate as f32 * duration_secs) as u32;
        let bits_per_sample: u16 = 16;
        let channels: u16 = 1;
        let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
        let block_align = channels * (bits_per_sample / 8);
        let data_size = num_samples * u32::from(channels) * u32::from(bits_per_sample / 8);
        let file_size = 36 + data_size;

        let mut buf = Vec::with_capacity(44 + data_size as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            // Fraction of the current beat period [0,1).
            let beat_phase = (t * beat_hz).fract();
            // Short percussive burst (first 5% of each beat) with a linear decay.
            let env = if beat_phase < 0.05 {
                1.0 - beat_phase / 0.05
            } else {
                0.0
            };
            let carrier = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let sample = env * carrier;
            let pcm = (sample * 32_767.0) as i16;
            buf.extend_from_slice(&pcm.to_le_bytes());
        }

        let path =
            std::env::temp_dir().join(format!("oximedia_mir_{}_{}.wav", name, std::process::id()));
        std::fs::write(&path, &buf).expect("write click WAV fixture");
        path
    }

    #[tokio::test]
    async fn test_load_audio_samples_missing_file() {
        let path = PathBuf::from("/nonexistent/audio.wav");
        let result = load_audio_samples(&path).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_audio_samples_real_wav_decodes() {
        let path = write_click_wav("decode", 2.0, 2.0);
        let result = load_audio_samples(&path).await;
        assert!(result.is_ok(), "real WAV must decode: {result:?}");
        let (samples, sr) = result.expect("should load samples");
        assert!(!samples.is_empty());
        assert!((sr - 44100.0).abs() < f32::EPSILON);
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_load_audio_samples_non_wav_errors_honestly() {
        // A non-WAV file must produce an honest error, never a fabricated tone.
        let path =
            std::env::temp_dir().join(format!("oximedia_mir_notawav_{}.mkv", std::process::id()));
        std::fs::write(&path, b"\x1a\x45\xdf\xa3not a wav").expect("write fake mkv");
        let result = load_audio_samples(&path).await;
        assert!(result.is_err(), "non-WAV input must error, not synthesize");
        std::fs::remove_file(&path).ok();
    }

    fn beat_only_config() -> MirConfig {
        MirConfig {
            enable_beat_tracking: true,
            enable_key_detection: false,
            enable_chord_recognition: false,
            enable_melody_extraction: false,
            enable_structure_analysis: false,
            enable_genre_classification: false,
            enable_mood_detection: false,
            enable_spectral_features: false,
            enable_rhythm_features: false,
            enable_harmonic_analysis: false,
            ..MirConfig::default()
        }
    }

    /// Quality-bar proof: two DIFFERENT real audio files (different beat rates
    /// written to WAV) must yield DIFFERENT tempo estimates. The old fabricated
    /// 440 Hz / 2 Hz tone produced identical results for every input.
    #[tokio::test]
    async fn test_tempo_differs_for_different_wavs() {
        let slow = write_click_wav("slow", 2.0, 8.0); // ~120 BPM click track
        let fast = write_click_wav("fast", 3.0, 8.0); // ~180 BPM click track

        let (slow_samples, slow_sr) = load_audio_samples(&slow).await.expect("decode slow WAV");
        let (fast_samples, fast_sr) = load_audio_samples(&fast).await.expect("decode fast WAV");

        let analyzer = MirAnalyzer::new(beat_only_config());
        let slow_res = analyzer
            .analyze(&slow_samples, slow_sr)
            .expect("analyze slow");
        let fast_res = analyzer
            .analyze(&fast_samples, fast_sr)
            .expect("analyze fast");

        let slow_bpm = slow_res.tempo.map(|t| t.bpm);
        let fast_bpm = fast_res.tempo.map(|t| t.bpm);

        std::fs::remove_file(&slow).ok();
        std::fs::remove_file(&fast).ok();

        match (slow_bpm, fast_bpm) {
            (Some(s), Some(f)) => {
                assert!(s.is_finite() && s > 0.0, "slow bpm invalid: {s}");
                assert!(f.is_finite() && f > 0.0, "fast bpm invalid: {f}");
                assert!(
                    (s - f).abs() > 1.0,
                    "tempo must differ for different click tracks: slow={s} fast={f}"
                );
            }
            other => panic!("both files must yield a tempo estimate, got {other:?}"),
        }
    }

    #[test]
    fn test_mir_config_selective() {
        let config = beat_only_config();
        assert!(config.enable_beat_tracking);
        assert!(!config.enable_key_detection);
    }
}
