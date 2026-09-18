//! Preset subcommand handler (list, show, create, template, import, export, remove).

use anyhow::Result;
use colored::Colorize;

use crate::commands::PresetCommand;
use crate::presets;

/// Handle preset subcommands.
pub(crate) async fn handle_preset_command(command: PresetCommand, json_output: bool) -> Result<()> {
    use presets::{PresetCategory, PresetManager};

    let custom_dir = PresetManager::default_custom_dir()?;
    let manager = PresetManager::with_custom_dir(&custom_dir)?;

    match command {
        PresetCommand::List {
            category,
            detail: verbose,
        } => {
            let presets = if let Some(cat_str) = category {
                let cat = PresetCategory::from_str(&cat_str)?;
                manager.list_presets_by_category(cat)
            } else {
                manager.list_presets()
            };

            if json_output {
                let json = serde_json::to_string_pretty(&presets)?;
                println!("{}", json);
            } else {
                println!("{}", "Available Presets".green().bold());
                println!("{}", "=".repeat(80));
                println!();

                let mut current_category = None;
                for preset in presets {
                    if current_category != Some(preset.category) {
                        current_category = Some(preset.category);
                        println!("{}", preset.category.name().cyan().bold());
                        println!("{}", preset.category.description().dimmed());
                        println!();
                    }

                    let builtin_badge = if preset.builtin {
                        "[built-in]".dimmed()
                    } else {
                        "[custom]".yellow()
                    };

                    println!("  {} {}", preset.name.green(), builtin_badge);

                    if verbose {
                        println!("    {}", preset.description);
                        println!(
                            "    Video: {} @ {}",
                            preset.video.codec,
                            preset
                                .video
                                .bitrate
                                .as_ref()
                                .map(|s| s.as_str())
                                .unwrap_or("CRF")
                        );
                        println!(
                            "    Audio: {} @ {}",
                            preset.audio.codec,
                            preset
                                .audio
                                .bitrate
                                .as_ref()
                                .map(|s| s.as_str())
                                .unwrap_or("default")
                        );
                        println!("    Container: {}", preset.container);
                        if !preset.tags.is_empty() {
                            println!("    Tags: {}", preset.tags.join(", "));
                        }
                        println!();
                    }
                }

                println!();
                println!("Total: {} presets", manager.preset_names().len());
                println!();
                println!(
                    "Use {} to see detailed information",
                    "oximedia preset show <name>".yellow()
                );
            }

            Ok(())
        }

        PresetCommand::Show { name, toml } => {
            let preset = manager.get_preset(&name)?;

            if json_output {
                let json = serde_json::to_string_pretty(preset)?;
                println!("{}", json);
            } else if toml {
                // Save to temp and read back
                let temp_dir = std::env::temp_dir();
                presets::custom::save_preset_to_file(preset, &temp_dir)?;
                let toml_path = temp_dir.join(format!("{}.toml", preset.name));
                let toml_content = std::fs::read_to_string(&toml_path)?;
                println!("{}", toml_content);
                let _ignore = std::fs::remove_file(&toml_path);
            } else {
                println!("{}", format!("Preset: {}", preset.name).green().bold());
                println!("{}", "=".repeat(80));
                println!();

                println!("{}: {}", "Description".cyan().bold(), preset.description);
                println!("{}: {}", "Category".cyan().bold(), preset.category.name());
                println!("{}: {}", "Container".cyan().bold(), preset.container);
                println!(
                    "{}: {}",
                    "Type".cyan().bold(),
                    if preset.builtin { "Built-in" } else { "Custom" }
                );

                if !preset.tags.is_empty() {
                    println!("{}: {}", "Tags".cyan().bold(), preset.tags.join(", "));
                }

                println!();
                println!("{}", "Video Configuration".yellow().bold());
                println!("{}", "-".repeat(40));
                println!("  Codec: {}", preset.video.codec);
                if let Some(ref bitrate) = preset.video.bitrate {
                    println!("  Bitrate: {}", bitrate);
                }
                if let Some(crf) = preset.video.crf {
                    println!("  CRF: {}", crf);
                }
                if let Some(width) = preset.video.width {
                    println!(
                        "  Resolution: {}x{}",
                        width,
                        preset.video.height.unwrap_or(0)
                    );
                }
                if let Some(fps) = preset.video.fps {
                    println!("  Frame rate: {}", fps);
                }
                if let Some(ref preset_name) = preset.video.preset {
                    println!("  Encoder preset: {}", preset_name);
                }
                if let Some(ref pix_fmt) = preset.video.pixel_format {
                    println!("  Pixel format: {}", pix_fmt);
                }
                println!("  Two-pass: {}", preset.video.two_pass);

                println!();
                println!("{}", "Audio Configuration".yellow().bold());
                println!("{}", "-".repeat(40));
                println!("  Codec: {}", preset.audio.codec);
                if let Some(ref bitrate) = preset.audio.bitrate {
                    println!("  Bitrate: {}", bitrate);
                }
                if let Some(sample_rate) = preset.audio.sample_rate {
                    println!("  Sample rate: {} Hz", sample_rate);
                }
                if let Some(channels) = preset.audio.channels {
                    println!("  Channels: {}", channels);
                }

                println!();
                println!(
                    "{}",
                    format!(
                        "oximedia transcode -i input.mkv -o output.{} --preset-name {}",
                        preset.container, preset.name
                    )
                    .yellow()
                );
            }

            Ok(())
        }

        PresetCommand::Create { output } => {
            let preset = presets::custom::create_preset_interactive()?;

            let out_dir = output.unwrap_or(custom_dir);
            if !out_dir.exists() {
                std::fs::create_dir_all(&out_dir)?;
            }

            presets::custom::save_preset_to_file(&preset, &out_dir)?;

            println!(
                "{} Preset '{}' created successfully!",
                "✓".green(),
                preset.name
            );
            println!(
                "Saved to: {}",
                out_dir.join(format!("{}.toml", preset.name)).display()
            );

            Ok(())
        }

        PresetCommand::Template { output } => {
            presets::custom::generate_template(&output)?;
            println!("{} Template generated: {}", "✓".green(), output.display());
            println!(
                "Edit the template and import it with: oximedia preset import {}",
                output.display()
            );
            Ok(())
        }

        PresetCommand::Import { file } => {
            let preset = presets::custom::load_preset_from_file(&file)?;

            if !custom_dir.exists() {
                std::fs::create_dir_all(&custom_dir)?;
            }

            presets::custom::save_preset_to_file(&preset, &custom_dir)?;

            println!(
                "{} Preset '{}' imported successfully!",
                "✓".green(),
                preset.name
            );

            Ok(())
        }

        PresetCommand::Export { name, output } => {
            let preset = manager.get_preset(&name)?;

            if preset.builtin {
                println!(
                    "{} Cannot export built-in preset '{}'. Use 'oximedia preset show {} --toml' instead.",
                    "!".yellow(),
                    name,
                    name
                );
                return Ok(());
            }

            let output_dir = output.parent().unwrap_or_else(|| std::path::Path::new("."));
            presets::custom::save_preset_to_file(preset, output_dir)?;

            println!("{} Preset exported to: {}", "✓".green(), output.display());

            Ok(())
        }

        PresetCommand::Remove { name } => {
            let preset = manager.get_preset(&name)?;

            if preset.builtin {
                return Err(anyhow::anyhow!("Cannot remove built-in preset '{}'", name));
            }

            let preset_path = custom_dir.join(format!("{}.toml", name));
            if preset_path.exists() {
                std::fs::remove_file(&preset_path)?;
                println!("{} Preset '{}' removed successfully!", "✓".green(), name);
            } else {
                println!(
                    "{} Preset '{}' not found in custom directory",
                    "!".yellow(),
                    name
                );
            }

            Ok(())
        }
    }
}
