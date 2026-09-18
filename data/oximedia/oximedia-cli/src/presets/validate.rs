//! Preset validation logic.
//!
//! Validates preset configurations for correctness and compatibility.

use super::{AudioConfig, Preset, VideoConfig};
use anyhow::{anyhow, Result};

/// Validate a complete preset.
pub fn validate_preset(preset: &Preset) -> Result<()> {
    // Validate name
    if preset.name.is_empty() {
        return Err(anyhow!("Preset name cannot be empty"));
    }

    if !preset
        .name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(anyhow!(
            "Preset name must contain only alphanumeric characters, hyphens, and underscores"
        ));
    }

    // Validate description
    if preset.description.is_empty() {
        return Err(anyhow!("Preset description cannot be empty"));
    }

    // Validate video config
    validate_video_config(&preset.video)?;

    // Validate audio config
    validate_audio_config(&preset.audio)?;

    // Validate container
    validate_container(&preset.container)?;

    Ok(())
}

/// Validate video configuration.
pub fn validate_video_config(video: &VideoConfig) -> Result<()> {
    // Validate codec
    validate_video_codec(&video.codec)?;

    // Validate bitrate if specified
    if let Some(ref bitrate) = video.bitrate {
        validate_bitrate_format(bitrate)?;
    }

    // Validate CRF if specified
    if let Some(crf) = video.crf {
        validate_crf(crf, &video.codec)?;
    }

    // Validate dimensions if specified
    if let Some(width) = video.width {
        if width == 0 || width > 8192 {
            return Err(anyhow!(
                "Video width must be between 1 and 8192, got {}",
                width
            ));
        }
        if width % 2 != 0 {
            return Err(anyhow!("Video width must be even, got {}", width));
        }
    }

    if let Some(height) = video.height {
        if height == 0 || height > 8192 {
            return Err(anyhow!(
                "Video height must be between 1 and 8192, got {}",
                height
            ));
        }
        if height % 2 != 0 {
            return Err(anyhow!("Video height must be even, got {}", height));
        }
    }

    // Validate frame rate if specified
    if let Some(fps) = video.fps {
        if fps <= 0.0 || fps > 240.0 {
            return Err(anyhow!("Frame rate must be between 0 and 240, got {}", fps));
        }
    }

    // Validate preset if specified
    if let Some(ref preset) = video.preset {
        validate_encoder_preset(preset)?;
    }

    // Validate pixel format if specified
    if let Some(ref pix_fmt) = video.pixel_format {
        validate_pixel_format(pix_fmt)?;
    }

    // Validate bitrate settings
    if let Some(ref max_bitrate) = video.max_bitrate {
        validate_bitrate_format(max_bitrate)?;
    }

    if let Some(ref min_bitrate) = video.min_bitrate {
        validate_bitrate_format(min_bitrate)?;
    }

    if let Some(ref buffer_size) = video.buffer_size {
        validate_bitrate_format(buffer_size)?;
    }

    // Validate keyframe intervals
    if let Some(keyframe) = video.keyframe_interval {
        if keyframe == 0 || keyframe > 600 {
            return Err(anyhow!(
                "Keyframe interval must be between 1 and 600, got {}",
                keyframe
            ));
        }
    }

    if let Some(min_keyframe) = video.min_keyframe_interval {
        if min_keyframe == 0 || min_keyframe > 600 {
            return Err(anyhow!(
                "Minimum keyframe interval must be between 1 and 600, got {}",
                min_keyframe
            ));
        }
    }

    // Validate aspect ratio if specified
    if let Some(ref aspect) = video.aspect_ratio {
        validate_aspect_ratio(aspect)?;
    }

    Ok(())
}

/// Validate audio configuration.
pub fn validate_audio_config(audio: &AudioConfig) -> Result<()> {
    // Validate codec
    validate_audio_codec(&audio.codec)?;

    // Validate bitrate if specified
    if let Some(ref bitrate) = audio.bitrate {
        validate_bitrate_format(bitrate)?;
    }

    // Validate sample rate if specified
    if let Some(sample_rate) = audio.sample_rate {
        let valid_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000];
        if !valid_rates.contains(&sample_rate) {
            return Err(anyhow!(
                "Invalid sample rate: {}. Common rates are: {:?}",
                sample_rate,
                valid_rates
            ));
        }
    }

    // Validate channels if specified
    if let Some(channels) = audio.channels {
        if channels == 0 || channels > 8 {
            return Err(anyhow!(
                "Audio channels must be between 1 and 8, got {}",
                channels
            ));
        }
    }

    // Validate quality if specified
    if let Some(quality) = audio.quality {
        if quality < 0.0 || quality > 10.0 {
            return Err(anyhow!(
                "Audio quality must be between 0 and 10, got {}",
                quality
            ));
        }
    }

    // Validate compression level if specified
    if let Some(level) = audio.compression_level {
        if level > 12 {
            return Err(anyhow!(
                "Compression level must be between 0 and 12, got {}",
                level
            ));
        }
    }

    Ok(())
}

/// Validate video codec name.
fn validate_video_codec(codec: &str) -> Result<()> {
    match codec.to_lowercase().as_str() {
        "av1" | "vp9" | "vp8" | "theora" => Ok(()),
        _ => Err(anyhow!(
            "Unsupported video codec '{}'. Supported: av1, vp9, vp8, theora",
            codec
        )),
    }
}

/// Validate audio codec name.
fn validate_audio_codec(codec: &str) -> Result<()> {
    match codec.to_lowercase().as_str() {
        "opus" | "vorbis" | "flac" | "pcm" => Ok(()),
        _ => Err(anyhow!(
            "Unsupported audio codec '{}'. Supported: opus, vorbis, flac, pcm",
            codec
        )),
    }
}

/// Validate container format.
fn validate_container(container: &str) -> Result<()> {
    match container.to_lowercase().as_str() {
        "webm" | "mkv" | "ogg" | "flac" | "wav" => Ok(()),
        _ => Err(anyhow!(
            "Unsupported container '{}'. Supported: webm, mkv, ogg, flac, wav",
            container
        )),
    }
}

/// Validate bitrate format string.
fn validate_bitrate_format(bitrate: &str) -> Result<()> {
    let bitrate = bitrate.trim();
    let has_suffix = bitrate.ends_with('M')
        || bitrate.ends_with('m')
        || bitrate.ends_with('K')
        || bitrate.ends_with('k');

    let numeric = if has_suffix {
        bitrate.trim_end_matches(|c: char| c.is_alphabetic())
    } else {
        bitrate
    };

    numeric.trim().parse::<f64>().map_err(|_| {
        anyhow!(
            "Invalid bitrate format '{}'. Use format like '5M', '128k', or '1000'",
            bitrate
        )
    })?;

    Ok(())
}

/// Validate CRF value for codec.
fn validate_crf(crf: u32, codec: &str) -> Result<()> {
    let max = match codec.to_lowercase().as_str() {
        "av1" => 255,
        "vp9" | "vp8" => 63,
        "theora" => 63,
        _ => return Err(anyhow!("Unknown codec for CRF validation: {}", codec)),
    };

    if crf > max {
        Err(anyhow!(
            "CRF {} is out of range for {} (max: {})",
            crf,
            codec,
            max
        ))
    } else {
        Ok(())
    }
}

/// Validate encoder preset name.
fn validate_encoder_preset(preset: &str) -> Result<()> {
    match preset.to_lowercase().as_str() {
        "ultrafast" | "superfast" | "veryfast" | "faster" | "fast" | "medium" | "slow"
        | "slower" | "veryslow" => Ok(()),
        _ => Err(anyhow!(
            "Invalid encoder preset '{}'. Valid presets: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow",
            preset
        )),
    }
}

/// Validate pixel format.
fn validate_pixel_format(pix_fmt: &str) -> Result<()> {
    match pix_fmt.to_lowercase().as_str() {
        "yuv420p" | "yuv422p" | "yuv444p" | "yuv420p10le" | "yuv422p10le" | "yuv444p10le"
        | "yuv420p12le" | "yuv422p12le" | "yuv444p12le" => Ok(()),
        _ => Err(anyhow!(
            "Unsupported pixel format '{}'. Common formats: yuv420p, yuv422p, yuv444p",
            pix_fmt
        )),
    }
}

/// Validate aspect ratio format.
fn validate_aspect_ratio(aspect: &str) -> Result<()> {
    let parts: Vec<&str> = aspect.split(':').collect();
    if parts.len() != 2 {
        return Err(anyhow!(
            "Invalid aspect ratio format '{}'. Use format like '16:9' or '4:3'",
            aspect
        ));
    }

    let width: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid aspect ratio width in '{}'", aspect))?;

    let height: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid aspect ratio height in '{}'", aspect))?;

    if width == 0 || height == 0 {
        return Err(anyhow!(
            "Aspect ratio dimensions must be positive in '{}'",
            aspect
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_bitrate_format() {
        assert!(validate_bitrate_format("5M").is_ok());
        assert!(validate_bitrate_format("2.5M").is_ok());
        assert!(validate_bitrate_format("128k").is_ok());
        assert!(validate_bitrate_format("1000").is_ok());
        assert!(validate_bitrate_format("invalid").is_err());
    }

    #[test]
    fn test_validate_video_codec() {
        assert!(validate_video_codec("av1").is_ok());
        assert!(validate_video_codec("vp9").is_ok());
        assert!(validate_video_codec("vp8").is_ok());
        assert!(validate_video_codec("theora").is_ok());
        assert!(validate_video_codec("h264").is_err());
    }

    #[test]
    fn test_validate_audio_codec() {
        assert!(validate_audio_codec("opus").is_ok());
        assert!(validate_audio_codec("vorbis").is_ok());
        assert!(validate_audio_codec("flac").is_ok());
        assert!(validate_audio_codec("pcm").is_ok());
        assert!(validate_audio_codec("aac").is_err());
    }

    #[test]
    fn test_validate_container() {
        assert!(validate_container("webm").is_ok());
        assert!(validate_container("mkv").is_ok());
        assert!(validate_container("ogg").is_ok());
        assert!(validate_container("mp4").is_err());
    }

    #[test]
    fn test_validate_aspect_ratio() {
        assert!(validate_aspect_ratio("16:9").is_ok());
        assert!(validate_aspect_ratio("4:3").is_ok());
        assert!(validate_aspect_ratio("21:9").is_ok());
        assert!(validate_aspect_ratio("invalid").is_err());
        assert!(validate_aspect_ratio("16:").is_err());
        assert!(validate_aspect_ratio("0:9").is_err());
    }
}
