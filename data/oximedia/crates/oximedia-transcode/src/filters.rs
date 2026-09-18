//! Video and audio filter chains for transcode operations.

use std::fmt;

/// Video filter chain.
#[derive(Debug, Clone)]
pub struct VideoFilter {
    filters: Vec<FilterNode>,
}

/// Audio filter chain.
#[derive(Debug, Clone)]
pub struct AudioFilter {
    filters: Vec<FilterNode>,
}

/// A single filter node in a filter chain.
#[derive(Debug, Clone)]
pub struct FilterNode {
    /// Filter name.
    pub name: String,
    /// Filter parameters.
    pub params: Vec<(String, String)>,
}

impl VideoFilter {
    /// Creates a new empty video filter chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Adds a scale filter.
    #[must_use]
    pub fn scale(mut self, width: u32, height: u32) -> Self {
        self.filters.push(FilterNode {
            name: "scale".to_string(),
            params: vec![
                ("width".to_string(), width.to_string()),
                ("height".to_string(), height.to_string()),
            ],
        });
        self
    }

    /// Adds a deinterlace filter.
    #[must_use]
    pub fn deinterlace(mut self) -> Self {
        self.filters.push(FilterNode {
            name: "yadif".to_string(),
            params: vec![("mode".to_string(), "1".to_string())],
        });
        self
    }

    /// Adds a crop filter.
    #[must_use]
    pub fn crop(mut self, width: u32, height: u32, x: u32, y: u32) -> Self {
        self.filters.push(FilterNode {
            name: "crop".to_string(),
            params: vec![
                ("width".to_string(), width.to_string()),
                ("height".to_string(), height.to_string()),
                ("x".to_string(), x.to_string()),
                ("y".to_string(), y.to_string()),
            ],
        });
        self
    }

    /// Adds a pad filter.
    #[must_use]
    pub fn pad(mut self, width: u32, height: u32, x: u32, y: u32, color: &str) -> Self {
        self.filters.push(FilterNode {
            name: "pad".to_string(),
            params: vec![
                ("width".to_string(), width.to_string()),
                ("height".to_string(), height.to_string()),
                ("x".to_string(), x.to_string()),
                ("y".to_string(), y.to_string()),
                ("color".to_string(), color.to_string()),
            ],
        });
        self
    }

    /// Adds a denoise filter.
    #[must_use]
    pub fn denoise(mut self, strength: f32) -> Self {
        self.filters.push(FilterNode {
            name: "hqdn3d".to_string(),
            params: vec![("luma_spatial".to_string(), strength.to_string())],
        });
        self
    }

    /// Adds a sharpen filter.
    #[must_use]
    pub fn sharpen(mut self, amount: f32) -> Self {
        self.filters.push(FilterNode {
            name: "unsharp".to_string(),
            params: vec![("luma_amount".to_string(), amount.to_string())],
        });
        self
    }

    /// Adds a color correction filter.
    #[must_use]
    pub fn color_correct(mut self, brightness: f32, contrast: f32, saturation: f32) -> Self {
        self.filters.push(FilterNode {
            name: "eq".to_string(),
            params: vec![
                ("brightness".to_string(), brightness.to_string()),
                ("contrast".to_string(), contrast.to_string()),
                ("saturation".to_string(), saturation.to_string()),
            ],
        });
        self
    }

    /// Adds a framerate conversion filter.
    #[must_use]
    pub fn framerate(mut self, fps: f64) -> Self {
        self.filters.push(FilterNode {
            name: "fps".to_string(),
            params: vec![("fps".to_string(), fps.to_string())],
        });
        self
    }

    /// Adds a rotate filter.
    #[must_use]
    pub fn rotate(mut self, degrees: f64) -> Self {
        self.filters.push(FilterNode {
            name: "rotate".to_string(),
            params: vec![("angle".to_string(), degrees.to_string())],
        });
        self
    }

    /// Adds a flip filter (vertical).
    #[must_use]
    pub fn vflip(mut self) -> Self {
        self.filters.push(FilterNode {
            name: "vflip".to_string(),
            params: Vec::new(),
        });
        self
    }

    /// Adds a flip filter (horizontal).
    #[must_use]
    pub fn hflip(mut self) -> Self {
        self.filters.push(FilterNode {
            name: "hflip".to_string(),
            params: Vec::new(),
        });
        self
    }

    /// Adds a custom filter.
    #[must_use]
    pub fn custom(mut self, name: impl Into<String>, params: Vec<(String, String)>) -> Self {
        self.filters.push(FilterNode {
            name: name.into(),
            params,
        });
        self
    }

    /// Converts the filter chain to a filter string.
    #[must_use]
    pub fn to_string(&self) -> String {
        self.filters
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Gets the number of filters in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Checks if the filter chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl Default for VideoFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter {
    /// Creates a new empty audio filter chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Adds a volume filter.
    #[must_use]
    pub fn volume(mut self, volume: f32) -> Self {
        self.filters.push(FilterNode {
            name: "volume".to_string(),
            params: vec![("volume".to_string(), volume.to_string())],
        });
        self
    }

    /// Adds a resample filter.
    #[must_use]
    pub fn resample(mut self, sample_rate: u32) -> Self {
        self.filters.push(FilterNode {
            name: "aresample".to_string(),
            params: vec![("sample_rate".to_string(), sample_rate.to_string())],
        });
        self
    }

    /// Adds a channel layout filter.
    #[must_use]
    pub fn channel_layout(mut self, layout: &str) -> Self {
        self.filters.push(FilterNode {
            name: "aformat".to_string(),
            params: vec![("channel_layouts".to_string(), layout.to_string())],
        });
        self
    }

    /// Adds an audio normalization filter.
    #[must_use]
    pub fn normalize(mut self, target_lufs: f64) -> Self {
        self.filters.push(FilterNode {
            name: "loudnorm".to_string(),
            params: vec![("I".to_string(), target_lufs.to_string())],
        });
        self
    }

    /// Adds a low-pass filter.
    #[must_use]
    pub fn lowpass(mut self, frequency: f64) -> Self {
        self.filters.push(FilterNode {
            name: "lowpass".to_string(),
            params: vec![("frequency".to_string(), frequency.to_string())],
        });
        self
    }

    /// Adds a high-pass filter.
    #[must_use]
    pub fn highpass(mut self, frequency: f64) -> Self {
        self.filters.push(FilterNode {
            name: "highpass".to_string(),
            params: vec![("frequency".to_string(), frequency.to_string())],
        });
        self
    }

    /// Adds an equalizer filter.
    #[must_use]
    pub fn equalizer(mut self, frequency: f64, width: f64, gain: f64) -> Self {
        self.filters.push(FilterNode {
            name: "equalizer".to_string(),
            params: vec![
                ("frequency".to_string(), frequency.to_string()),
                ("width_type".to_string(), "h".to_string()),
                ("width".to_string(), width.to_string()),
                ("gain".to_string(), gain.to_string()),
            ],
        });
        self
    }

    /// Adds a compressor filter.
    #[must_use]
    pub fn compress(mut self, threshold: f64, ratio: f64) -> Self {
        self.filters.push(FilterNode {
            name: "acompressor".to_string(),
            params: vec![
                ("threshold".to_string(), threshold.to_string()),
                ("ratio".to_string(), ratio.to_string()),
            ],
        });
        self
    }

    /// Adds a delay filter.
    #[must_use]
    pub fn delay(mut self, milliseconds: u32) -> Self {
        self.filters.push(FilterNode {
            name: "adelay".to_string(),
            params: vec![("delays".to_string(), format!("{milliseconds}ms"))],
        });
        self
    }

    /// Adds a fade-in filter.
    #[must_use]
    pub fn fade_in(mut self, duration: f64) -> Self {
        self.filters.push(FilterNode {
            name: "afade".to_string(),
            params: vec![
                ("type".to_string(), "in".to_string()),
                ("duration".to_string(), duration.to_string()),
            ],
        });
        self
    }

    /// Adds a fade-out filter.
    #[must_use]
    pub fn fade_out(mut self, start: f64, duration: f64) -> Self {
        self.filters.push(FilterNode {
            name: "afade".to_string(),
            params: vec![
                ("type".to_string(), "out".to_string()),
                ("start_time".to_string(), start.to_string()),
                ("duration".to_string(), duration.to_string()),
            ],
        });
        self
    }

    /// Adds a custom filter.
    #[must_use]
    pub fn custom(mut self, name: impl Into<String>, params: Vec<(String, String)>) -> Self {
        self.filters.push(FilterNode {
            name: name.into(),
            params,
        });
        self
    }

    /// Converts the filter chain to a filter string.
    #[must_use]
    pub fn to_string(&self) -> String {
        self.filters
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Gets the number of filters in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Checks if the filter chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl Default for AudioFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl FilterNode {
    /// Creates a new filter node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
        }
    }

    /// Adds a parameter to the filter.
    #[must_use]
    pub fn param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.push((key.into(), value.into()));
        self
    }
}

impl fmt::Display for FilterNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.params.is_empty() {
            write!(f, "{}", self.name)
        } else {
            let params = self
                .params
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(":");
            write!(f, "{}={}", self.name, params)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_filter_scale() {
        let filter = VideoFilter::new().scale(1920, 1080);
        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_video_filter_chain() {
        let filter = VideoFilter::new()
            .scale(1920, 1080)
            .deinterlace()
            .denoise(3.0)
            .sharpen(1.5);

        assert_eq!(filter.len(), 4);
        let filter_str = filter.to_string();
        assert!(filter_str.contains("scale"));
        assert!(filter_str.contains("yadif"));
    }

    #[test]
    fn test_video_filter_crop() {
        let filter = VideoFilter::new().crop(1920, 1080, 0, 0);
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_video_filter_color_correct() {
        let filter = VideoFilter::new().color_correct(0.1, 1.2, 1.0);
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_video_filter_rotate() {
        let filter = VideoFilter::new().rotate(90.0);
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_video_filter_flip() {
        let filter = VideoFilter::new().vflip().hflip();
        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn test_audio_filter_volume() {
        let filter = AudioFilter::new().volume(1.5);
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_audio_filter_resample() {
        let filter = AudioFilter::new().resample(48000);
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_audio_filter_normalize() {
        let filter = AudioFilter::new().normalize(-23.0);
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_audio_filter_chain() {
        let filter = AudioFilter::new()
            .volume(1.0)
            .resample(48000)
            .normalize(-23.0)
            .compress(-20.0, 3.0);

        assert_eq!(filter.len(), 4);
    }

    #[test]
    fn test_audio_filter_eq() {
        let filter = AudioFilter::new()
            .equalizer(100.0, 200.0, 5.0)
            .equalizer(1000.0, 200.0, -3.0);

        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn test_audio_filter_fade() {
        let filter = AudioFilter::new().fade_in(2.0).fade_out(58.0, 2.0);

        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn test_filter_node_display() {
        let node = FilterNode {
            name: "scale".to_string(),
            params: vec![
                ("width".to_string(), "1920".to_string()),
                ("height".to_string(), "1080".to_string()),
            ],
        };

        let display = format!("{node}");
        assert!(display.contains("scale"));
        assert!(display.contains("width=1920"));
        assert!(display.contains("height=1080"));
    }

    #[test]
    fn test_empty_filter_chain() {
        let video_filter = VideoFilter::new();
        assert!(video_filter.is_empty());
        assert_eq!(video_filter.len(), 0);

        let audio_filter = AudioFilter::new();
        assert!(audio_filter.is_empty());
        assert_eq!(audio_filter.len(), 0);
    }
}
