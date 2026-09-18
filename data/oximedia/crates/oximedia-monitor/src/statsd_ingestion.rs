//! StatsD ingestion endpoint for accepting metrics from external processes.
//!
//! Implements parsing of the StatsD line protocol, allowing `oximedia-monitor`
//! to receive metrics forwarded from legacy or external applications that emit
//! StatsD datagrams.
//!
//! # StatsD Protocol
//!
//! Each line has the form:
//!
//! ```text
//! metric.name:value|type[|@sample_rate][|#tag:val,tag:val]
//! ```
//!
//! Supported metric types:
//!
//! | Type  | Symbol | Description                      |
//! |-------|--------|----------------------------------|
//! | Gauge | `g`    | Instantaneous value              |
//! | Counter | `c`  | Monotonically increasing counter |
//! | Timer | `ms`   | Timing sample (in milliseconds)  |
//! | Histogram | `h` | Distribution sample             |
//! | Set   | `s`    | Unique element count (distinct)  |
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::statsd_ingestion::{StatsdLine, StatsdMetricType, StatsdParser};
//!
//! let line = "cpu.usage:72.5|g";
//! let metric = StatsdParser::parse_line(line).expect("valid line");
//! assert_eq!(metric.name, "cpu.usage");
//! assert!((metric.value - 72.5).abs() < f64::EPSILON);
//! assert_eq!(metric.metric_type, StatsdMetricType::Gauge);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// Metric type
// ---------------------------------------------------------------------------

/// StatsD metric type indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatsdMetricType {
    /// Gauge: `g`.
    Gauge,
    /// Counter: `c`.
    Counter,
    /// Timer: `ms`.
    Timer,
    /// Histogram: `h`.
    Histogram,
    /// Set (unique count): `s`.
    Set,
}

impl StatsdMetricType {
    /// Return the StatsD type symbol.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Gauge => "g",
            Self::Counter => "c",
            Self::Timer => "ms",
            Self::Histogram => "h",
            Self::Set => "s",
        }
    }

    /// Parse from the StatsD type symbol.
    ///
    /// # Errors
    ///
    /// Returns an error if the symbol is unrecognized.
    pub fn from_symbol(s: &str) -> MonitorResult<Self> {
        match s {
            "g" => Ok(Self::Gauge),
            "c" => Ok(Self::Counter),
            "ms" => Ok(Self::Timer),
            "h" => Ok(Self::Histogram),
            "s" => Ok(Self::Set),
            other => Err(MonitorError::Other(format!(
                "unknown StatsD metric type: '{other}'"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsed line
// ---------------------------------------------------------------------------

/// A successfully parsed StatsD metric line.
#[derive(Debug, Clone)]
pub struct StatsdLine {
    /// Metric name (the part before `:`).
    pub name: String,
    /// Parsed numeric value.
    pub value: f64,
    /// Metric type.
    pub metric_type: StatsdMetricType,
    /// Optional sampling rate (1.0 means all samples; 0.1 means 10% sampled).
    pub sample_rate: f64,
    /// DogStatsD-style tags parsed as key-value pairs.
    /// Tags without `key:value` structure are stored with an empty-string value
    /// under their literal text as key.
    pub tags: HashMap<String, String>,
    /// Raw sign prefix: `+` or `-` for gauge deltas, or empty.
    pub sign: Option<char>,
}

impl StatsdLine {
    /// Return the effective value, adjusted for sampling rate.
    ///
    /// For counters, the value is scaled by `1.0 / sample_rate` to estimate
    /// the true count. For all other metric types, the raw value is returned.
    #[must_use]
    pub fn effective_value(&self) -> f64 {
        if self.metric_type == StatsdMetricType::Counter && self.sample_rate > 0.0 {
            self.value / self.sample_rate
        } else {
            self.value
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// StatsD line protocol parser.
pub struct StatsdParser;

impl StatsdParser {
    /// Parse a single StatsD datagram line.
    ///
    /// # Errors
    ///
    /// Returns an error if the line format is invalid.
    pub fn parse_line(line: &str) -> MonitorResult<StatsdLine> {
        let line = line.trim();
        if line.is_empty() {
            return Err(MonitorError::Other("empty StatsD line".to_string()));
        }

        // Split on ':' to separate name from value+type+options.
        let colon_pos = line
            .find(':')
            .ok_or_else(|| MonitorError::Other(format!("missing ':' in StatsD line: '{line}'")))?;

        let name = line[..colon_pos].trim().to_string();
        if name.is_empty() {
            return Err(MonitorError::Other(
                "StatsD metric name is empty".to_string(),
            ));
        }

        let rest = &line[colon_pos + 1..];

        // The rest is: value|type[|@sample_rate][|#tag:val,tag:val,...]
        // Split on '|'.
        let mut segments: Vec<&str> = rest.split('|').collect();
        if segments.len() < 2 {
            return Err(MonitorError::Other(format!(
                "StatsD line missing '|type' section: '{line}'"
            )));
        }

        // Parse value (may have leading +/- for gauge delta).
        let value_str = segments[0].trim();
        let (sign, numeric_str) = if value_str.starts_with('+') {
            (Some('+'), &value_str[1..])
        } else if value_str.starts_with('-') {
            (Some('-'), &value_str[1..])
        } else {
            (None, value_str)
        };

        let mut value: f64 = numeric_str
            .parse()
            .map_err(|_| MonitorError::Other(format!("invalid StatsD value: '{value_str}'")))?;
        if sign == Some('-') {
            value = -value;
        }

        // Parse metric type.
        let metric_type = StatsdMetricType::from_symbol(segments[1].trim())?;

        // Parse optional extensions.
        let mut sample_rate = 1.0_f64;
        let mut tags: HashMap<String, String> = HashMap::new();

        for seg in segments.drain(2..) {
            let seg = seg.trim();
            if let Some(rate_str) = seg.strip_prefix('@') {
                // Sampling rate.
                sample_rate = rate_str.parse().unwrap_or(1.0);
                sample_rate = sample_rate.clamp(0.0, 1.0);
            } else if let Some(tag_str) = seg.strip_prefix('#') {
                // DogStatsD tags.
                for tag in tag_str.split(',') {
                    let tag = tag.trim();
                    if let Some(eq_pos) = tag.find(':') {
                        tags.insert(tag[..eq_pos].to_string(), tag[eq_pos + 1..].to_string());
                    } else if !tag.is_empty() {
                        tags.insert(tag.to_string(), String::new());
                    }
                }
            }
        }

        Ok(StatsdLine {
            name,
            value,
            metric_type,
            sample_rate,
            tags,
            sign: sign.map(|c| if c == '+' { '+' } else { '-' }),
        })
    }

    /// Parse multiple lines from a datagram payload (lines separated by `\n`).
    ///
    /// Empty lines and parse errors are silently skipped (following StatsD
    /// convention of best-effort UDP delivery).
    #[must_use]
    pub fn parse_datagram(data: &str) -> Vec<StatsdLine> {
        data.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| Self::parse_line(l).ok())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// In-process accumulator
// ---------------------------------------------------------------------------

/// Accumulated state for a single StatsD metric.
#[derive(Debug, Clone)]
pub struct StatsdAccumulator {
    /// Current gauge value or counter total.
    pub value: f64,
    /// Number of samples aggregated.
    pub sample_count: u64,
    /// For timers and histograms: all values seen.
    pub samples: Vec<f64>,
    /// Metric type.
    pub metric_type: StatsdMetricType,
}

impl StatsdAccumulator {
    fn new_gauge(value: f64) -> Self {
        Self {
            value,
            sample_count: 1,
            samples: Vec::new(),
            metric_type: StatsdMetricType::Gauge,
        }
    }

    fn new_counter(value: f64) -> Self {
        Self {
            value,
            sample_count: 1,
            samples: Vec::new(),
            metric_type: StatsdMetricType::Counter,
        }
    }

    fn new_timer_or_histogram(value: f64, metric_type: StatsdMetricType) -> Self {
        Self {
            value,
            sample_count: 1,
            samples: vec![value],
            metric_type,
        }
    }

    /// Apply a new StatsD line to this accumulator.
    fn apply(&mut self, line: &StatsdLine) {
        self.sample_count += 1;
        match line.metric_type {
            StatsdMetricType::Gauge => {
                // Gauge delta: +/- prefix adjusts; otherwise replaces.
                if line.sign == Some('+') {
                    self.value += line.value.abs();
                } else if line.sign == Some('-') {
                    self.value -= line.value.abs();
                } else {
                    self.value = line.value;
                }
            }
            StatsdMetricType::Counter => {
                self.value += line.effective_value();
            }
            StatsdMetricType::Timer | StatsdMetricType::Histogram => {
                self.samples.push(line.value);
                self.value = line.value; // store last value too
            }
            StatsdMetricType::Set => {
                // For sets we just track unique count — simplified to counting.
                self.value += 1.0;
            }
        }
    }

    /// Timer/histogram p50 (median).
    #[must_use]
    pub fn percentile(&self, pct: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((pct / 100.0) * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Mean of timer/histogram samples.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
    }
}

/// In-process StatsD metric store: accumulates parsed lines.
#[derive(Debug, Default)]
pub struct StatsdStore {
    metrics: HashMap<String, StatsdAccumulator>,
}

impl StatsdStore {
    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a parsed StatsD line.
    pub fn ingest(&mut self, line: StatsdLine) {
        if let Some(acc) = self.metrics.get_mut(&line.name) {
            acc.apply(&line);
        } else {
            let acc = match line.metric_type {
                StatsdMetricType::Gauge => StatsdAccumulator::new_gauge(line.value),
                StatsdMetricType::Counter => StatsdAccumulator::new_counter(line.effective_value()),
                other => StatsdAccumulator::new_timer_or_histogram(line.value, other),
            };
            self.metrics.insert(line.name.clone(), acc);
        }
    }

    /// Ingest all lines from a datagram.
    pub fn ingest_datagram(&mut self, data: &str) {
        for line in StatsdParser::parse_datagram(data) {
            self.ingest(line);
        }
    }

    /// Get the accumulator for a metric.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StatsdAccumulator> {
        self.metrics.get(name)
    }

    /// Number of tracked metrics.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Get the current value of a metric (or `None` if not seen).
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).map(|a| a.value)
    }

    /// Reset the store (clear all accumulated data).
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Names of all tracked metrics.
    #[must_use]
    pub fn metric_names(&self) -> Vec<&str> {
        self.metrics.keys().map(String::as_str).collect()
    }
}

// ---------------------------------------------------------------------------
// StatsDMetric enum (typed value form)
// ---------------------------------------------------------------------------

/// A StatsD metric value with its type discriminant.
///
/// This is the high-level, typed representation produced by
/// [`StatsDMetricRecord::from_line`], complementing the lower-level
/// [`StatsdLine`] struct that carries full parse metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsDMetric {
    /// Counter value (cumulative sum, possibly scaled by sample rate).
    Counter(f64),
    /// Gauge value (instantaneous measurement).
    Gauge(f64),
    /// Timer sample in milliseconds.
    Timer(f64),
    /// Unique set element (stored as its string representation).
    Set(String),
    /// Histogram sample.
    Histogram(f64),
}

impl StatsDMetric {
    /// Human-readable metric type name.
    #[must_use]
    pub fn metric_type_name(&self) -> &'static str {
        match self {
            Self::Counter(_) => "counter",
            Self::Gauge(_) => "gauge",
            Self::Timer(_) => "timer",
            Self::Set(_) => "set",
            Self::Histogram(_) => "histogram",
        }
    }

    /// Extract the numeric value if the metric type carries one.
    ///
    /// Returns `None` for [`StatsDMetric::Set`] variants.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Counter(v) | Self::Gauge(v) | Self::Timer(v) | Self::Histogram(v) => Some(*v),
            Self::Set(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// StatsDMetricRecord
// ---------------------------------------------------------------------------

/// A timestamped, typed StatsD metric record.
///
/// Produced by converting a [`StatsdLine`] via [`StatsDMetricRecord::from_line`].
#[derive(Debug, Clone)]
pub struct StatsDMetricRecord {
    /// Metric name.
    pub name: String,
    /// Typed metric value.
    pub metric: StatsDMetric,
    /// Sampling rate at which this metric was emitted.
    pub sample_rate: f64,
    /// Milliseconds since the Unix epoch when this record was created.
    pub timestamp_ms: u64,
}

impl StatsDMetricRecord {
    /// Convert a parsed [`StatsdLine`] into a typed record.
    ///
    /// The timestamp is set to the current wall-clock time.
    #[must_use]
    pub fn from_line(line: &StatsdLine) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let metric = match line.metric_type {
            StatsdMetricType::Counter => StatsDMetric::Counter(line.effective_value()),
            StatsdMetricType::Gauge => StatsDMetric::Gauge(line.value),
            StatsdMetricType::Timer => StatsDMetric::Timer(line.value),
            StatsdMetricType::Histogram => StatsDMetric::Histogram(line.value),
            StatsdMetricType::Set => StatsDMetric::Set(line.value.to_string()),
        };

        Self {
            name: line.name.clone(),
            metric,
            sample_rate: line.sample_rate,
            timestamp_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// StatsDListener (configuration wrapper)
// ---------------------------------------------------------------------------

/// UDP address configuration for a StatsD listener.
///
/// Encapsulates the bind address used by [`StatsDIngester::start_udp`].
#[derive(Debug, Clone)]
pub struct StatsDListener {
    bind_addr: String,
}

impl StatsDListener {
    /// Create a new listener configuration bound to `bind_addr`.
    #[must_use]
    pub fn new(bind_addr: impl Into<String>) -> Self {
        Self {
            bind_addr: bind_addr.into(),
        }
    }

    /// The UDP bind address string.
    #[must_use]
    pub fn bind_addr(&self) -> &str {
        &self.bind_addr
    }
}

// ---------------------------------------------------------------------------
// StatsDIngester — background UDP receiver
// ---------------------------------------------------------------------------

/// Background UDP receiver that accumulates parsed StatsD metrics.
///
/// Spawn a listener with [`StatsDIngester::start_udp`], then poll
/// [`StatsDIngester::received_metrics`] for a snapshot of what has been
/// received so far. Call [`StatsDIngester::stop`] to shut the background
/// thread down cleanly.
///
/// # Example
///
/// ```no_run
/// use oximedia_monitor::statsd_ingestion::StatsDIngester;
///
/// let ingester = StatsDIngester::new();
/// ingester.start_udp("127.0.0.1:8125").expect("bind UDP");
/// // … receive metrics …
/// let metrics = ingester.received_metrics();
/// ingester.stop();
/// ```
pub struct StatsDIngester {
    store: std::sync::Arc<std::sync::Mutex<Vec<StatsDMetricRecord>>>,
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl std::fmt::Debug for StatsDIngester {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.store.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("StatsDIngester")
            .field("metric_count", &count)
            .field(
                "running",
                &self.running.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for StatsDIngester {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsDIngester {
    /// Create a new, idle ingester.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Bind a non-blocking UDP socket on `addr` and spawn a background thread
    /// that reads datagrams, parses them with [`StatsdParser`], converts each
    /// [`StatsdLine`] to a [`StatsDMetricRecord`], and appends it to the
    /// internal store.
    ///
    /// The thread exits when [`StatsDIngester::stop`] is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket cannot be bound.
    pub fn start_udp(&self, addr: &str) -> crate::error::MonitorResult<()> {
        use std::io::ErrorKind;
        use std::net::UdpSocket;
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        let socket = UdpSocket::bind(addr).map_err(|e| MonitorError::Io(e))?;
        socket
            .set_nonblocking(true)
            .map_err(|e| MonitorError::Io(e))?;

        self.running.store(true, Ordering::SeqCst);

        let store = std::sync::Arc::clone(&self.store);
        let running = std::sync::Arc::clone(&self.running);

        std::thread::spawn(move || {
            let mut buf = vec![0u8; 65_507];
            while running.load(Ordering::Relaxed) {
                match socket.recv_from(&mut buf) {
                    Ok((len, _src)) => {
                        // Parse as UTF-8; skip invalid datagrams.
                        if let Ok(text) = std::str::from_utf8(&buf[..len]) {
                            let lines = StatsdParser::parse_datagram(text);
                            if let Ok(mut guard) = store.lock() {
                                for line in lines {
                                    guard.push(StatsDMetricRecord::from_line(&line));
                                }
                            }
                        }
                    }
                    Err(ref e)
                        if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut =>
                    {
                        // No data available yet — yield briefly.
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => {
                        // Other socket errors — break loop.
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Return a snapshot of all metrics received since the ingester started.
    #[must_use]
    pub fn received_metrics(&self) -> Vec<StatsDMetricRecord> {
        self.store.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Number of metrics received so far.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.store.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Signal the background thread to stop.
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- StatsdMetricType --

    #[test]
    fn test_metric_type_symbols() {
        assert_eq!(StatsdMetricType::Gauge.symbol(), "g");
        assert_eq!(StatsdMetricType::Counter.symbol(), "c");
        assert_eq!(StatsdMetricType::Timer.symbol(), "ms");
        assert_eq!(StatsdMetricType::Histogram.symbol(), "h");
        assert_eq!(StatsdMetricType::Set.symbol(), "s");
    }

    #[test]
    fn test_metric_type_from_symbol() {
        assert_eq!(
            StatsdMetricType::from_symbol("g").expect("ok"),
            StatsdMetricType::Gauge
        );
        assert_eq!(
            StatsdMetricType::from_symbol("c").expect("ok"),
            StatsdMetricType::Counter
        );
        assert_eq!(
            StatsdMetricType::from_symbol("ms").expect("ok"),
            StatsdMetricType::Timer
        );
        assert_eq!(
            StatsdMetricType::from_symbol("h").expect("ok"),
            StatsdMetricType::Histogram
        );
        assert_eq!(
            StatsdMetricType::from_symbol("s").expect("ok"),
            StatsdMetricType::Set
        );
    }

    #[test]
    fn test_metric_type_unknown_fails() {
        assert!(StatsdMetricType::from_symbol("x").is_err());
        assert!(StatsdMetricType::from_symbol("").is_err());
    }

    // -- StatsdParser::parse_line --

    #[test]
    fn test_parse_gauge() {
        let m = StatsdParser::parse_line("cpu.usage:72.5|g").expect("valid");
        assert_eq!(m.name, "cpu.usage");
        assert!((m.value - 72.5).abs() < f64::EPSILON);
        assert_eq!(m.metric_type, StatsdMetricType::Gauge);
        assert!((m.sample_rate - 1.0).abs() < f64::EPSILON);
        assert!(m.tags.is_empty());
    }

    #[test]
    fn test_parse_counter() {
        let m = StatsdParser::parse_line("requests:1|c").expect("valid");
        assert_eq!(m.name, "requests");
        assert!((m.value - 1.0).abs() < f64::EPSILON);
        assert_eq!(m.metric_type, StatsdMetricType::Counter);
    }

    #[test]
    fn test_parse_timer() {
        let m = StatsdParser::parse_line("response.time:320|ms").expect("valid");
        assert_eq!(m.metric_type, StatsdMetricType::Timer);
        assert!((m.value - 320.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_histogram() {
        let m = StatsdParser::parse_line("frame.size:4096|h").expect("valid");
        assert_eq!(m.metric_type, StatsdMetricType::Histogram);
    }

    #[test]
    fn test_parse_set() {
        let m = StatsdParser::parse_line("unique.users:123|s").expect("valid");
        assert_eq!(m.metric_type, StatsdMetricType::Set);
    }

    #[test]
    fn test_parse_with_sample_rate() {
        let m = StatsdParser::parse_line("errors:1|c|@0.1").expect("valid");
        assert!((m.sample_rate - 0.1).abs() < f64::EPSILON);
        assert!((m.effective_value() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_with_tags() {
        let m = StatsdParser::parse_line("cpu:50|g|#host:server1,env:prod").expect("valid");
        assert_eq!(m.tags.get("host").map(String::as_str), Some("server1"));
        assert_eq!(m.tags.get("env").map(String::as_str), Some("prod"));
    }

    #[test]
    fn test_parse_with_sample_rate_and_tags() {
        let m = StatsdParser::parse_line("requests:1|c|@0.5|#service:api,region:us-east")
            .expect("valid");
        assert!((m.sample_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(m.tags.get("service").map(String::as_str), Some("api"));
        assert_eq!(m.tags.get("region").map(String::as_str), Some("us-east"));
    }

    #[test]
    fn test_parse_gauge_delta_positive() {
        let m = StatsdParser::parse_line("memory:+10|g").expect("valid");
        assert_eq!(m.sign, Some('+'));
        assert!((m.value - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_gauge_delta_negative() {
        let m = StatsdParser::parse_line("memory:-5|g").expect("valid");
        assert_eq!(m.sign, Some('-'));
        assert!((m.value - (-5.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_empty_line_fails() {
        assert!(StatsdParser::parse_line("").is_err());
        assert!(StatsdParser::parse_line("   ").is_err());
    }

    #[test]
    fn test_parse_missing_colon_fails() {
        assert!(StatsdParser::parse_line("cpu_usage|g").is_err());
    }

    #[test]
    fn test_parse_missing_pipe_fails() {
        assert!(StatsdParser::parse_line("cpu:50").is_err());
    }

    #[test]
    fn test_parse_invalid_value_fails() {
        assert!(StatsdParser::parse_line("cpu:notanumber|g").is_err());
    }

    #[test]
    fn test_parse_unknown_type_fails() {
        assert!(StatsdParser::parse_line("cpu:50|z").is_err());
    }

    #[test]
    fn test_parse_trims_whitespace() {
        let m = StatsdParser::parse_line("  cpu : 50 | g  ").expect("valid after trim");
        assert_eq!(m.name, "cpu");
        assert!((m.value - 50.0).abs() < f64::EPSILON);
    }

    // -- StatsdParser::parse_datagram --

    #[test]
    fn test_parse_datagram_multiple_lines() {
        let data = "cpu:50|g\nmem:70|g\nreqs:1|c";
        let lines = StatsdParser::parse_datagram(data);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_parse_datagram_skips_invalid() {
        let data = "cpu:50|g\nbad_line\nreqs:1|c";
        let lines = StatsdParser::parse_datagram(data);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_parse_datagram_skips_empty_lines() {
        let data = "cpu:50|g\n\n\nreqs:1|c\n";
        let lines = StatsdParser::parse_datagram(data);
        assert_eq!(lines.len(), 2);
    }

    // -- StatsdStore --

    #[test]
    fn test_store_ingest_gauge() {
        let mut store = StatsdStore::new();
        let line = StatsdParser::parse_line("cpu:72.5|g").expect("valid");
        store.ingest(line);
        assert!((store.value("cpu").expect("exists") - 72.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_ingest_counter_accumulates() {
        let mut store = StatsdStore::new();
        store.ingest(StatsdParser::parse_line("reqs:1|c").expect("valid"));
        store.ingest(StatsdParser::parse_line("reqs:1|c").expect("valid"));
        store.ingest(StatsdParser::parse_line("reqs:3|c").expect("valid"));
        assert!((store.value("reqs").expect("exists") - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_ingest_gauge_replaces() {
        let mut store = StatsdStore::new();
        store.ingest(StatsdParser::parse_line("cpu:50|g").expect("valid"));
        store.ingest(StatsdParser::parse_line("cpu:80|g").expect("valid"));
        assert!((store.value("cpu").expect("exists") - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_ingest_gauge_delta() {
        let mut store = StatsdStore::new();
        store.ingest(StatsdParser::parse_line("mem:100|g").expect("valid"));
        store.ingest(StatsdParser::parse_line("mem:+20|g").expect("valid"));
        assert!((store.value("mem").expect("exists") - 120.0).abs() < f64::EPSILON);
        store.ingest(StatsdParser::parse_line("mem:-10|g").expect("valid"));
        assert!((store.value("mem").expect("exists") - 110.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_ingest_timer_samples() {
        let mut store = StatsdStore::new();
        for ms in [100, 200, 150, 300, 250] {
            store.ingest(StatsdParser::parse_line(&format!("latency:{ms}|ms")).expect("valid"));
        }
        let acc = store.get("latency").expect("exists");
        assert_eq!(acc.samples.len(), 5);
        let p50 = acc.percentile(50.0).expect("p50");
        assert!((p50 - 200.0).abs() < f64::EPSILON, "p50={p50}");
    }

    #[test]
    fn test_store_ingest_counter_with_sample_rate() {
        let mut store = StatsdStore::new();
        // sample_rate=0.1 means 10% sampled → effective value = 10.
        store.ingest(StatsdParser::parse_line("errors:1|c|@0.1").expect("valid"));
        assert!((store.value("errors").expect("exists") - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_ingest_datagram() {
        let mut store = StatsdStore::new();
        store.ingest_datagram("cpu:50|g\nmem:70|g\nreqs:5|c");
        assert_eq!(store.metric_count(), 3);
        assert!((store.value("cpu").expect("cpu") - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_metric_count() {
        let mut store = StatsdStore::new();
        store.ingest(StatsdParser::parse_line("a:1|g").expect("valid"));
        store.ingest(StatsdParser::parse_line("b:2|g").expect("valid"));
        store.ingest(StatsdParser::parse_line("a:3|g").expect("valid")); // update, not new
        assert_eq!(store.metric_count(), 2);
    }

    #[test]
    fn test_store_clear() {
        let mut store = StatsdStore::new();
        store.ingest(StatsdParser::parse_line("cpu:50|g").expect("valid"));
        store.clear();
        assert_eq!(store.metric_count(), 0);
        assert!(store.value("cpu").is_none());
    }

    #[test]
    fn test_store_metric_names() {
        let mut store = StatsdStore::new();
        store.ingest(StatsdParser::parse_line("cpu:50|g").expect("valid"));
        store.ingest(StatsdParser::parse_line("mem:70|g").expect("valid"));
        let mut names = store.metric_names();
        names.sort();
        assert_eq!(names, vec!["cpu", "mem"]);
    }

    // -- StatsdAccumulator percentile / mean --

    #[test]
    fn test_accumulator_percentile_empty() {
        let acc = StatsdAccumulator::new_gauge(0.0);
        assert!(acc.percentile(50.0).is_none());
    }

    #[test]
    fn test_accumulator_mean() {
        let mut acc = StatsdAccumulator::new_timer_or_histogram(10.0, StatsdMetricType::Timer);
        acc.samples.push(20.0);
        acc.samples.push(30.0);
        let mean = acc.mean().expect("mean exists");
        assert!((mean - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_accumulator_mean_empty() {
        let acc = StatsdAccumulator::new_gauge(0.0);
        assert!(acc.mean().is_none());
    }

    // -- Integration: full pipeline --

    #[test]
    fn test_full_statsd_ingest_pipeline() {
        let mut store = StatsdStore::new();

        // Simulate a burst of StatsD messages from a media encoder.
        let messages = [
            "encoder.frames:1|c",
            "encoder.frames:1|c",
            "encoder.bitrate:4500|g",
            "encoder.latency:16|ms",
            "encoder.latency:17|ms",
            "encoder.latency:15|ms",
            "encoder.drops:0|g",
        ];
        for msg in &messages {
            store.ingest(StatsdParser::parse_line(msg).expect("valid"));
        }

        assert!((store.value("encoder.frames").expect("frames") - 2.0).abs() < f64::EPSILON);
        assert!((store.value("encoder.bitrate").expect("bitrate") - 4500.0).abs() < f64::EPSILON);
        assert!((store.value("encoder.drops").expect("drops") - 0.0).abs() < f64::EPSILON);

        let latency_acc = store.get("encoder.latency").expect("latency");
        assert_eq!(latency_acc.samples.len(), 3);
        let p50 = latency_acc.percentile(50.0).expect("p50");
        assert!((p50 - 16.0).abs() < f64::EPSILON, "p50={p50}");
    }

    // -- StatsDMetric enum --

    #[test]
    fn test_statsd_metric_counter_value() {
        let m = StatsDMetric::Counter(42.0);
        assert_eq!(m.as_f64(), Some(42.0));
    }

    #[test]
    fn test_statsd_metric_gauge_value() {
        let m = StatsDMetric::Gauge(3.14);
        assert_eq!(m.as_f64(), Some(3.14));
    }

    #[test]
    fn test_statsd_metric_timer_value() {
        let m = StatsDMetric::Timer(120.0);
        assert_eq!(m.as_f64(), Some(120.0));
    }

    #[test]
    fn test_statsd_metric_histogram_value() {
        let m = StatsDMetric::Histogram(99.9);
        assert_eq!(m.as_f64(), Some(99.9));
    }

    #[test]
    fn test_statsd_metric_set_value() {
        let m = StatsDMetric::Set("user-42".to_string());
        assert!(m.as_f64().is_none());
    }

    #[test]
    fn test_statsd_metric_type_names() {
        assert_eq!(StatsDMetric::Counter(1.0).metric_type_name(), "counter");
        assert_eq!(StatsDMetric::Gauge(1.0).metric_type_name(), "gauge");
        assert_eq!(StatsDMetric::Timer(1.0).metric_type_name(), "timer");
        assert_eq!(StatsDMetric::Histogram(1.0).metric_type_name(), "histogram");
        assert_eq!(StatsDMetric::Set("x".into()).metric_type_name(), "set");
    }

    #[test]
    fn test_statsd_metric_as_f64_counter() {
        assert_eq!(StatsDMetric::Counter(7.0).as_f64(), Some(7.0));
    }

    #[test]
    fn test_statsd_metric_as_f64_set_is_none() {
        assert!(StatsDMetric::Set("abc".to_string()).as_f64().is_none());
    }

    // -- StatsDMetricRecord --

    #[test]
    fn test_statsd_metric_record_from_line_counter() {
        let line = StatsdParser::parse_line("requests:5|c").expect("valid");
        let rec = StatsDMetricRecord::from_line(&line);
        assert_eq!(rec.name, "requests");
        assert!(matches!(rec.metric, StatsDMetric::Counter(_)));
        if let StatsDMetric::Counter(v) = rec.metric {
            assert!((v - 5.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_statsd_metric_record_from_line_gauge() {
        let line = StatsdParser::parse_line("cpu:88.5|g").expect("valid");
        let rec = StatsDMetricRecord::from_line(&line);
        assert!(matches!(rec.metric, StatsDMetric::Gauge(_)));
        if let StatsDMetric::Gauge(v) = rec.metric {
            assert!((v - 88.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_statsd_metric_record_from_line_timer() {
        let line = StatsdParser::parse_line("latency:200|ms").expect("valid");
        let rec = StatsDMetricRecord::from_line(&line);
        assert!(matches!(rec.metric, StatsDMetric::Timer(_)));
    }

    #[test]
    fn test_statsd_metric_record_from_line_histogram() {
        let line = StatsdParser::parse_line("frame.size:1024|h").expect("valid");
        let rec = StatsDMetricRecord::from_line(&line);
        assert!(matches!(rec.metric, StatsDMetric::Histogram(_)));
    }

    #[test]
    fn test_statsd_metric_record_from_line_set() {
        let line = StatsdParser::parse_line("users:42|s").expect("valid");
        let rec = StatsDMetricRecord::from_line(&line);
        // Set type: value stored as string representation
        assert!(matches!(rec.metric, StatsDMetric::Set(_)));
    }

    // -- StatsDListener --

    #[test]
    fn test_statsd_listener_new() {
        let listener = StatsDListener::new("127.0.0.1:9125");
        assert_eq!(listener.bind_addr(), "127.0.0.1:9125");
    }

    #[test]
    fn test_statsd_listener_bind_addr() {
        let listener = StatsDListener::new("0.0.0.0:8125");
        assert_eq!(listener.bind_addr(), "0.0.0.0:8125");
    }

    // -- StatsDIngester --

    #[test]
    fn test_statsd_ingester_new() {
        let ingester = StatsDIngester::new();
        assert_eq!(ingester.metric_count(), 0);
        assert!(ingester.received_metrics().is_empty());
    }

    #[test]
    fn test_statsd_ingester_start_stop() {
        let ingester = StatsDIngester::new();
        ingester
            .start_udp("127.0.0.1:19125")
            .expect("should bind UDP");
        // Give thread a moment to start.
        std::thread::sleep(std::time::Duration::from_millis(20));
        ingester.stop();
    }

    #[test]
    fn test_statsd_ingester_send_and_receive() {
        use std::net::UdpSocket;

        let ingester = StatsDIngester::new();
        ingester
            .start_udp("127.0.0.1:19126")
            .expect("should bind UDP on 19126");

        // Send one metric via UDP.
        let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
        sender
            .send_to(b"cpu:72.5|g", "127.0.0.1:19126")
            .expect("send");

        // Wait for background thread to process.
        std::thread::sleep(std::time::Duration::from_millis(150));

        let metrics = ingester.received_metrics();
        assert_eq!(metrics.len(), 1, "expected 1 metric, got {}", metrics.len());
        assert_eq!(metrics[0].name, "cpu");
        if let StatsDMetric::Gauge(v) = &metrics[0].metric {
            assert!((*v - 72.5).abs() < f64::EPSILON);
        } else {
            panic!("expected Gauge metric");
        }

        ingester.stop();
    }

    #[test]
    fn test_statsd_ingester_multiple_metrics() {
        use std::net::UdpSocket;

        let ingester = StatsDIngester::new();
        ingester
            .start_udp("127.0.0.1:19127")
            .expect("should bind UDP on 19127");

        let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
        sender
            .send_to(b"frames:1|c", "127.0.0.1:19127")
            .expect("send frames");
        sender
            .send_to(b"bitrate:4500|g", "127.0.0.1:19127")
            .expect("send bitrate");
        sender
            .send_to(b"latency:16|ms", "127.0.0.1:19127")
            .expect("send latency");

        std::thread::sleep(std::time::Duration::from_millis(150));

        let metrics = ingester.received_metrics();
        assert_eq!(
            metrics.len(),
            3,
            "expected 3 metrics, got {}",
            metrics.len()
        );

        ingester.stop();
    }

    #[test]
    fn test_statsd_ingester_metric_count() {
        use std::net::UdpSocket;

        let ingester = StatsDIngester::new();
        ingester
            .start_udp("127.0.0.1:19128")
            .expect("should bind UDP on 19128");

        let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
        sender.send_to(b"a:1|g", "127.0.0.1:19128").expect("send a");
        sender.send_to(b"b:2|c", "127.0.0.1:19128").expect("send b");

        std::thread::sleep(std::time::Duration::from_millis(150));

        assert_eq!(ingester.metric_count(), 2);

        ingester.stop();
    }
}
