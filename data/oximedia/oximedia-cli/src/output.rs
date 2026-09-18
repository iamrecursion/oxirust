//! Output format helpers for the OxiMedia CLI.
//!
//! Provides [`OutputFormat`] resolution and [`NdjsonWriter`] for streaming
//! newline-delimited JSON records to any [`Write`] sink.

use std::io::{self, Write};

use serde::Serialize;

/// The output format chosen for a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable coloured text (default).
    Plain,
    /// Single JSON document (pretty-printed).
    Json,
    /// One JSON object per line (NDJSON / JSON Lines).
    Ndjson,
}

/// Resolve the output format from the two global flags.
///
/// `--ndjson` takes priority over `--json` when both are set (though clap
/// already rejects that combination via `conflicts_with`).
pub fn resolve_format(json: bool, ndjson: bool) -> OutputFormat {
    if ndjson {
        OutputFormat::Ndjson
    } else if json {
        OutputFormat::Json
    } else {
        OutputFormat::Plain
    }
}

/// A streaming writer that emits one JSON record per line.
pub struct NdjsonWriter<W: Write> {
    writer: W,
}

impl<W: Write> NdjsonWriter<W> {
    /// Wrap any [`Write`] sink.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Serialise `item` as compact JSON and write it followed by `\n`, then
    /// flush the underlying writer so each record is observable immediately.
    pub fn emit<T: Serialize>(&mut self, item: &T) -> io::Result<()> {
        serde_json::to_writer(&mut self.writer, item)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize)]
    struct TestRecord {
        id: u32,
        value: String,
    }

    #[test]
    fn ndjson_writer_round_trip() {
        let mut buf = Vec::new();
        let mut writer = NdjsonWriter::new(&mut buf);
        writer
            .emit(&TestRecord {
                id: 1,
                value: "hello".into(),
            })
            .expect("emit 1");
        writer
            .emit(&TestRecord {
                id: 2,
                value: "world".into(),
            })
            .expect("emit 2");
        let text = String::from_utf8(buf).expect("utf8");
        let lines: Vec<&str> = text.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 2);
        let v1: serde_json::Value = serde_json::from_str(lines[0]).expect("json line 1");
        assert_eq!(v1["id"], 1);
        let v2: serde_json::Value = serde_json::from_str(lines[1]).expect("json line 2");
        assert_eq!(v2["id"], 2);
    }

    #[test]
    fn resolve_format_ndjson_wins() {
        assert_eq!(resolve_format(true, true), OutputFormat::Ndjson);
        assert_eq!(resolve_format(false, true), OutputFormat::Ndjson);
        assert_eq!(resolve_format(true, false), OutputFormat::Json);
        assert_eq!(resolve_format(false, false), OutputFormat::Plain);
    }
}
