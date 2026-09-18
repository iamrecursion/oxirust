//! I/O reliability, concurrency, and edge-behavior pinning tests for
//! `oximedia-io` primitives.
//!
//! These integration tests exercise crash-relevant serialization, concurrent
//! producer/consumer streaming, windowed memory mapping, format-detection edge
//! cases, and progress-reporting accuracy.
//!
//! ## Crash-recovery caveat
//!
//! [`oximedia_io::write_journal::WriteJournal`] is an **in-memory,
//! serialize-only** journal. There is no on-disk persistence or `replay()`
//! API. The "crash recovery" tests below therefore pin the *real* surface that
//! the journal exposes: the fixed-size 40-byte
//! [`JournalEntry::to_bytes`]/[`JournalEntry::from_bytes`] codec. We model a
//! torn write by truncating a concatenated entry stream mid-record and assert
//! that decoding in 40-byte frames recovers exactly the intact records and
//! rejects the partial tail. This is the genuine durability seam that a
//! file-backed WAL would build on top of.

use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use oximedia_io::format_detector::{FormatDetector, MediaFormat};
use oximedia_io::mmap::MmapFile;
use oximedia_io::progress_reader::ProgressReader;
use oximedia_io::ring_buffer::spsc_ring_buffer;
use oximedia_io::write_journal::{JournalEntry, WriteJournal};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Size of one serialized [`JournalEntry`] in bytes.
const ENTRY_BYTES: usize = 40;

/// Build a unique temp-file path keyed by the current process id and a label so
/// concurrent test binaries do not collide.
fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oximedia_io_{}_{}.bin", label, std::process::id()))
}

/// Deterministic xorshift64 byte generator (matches the seeded stream used by
/// the SPSC concurrency test).
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_byte(&mut self) -> u8 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        (s & 0xff) as u8
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// write_journal: crash-recovery / torn-write surface
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn journal_replay_recovers_intact_records_after_torn_write() {
    // Record 8 known writes with distinct offsets and payload lengths.
    let mut journal = WriteJournal::with_defaults();
    let payloads: Vec<Vec<u8>> = (0..8u64)
        .map(|i| {
            // Each payload has a distinct, deterministic length and content.
            let len = (i as usize + 1) * 3;
            (0..len).map(|b| (b as u64 + i) as u8).collect()
        })
        .collect();
    let mut originals: Vec<(u64, u64, u32)> = Vec::new(); // (seq, offset, data_len)
    for (i, payload) in payloads.iter().enumerate() {
        let offset = (i as u64) * 1024;
        let seq = journal.record_write(offset, payload);
        originals.push((seq, offset, payload.len() as u32));
    }
    assert_eq!(journal.entry_count(), 8);

    // Serialize each entry via the only serialization path (40-byte frames),
    // concatenated into a single buffer.
    let mut buf: Vec<u8> = Vec::with_capacity(8 * ENTRY_BYTES);
    for entry in journal.all_entries() {
        buf.extend_from_slice(&entry.to_bytes());
    }
    assert_eq!(buf.len(), 8 * ENTRY_BYTES);

    // Persist to disk and simulate a torn write: keep 7 full records plus a
    // partial 8th record of 17 bytes (less than a full 40-byte frame).
    let torn_len = 7 * ENTRY_BYTES + 17;
    let path = temp_path("journal_recovery");
    {
        let mut f = fs::File::create(&path).expect("create journal file");
        f.write_all(&buf).expect("write journal bytes");
        f.flush().expect("flush journal");
    }
    {
        // Truncate to the torn length to emulate an interrupted append.
        let f = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen for truncate");
        f.set_len(torn_len as u64).expect("truncate to torn length");
    }

    // Re-open and decode in 40-byte chunks, stopping on a short final chunk.
    let mut file = fs::File::open(&path).expect("reopen journal");
    let mut recovered: Vec<JournalEntry> = Vec::new();
    let mut rejected_tail = 0usize;
    loop {
        let mut frame = [0u8; ENTRY_BYTES];
        let mut filled = 0usize;
        // Read exactly one frame, tolerating short reads from the OS.
        while filled < ENTRY_BYTES {
            let n = file.read(&mut frame[filled..]).expect("read journal frame");
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break; // clean EOF on a frame boundary
        }
        if filled < ENTRY_BYTES {
            // Torn / partial final record — reject it.
            rejected_tail = filled;
            break;
        }
        recovered.push(JournalEntry::from_bytes(&frame));
    }

    let _ = fs::remove_file(&path);

    // Exactly 7 intact records recover; the 17-byte tail is rejected.
    assert_eq!(recovered.len(), 7, "expected 7 intact records");
    assert_eq!(rejected_tail, 17, "torn tail must be the partial 17 bytes");

    // Seqs are 1..=7, contiguous and strictly monotonic.
    for (i, entry) in recovered.iter().enumerate() {
        let expected_seq = (i as u64) + 1;
        assert_eq!(entry.seq, expected_seq, "seq must be contiguous from 1");
        if i > 0 {
            assert!(
                entry.seq > recovered[i - 1].seq,
                "seqs must be strictly monotonic"
            );
        }
        // Each recovered offset / data_len equals the originals.
        let (orig_seq, orig_offset, orig_len) = originals[i];
        assert_eq!(entry.seq, orig_seq);
        assert_eq!(entry.offset, orig_offset, "offset must round-trip");
        assert_eq!(entry.data_len, orig_len, "data_len must round-trip");
    }
}

#[test]
fn journal_zero_length_file_replays_empty() {
    // A 0-byte journal file must decode to zero entries without panicking.
    let path = temp_path("journal_zero");
    {
        let _f = fs::File::create(&path).expect("create empty journal");
    }
    let mut file = fs::File::open(&path).expect("reopen empty journal");

    let mut recovered: Vec<JournalEntry> = Vec::new();
    let mut frame = [0u8; ENTRY_BYTES];
    let mut filled = 0usize;
    loop {
        let n = file.read(&mut frame[filled..]).expect("read empty journal");
        if n == 0 {
            break;
        }
        filled += n;
        if filled == ENTRY_BYTES {
            recovered.push(JournalEntry::from_bytes(&frame));
            filled = 0;
        }
    }

    let _ = fs::remove_file(&path);

    assert_eq!(filled, 0, "no partial frame from an empty file");
    assert!(recovered.is_empty(), "empty file yields zero entries");
}

#[test]
fn journal_checkpoint_then_recover_only_post_checkpoint() {
    let mut journal = WriteJournal::with_defaults();
    // Record 3 entries (seq 1..=3).
    journal.record_write(0, b"pre-a");
    journal.record_write(10, b"pre-b");
    journal.record_write(20, b"pre-c");
    assert_eq!(journal.entry_count(), 3);

    // Checkpoint clears the entries but does NOT reset the sequence counter.
    let cp = journal.checkpoint();
    assert_eq!(cp.entry_count, 3);
    assert_eq!(journal.entry_count(), 0);
    assert_eq!(
        journal.next_seq(),
        4,
        "checkpoint preserves the seq counter"
    );

    // Record 2 more entries (seq 4..=5).
    let seq_d = journal.record_write(100, b"post-d");
    let seq_e = journal.record_write(200, b"post-e-longer");
    assert_eq!(seq_d, 4);
    assert_eq!(seq_e, 5);
    assert_eq!(journal.next_seq(), 6);

    // Serialize only the post-checkpoint set (all_entries holds the live ones).
    let live = journal.all_entries();
    assert_eq!(live.len(), 2, "only post-checkpoint entries remain live");
    let mut buf: Vec<u8> = Vec::new();
    for entry in &live {
        buf.extend_from_slice(&entry.to_bytes());
    }
    assert_eq!(buf.len(), 2 * ENTRY_BYTES);

    // Recover the 2 entries from the serialized buffer.
    let mut recovered: Vec<JournalEntry> = Vec::new();
    for chunk in buf.chunks_exact(ENTRY_BYTES) {
        let mut frame = [0u8; ENTRY_BYTES];
        frame.copy_from_slice(chunk);
        recovered.push(JournalEntry::from_bytes(&frame));
    }
    assert_eq!(recovered.len(), 2, "exactly the 2 post-checkpoint entries");
    assert_eq!(recovered[0].seq, 4);
    assert_eq!(recovered[0].offset, 100);
    assert_eq!(recovered[1].seq, 5);
    assert_eq!(recovered[1].offset, 200);
    assert_eq!(recovered[1].data_len, b"post-e-longer".len() as u32);
}

// ──────────────────────────────────────────────────────────────────────────────
// format_detector: truncated and zero-length inputs
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn format_detector_truncated_and_zero_length() {
    // Empty input → Unknown, confidence exactly 0.0.
    let empty = FormatDetector::detect(&[]);
    assert_eq!(empty.format, MediaFormat::Unknown);
    assert_eq!(empty.confidence, 0.0);

    // 1-byte JPEG SOI prefix (full SOI marker is 0xFF 0xD8): too short to match.
    let jpeg_prefix = [0xFFu8];
    let det = FormatDetector::detect(&jpeg_prefix);
    assert_ne!(det.format, MediaFormat::Jpeg, "1 byte cannot be JPEG");

    // 7-byte PNG prefix (PNG signature needs 8 bytes): must not match PNG.
    let png_prefix = [0x89u8, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A];
    let det = FormatDetector::detect(&png_prefix);
    assert_ne!(det.format, MediaFormat::Png, "7 bytes cannot be PNG");

    // 11-byte RIFF buffer, one byte short of the 12-byte WAVE tag region.
    let riff_short = b"RIFF\0\0\0\0WAV";
    assert_eq!(riff_short.len(), 11);
    let det = FormatDetector::detect(riff_short);
    assert_ne!(det.format, MediaFormat::Wav, "11 bytes cannot be WAV");
    assert_ne!(det.format, MediaFormat::Avi);
    assert_ne!(det.format, MediaFormat::Webp);

    // 3-byte EBML prefix (full magic is 4 bytes): must not match MKV/WebM.
    let ebml_prefix = [0x1Au8, 0x45, 0xDF];
    let det = FormatDetector::detect(&ebml_prefix);
    assert_ne!(det.format, MediaFormat::Mkv, "3 bytes cannot be MKV");
    assert_ne!(det.format, MediaFormat::Webm);

    // Positive control: a full RIFF/WAVE header detects as WAV with full
    // confidence. (Confirmed against src: `RIFF` + 4 size bytes + `WAVE`.)
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&[0x24, 0x00, 0x00, 0x00]); // chunk size
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt "); // realistic continuation
    let det = FormatDetector::detect(&wav);
    assert_eq!(det.format, MediaFormat::Wav, "full RIFF/WAVE must be WAV");
    assert_eq!(
        det.confidence, 1.0,
        "definitive WAV match is confidence 1.0"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// mmap: windowed sequential remap
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn mmap_windowed_sequential_remap() {
    // 64 KiB logical payload mapped as 16 windows of 4 KiB each. We model the
    // "next window" of a large-file mapping as advancing the region index,
    // verifying that the simulation tracks offsets and contiguity exactly.
    const WINDOW: usize = 4096;
    const WINDOWS: usize = 16;
    const TOTAL: usize = WINDOW * WINDOWS; // 65536

    let mut file = MmapFile::new("windowed.raw");
    for w in 0..WINDOWS {
        let chunk: Vec<u8> = (0..WINDOW)
            .map(|j| {
                let global_index = w * WINDOW + j;
                (global_index % 251) as u8
            })
            .collect();
        let idx = file.map_region(chunk, (w as u64) * WINDOW as u64);
        assert_eq!(idx, w, "region index advances sequentially");
    }

    assert_eq!(
        file.total_mapped_bytes(),
        TOTAL as u64,
        "all 64 KiB accounted for"
    );

    // Verify each window's offset, length, and content; check cross-window
    // contiguity (last byte of window w then first byte of window w+1).
    let mut prev_last: Option<u8> = None;
    for w in 0..WINDOWS {
        let region = file.get_region(w).expect("window must exist");
        assert_eq!(region.offset, (w as u64) * WINDOW as u64);
        assert_eq!(region.length, WINDOW as u64);

        let slice = region
            .slice(0, WINDOW)
            .expect("full window slice must succeed");
        for (j, &byte) in slice.iter().enumerate() {
            let global_index = w * WINDOW + j;
            assert_eq!(byte, (global_index % 251) as u8, "byte must match payload");
        }

        // Cross-window contiguity: the first byte of this window continues the
        // global modular sequence after the previous window's last byte.
        let first_global = w * WINDOW;
        assert_eq!(slice[0], (first_global % 251) as u8);
        if let Some(last) = prev_last {
            let expected_first = (first_global % 251) as u8;
            // The previous window ended at global index (first_global - 1).
            let expected_prev_last = ((first_global - 1) % 251) as u8;
            assert_eq!(last, expected_prev_last);
            assert_eq!(slice[0], expected_first);
        }
        prev_last = Some(slice[WINDOW - 1]);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// progress_reader: callback accuracy across read granularities
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn progress_reader_callback_accuracy_varying_granularity() {
    const TOTAL: usize = 10_000;
    let source: Vec<u8> = (0..TOTAL).map(|i| (i % 256) as u8).collect();

    // For each read-buffer granularity, the reader must consume exactly TOTAL
    // bytes, the final callback must observe bytes_read == TOTAL, and the
    // fraction must reach 1.0 at the end.
    for &granularity in &[1usize, 7, 64, 4096, 16384] {
        let last_cb_bytes = Arc::new(AtomicU64::new(0));
        let final_fraction = Arc::new(AtomicU64::new(u64::MAX)); // sentinel
        let lcb = Arc::clone(&last_cb_bytes);
        let ff = Arc::clone(&final_fraction);

        let cursor = Cursor::new(source.clone());
        let mut reader = ProgressReader::new(cursor, move |p| {
            lcb.store(p.bytes_read, Ordering::Relaxed);
            // Encode the fraction's "is it 1.0?" answer as a flag; we cannot
            // store f64 atomically without bit-casting, so record the bytes and
            // total relationship instead.
            if let Some(frac) = p.fraction() {
                // store 1 if fraction == 1.0 exactly, else 0
                ff.store(
                    u64::from((frac - 1.0).abs() < f64::EPSILON),
                    Ordering::Relaxed,
                );
            }
        })
        .with_total(TOTAL as u64);

        let mut buf = vec![0u8; granularity];
        let mut total = 0usize;
        loop {
            let n = reader.read(&mut buf).expect("read must succeed");
            if n == 0 {
                break;
            }
            total += n;
        }

        assert_eq!(total, TOTAL, "granularity {granularity}: total bytes");
        assert_eq!(
            reader.bytes_read(),
            TOTAL as u64,
            "granularity {granularity}: reader.bytes_read()"
        );
        assert_eq!(
            last_cb_bytes.load(Ordering::Relaxed),
            TOTAL as u64,
            "granularity {granularity}: final callback bytes_read"
        );
        // The last callback observed fraction == 1.0 (default report_interval=0
        // fires on every read, including the final read that reaches EOF).
        assert_eq!(
            final_fraction.load(Ordering::Relaxed),
            1,
            "granularity {granularity}: final fraction must be 1.0"
        );
    }

    // With a report interval of 2500 over 10_000 bytes, the callback must fire
    // at least 4 times (10_000 / 2500 == 4). Use a small granularity so the
    // accumulator crosses each 2500 boundary.
    let count = Arc::new(AtomicU64::new(0));
    let cc = Arc::clone(&count);
    let last = Arc::new(AtomicU64::new(0));
    let lc = Arc::clone(&last);
    let cursor = Cursor::new(source.clone());
    let mut reader = ProgressReader::new(cursor, move |p| {
        cc.fetch_add(1, Ordering::Relaxed);
        lc.store(p.bytes_read, Ordering::Relaxed);
    })
    .with_total(TOTAL as u64)
    .with_report_interval(2500);

    let mut buf = [0u8; 100];
    while reader.read(&mut buf).expect("read must succeed") > 0 {}

    assert!(
        count.load(Ordering::Relaxed) >= 4,
        "expected >= 4 reports with interval 2500, got {}",
        count.load(Ordering::Relaxed)
    );
    assert_eq!(
        last.load(Ordering::Relaxed),
        TOTAL as u64,
        "last interval-gated callback lands at the final byte (10_000 is a multiple of 2500)"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// ring_buffer (SPSC): concurrent producer / consumer at varying rates
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn spsc_concurrent_producer_consumer_varying_rates() {
    const TOTAL: usize = 100_000;
    const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    let (producer, consumer) = spsc_ring_buffer(256).expect("spsc ring buffer must construct");

    // Producer thread: push the seeded xorshift stream, looping on partial
    // pushes (the buffer only accepts what currently fits).
    let producer_handle = thread::spawn(move || {
        let mut rng = XorShift::new(SEED);
        let stream: Vec<u8> = (0..TOTAL).map(|_| rng.next_byte()).collect();
        let mut offset = 0usize;
        while offset < stream.len() {
            let pushed = producer.push(&stream[offset..]);
            if pushed == 0 {
                // Buffer full; yield and retry so the consumer can drain.
                thread::yield_now();
            } else {
                offset += pushed;
            }
        }
    });

    // Consumer thread: pop into a small buffer, occasionally yielding to vary
    // the consumption rate, accumulating everything received.
    let consumer_handle = thread::spawn(move || {
        let mut received: Vec<u8> = Vec::with_capacity(TOTAL);
        let mut buf = [0u8; 37]; // small, non-power-of-two read size
        let mut spins = 0u64;
        while received.len() < TOTAL {
            let n = consumer.pop(&mut buf);
            if n == 0 {
                thread::yield_now();
            } else {
                received.extend_from_slice(&buf[..n]);
                // Vary the rate: every few successful pops, yield.
                spins += 1;
                if spins % 5 == 0 {
                    thread::yield_now();
                }
            }
        }
        received
    });

    producer_handle.join().expect("producer thread must finish");
    let received = consumer_handle.join().expect("consumer thread must finish");

    // Regenerate the expected seeded stream and compare for exact FIFO order.
    let mut rng = XorShift::new(SEED);
    let expected: Vec<u8> = (0..TOTAL).map(|_| rng.next_byte()).collect();

    assert_eq!(
        received.len(),
        TOTAL,
        "consumer received exactly TOTAL bytes"
    );
    assert_eq!(
        received, expected,
        "FIFO order and content must match the seeded stream"
    );
}
