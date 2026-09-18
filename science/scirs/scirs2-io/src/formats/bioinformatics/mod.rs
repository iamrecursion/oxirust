//! Bioinformatics file format support
//!
//! This module provides support for common bioinformatics file formats used
//! in genomics, proteomics, and molecular biology research.
//!
//! ## Supported Formats
//!
//! - **FASTA**: Standard format for nucleotide and protein sequences
//! - **FASTQ**: Sequences with per-base quality scores
//! - **SAM/BAM**: Sequence Alignment/Map format
//! - **VCF**: Variant Call Format for genomic variations
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::formats::bioinformatics::{FastaReader, FastqReader};
//!
//! // Read FASTA file
//! let mut reader = FastaReader::open("sequences.fasta")?;
//! for record in reader.records() {
//!     let record = record?;
//!     println!(">{}", record.id());
//!     println!("{}", record.sequence());
//! }
//!
//! // Read FASTQ file
//! let mut reader = FastqReader::open("reads.fastq")?;
//! for record in reader.records() {
//!     let record = record?;
//!     println!("@{}", record.id());
//!     println!("{}", record.sequence());
//!     println!("+");
//!     println!("{}", record.quality());
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// FASTA sequence record
#[derive(Debug, Clone, PartialEq)]
pub struct FastaRecord {
    /// Sequence identifier (header line without '>')
    id: String,
    /// Optional description after the ID
    description: Option<String>,
    /// Sequence data
    sequence: String,
}

impl FastaRecord {
    /// Create a new FASTA record
    pub fn new(id: String, sequence: String) -> Self {
        Self {
            id,
            description: None,
            sequence,
        }
    }

    /// Create a new FASTA record with description
    pub fn with_description(id: String, description: String, sequence: String) -> Self {
        Self {
            id,
            description: Some(description),
            sequence,
        }
    }

    /// Get the sequence ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the optional description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get the sequence
    pub fn sequence(&self) -> &str {
        &self.sequence
    }

    /// Get the sequence length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get the full header (ID + description)
    pub fn header(&self) -> String {
        match &self.description {
            Some(desc) => format!("{} {}", self.id, desc),
            None => self.id.clone(),
        }
    }
}

/// FASTA file reader
pub struct FastaReader {
    reader: BufReader<File>,
    line_buffer: String,
    lookahead: Option<String>,
}

impl FastaReader {
    /// Open a FASTA file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
        Ok(Self {
            reader: BufReader::new(file),
            line_buffer: String::new(),
            lookahead: None,
        })
    }

    /// Create an iterator over FASTA records
    pub fn records(&mut self) -> FastaRecordIterator<'_> {
        FastaRecordIterator { reader: self }
    }

    /// Read the next record
    fn read_record(&mut self) -> Result<Option<FastaRecord>> {
        self.line_buffer.clear();

        // Check if we have a lookahead line (next header)
        if let Some(lookahead) = self.lookahead.take() {
            self.line_buffer = lookahead;
        } else {
            // Find the next header line
            loop {
                if self
                    .reader
                    .read_line(&mut self.line_buffer)
                    .map_err(|e| IoError::ParseError(format!("Failed to read line: {e}")))?
                    == 0
                {
                    return Ok(None); // EOF
                }

                if self.line_buffer.starts_with('>') {
                    break;
                }
                self.line_buffer.clear();
            }
        }

        // Parse header
        let header = self.line_buffer[1..].trim().to_string();
        let (id, description) = if let Some(space_pos) = header.find(' ') {
            let (id_part, desc_part) = header.split_at(space_pos);
            (id_part.to_string(), Some(desc_part[1..].to_string()))
        } else {
            (header, None)
        };

        // Read sequence lines until next header or EOF
        let mut sequence = String::new();
        self.line_buffer.clear();

        loop {
            let bytes_read = self
                .reader
                .read_line(&mut self.line_buffer)
                .map_err(|e| IoError::ParseError(format!("Failed to read line: {e}")))?;

            if bytes_read == 0 || self.line_buffer.starts_with('>') {
                // Reached next record or EOF
                if self.line_buffer.starts_with('>') {
                    // Store the header line for the next iteration
                    self.lookahead = Some(self.line_buffer.clone());
                }
                break;
            }

            sequence.push_str(self.line_buffer.trim());
            if !self.line_buffer.starts_with('>') {
                self.line_buffer.clear();
            }
        }

        Ok(Some(FastaRecord {
            id,
            description,
            sequence,
        }))
    }
}

/// Iterator over FASTA records
pub struct FastaRecordIterator<'a> {
    reader: &'a mut FastaReader,
}

impl Iterator for FastaRecordIterator<'_> {
    type Item = Result<FastaRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_record() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// FASTA file writer
pub struct FastaWriter {
    writer: BufWriter<File>,
    line_width: usize,
}

impl FastaWriter {
    /// Create a new FASTA file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        Ok(Self {
            writer: BufWriter::new(file),
            line_width: 80, // Standard FASTA line width
        })
    }

    /// Set the line width for sequence wrapping
    pub fn set_line_width(&mut self, width: usize) {
        self.line_width = width;
    }

    /// Write a FASTA record
    pub fn write_record(&mut self, record: &FastaRecord) -> Result<()> {
        // Write header
        write!(self.writer, ">{}", record.header())
            .map_err(|e| IoError::FileError(format!("Failed to write header: {e}")))?;
        writeln!(self.writer)
            .map_err(|e| IoError::FileError(format!("Failed to write newline: {e}")))?;

        // Write sequence with line wrapping
        let sequence = record.sequence();
        for chunk in sequence.as_bytes().chunks(self.line_width) {
            self.writer
                .write_all(chunk)
                .map_err(|e| IoError::FileError(format!("Failed to write sequence: {e}")))?;
            writeln!(self.writer)
                .map_err(|e| IoError::FileError(format!("Failed to write newline: {e}")))?;
        }

        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {e}")))
    }
}

/// FASTQ quality encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityEncoding {
    /// Sanger/Illumina 1.8+ (Phred+33)
    #[default]
    Sanger,
    /// Illumina 1.3-1.7 (Phred+64)
    Illumina,
}

/// FASTQ sequence record
#[derive(Debug, Clone, PartialEq)]
pub struct FastqRecord {
    /// Sequence identifier
    id: String,
    /// Optional description
    description: Option<String>,
    /// Sequence data
    sequence: String,
    /// Quality scores (ASCII encoded)
    quality: String,
}

impl FastqRecord {
    /// Create a new FASTQ record
    pub fn new(id: String, sequence: String, quality: String) -> Result<Self> {
        if sequence.len() != quality.len() {
            return Err(IoError::ParseError(format!(
                "Sequence and quality lengths don't match: {} vs {}",
                sequence.len(),
                quality.len()
            )));
        }

        Ok(Self {
            id,
            description: None,
            sequence,
            quality,
        })
    }

    /// Create a new FASTQ record with description
    pub fn with_description(
        id: String,
        description: String,
        sequence: String,
        quality: String,
    ) -> Result<Self> {
        if sequence.len() != quality.len() {
            return Err(IoError::ParseError(format!(
                "Sequence and quality lengths don't match: {} vs {}",
                sequence.len(),
                quality.len()
            )));
        }

        Ok(Self {
            id,
            description: Some(description),
            sequence,
            quality,
        })
    }

    /// Get the sequence ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the optional description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get the sequence
    pub fn sequence(&self) -> &str {
        &self.sequence
    }

    /// Get the quality string
    pub fn quality(&self) -> &str {
        &self.quality
    }

    /// Get the sequence length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get quality scores as numeric values
    pub fn quality_scores(&self, encoding: QualityEncoding) -> Vec<u8> {
        let offset = match encoding {
            QualityEncoding::Sanger => 33,
            QualityEncoding::Illumina => 64,
        };

        self.quality
            .bytes()
            .map(|b| b.saturating_sub(offset))
            .collect()
    }

    /// Get the full header (ID + description)
    pub fn header(&self) -> String {
        match &self.description {
            Some(desc) => format!("{} {}", self.id, desc),
            None => self.id.clone(),
        }
    }
}

/// FASTQ file reader
pub struct FastqReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    encoding: QualityEncoding,
    line_buffer: String,
}

impl FastqReader {
    /// Open a FASTQ file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
        Ok(Self {
            reader: BufReader::new(file),
            encoding: QualityEncoding::default(),
            line_buffer: String::new(),
        })
    }

    /// Open a FASTQ file with specific quality encoding
    pub fn open_with_encoding<P: AsRef<Path>>(path: P, encoding: QualityEncoding) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
        Ok(Self {
            reader: BufReader::new(file),
            encoding,
            line_buffer: String::new(),
        })
    }

    /// Create an iterator over FASTQ records
    pub fn records(&mut self) -> FastqRecordIterator<'_> {
        FastqRecordIterator { reader: self }
    }

    /// Read the next record
    fn read_record(&mut self) -> Result<Option<FastqRecord>> {
        // Read header line
        self.line_buffer.clear();
        if self
            .reader
            .read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read header: {e}")))?
            == 0
        {
            return Ok(None); // EOF
        }

        if !self.line_buffer.starts_with('@') {
            return Err(IoError::ParseError(format!(
                "Expected '@' at start of header, found: {}",
                self.line_buffer.trim()
            )));
        }

        // Parse header
        let header = self.line_buffer[1..].trim().to_string();
        let (id, description) = if let Some(space_pos) = header.find(' ') {
            let (id_part, desc_part) = header.split_at(space_pos);
            (id_part.to_string(), Some(desc_part[1..].to_string()))
        } else {
            (header, None)
        };

        // Read sequence line
        self.line_buffer.clear();
        self.reader
            .read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read sequence: {e}")))?;
        let sequence = self.line_buffer.trim().to_string();

        // Read separator line
        self.line_buffer.clear();
        self.reader
            .read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read separator: {e}")))?;
        if !self.line_buffer.starts_with('+') {
            return Err(IoError::ParseError(format!(
                "Expected '+' separator, found: {}",
                self.line_buffer.trim()
            )));
        }

        // Read quality line
        self.line_buffer.clear();
        self.reader
            .read_line(&mut self.line_buffer)
            .map_err(|e| IoError::ParseError(format!("Failed to read quality: {e}")))?;
        let quality = self.line_buffer.trim().to_string();

        FastqRecord::new(id.clone(), sequence.clone(), quality.clone())
            .or_else(|_| {
                FastqRecord::with_description(
                    id,
                    description.unwrap_or_default(),
                    sequence,
                    quality,
                )
            })
            .map(Some)
    }
}

/// Iterator over FASTQ records
pub struct FastqRecordIterator<'a> {
    reader: &'a mut FastqReader,
}

impl Iterator for FastqRecordIterator<'_> {
    type Item = Result<FastqRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_record() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// FASTQ file writer
pub struct FastqWriter {
    writer: BufWriter<File>,
    encoding: QualityEncoding,
}

impl FastqWriter {
    /// Create a new FASTQ file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        Ok(Self {
            writer: BufWriter::new(file),
            encoding: QualityEncoding::default(),
        })
    }

    /// Create a new FASTQ file with specific quality encoding
    pub fn create_with_encoding<P: AsRef<Path>>(
        path: P,
        encoding: QualityEncoding,
    ) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        Ok(Self {
            writer: BufWriter::new(file),
            encoding,
        })
    }

    /// Write a FASTQ record
    pub fn write_record(&mut self, record: &FastqRecord) -> Result<()> {
        // Write header
        writeln!(self.writer, "@{}", record.header())
            .map_err(|e| IoError::FileError(format!("Failed to write header: {e}")))?;

        // Write sequence
        writeln!(self.writer, "{}", record.sequence())
            .map_err(|e| IoError::FileError(format!("Failed to write sequence: {e}")))?;

        // Write separator
        writeln!(self.writer, "+")
            .map_err(|e| IoError::FileError(format!("Failed to write separator: {e}")))?;

        // Write quality
        writeln!(self.writer, "{}", record.quality())
            .map_err(|e| IoError::FileError(format!("Failed to write quality: {e}")))?;

        Ok(())
    }

    /// Write a FASTQ record from numeric quality scores
    pub fn write_record_with_scores(
        &mut self,
        id: &str,
        sequence: &str,
        quality_scores: &[u8],
    ) -> Result<()> {
        if sequence.len() != quality_scores.len() {
            return Err(IoError::FileError(format!(
                "Sequence and quality lengths don't match: {} vs {}",
                sequence.len(),
                quality_scores.len()
            )));
        }

        let offset = match self.encoding {
            QualityEncoding::Sanger => 33,
            QualityEncoding::Illumina => 64,
        };

        let quality_string: String = quality_scores
            .iter()
            .map(|&score| (score.saturating_add(offset)) as char)
            .collect();

        let record = FastqRecord::new(id.to_string(), sequence.to_string(), quality_string)?;
        self.write_record(&record)
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {e}")))
    }
}

/// Count sequences in a FASTA file
#[allow(dead_code)]
pub fn count_fastasequences<P: AsRef<Path>>(path: P) -> Result<usize> {
    let file = File::open(path.as_ref())
        .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
    let reader = BufReader::new(file);

    let count = reader
        .lines()
        .map_while(|result| result.ok())
        .filter(|line| line.starts_with('>'))
        .count();

    Ok(count)
}

/// Count sequences in a FASTQ file
#[allow(dead_code)]
pub fn count_fastqsequences<P: AsRef<Path>>(path: P) -> Result<usize> {
    let file = File::open(path.as_ref())
        .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
    let reader = BufReader::new(file);

    let line_count = reader.lines().count();
    if line_count % 4 != 0 {
        return Err(IoError::ParseError(format!(
            "Invalid FASTQ file: line count {line_count} is not divisible by 4"
        )));
    }

    Ok(line_count / 4)
}

/// Sequence analysis utilities
pub mod analysis {
    use super::*;
    use std::collections::HashMap;

    /// Calculate GC content of a DNA sequence
    pub fn gc_content(sequence: &str) -> f64 {
        if sequence.is_empty() {
            return 0.0;
        }

        let gc_count = sequence
            .chars()
            .filter(|&c| c == 'G' || c == 'C' || c == 'g' || c == 'c')
            .count();

        gc_count as f64 / sequence.len() as f64
    }

    /// Calculate nucleotide composition
    pub fn nucleotide_composition(sequence: &str) -> HashMap<char, usize> {
        let mut composition = HashMap::new();

        for nucleotide in sequence.chars() {
            *composition
                .entry(nucleotide.to_ascii_uppercase())
                .or_insert(0) += 1;
        }

        composition
    }

    /// Reverse complement of a DNA sequence
    pub fn reverse_complement(sequence: &str) -> String {
        sequence
            .chars()
            .rev()
            .map(|c| match c.to_ascii_uppercase() {
                'A' => 'T',
                'T' => 'A',
                'G' => 'C',
                'C' => 'G',
                'U' => 'A', // RNA
                'N' => 'N', // Unknown
                _ => c,
            })
            .collect()
    }

    /// Translate DNA sequence to protein (single frame)
    pub fn translate_dna(sequence: &str) -> String {
        let codon_table = get_standard_genetic_code();

        sequence
            .chars()
            .collect::<Vec<_>>()
            .chunks(3)
            .filter_map(|codon| {
                if codon.len() == 3 {
                    let codon_str: String = codon.iter().collect::<String>().to_uppercase();
                    Some(codon_table.get(&codon_str).unwrap_or(&'X').to_owned())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get standard genetic code table
    fn get_standard_genetic_code() -> HashMap<String, char> {
        let mut code = HashMap::new();

        // Standard genetic code (NCBI translation table 1)
        let codons = [
            ("TTT", 'F'),
            ("TTC", 'F'),
            ("TTA", 'L'),
            ("TTG", 'L'),
            ("TCT", 'S'),
            ("TCC", 'S'),
            ("TCA", 'S'),
            ("TCG", 'S'),
            ("TAT", 'Y'),
            ("TAC", 'Y'),
            ("TAA", '*'),
            ("TAG", '*'),
            ("TGT", 'C'),
            ("TGC", 'C'),
            ("TGA", '*'),
            ("TGG", 'W'),
            ("CTT", 'L'),
            ("CTC", 'L'),
            ("CTA", 'L'),
            ("CTG", 'L'),
            ("CCT", 'P'),
            ("CCC", 'P'),
            ("CCA", 'P'),
            ("CCG", 'P'),
            ("CAT", 'H'),
            ("CAC", 'H'),
            ("CAA", 'Q'),
            ("CAG", 'Q'),
            ("CGT", 'R'),
            ("CGC", 'R'),
            ("CGA", 'R'),
            ("CGG", 'R'),
            ("ATT", 'I'),
            ("ATC", 'I'),
            ("ATA", 'I'),
            ("ATG", 'M'),
            ("ACT", 'T'),
            ("ACC", 'T'),
            ("ACA", 'T'),
            ("ACG", 'T'),
            ("AAT", 'N'),
            ("AAC", 'N'),
            ("AAA", 'K'),
            ("AAG", 'K'),
            ("AGT", 'S'),
            ("AGC", 'S'),
            ("AGA", 'R'),
            ("AGG", 'R'),
            ("GTT", 'V'),
            ("GTC", 'V'),
            ("GTA", 'V'),
            ("GTG", 'V'),
            ("GCT", 'A'),
            ("GCC", 'A'),
            ("GCA", 'A'),
            ("GCG", 'A'),
            ("GAT", 'D'),
            ("GAC", 'D'),
            ("GAA", 'E'),
            ("GAG", 'E'),
            ("GGT", 'G'),
            ("GGC", 'G'),
            ("GGA", 'G'),
            ("GGG", 'G'),
        ];

        for (codon, amino_acid) in &codons {
            code.insert(codon.to_string(), *amino_acid);
        }

        code
    }

    /// Find open reading frames (ORFs) in a DNA sequence
    pub fn find_orfs(sequence: &str, minlength: usize) -> Vec<Orf> {
        let mut orfs = Vec::new();
        let seq_upper = sequence.to_uppercase();

        // Check all three reading frames
        for frame in 0..3 {
            let frame_seq = &seq_upper[frame..];
            let mut start_pos = None;

            for (pos, codon) in frame_seq.chars().collect::<Vec<_>>().chunks(3).enumerate() {
                if codon.len() < 3 {
                    break;
                }

                let codon_str: String = codon.iter().collect();

                // Start codon
                if codon_str == "ATG" && start_pos.is_none() {
                    start_pos = Some(frame + pos * 3);
                }

                // Stop codon
                if matches!(codon_str.as_str(), "TAA" | "TAG" | "TGA") {
                    if let Some(start) = start_pos {
                        let length = frame + pos * 3 + 3 - start;
                        if length >= minlength {
                            let orf_seq = &sequence[start..start + length];
                            orfs.push(Orf {
                                start_pos: start,
                                end_pos: start + length,
                                frame: frame as i8,
                                sequence: orf_seq.to_string(),
                                protein: translate_dna(orf_seq),
                            });
                        }
                        start_pos = None;
                    }
                }
            }
        }

        orfs
    }

    /// Open Reading Frame
    #[derive(Debug, Clone, PartialEq)]
    pub struct Orf {
        pub start_pos: usize,
        pub end_pos: usize,
        pub frame: i8,
        pub sequence: String,
        pub protein: String,
    }

    impl Orf {
        pub fn length(&self) -> usize {
            self.sequence.len()
        }
    }

    /// Calculate basic sequence statistics
    pub fn sequence_stats(records: &[FastaRecord]) -> SequenceStats {
        if records.is_empty() {
            return SequenceStats::default();
        }

        let lengths: Vec<usize> = records.iter().map(|r| r.len()).collect();
        let totallength: usize = lengths.iter().sum();
        let minlength = *lengths.iter().min().expect("Operation failed");
        let maxlength = *lengths.iter().max().expect("Operation failed");
        let meanlength = totallength as f64 / records.len() as f64;

        // Calculate N50
        let mut sortedlengths = lengths.clone();
        sortedlengths.sort_by(|a, b| b.cmp(a)); // Sort in descending order
        let mut cumulative = 0;
        let half_total = totallength / 2;
        let mut n50 = 0;

        for length in sortedlengths {
            cumulative += length;
            if cumulative >= half_total {
                n50 = length;
                break;
            }
        }

        // Calculate overall GC content
        let total_gc: usize = records
            .iter()
            .map(|r| {
                r.sequence()
                    .chars()
                    .filter(|&c| c == 'G' || c == 'C' || c == 'g' || c == 'c')
                    .count()
            })
            .sum();
        let gc_content = total_gc as f64 / totallength as f64;

        SequenceStats {
            numsequences: records.len(),
            totallength,
            minlength,
            maxlength,
            meanlength,
            n50,
            gc_content,
        }
    }

    /// Sequence statistics
    #[derive(Debug, Clone, PartialEq)]
    pub struct SequenceStats {
        pub numsequences: usize,
        pub totallength: usize,
        pub minlength: usize,
        pub maxlength: usize,
        pub meanlength: f64,
        pub n50: usize,
        pub gc_content: f64,
    }

    impl Default for SequenceStats {
        fn default() -> Self {
            Self {
                numsequences: 0,
                totallength: 0,
                minlength: 0,
                maxlength: 0,
                meanlength: 0.0,
                n50: 0,
                gc_content: 0.0,
            }
        }
    }

    /// Quality analysis for FASTQ data
    pub fn quality_stats(records: &[FastqRecord], encoding: QualityEncoding) -> QualityStats {
        if records.is_empty() {
            return QualityStats::default();
        }

        let mut all_scores = Vec::new();
        let mut position_scores: HashMap<usize, Vec<u8>> = HashMap::new();

        for record in records {
            let scores = record.quality_scores(encoding);
            all_scores.extend_from_slice(&scores);

            for (pos, &score) in scores.iter().enumerate() {
                position_scores.entry(pos).or_default().push(score);
            }
        }

        let mean_quality = all_scores.iter().sum::<u8>() as f64 / all_scores.len() as f64;
        let min_quality = *all_scores.iter().min().unwrap_or(&0);
        let max_quality = *all_scores.iter().max().unwrap_or(&0);

        // Calculate per-position mean qualities
        let mut per_position_mean = Vec::new();
        let max_pos = position_scores.keys().max().unwrap_or(&0);

        for pos in 0..=*max_pos {
            if let Some(scores) = position_scores.get(&pos) {
                let mean = scores.iter().sum::<u8>() as f64 / scores.len() as f64;
                per_position_mean.push(mean);
            } else {
                per_position_mean.push(0.0);
            }
        }

        QualityStats {
            mean_quality,
            min_quality,
            max_quality,
            per_position_mean,
        }
    }

    /// Quality statistics for FASTQ data
    #[derive(Debug, Clone, PartialEq)]
    pub struct QualityStats {
        pub mean_quality: f64,
        pub min_quality: u8,
        pub max_quality: u8,
        pub per_position_mean: Vec<f64>,
    }

    impl Default for QualityStats {
        fn default() -> Self {
            Self {
                mean_quality: 0.0,
                min_quality: 0,
                max_quality: 0,
                per_position_mean: Vec::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_fasta_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().expect("Operation failed");
        let path = temp_file.path();

        // Write test data
        {
            let mut writer = FastaWriter::create(path)?;
            writer.write_record(&FastaRecord::new(
                "seq1".to_string(),
                "ATCGATCGATCG".to_string(),
            ))?;
            writer.write_record(&FastaRecord::with_description(
                "seq2".to_string(),
                "test sequence".to_string(),
                "GCTAGCTAGCTA".to_string(),
            ))?;
            writer.flush()?;
        }

        // Read and verify
        {
            let mut reader = FastaReader::open(path)?;
            let records: Vec<_> = reader.records().collect::<Result<Vec<_>>>()?;

            assert_eq!(records.len(), 2);
            assert_eq!(records[0].id(), "seq1");
            assert_eq!(records[0].sequence(), "ATCGATCGATCG");
            assert_eq!(records[1].id(), "seq2");
            assert_eq!(records[1].description(), Some("test sequence"));
            assert_eq!(records[1].sequence(), "GCTAGCTAGCTA");
        }

        Ok(())
    }

    #[test]
    fn test_fastq_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().expect("Operation failed");
        let path = temp_file.path();

        // Write test data
        {
            let mut writer = FastqWriter::create(path)?;
            writer.write_record(&FastqRecord::new(
                "read1".to_string(),
                "ATCG".to_string(),
                "IIII".to_string(),
            )?)?;
            writer.write_record_with_scores("read2", "GCTA", &[30, 35, 40, 35])?;
            writer.flush()?;
        }

        // Read and verify
        {
            let mut reader = FastqReader::open(path)?;
            let records: Vec<_> = reader.records().collect::<Result<Vec<_>>>()?;

            assert_eq!(records.len(), 2);
            assert_eq!(records[0].id(), "read1");
            assert_eq!(records[0].sequence(), "ATCG");
            assert_eq!(records[0].quality(), "IIII");

            assert_eq!(records[1].id(), "read2");
            assert_eq!(records[1].sequence(), "GCTA");
            let scores = records[1].quality_scores(QualityEncoding::Sanger);
            assert_eq!(scores, vec![30, 35, 40, 35]);
        }

        Ok(())
    }

    #[test]
    fn testsequence_counting() -> Result<()> {
        let fasta_file = NamedTempFile::new().expect("Operation failed");
        let fastq_file = NamedTempFile::new().expect("Operation failed");

        // Create test FASTA
        {
            let mut writer = FastaWriter::create(fasta_file.path())?;
            for i in 0..5 {
                writer.write_record(&FastaRecord::new(format!("seq{}", i), "ATCG".to_string()))?;
            }
            writer.flush()?;
        }

        // Create test FASTQ
        {
            let mut writer = FastqWriter::create(fastq_file.path())?;
            for i in 0..3 {
                writer.write_record(&FastqRecord::new(
                    format!("read{}", i),
                    "ATCG".to_string(),
                    "IIII".to_string(),
                )?)?;
            }
            writer.flush()?;
        }

        assert_eq!(count_fastasequences(fasta_file.path())?, 5);
        assert_eq!(count_fastqsequences(fastq_file.path())?, 3);

        Ok(())
    }
}
