//! Lossless JPEG decoder (ITU-T T.81 Annex H, process 14, selection value 1).
//!
//! DNG stores raw Bayer CFA data using *lossless* JPEG (TIFF compression 7),
//! which is an entirely different algorithm from baseline DCT JPEG and from
//! JPEG-LS (ITU-T T.87). Lossless JPEG predicts each sample from neighbouring
//! samples and Huffman-codes the prediction *difference* as a magnitude
//! category plus mantissa bits — there is no DCT, no quantization.
//!
//! # Algorithm (T.81 §H.1)
//!
//! For every sample of every component, a predictor `Px` is formed from the
//! already-decoded neighbours `Ra` (left), `Rb` (above) and `Rc` (above-left):
//!
//! | Selector | Prediction                          |
//! |----------|-------------------------------------|
//! | 0        | no prediction (only DC differential)|
//! | 1        | `Px = Ra`                           |
//! | 2        | `Px = Rb`                           |
//! | 3        | `Px = Rc`                           |
//! | 4        | `Px = Ra + Rb - Rc`                 |
//! | 5        | `Px = Ra + ((Rb - Rc) >> 1)`        |
//! | 6        | `Px = Rb + ((Ra - Rc) >> 1)`        |
//! | 7        | `Px = (Ra + Rb) >> 1`               |
//!
//! At the start of the first line `Px = 2^(P-Pt-1)` (`Ra`-substitute); at the
//! start of every subsequent line `Px = Rb`. The reconstructed sample is
//! `Px + DIFF`, taken modulo `2^16` per the standard.
//!
//! # DNG CFA packing
//!
//! Adobe's DNG converter packs a Bayer mosaic into a lossless-JPEG image whose
//! component count equals the number of CFA columns in the tile width divisor
//! (commonly 2): the JPEG is `cfa_width/2` wide with 2 components, and the two
//! components are the left/right sample of every horizontal pixel pair. The
//! caller supplies the true CFA dimensions and this decoder de-interleaves the
//! components back into a single raster.

use crate::error::{ImageError, ImageResult};

/// Lossless-JPEG Start-of-Frame marker (process 14, Huffman, non-differential).
const MARKER_SOF3: u8 = 0xC3;
/// Define-Huffman-Table marker.
const MARKER_DHT: u8 = 0xC4;
/// Start-of-Scan marker.
const MARKER_SOS: u8 = 0xDA;
/// Define-Restart-Interval marker.
const MARKER_DRI: u8 = 0xDD;
/// Start-of-Image marker.
const MARKER_SOI: u8 = 0xD8;
/// End-of-Image marker.
const MARKER_EOI: u8 = 0xD9;

/// One Huffman table decoded from a DHT segment.
///
/// Lossless JPEG only uses DC (class 0) tables; the symbol is the magnitude
/// category `SSSS` (0..=16). We pre-expand a fast lookup over the canonical
/// codes.
#[derive(Clone, Default)]
struct HuffTable {
    /// `max_code[l]` = largest canonical code of length `l` (1..=16), or -1.
    max_code: [i32; 17],
    /// `min_code[l]` = smallest canonical code of length `l` (1..=16).
    min_code: [i32; 17],
    /// `val_ptr[l]` = index into `values` for the first symbol of length `l`.
    val_ptr: [usize; 17],
    /// Symbols in canonical (code) order.
    values: Vec<u8>,
}

impl HuffTable {
    /// Build the canonical decoding tables from per-length code counts.
    fn build(counts: &[u8; 16], values: Vec<u8>) -> Self {
        let mut table = HuffTable {
            values,
            ..Default::default()
        };
        // Canonical-code generation per T.81 Annex C / F.
        let mut code: i32 = 0;
        let mut value_index = 0usize;
        for length in 1..=16usize {
            let count = counts[length - 1] as usize;
            if count == 0 {
                table.max_code[length] = -1;
            } else {
                table.val_ptr[length] = value_index;
                table.min_code[length] = code;
                code += count as i32;
                table.max_code[length] = code - 1;
                value_index += count;
            }
            code <<= 1;
        }
        table
    }

    /// Decode one symbol (magnitude category) from the bit reader.
    fn decode_symbol(&self, reader: &mut BitReader<'_>) -> ImageResult<u8> {
        let mut code: i32 = 0;
        for length in 1..=16usize {
            code = (code << 1) | i32::from(reader.read_bit()?);
            let max = self.max_code[length];
            if max >= 0 && code <= max {
                let index = self.val_ptr[length] + (code - self.min_code[length]) as usize;
                return self
                    .values
                    .get(index)
                    .copied()
                    .ok_or_else(|| ImageError::compression("Lossless JPEG: bad Huffman symbol"));
            }
        }
        Err(ImageError::compression(
            "Lossless JPEG: Huffman code not found within 16 bits",
        ))
    }
}

/// MSB-first bit reader over an entropy-coded segment.
///
/// Handles JPEG byte-stuffing (`0xFF 0x00` → `0xFF`) and stops cleanly at a
/// non-stuff marker (the next `0xFF Xx`). Restart markers are consumed by the
/// caller via [`BitReader::skip_to_marker`].
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buffer: u32,
    bits_in_buffer: u32,
    /// Set when the stream hit a real marker (anything but `0xFF00`).
    hit_marker: Option<u8>,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit_buffer: 0,
            bits_in_buffer: 0,
            hit_marker: None,
        }
    }

    /// Fetch the next entropy byte, transparently de-stuffing.
    ///
    /// Returns `None` once a real marker is reached (recorded in `hit_marker`).
    fn next_byte(&mut self) -> Option<u8> {
        if self.pos >= self.data.len() {
            return None;
        }
        let byte = self.data[self.pos];
        if byte != 0xFF {
            self.pos += 1;
            return Some(byte);
        }
        // 0xFF: inspect the following byte.
        if self.pos + 1 >= self.data.len() {
            self.pos += 1;
            return None;
        }
        let marker = self.data[self.pos + 1];
        if marker == 0x00 {
            // Stuffed 0xFF.
            self.pos += 2;
            Some(0xFF)
        } else if marker == 0xFF {
            // Fill byte — skip a single 0xFF and retry.
            self.pos += 1;
            self.next_byte()
        } else {
            // Genuine marker — do not consume; report it.
            self.hit_marker = Some(marker);
            None
        }
    }

    /// Read a single bit (MSB-first).
    fn read_bit(&mut self) -> ImageResult<u8> {
        if self.bits_in_buffer == 0 {
            let byte = self.next_byte().ok_or_else(|| {
                ImageError::compression("Lossless JPEG: entropy data ended prematurely")
            })?;
            self.bit_buffer = u32::from(byte);
            self.bits_in_buffer = 8;
        }
        self.bits_in_buffer -= 1;
        Ok(((self.bit_buffer >> self.bits_in_buffer) & 1) as u8)
    }

    /// Read `n` bits (0..=16) as an unsigned MSB-first integer.
    fn read_bits(&mut self, n: u8) -> ImageResult<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Discard buffered bits and advance to the next marker byte.
    ///
    /// Returns the marker code (the byte after `0xFF`) or `None` at EOF.
    fn skip_to_marker(&mut self) -> Option<u8> {
        self.bit_buffer = 0;
        self.bits_in_buffer = 0;
        if let Some(marker) = self.hit_marker.take() {
            // We are positioned on the `0xFF`; step past `0xFF Xx`.
            self.pos += 2;
            return Some(marker);
        }
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let marker = self.data[self.pos + 1];
                if marker != 0x00 && marker != 0xFF {
                    self.pos += 2;
                    return Some(marker);
                }
            }
            self.pos += 1;
        }
        None
    }
}

/// Extend a magnitude-coded value to its signed difference (T.81 Table H.2).
///
/// `ssss` is the magnitude category; `bits` holds the `ssss` mantissa bits.
fn extend_difference(bits: u32, ssss: u8) -> i32 {
    if ssss == 0 {
        return 0;
    }
    // Special category 16 (lossless only): DIFF == 32768.
    if ssss == 16 {
        return 32768;
    }
    let vt = 1i32 << (ssss - 1);
    let v = bits as i32;
    if v < vt {
        v - (1 << ssss) + 1
    } else {
        v
    }
}

/// A component descriptor parsed from the SOF3 segment.
#[derive(Clone, Copy)]
struct FrameComponent {
    /// Component identifier (matched against the scan component selectors).
    id: u8,
    /// Horizontal sampling factor.
    h: u8,
    /// Vertical sampling factor.
    v: u8,
}

/// SOF3 frame parameters.
struct FrameHeader {
    /// Sample precision in bits (`P`).
    precision: u8,
    /// Number of lines (`Y`).
    lines: u16,
    /// Samples per line (`X`).
    samples_per_line: u16,
    /// Frame components.
    components: Vec<FrameComponent>,
}

/// A scan component, pairing a frame component with its Huffman table id.
#[derive(Clone, Copy)]
struct ScanComponent {
    /// Index into [`FrameHeader::components`].
    frame_index: usize,
    /// DC Huffman table selector (`Td`).
    huff_id: u8,
}

/// Decoded lossless-JPEG image: one interleaved sample raster.
pub struct LosslessJpegImage {
    /// Width in samples (per component).
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Number of components.
    pub components: u8,
    /// Sample precision in bits.
    pub precision: u8,
    /// Component-interleaved samples, row-major: `samples[(y*width + x)*components + c]`.
    pub samples: Vec<u16>,
}

/// Decode a complete lossless-JPEG (SOF3) datastream.
///
/// # Errors
///
/// Returns an error if the stream is not a lossless JPEG, uses an unsupported
/// predictor/precision, or the entropy data is malformed.
pub fn decode_lossless_jpeg(data: &[u8]) -> ImageResult<LosslessJpegImage> {
    let mut parser = SegmentParser::new(data)?;

    let mut frame: Option<FrameHeader> = None;
    let mut huff_tables: [Option<HuffTable>; 4] = Default::default();
    let mut restart_interval: u32 = 0;

    loop {
        let marker = parser
            .next_marker()
            .ok_or_else(|| ImageError::compression("Lossless JPEG: no SOS marker found"))?;
        match marker {
            MARKER_SOF3 => {
                frame = Some(parser.parse_sof3()?);
            }
            MARKER_DHT => {
                parser.parse_dht(&mut huff_tables)?;
            }
            MARKER_DRI => {
                restart_interval = parser.parse_dri()?;
            }
            MARKER_SOS => {
                let frame = frame
                    .ok_or_else(|| ImageError::compression("Lossless JPEG: SOS before SOF3"))?;
                let (scan_components, predictor, point_transform) = parser.parse_sos(&frame)?;
                let entropy = &data[parser.pos..];
                return decode_scan(
                    &frame,
                    &scan_components,
                    &huff_tables,
                    predictor,
                    point_transform,
                    restart_interval,
                    entropy,
                );
            }
            MARKER_EOI => {
                return Err(ImageError::compression(
                    "Lossless JPEG: reached EOI before scan data",
                ));
            }
            // SOF0/1/2/5.. and other DCT processes are not lossless JPEG.
            0xC0 | 0xC1 | 0xC2 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF => {
                return Err(ImageError::unsupported(
                    "DNG strip is a DCT JPEG, not lossless JPEG (process 14)",
                ));
            }
            _ => {
                // APPn, COM, DAC, DNL, etc. — skip the segment by length.
                parser.skip_segment()?;
            }
        }
    }
}

/// Stateful walker over JPEG marker segments.
struct SegmentParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SegmentParser<'a> {
    fn new(data: &'a [u8]) -> ImageResult<Self> {
        if data.len() < 2 || data[0] != 0xFF || data[1] != MARKER_SOI {
            return Err(ImageError::compression("Lossless JPEG: missing SOI marker"));
        }
        Ok(SegmentParser { data, pos: 2 })
    }

    /// Advance to the next marker, returning its code (byte after `0xFF`).
    fn next_marker(&mut self) -> Option<u8> {
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let marker = self.data[self.pos + 1];
                if marker != 0x00 && marker != 0xFF {
                    self.pos += 2;
                    return Some(marker);
                }
            }
            self.pos += 1;
        }
        None
    }

    /// Read a big-endian `u16` at the current position and advance.
    fn read_u16(&mut self) -> ImageResult<u16> {
        if self.pos + 2 > self.data.len() {
            return Err(ImageError::compression(
                "Lossless JPEG: truncated segment length",
            ));
        }
        let value = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(value)
    }

    fn read_u8(&mut self) -> ImageResult<u8> {
        let byte = self
            .data
            .get(self.pos)
            .copied()
            .ok_or_else(|| ImageError::compression("Lossless JPEG: truncated segment"))?;
        self.pos += 1;
        Ok(byte)
    }

    /// Skip a length-prefixed marker segment.
    fn skip_segment(&mut self) -> ImageResult<()> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            return Err(ImageError::compression(
                "Lossless JPEG: invalid segment length",
            ));
        }
        self.pos = self
            .pos
            .checked_add(length - 2)
            .filter(|&p| p <= self.data.len())
            .ok_or_else(|| ImageError::compression("Lossless JPEG: segment overruns data"))?;
        Ok(())
    }

    /// Parse an SOF3 (lossless, Huffman) frame header.
    fn parse_sof3(&mut self) -> ImageResult<FrameHeader> {
        let length = self.read_u16()? as usize;
        let segment_end = self.pos + length - 2;
        let precision = self.read_u8()?;
        let lines = self.read_u16()?;
        let samples_per_line = self.read_u16()?;
        let component_count = self.read_u8()? as usize;
        if !(1..=4).contains(&component_count) {
            return Err(ImageError::unsupported(format!(
                "Lossless JPEG: unsupported component count {component_count}"
            )));
        }
        if !(2..=16).contains(&precision) {
            return Err(ImageError::unsupported(format!(
                "Lossless JPEG: unsupported precision {precision}"
            )));
        }
        let mut components = Vec::with_capacity(component_count);
        for _ in 0..component_count {
            let id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let _quant_table = self.read_u8()?; // unused in lossless
            components.push(FrameComponent {
                id,
                h: (sampling >> 4) & 0x0F,
                v: sampling & 0x0F,
            });
        }
        if self.pos > segment_end {
            return Err(ImageError::compression(
                "Lossless JPEG: SOF3 segment overran declared length",
            ));
        }
        self.pos = segment_end;
        Ok(FrameHeader {
            precision,
            lines,
            samples_per_line,
            components,
        })
    }

    /// Parse a DHT segment (may carry several tables).
    fn parse_dht(&mut self, tables: &mut [Option<HuffTable>; 4]) -> ImageResult<()> {
        let length = self.read_u16()? as usize;
        let segment_end = self.pos + length - 2;
        while self.pos < segment_end {
            let tc_th = self.read_u8()?;
            let table_class = (tc_th >> 4) & 0x0F;
            let table_id = (tc_th & 0x0F) as usize;
            if table_id >= 4 {
                return Err(ImageError::compression(
                    "Lossless JPEG: Huffman table id out of range",
                ));
            }
            // Lossless JPEG uses only DC (class 0) tables; tolerate class 1
            // tables in the stream by parsing them but storing under the id.
            let mut counts = [0u8; 16];
            let mut total = 0usize;
            for slot in &mut counts {
                *slot = self.read_u8()?;
                total += *slot as usize;
            }
            let mut values = Vec::with_capacity(total);
            for _ in 0..total {
                values.push(self.read_u8()?);
            }
            let _ = table_class;
            tables[table_id] = Some(HuffTable::build(&counts, values));
        }
        self.pos = segment_end;
        Ok(())
    }

    /// Parse a DRI segment, returning the restart interval (MCU count).
    fn parse_dri(&mut self) -> ImageResult<u32> {
        let length = self.read_u16()? as usize;
        if length != 4 {
            return Err(ImageError::compression(
                "Lossless JPEG: malformed DRI segment",
            ));
        }
        Ok(u32::from(self.read_u16()?))
    }

    /// Parse an SOS segment.
    ///
    /// Returns the scan components, the predictor selector `Ss`, and the point
    /// transform `Al`.
    fn parse_sos(&mut self, frame: &FrameHeader) -> ImageResult<(Vec<ScanComponent>, u8, u8)> {
        let length = self.read_u16()? as usize;
        let segment_end = self.pos + length - 2;
        let scan_count = self.read_u8()? as usize;
        if scan_count == 0 || scan_count > frame.components.len() {
            return Err(ImageError::compression(
                "Lossless JPEG: invalid scan component count",
            ));
        }
        let mut scan_components = Vec::with_capacity(scan_count);
        for _ in 0..scan_count {
            let component_selector = self.read_u8()?;
            let td_ta = self.read_u8()?;
            let huff_id = (td_ta >> 4) & 0x0F;
            let frame_index = frame
                .components
                .iter()
                .position(|c| c.id == component_selector)
                .ok_or_else(|| {
                    ImageError::compression("Lossless JPEG: scan references unknown component")
                })?;
            scan_components.push(ScanComponent {
                frame_index,
                huff_id,
            });
        }
        // Ss = predictor selector, Se = 0, Ah/Al: Al is the point transform.
        let predictor = self.read_u8()?;
        let _se = self.read_u8()?;
        let ah_al = self.read_u8()?;
        let point_transform = ah_al & 0x0F;
        if predictor > 7 {
            return Err(ImageError::unsupported(format!(
                "Lossless JPEG: predictor selector {predictor} out of range 0..=7"
            )));
        }
        self.pos = segment_end;
        Ok((scan_components, predictor, point_transform))
    }
}

/// Apply the selected lossless predictor (T.81 §H.1.2.1).
///
/// `a` = left, `b` = above, `c` = above-left.
fn predict(selector: u8, a: i32, b: i32, c: i32) -> i32 {
    match selector {
        1 => a,
        2 => b,
        3 => c,
        4 => a + b - c,
        5 => a + ((b - c) >> 1),
        6 => b + ((a - c) >> 1),
        7 => (a + b) >> 1,
        // Selector 0 carries no spatial prediction.
        _ => 0,
    }
}

/// Decode the lossless scan into an interleaved sample raster.
fn decode_scan(
    frame: &FrameHeader,
    scan: &[ScanComponent],
    huff_tables: &[Option<HuffTable>; 4],
    predictor: u8,
    point_transform: u8,
    restart_interval: u32,
    entropy: &[u8],
) -> ImageResult<LosslessJpegImage> {
    let width = frame.samples_per_line as usize;
    let height = frame.lines as usize;
    let n_comp = scan.len();
    if width == 0 || height == 0 {
        return Err(ImageError::compression("Lossless JPEG: zero-sized frame"));
    }
    // Defend against an allocation bomb: the embedded DNG-JPEG width/height and
    // component count drive the `vec![0u16; width*height*n_comp]` sample buffer
    // below (potentially tens of GB). Reject impossibly large frames first.
    crate::limits::checked_dims(width, height, n_comp.max(1), 2)
        .map_err(ImageError::InvalidFormat)?;
    // Lossless JPEG in DNG always uses 1x1 sampling for every component.
    for sc in scan {
        let fc = &frame.components[sc.frame_index];
        if fc.h != 1 || fc.v != 1 {
            return Err(ImageError::unsupported(
                "Lossless JPEG: non-1x1 sampling factors are not supported",
            ));
        }
    }

    // Resolve a Huffman table per scan component.
    let mut component_tables: Vec<&HuffTable> = Vec::with_capacity(n_comp);
    for sc in scan {
        let table = huff_tables[sc.huff_id as usize].as_ref().ok_or_else(|| {
            ImageError::compression("Lossless JPEG: scan references undefined Huffman table")
        })?;
        component_tables.push(table);
    }

    let precision = frame.precision;
    // The default sample at the very first pixel: 2^(P - Pt - 1).
    let default_shift = precision.saturating_sub(point_transform);
    let default_sample: i32 = if default_shift == 0 {
        0
    } else {
        1i32 << (default_shift - 1)
    };
    // All reconstruction is modulo 2^16 (T.81 §H.1.2.1, with 16-bit samples).
    let modulo_mask: i32 = 0xFFFF;

    let mut reader = BitReader::new(entropy);
    let mut samples = vec![0u16; width * height * n_comp];

    // Restart bookkeeping: lossless JPEG counts restart intervals in MCUs;
    // for 1x1 sampling each MCU is one pixel position across all components.
    let mut mcu_since_restart: u32 = 0;

    for y in 0..height {
        for x in 0..width {
            if restart_interval != 0 && mcu_since_restart == restart_interval {
                handle_restart(&mut reader)?;
                mcu_since_restart = 0;
            }
            for c in 0..n_comp {
                let table = component_tables[c];
                let ssss = table.decode_symbol(&mut reader)?;
                if ssss > 16 {
                    return Err(ImageError::compression(
                        "Lossless JPEG: magnitude category exceeds 16",
                    ));
                }
                let mantissa = if ssss == 16 {
                    0
                } else {
                    reader.read_bits(ssss)?
                };
                let diff = extend_difference(mantissa, ssss);

                // Form the predictor from already-decoded neighbours.
                let px = compute_predictor(
                    &samples,
                    width,
                    n_comp,
                    x,
                    y,
                    c,
                    predictor,
                    default_sample,
                    restart_interval != 0 && mcu_since_restart == 0,
                );
                let value = (px + diff) & modulo_mask;
                samples[(y * width + x) * n_comp + c] = value as u16;
            }
            mcu_since_restart += 1;
        }
    }

    Ok(LosslessJpegImage {
        width: width as u32,
        height: height as u32,
        components: n_comp as u8,
        precision,
        samples,
    })
}

/// Consume an expected `RSTn` (0xD0..=0xD7) marker between restart intervals.
fn handle_restart(reader: &mut BitReader<'_>) -> ImageResult<()> {
    match reader.skip_to_marker() {
        Some(marker) if (0xD0..=0xD7).contains(&marker) => Ok(()),
        Some(MARKER_EOI) => Err(ImageError::compression(
            "Lossless JPEG: hit EOI while expecting restart marker",
        )),
        Some(other) => Err(ImageError::compression(format!(
            "Lossless JPEG: expected RSTn, found marker 0x{other:02X}"
        ))),
        None => Err(ImageError::compression(
            "Lossless JPEG: stream ended while expecting restart marker",
        )),
    }
}

/// Compute the predictor `Px` for component `c` at `(x, y)`.
///
/// `at_restart_boundary` is true for the first MCU of a restart interval,
/// where prediction is reset exactly as at the start of the image.
#[allow(clippy::too_many_arguments)]
fn compute_predictor(
    samples: &[u16],
    width: usize,
    n_comp: usize,
    x: usize,
    y: usize,
    c: usize,
    selector: u8,
    default_sample: i32,
    at_restart_boundary: bool,
) -> i32 {
    let sample_at =
        |sx: usize, sy: usize| -> i32 { i32::from(samples[(sy * width + sx) * n_comp + c]) };

    // T.81 §H.1.2.1 boundary rules:
    //   * first sample of the image / restart interval: Px = 2^(P-Pt-1)
    //   * first sample of a line (x == 0): Px = Rb (sample directly above),
    //     or the default when there is no line above.
    //   * first line (y == 0): Px = Ra (sample to the left).
    let first_pixel = (x == 0 && y == 0) || at_restart_boundary;
    if first_pixel {
        return default_sample;
    }
    if x == 0 {
        // Start of a line: predictor is the sample above.
        return sample_at(0, y - 1);
    }
    if y == 0 {
        // First line: predictor is the sample to the left.
        return sample_at(x - 1, 0);
    }
    let ra = sample_at(x - 1, y);
    let rb = sample_at(x, y - 1);
    let rc = sample_at(x - 1, y - 1);
    predict(selector, ra, rb, rc)
}

/// De-interleave a decoded lossless-JPEG image into a single CFA raster.
///
/// DNG packs a Bayer mosaic of `cfa_width × cfa_height` into a lossless-JPEG
/// image of `components` planes. The reconstructed CFA sample at column `cx`
/// is taken from component `cx % components` of JPEG column `cx / components`.
/// When the JPEG is single-component it is already a plain raster and is
/// returned unchanged (clipped/padded to the CFA dimensions).
///
/// # Errors
///
/// Returns an error if the JPEG geometry cannot cover the requested CFA size.
pub fn deinterleave_cfa(
    image: &LosslessJpegImage,
    cfa_width: u32,
    cfa_height: u32,
) -> ImageResult<Vec<u16>> {
    let cfa_w = cfa_width as usize;
    let cfa_h = cfa_height as usize;
    let comps = image.components as usize;
    let jpeg_w = image.width as usize;
    let jpeg_h = image.height as usize;

    // The decoded JPEG must provide at least as many samples per row as the
    // CFA needs, and at least as many rows.
    let provided_columns = jpeg_w * comps;
    if provided_columns < cfa_w || jpeg_h < cfa_h {
        return Err(ImageError::compression(format!(
            "Lossless JPEG geometry {}x{}x{} cannot fill CFA {}x{}",
            jpeg_w, jpeg_h, comps, cfa_w, cfa_h
        )));
    }

    let mut raster = vec![0u16; cfa_w * cfa_h];
    for y in 0..cfa_h {
        for cx in 0..cfa_w {
            let component = cx % comps;
            let jpeg_x = cx / comps;
            let sample = image.samples[(y * jpeg_w + jpeg_x) * comps + component];
            raster[y * cfa_w + cx] = sample;
        }
    }
    Ok(raster)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer used to synthesise lossless-JPEG entropy data.
    struct BitWriter {
        bytes: Vec<u8>,
        current: u8,
        filled: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            BitWriter {
                bytes: Vec::new(),
                current: 0,
                filled: 0,
            }
        }

        fn write_bit(&mut self, bit: u8) {
            self.current = (self.current << 1) | (bit & 1);
            self.filled += 1;
            if self.filled == 8 {
                self.bytes.push(self.current);
                // JPEG byte stuffing.
                if self.current == 0xFF {
                    self.bytes.push(0x00);
                }
                self.current = 0;
                self.filled = 0;
            }
        }

        fn write_bits(&mut self, value: u32, n: u8) {
            for i in (0..n).rev() {
                self.write_bit(((value >> i) & 1) as u8);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.filled > 0 {
                // Pad the final byte with 1-bits (JPEG convention).
                while self.filled != 0 {
                    self.write_bit(1);
                }
            }
            self.bytes
        }
    }

    /// Magnitude category of a signed difference.
    fn category(diff: i32) -> u8 {
        let mut magnitude = diff.unsigned_abs();
        let mut bits = 0u8;
        while magnitude != 0 {
            magnitude >>= 1;
            bits += 1;
        }
        bits
    }

    /// Encode the mantissa bits for a signed difference of the given category.
    fn mantissa(diff: i32, ssss: u8) -> u32 {
        if ssss == 0 {
            return 0;
        }
        if diff >= 0 {
            diff as u32
        } else {
            (diff - 1 + (1 << ssss)) as u32
        }
    }

    /// Build a DHT segment for a single DC table covering categories 0..=16.
    ///
    /// We use a flat table: every category gets a distinct fixed-length code.
    /// 17 symbols fit in 5-bit codes (32 slots), so all codes have length 5.
    fn build_flat_dht() -> (Vec<u8>, [u8; 16], Vec<u8>) {
        let mut counts = [0u8; 16];
        counts[4] = 17; // 17 codes of length 5
        let values: Vec<u8> = (0u8..=16).collect();
        let mut segment = Vec::new();
        segment.push(0x00); // Tc=0 (DC), Th=0
        segment.extend_from_slice(&counts);
        segment.extend_from_slice(&values);
        (segment, counts, values)
    }

    /// Return the canonical 5-bit code for `symbol` in the flat table.
    fn flat_code(symbol: u8) -> u32 {
        // Canonical assignment: codes 0,1,2,... of length 5 in symbol order.
        u32::from(symbol)
    }

    /// Assemble a complete lossless-JPEG datastream.
    fn build_lossless_jpeg(
        precision: u8,
        width: u16,
        height: u16,
        components: u8,
        predictor: u8,
        restart_interval: u16,
        samples: &[u16],
    ) -> Vec<u8> {
        let (dht_segment, _counts, _values) = build_flat_dht();

        let mut out = Vec::new();
        // SOI
        out.extend_from_slice(&[0xFF, MARKER_SOI]);

        // SOF3
        let mut sof = Vec::new();
        sof.push(precision);
        sof.extend_from_slice(&height.to_be_bytes());
        sof.extend_from_slice(&width.to_be_bytes());
        sof.push(components);
        for c in 0..components {
            sof.push(c + 1); // component id
            sof.push(0x11); // 1x1 sampling
            sof.push(0x00); // quant table (unused)
        }
        out.extend_from_slice(&[0xFF, MARKER_SOF3]);
        out.extend_from_slice(&((sof.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&sof);

        // DHT
        out.extend_from_slice(&[0xFF, MARKER_DHT]);
        out.extend_from_slice(&((dht_segment.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&dht_segment);

        // DRI (optional)
        if restart_interval != 0 {
            out.extend_from_slice(&[0xFF, MARKER_DRI]);
            out.extend_from_slice(&4u16.to_be_bytes());
            out.extend_from_slice(&restart_interval.to_be_bytes());
        }

        // SOS
        let mut sos = Vec::new();
        sos.push(components);
        for c in 0..components {
            sos.push(c + 1); // component selector
            sos.push(0x00); // Td=0, Ta=0
        }
        sos.push(predictor); // Ss
        sos.push(0x00); // Se
        sos.push(0x00); // Ah/Al
        out.extend_from_slice(&[0xFF, MARKER_SOS]);
        out.extend_from_slice(&((sos.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&sos);

        // Entropy data: re-derive predictions exactly as the decoder will.
        let w = width as usize;
        let h = height as usize;
        let n = components as usize;
        let default_sample: i32 = if precision == 0 {
            0
        } else {
            1i32 << (precision - 1)
        };
        let mut writer = BitWriter::new();
        let mut mcu_count: u32 = 0;
        for y in 0..h {
            for x in 0..w {
                if restart_interval != 0 && mcu_count == u32::from(restart_interval) {
                    // Emit an RST marker (cycling D0..D7).
                    let bytes = writer.finish();
                    out.extend_from_slice(&bytes);
                    let rst = 0xD0 + ((mcu_count / u32::from(restart_interval) - 1) % 8) as u8;
                    out.extend_from_slice(&[0xFF, rst]);
                    writer = BitWriter::new();
                    mcu_count = 0;
                }
                for c in 0..n {
                    let actual = i32::from(samples[(y * w + x) * n + c]);
                    let at_boundary = restart_interval != 0 && mcu_count == 0;
                    let px = predictor_for_encode(
                        samples,
                        w,
                        n,
                        x,
                        y,
                        c,
                        predictor,
                        default_sample,
                        at_boundary,
                    );
                    let diff = ((actual - px) & 0xFFFF) as i16 as i32;
                    let ssss = category(diff);
                    writer.write_bits(flat_code(ssss), 5);
                    if ssss > 0 {
                        writer.write_bits(mantissa(diff, ssss), ssss);
                    }
                }
                mcu_count += 1;
            }
        }
        let tail = writer.finish();
        out.extend_from_slice(&tail);

        // EOI
        out.extend_from_slice(&[0xFF, MARKER_EOI]);
        out
    }

    /// Mirror of [`compute_predictor`] for the test encoder.
    #[allow(clippy::too_many_arguments)]
    fn predictor_for_encode(
        samples: &[u16],
        width: usize,
        n_comp: usize,
        x: usize,
        y: usize,
        c: usize,
        selector: u8,
        default_sample: i32,
        at_restart_boundary: bool,
    ) -> i32 {
        let sample_at =
            |sx: usize, sy: usize| -> i32 { i32::from(samples[(sy * width + sx) * n_comp + c]) };
        let first_pixel = (x == 0 && y == 0) || at_restart_boundary;
        if first_pixel {
            return default_sample;
        }
        if x == 0 {
            return sample_at(0, y - 1);
        }
        if y == 0 {
            return sample_at(x - 1, 0);
        }
        let ra = sample_at(x - 1, y);
        let rb = sample_at(x, y - 1);
        let rc = sample_at(x - 1, y - 1);
        predict(selector, ra, rb, rc)
    }

    #[test]
    fn extend_difference_matches_t81_table() {
        // Category 1: bit 0 -> -1, bit 1 -> +1.
        assert_eq!(extend_difference(0, 1), -1);
        assert_eq!(extend_difference(1, 1), 1);
        // Category 2: 00->-3 01->-2 10->2 11->3.
        assert_eq!(extend_difference(0, 2), -3);
        assert_eq!(extend_difference(1, 2), -2);
        assert_eq!(extend_difference(2, 2), 2);
        assert_eq!(extend_difference(3, 2), 3);
        // Category 0 always 0; category 16 is the special 32768.
        assert_eq!(extend_difference(0, 0), 0);
        assert_eq!(extend_difference(0, 16), 32768);
    }

    #[test]
    fn predictors_follow_t81_h1() {
        assert_eq!(predict(1, 10, 20, 5), 10);
        assert_eq!(predict(2, 10, 20, 5), 20);
        assert_eq!(predict(3, 10, 20, 5), 5);
        assert_eq!(predict(4, 10, 20, 5), 25);
        assert_eq!(predict(5, 10, 20, 4), 10 + ((20 - 4) >> 1));
        assert_eq!(predict(6, 10, 20, 4), 20 + ((10 - 4) >> 1));
        assert_eq!(predict(7, 10, 20, 5), 15);
    }

    #[test]
    fn decode_single_component_predictor1() {
        // 4x3, predictor 1 (left), 16-bit precision.
        let width = 4u16;
        let height = 3u16;
        let samples: Vec<u16> = vec![
            1000, 1010, 1005, 990, // row 0
            2000, 2100, 2050, 2080, // row 1
            500, 600, 700, 800, // row 2
        ];
        let stream = build_lossless_jpeg(16, width, height, 1, 1, 0, &samples);
        let image = decode_lossless_jpeg(&stream).expect("decode");
        assert_eq!(image.width, 4);
        assert_eq!(image.height, 3);
        assert_eq!(image.components, 1);
        assert_eq!(image.samples, samples);
    }

    #[test]
    fn decode_predictor4_roundtrip() {
        let width = 5u16;
        let height = 4u16;
        let samples: Vec<u16> = (0..20u16)
            .map(|v| v.wrapping_mul(137).wrapping_add(7))
            .collect();
        let stream = build_lossless_jpeg(16, width, height, 1, 4, 0, &samples);
        let image = decode_lossless_jpeg(&stream).expect("decode");
        assert_eq!(image.samples, samples);
    }

    #[test]
    fn decode_predictor6_roundtrip() {
        let width = 6u16;
        let height = 5u16;
        let samples: Vec<u16> = (0..30u16)
            .map(|v| 8000u16.wrapping_add(v.wrapping_mul(211)))
            .collect();
        let stream = build_lossless_jpeg(16, width, height, 1, 6, 0, &samples);
        let image = decode_lossless_jpeg(&stream).expect("decode");
        assert_eq!(image.samples, samples);
    }

    #[test]
    fn decode_two_component_cfa_packing() {
        // DNG-style: 3x4 JPEG with 2 components encodes a 6x4 CFA mosaic.
        let jpeg_w = 3u16;
        let jpeg_h = 4u16;
        let comps = 2u8;
        // samples interleaved: [c0,c1, c0,c1, c0,c1] per row.
        let samples: Vec<u16> = (0..(jpeg_w as usize * jpeg_h as usize * comps as usize))
            .map(|i| (i as u16).wrapping_mul(97).wrapping_add(1234))
            .collect();
        let stream = build_lossless_jpeg(16, jpeg_w, jpeg_h, comps, 1, 0, &samples);
        let image = decode_lossless_jpeg(&stream).expect("decode");
        assert_eq!(image.components, 2);

        let cfa = deinterleave_cfa(&image, 6, 4).expect("deinterleave");
        assert_eq!(cfa.len(), 24);
        // CFA column cx -> component cx%2 of jpeg column cx/2.
        for y in 0..4usize {
            for cx in 0..6usize {
                let comp = cx % 2;
                let jx = cx / 2;
                let expected = samples[(y * 3 + jx) * 2 + comp];
                assert_eq!(cfa[y * 6 + cx], expected, "mismatch at ({cx},{y})");
            }
        }
    }

    #[test]
    fn decode_with_restart_markers() {
        // Restart every 3 MCUs; 12-pixel image -> 4 intervals.
        let width = 6u16;
        let height = 2u16;
        let samples: Vec<u16> = (0..12u16)
            .map(|v| 30000u16.wrapping_add(v.wrapping_mul(503)))
            .collect();
        let stream = build_lossless_jpeg(16, width, height, 1, 1, 3, &samples);
        let image = decode_lossless_jpeg(&stream).expect("decode");
        assert_eq!(image.samples, samples);
    }

    #[test]
    fn rejects_non_jpeg() {
        let err = decode_lossless_jpeg(&[0x00, 0x01, 0x02, 0x03]);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_dct_jpeg() {
        // SOI followed by SOF0 (baseline DCT) must be rejected as non-lossless.
        let mut data = vec![0xFF, MARKER_SOI, 0xFF, 0xC0];
        data.extend_from_slice(&8u16.to_be_bytes()); // length
        data.extend_from_slice(&[8, 0, 1, 0, 1, 1]); // minimal SOF0 body
        let err = decode_lossless_jpeg(&data);
        assert!(err.is_err());
    }

    #[test]
    fn deinterleave_rejects_undersized_geometry() {
        let image = LosslessJpegImage {
            width: 2,
            height: 2,
            components: 1,
            precision: 16,
            samples: vec![1, 2, 3, 4],
        };
        // Request a CFA wider than the JPEG can supply.
        assert!(deinterleave_cfa(&image, 8, 2).is_err());
    }

    #[test]
    fn decode_12bit_precision() {
        let width = 4u16;
        let height = 4u16;
        let samples: Vec<u16> = (0..16u16).map(|v| (v.wrapping_mul(241)) & 0x0FFF).collect();
        let stream = build_lossless_jpeg(12, width, height, 1, 1, 0, &samples);
        let image = decode_lossless_jpeg(&stream).expect("decode");
        assert_eq!(image.precision, 12);
        assert_eq!(image.samples, samples);
    }
}
