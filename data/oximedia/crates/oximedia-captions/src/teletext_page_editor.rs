//! Teletext page layout editor (Level 1.5).
//!
//! Provides structures for composing Teletext page layouts conforming to EBU EN 300 706
//! Level 1.5. This includes:
//! - 25-row × 40-column character grid
//! - Level 1.5 colour attributes (foreground/background from the 8-colour CLUT)
//! - CLUT (Colour Look-Up Table) with the 8 standard Teletext colours
//! - Double-height, graphics-mosaic, and hold-mosaic flags (Level 1.5)
//! - Page header (page number, subpage, time, national character set)
//! - Serialise to raw packet bytes (transport-layer aware)
//!
//! # Reference
//! EBU EN 300 706 v1.2.1 (Teletext, Level 1.5)

#![allow(dead_code)]

use std::fmt;

// ── Colour ────────────────────────────────────────────────────────────────────

/// The 8 standard Teletext Level 1.5 foreground/background colours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TeletextColour {
    /// Black (0, 0, 0).
    Black = 0,
    /// Red (255, 0, 0).
    Red = 1,
    /// Green (0, 255, 0).
    Green = 2,
    /// Yellow (255, 255, 0).
    Yellow = 3,
    /// Blue (0, 0, 255).
    Blue = 4,
    /// Magenta (255, 0, 255).
    Magenta = 5,
    /// Cyan (0, 255, 255).
    Cyan = 6,
    /// White (255, 255, 255).
    White = 7,
}

impl TeletextColour {
    /// Return the approximate sRGB triplet for this colour.
    #[must_use]
    pub const fn srgb(self) -> (u8, u8, u8) {
        match self {
            Self::Black => (0, 0, 0),
            Self::Red => (255, 0, 0),
            Self::Green => (0, 255, 0),
            Self::Yellow => (255, 255, 0),
            Self::Blue => (0, 0, 255),
            Self::Magenta => (255, 0, 255),
            Self::Cyan => (0, 255, 255),
            Self::White => (255, 255, 255),
        }
    }

    /// Try to construct a `TeletextColour` from a numeric index `0–7`.
    #[must_use]
    pub fn from_index(idx: u8) -> Option<Self> {
        Some(match idx {
            0 => Self::Black,
            1 => Self::Red,
            2 => Self::Green,
            3 => Self::Yellow,
            4 => Self::Blue,
            5 => Self::Magenta,
            6 => Self::Cyan,
            7 => Self::White,
            _ => return None,
        })
    }
}

impl Default for TeletextColour {
    fn default() -> Self {
        Self::White
    }
}

impl fmt::Display for TeletextColour {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Black => "Black",
            Self::Red => "Red",
            Self::Green => "Green",
            Self::Yellow => "Yellow",
            Self::Blue => "Blue",
            Self::Magenta => "Magenta",
            Self::Cyan => "Cyan",
            Self::White => "White",
        };
        write!(f, "{name}")
    }
}

// ── CharacterAttributes ───────────────────────────────────────────────────────

/// Display attributes for a single Teletext character cell (Level 1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterAttributes {
    /// Foreground colour.
    pub foreground: TeletextColour,
    /// Background colour.
    pub background: TeletextColour,
    /// Double-height rendering flag.
    pub double_height: bool,
    /// Mosaic graphics mode (uses Level 1 graphic character set).
    pub mosaic: bool,
    /// Flash (blink) flag.
    pub flash: bool,
    /// Conceal flag (hidden until user reveals).
    pub conceal: bool,
    /// Hold-mosaic: retain the last mosaic character during spacing attribute transitions.
    pub hold_mosaic: bool,
}

impl Default for CharacterAttributes {
    fn default() -> Self {
        Self {
            foreground: TeletextColour::White,
            background: TeletextColour::Black,
            double_height: false,
            mosaic: false,
            flash: false,
            conceal: false,
            hold_mosaic: false,
        }
    }
}

// ── Cell ─────────────────────────────────────────────────────────────────────

/// A single cell on the Teletext character grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TeletextCell {
    /// Character at this cell (ASCII or national subset character).
    pub character: char,
    /// Display attributes.
    pub attrs: CharacterAttributes,
}

impl TeletextCell {
    /// Create a plain white-on-black cell.
    #[must_use]
    pub fn plain(character: char) -> Self {
        Self {
            character,
            attrs: CharacterAttributes::default(),
        }
    }

    /// Create a coloured cell.
    #[must_use]
    pub fn coloured(
        character: char,
        foreground: TeletextColour,
        background: TeletextColour,
    ) -> Self {
        Self {
            character,
            attrs: CharacterAttributes {
                foreground,
                background,
                ..Default::default()
            },
        }
    }

    /// Create a space cell (blank background filler).
    #[must_use]
    pub fn space() -> Self {
        Self::plain(' ')
    }
}

impl Default for TeletextCell {
    fn default() -> Self {
        Self::space()
    }
}

// ── PageHeader ────────────────────────────────────────────────────────────────

/// Teletext page header (row 0 metadata).
#[derive(Debug, Clone, PartialEq)]
pub struct TeletextPageHeader {
    /// Magazine number (1–8).
    pub magazine: u8,
    /// Page number (0x00–0xFF hex-BCD).
    pub page_number: u8,
    /// Sub-page number (0–0x3F7F).
    pub subpage: u16,
    /// National option character subset index (0–7, per EBU EN 300 706 §15.2).
    pub national_subset: u8,
    /// Whether the page is a subtitle page (C6 flag).
    pub is_subtitle: bool,
    /// Whether the page should be suppressed from display header (C7/C8).
    pub suppress_header: bool,
}

impl Default for TeletextPageHeader {
    fn default() -> Self {
        Self {
            magazine: 1,
            page_number: 0x00,
            subpage: 0,
            national_subset: 0, // English/Western European
            is_subtitle: true,
            suppress_header: false,
        }
    }
}

// ── TeletextPage ─────────────────────────────────────────────────────────────

/// The maximum number of rows on a Teletext page (rows 0–24).
pub const TELETEXT_ROWS: usize = 25;

/// The number of displayable columns per row.
pub const TELETEXT_COLS: usize = 40;

/// A full Teletext page (25 rows × 40 columns).
#[derive(Debug, Clone)]
pub struct TeletextPage {
    /// Page header.
    pub header: TeletextPageHeader,
    /// Character grid: `grid[row][col]`.
    grid: [[TeletextCell; TELETEXT_COLS]; TELETEXT_ROWS],
}

impl TeletextPage {
    /// Create an empty (space-filled) page.
    #[must_use]
    pub fn new(header: TeletextPageHeader) -> Self {
        let row: [TeletextCell; TELETEXT_COLS] = [(); TELETEXT_COLS].map(|_| TeletextCell::space());
        Self {
            header,
            grid: [(); TELETEXT_ROWS].map(|_| row.clone()),
        }
    }

    /// Write a string to `row` starting at `col`, using the given attributes.
    ///
    /// Characters beyond column 39 are silently dropped.
    pub fn write_text(
        &mut self,
        row: usize,
        col: usize,
        text: &str,
        attrs: CharacterAttributes,
    ) -> usize {
        let row = row.min(TELETEXT_ROWS - 1);
        let mut written = 0;
        for (i, ch) in text.chars().enumerate() {
            let c = col + i;
            if c >= TELETEXT_COLS {
                break;
            }
            self.grid[row][c] = TeletextCell {
                character: ch,
                attrs,
            };
            written += 1;
        }
        written
    }

    /// Write plain white-on-black text.
    pub fn write_plain(&mut self, row: usize, col: usize, text: &str) -> usize {
        self.write_text(row, col, text, CharacterAttributes::default())
    }

    /// Clear a row back to spaces.
    pub fn clear_row(&mut self, row: usize) {
        let row = row.min(TELETEXT_ROWS - 1);
        for col in 0..TELETEXT_COLS {
            self.grid[row][col] = TeletextCell::space();
        }
    }

    /// Clear the entire page (rows 1–24; row 0 is the header row).
    pub fn clear(&mut self) {
        for row in 1..TELETEXT_ROWS {
            self.clear_row(row);
        }
    }

    /// Get a reference to a cell.
    #[must_use]
    pub fn cell(&self, row: usize, col: usize) -> &TeletextCell {
        &self.grid[row.min(TELETEXT_ROWS - 1)][col.min(TELETEXT_COLS - 1)]
    }

    /// Get a mutable reference to a cell.
    pub fn cell_mut(&mut self, row: usize, col: usize) -> &mut TeletextCell {
        &mut self.grid[row.min(TELETEXT_ROWS - 1)][col.min(TELETEXT_COLS - 1)]
    }

    /// Extract the plain text of row `r` (stripping attributes).
    #[must_use]
    pub fn row_text(&self, row: usize) -> String {
        let row = row.min(TELETEXT_ROWS - 1);
        self.grid[row].iter().map(|c| c.character).collect()
    }

    /// Extract all non-blank rows as plain text lines.
    #[must_use]
    pub fn text_lines(&self) -> Vec<(usize, String)> {
        (0..TELETEXT_ROWS)
            .filter_map(|r| {
                let text = self.row_text(r);
                if text.trim().is_empty() {
                    None
                } else {
                    Some((r, text.trim_end().to_string()))
                }
            })
            .collect()
    }

    /// Serialise the page to a flat byte buffer (one byte per cell, ASCII-mapped).
    ///
    /// The buffer is `TELETEXT_ROWS * TELETEXT_COLS` bytes.  Control characters
    /// outside the printable ASCII range are replaced with spaces.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TELETEXT_ROWS * TELETEXT_COLS);
        for row in &self.grid {
            for cell in row {
                let byte = if cell.character.is_ascii() && cell.character >= ' ' {
                    cell.character as u8
                } else {
                    b' '
                };
                out.push(byte);
            }
        }
        out
    }

    /// Centre `text` on `row` with given attributes.
    pub fn write_centred(&mut self, row: usize, text: &str, attrs: CharacterAttributes) -> usize {
        let len = text.chars().count().min(TELETEXT_COLS);
        let col = (TELETEXT_COLS.saturating_sub(len)) / 2;
        self.write_text(row, col, text, attrs)
    }
}

impl Default for TeletextPage {
    fn default() -> Self {
        Self::new(TeletextPageHeader::default())
    }
}

// ── TeletextPageEditor ────────────────────────────────────────────────────────

/// Interactive editor for composing Teletext pages.
///
/// Maintains a current cursor position and active attributes, providing
/// convenience helpers for common authoring patterns.
#[derive(Debug, Clone)]
pub struct TeletextPageEditor {
    /// The page being edited.
    pub page: TeletextPage,
    /// Current row cursor (0–24).
    pub cursor_row: usize,
    /// Current column cursor (0–39).
    pub cursor_col: usize,
    /// Currently active character attributes.
    pub active_attrs: CharacterAttributes,
}

impl TeletextPageEditor {
    /// Create a new editor with a blank page.
    #[must_use]
    pub fn new(header: TeletextPageHeader) -> Self {
        Self {
            page: TeletextPage::new(header),
            cursor_row: 1, // row 0 is the page header
            cursor_col: 0,
            active_attrs: CharacterAttributes::default(),
        }
    }

    /// Move the cursor to `(row, col)`.
    pub fn move_cursor(&mut self, row: usize, col: usize) {
        self.cursor_row = row.min(TELETEXT_ROWS - 1);
        self.cursor_col = col.min(TELETEXT_COLS - 1);
    }

    /// Set the active foreground colour.
    pub fn set_foreground(&mut self, colour: TeletextColour) {
        self.active_attrs.foreground = colour;
    }

    /// Set the active background colour.
    pub fn set_background(&mut self, colour: TeletextColour) {
        self.active_attrs.background = colour;
    }

    /// Enable or disable double-height for new characters.
    pub fn set_double_height(&mut self, on: bool) {
        self.active_attrs.double_height = on;
    }

    /// Type `text` at the current cursor position, advancing the cursor.
    ///
    /// Returns the number of characters written.
    pub fn type_text(&mut self, text: &str) -> usize {
        let written =
            self.page
                .write_text(self.cursor_row, self.cursor_col, text, self.active_attrs);
        self.cursor_col = (self.cursor_col + written).min(TELETEXT_COLS - 1);
        written
    }

    /// Move to the next row and reset column to 0.
    pub fn newline(&mut self) {
        self.cursor_row = (self.cursor_row + 1).min(TELETEXT_ROWS - 1);
        self.cursor_col = 0;
    }

    /// Clear the current row and move the cursor to column 0.
    pub fn clear_current_row(&mut self) {
        self.page.clear_row(self.cursor_row);
        self.cursor_col = 0;
    }

    /// Finalise the page and return it.
    #[must_use]
    pub fn build(self) -> TeletextPage {
        self.page
    }
}

impl Default for TeletextPageEditor {
    fn default() -> Self {
        Self::new(TeletextPageHeader::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teletext_colour_srgb() {
        assert_eq!(TeletextColour::White.srgb(), (255, 255, 255));
        assert_eq!(TeletextColour::Black.srgb(), (0, 0, 0));
        assert_eq!(TeletextColour::Red.srgb(), (255, 0, 0));
    }

    #[test]
    fn test_teletext_colour_from_index() {
        assert_eq!(TeletextColour::from_index(0), Some(TeletextColour::Black));
        assert_eq!(TeletextColour::from_index(7), Some(TeletextColour::White));
        assert_eq!(TeletextColour::from_index(8), None);
    }

    #[test]
    fn test_teletext_colour_display() {
        assert_eq!(format!("{}", TeletextColour::Cyan), "Cyan");
        assert_eq!(format!("{}", TeletextColour::Yellow), "Yellow");
    }

    #[test]
    fn test_character_attributes_defaults() {
        let a = CharacterAttributes::default();
        assert_eq!(a.foreground, TeletextColour::White);
        assert_eq!(a.background, TeletextColour::Black);
        assert!(!a.double_height);
        assert!(!a.conceal);
    }

    #[test]
    fn test_teletext_cell_plain() {
        let cell = TeletextCell::plain('A');
        assert_eq!(cell.character, 'A');
        assert_eq!(cell.attrs.foreground, TeletextColour::White);
    }

    #[test]
    fn test_teletext_cell_coloured() {
        let cell = TeletextCell::coloured('X', TeletextColour::Red, TeletextColour::Blue);
        assert_eq!(cell.attrs.foreground, TeletextColour::Red);
        assert_eq!(cell.attrs.background, TeletextColour::Blue);
    }

    #[test]
    fn test_page_write_plain() {
        let mut page = TeletextPage::default();
        let written = page.write_plain(3, 0, "Hello");
        assert_eq!(written, 5);
        assert_eq!(page.cell(3, 0).character, 'H');
        assert_eq!(page.cell(3, 4).character, 'o');
    }

    #[test]
    fn test_page_write_truncates_at_column_boundary() {
        let mut page = TeletextPage::default();
        let text = "A".repeat(50);
        let written = page.write_plain(1, 0, &text);
        assert_eq!(written, TELETEXT_COLS);
    }

    #[test]
    fn test_page_row_text() {
        let mut page = TeletextPage::default();
        page.write_plain(5, 0, "HELLO");
        let text = page.row_text(5);
        assert!(text.starts_with("HELLO"));
        assert_eq!(text.len(), TELETEXT_COLS);
    }

    #[test]
    fn test_page_clear_row() {
        let mut page = TeletextPage::default();
        page.write_plain(2, 0, "TEXT");
        page.clear_row(2);
        let text = page.row_text(2);
        assert!(text.chars().all(|c| c == ' '));
    }

    #[test]
    fn test_page_text_lines() {
        let mut page = TeletextPage::default();
        page.write_plain(3, 0, "Line one");
        page.write_plain(7, 0, "Line two");
        let lines = page.text_lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].0, 3);
        assert!(lines[0].1.contains("Line one"));
    }

    #[test]
    fn test_page_to_bytes_length() {
        let page = TeletextPage::default();
        let bytes = page.to_bytes();
        assert_eq!(bytes.len(), TELETEXT_ROWS * TELETEXT_COLS);
    }

    #[test]
    fn test_page_to_bytes_content() {
        let mut page = TeletextPage::default();
        page.write_plain(0, 0, "HELLO");
        let bytes = page.to_bytes();
        assert_eq!(bytes[0], b'H');
        assert_eq!(bytes[4], b'O');
    }

    #[test]
    fn test_page_write_centred() {
        let mut page = TeletextPage::default();
        let written = page.write_centred(10, "Hi", CharacterAttributes::default());
        assert_eq!(written, 2);
        // Text should be at column 19 (centred in 40 columns)
        let row_text = page.row_text(10);
        assert!(row_text.contains("Hi"));
    }

    #[test]
    fn test_editor_type_text_advances_cursor() {
        let mut editor = TeletextPageEditor::default();
        editor.move_cursor(5, 0);
        editor.type_text("HELLO");
        assert_eq!(editor.cursor_col, 5);
    }

    #[test]
    fn test_editor_newline() {
        let mut editor = TeletextPageEditor::default();
        editor.move_cursor(5, 20);
        editor.newline();
        assert_eq!(editor.cursor_row, 6);
        assert_eq!(editor.cursor_col, 0);
    }

    #[test]
    fn test_editor_set_foreground() {
        let mut editor = TeletextPageEditor::default();
        editor.set_foreground(TeletextColour::Cyan);
        editor.type_text("X");
        let cell = editor.page.cell(editor.cursor_row, 0);
        assert_eq!(cell.attrs.foreground, TeletextColour::Cyan);
    }

    #[test]
    fn test_editor_build() {
        let mut editor = TeletextPageEditor::new(TeletextPageHeader {
            is_subtitle: true,
            ..Default::default()
        });
        editor.type_text("Test");
        let page = editor.build();
        assert!(page.header.is_subtitle);
    }

    #[test]
    fn test_page_header_defaults() {
        let header = TeletextPageHeader::default();
        assert_eq!(header.magazine, 1);
        assert!(header.is_subtitle);
        assert!(!header.suppress_header);
    }
}
