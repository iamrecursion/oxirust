//! CEA-708 DTVCC (Digital Television Closed Caption) decoder.
//!
//! CEA-708 is the standard for digital television closed captions in the USA.
//! It uses the DTVCC (Digital Television Closed Caption) transport layer.
//!
//! # Window Model
//!
//! CEA-708 supports up to 8 concurrent caption windows. Each window has
//! its own text buffer, visibility state, and positioning attributes.
//!
//! # Packet Structure
//!
//! DTVCC packets carry a sequence number and a payload of service blocks.
//! Each service block addresses a specific caption service (1-63).

#![allow(dead_code)]

/// A raw DTVCC packet from the transport layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Dtvcc708Packet {
    /// Sequence counter (0-3, wraps around).
    pub sequence: u8,
    /// Raw packet payload bytes.
    pub data: Vec<u8>,
}

impl Dtvcc708Packet {
    /// Create a new DTVCC packet.
    #[must_use]
    pub fn new(sequence: u8, data: Vec<u8>) -> Self {
        Self { sequence, data }
    }
}

/// A CEA-708 caption window.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptionWindow {
    /// Window identifier (0-7).
    pub id: u8,
    /// Whether this window is currently visible.
    pub visible: bool,
    /// Number of rows in the window (1-15).
    pub row_count: u8,
    /// Number of columns in the window (1-63).
    pub col_count: u8,
    /// Text content of the window.
    pub text: String,
    /// Current pen row (0-based).
    pub pen_row: u8,
    /// Current pen column (0-based).
    pub pen_col: u8,
}

impl CaptionWindow {
    /// Create a new empty caption window with default dimensions.
    #[must_use]
    pub fn new(id: u8) -> Self {
        Self {
            id,
            visible: false,
            row_count: 15,
            col_count: 32,
            text: String::new(),
            pen_row: 0,
            pen_col: 0,
        }
    }
}

impl Default for CaptionWindow {
    fn default() -> Self {
        Self::new(0)
    }
}

/// CEA-708 DTVCC command decoded from a packet.
#[derive(Clone, Debug, PartialEq)]
pub enum Dtvcc708Command {
    /// Set the current active window (window id 0-7).
    SetCurrentWindow(u8),
    /// Delete windows specified by the bitmask.
    DeleteWindows(u8),
    /// Make windows specified by the bitmask visible.
    DisplayWindows(u8),
    /// Clear (erase) text from windows specified by the bitmask.
    ClearWindows(u8),
    /// Hide windows specified by the bitmask (but retain text).
    HideWindows(u8),
    /// Set window display attributes (parameters consumed but not fully decoded).
    SetWindowAttributes,
    /// Set pen text attributes (parameters consumed but not fully decoded).
    SetPenAttributes,
    /// Set pen foreground/background color (parameters consumed but not fully decoded).
    SetPenColor,
    /// Move the pen to the specified row and column.
    SetPenLocation {
        /// Row index (0-based).
        row: u8,
        /// Column index (0-based).
        col: u8,
    },
    /// Printable text accumulated from 0x20-0x7F bytes.
    Text(String),
    /// Backspace — delete last character in current window.
    Backspace,
    /// Form feed / clear screen for current window.
    FormFeed,
    /// Carriage return — move to next line.
    CarriageReturn,
    /// An unrecognized command byte.
    Unknown(u8),
}

// ============================================================================
// Number of caption windows per CEA-708 spec
// ============================================================================
const NUM_WINDOWS: usize = 8;

/// CEA-708 DTVCC stateful decoder.
///
/// Maintains the state of up to 8 caption windows across multiple packets.
pub struct Dtvcc708Decoder {
    /// The 8 caption windows.
    windows: [CaptionWindow; NUM_WINDOWS],
    /// Index of the current active window (0-7).
    current_window: usize,
}

impl Dtvcc708Decoder {
    /// Create a new decoder with 8 empty, invisible windows.
    #[must_use]
    pub fn new() -> Self {
        Self {
            windows: [
                CaptionWindow::new(0),
                CaptionWindow::new(1),
                CaptionWindow::new(2),
                CaptionWindow::new(3),
                CaptionWindow::new(4),
                CaptionWindow::new(5),
                CaptionWindow::new(6),
                CaptionWindow::new(7),
            ],
            current_window: 0,
        }
    }

    /// Decode a DTVCC packet and update internal window state.
    ///
    /// Returns the current state of all 8 windows after processing.
    pub fn decode(&mut self, packet: &Dtvcc708Packet) -> Vec<CaptionWindow> {
        let commands = Self::parse_commands(&packet.data);
        for cmd in commands {
            self.apply_command(&cmd);
        }
        self.windows.to_vec()
    }

    /// Get the current active window index.
    #[must_use]
    pub fn current_window(&self) -> usize {
        self.current_window
    }

    /// Get a reference to all 8 windows.
    #[must_use]
    pub fn windows(&self) -> &[CaptionWindow; NUM_WINDOWS] {
        &self.windows
    }

    // ---- command parsing ----

    /// Parse raw packet bytes into a sequence of commands.
    fn parse_commands(data: &[u8]) -> Vec<Dtvcc708Command> {
        let mut commands = Vec::new();
        let mut i = 0;
        let mut text_buf = String::new();

        while i < data.len() {
            let byte = data[i];

            match byte {
                // Printable ASCII
                0x20..=0x7E => {
                    text_buf.push(byte as char);
                    i += 1;
                }
                // DEL — treated as printable in some contexts; skip
                0x7F => {
                    i += 1;
                }
                // Non-printable control codes — flush any pending text first
                _ => {
                    if !text_buf.is_empty() {
                        commands.push(Dtvcc708Command::Text(text_buf.clone()));
                        text_buf.clear();
                    }

                    match byte {
                        // Backspace
                        0x08 => {
                            commands.push(Dtvcc708Command::Backspace);
                            i += 1;
                        }
                        // Form Feed / clear screen
                        0x0C => {
                            commands.push(Dtvcc708Command::FormFeed);
                            i += 1;
                        }
                        // Carriage Return
                        0x0D => {
                            commands.push(Dtvcc708Command::CarriageReturn);
                            i += 1;
                        }
                        // SetCurrentWindow (primary range 0x80-0x87)
                        0x80..=0x87 => {
                            let win_id = byte & 0x07;
                            commands.push(Dtvcc708Command::SetCurrentWindow(win_id));
                            i += 1;
                        }
                        // ClearWindows (0x88) — next byte is bitmask
                        0x88 => {
                            i += 1;
                            let mask = data.get(i).copied().unwrap_or(0);
                            commands.push(Dtvcc708Command::ClearWindows(mask));
                            i += 1;
                        }
                        // DisplayWindows (0x89) — next byte is bitmask
                        0x89 => {
                            i += 1;
                            let mask = data.get(i).copied().unwrap_or(0);
                            commands.push(Dtvcc708Command::DisplayWindows(mask));
                            i += 1;
                        }
                        // HideWindows (0x8A) — next byte is bitmask
                        0x8A => {
                            i += 1;
                            let mask = data.get(i).copied().unwrap_or(0);
                            commands.push(Dtvcc708Command::HideWindows(mask));
                            i += 1;
                        }
                        // ToggleWindows (0x8B) — next byte is bitmask — treat as unknown
                        0x8B => {
                            i += 2; // consume command + bitmask byte
                        }
                        // DeleteWindows (0x8C) — next byte is bitmask
                        0x8C => {
                            i += 1;
                            let mask = data.get(i).copied().unwrap_or(0);
                            commands.push(Dtvcc708Command::DeleteWindows(mask));
                            i += 1;
                        }
                        // Delay (0x8D) — next byte is tenths of a second — consume
                        0x8D => {
                            i += 2;
                        }
                        // DelayCancel (0x8E)
                        0x8E => {
                            i += 1;
                        }
                        // Reset (0x8F)
                        0x8F => {
                            i += 1;
                        }
                        // SetCurrentWindow (secondary range 0x91-0x97, mirrors primary)
                        0x91..=0x97 => {
                            let win_id = byte & 0x07;
                            commands.push(Dtvcc708Command::SetCurrentWindow(win_id));
                            i += 1;
                        }
                        // SetWindowAttributes (0x98) — 4 param bytes
                        0x98 => {
                            commands.push(Dtvcc708Command::SetWindowAttributes);
                            i += 1 + 4; // skip 4 param bytes
                        }
                        // SetPenAttributes (0x99) — 2 param bytes
                        0x99 => {
                            commands.push(Dtvcc708Command::SetPenAttributes);
                            i += 1 + 2;
                        }
                        // SetPenColor (0x9A) — 3 param bytes
                        0x9A => {
                            commands.push(Dtvcc708Command::SetPenColor);
                            i += 1 + 3;
                        }
                        // SetPenLocation (0x9B) — 2 param bytes: row, col
                        0x9B => {
                            i += 1;
                            let row = data.get(i).copied().unwrap_or(0) & 0x0F;
                            i += 1;
                            let col = data.get(i).copied().unwrap_or(0) & 0x3F;
                            i += 1;
                            commands.push(Dtvcc708Command::SetPenLocation { row, col });
                        }
                        // Extended control codes (0x10-0x17) — consume 1 additional byte
                        0x10..=0x17 => {
                            i += 2;
                        }
                        // Everything else — unknown
                        _ => {
                            commands.push(Dtvcc708Command::Unknown(byte));
                            i += 1;
                        }
                    }
                }
            }
        }

        // Flush any remaining text
        if !text_buf.is_empty() {
            commands.push(Dtvcc708Command::Text(text_buf));
        }

        commands
    }

    // ---- command application ----

    fn apply_command(&mut self, cmd: &Dtvcc708Command) {
        match cmd {
            Dtvcc708Command::SetCurrentWindow(id) => {
                let win_id = (*id as usize).min(NUM_WINDOWS - 1);
                self.current_window = win_id;
            }
            Dtvcc708Command::DeleteWindows(mask) => {
                for bit in 0..NUM_WINDOWS {
                    if mask & (1 << bit) != 0 {
                        self.windows[bit].text.clear();
                        self.windows[bit].visible = false;
                        self.windows[bit].pen_row = 0;
                        self.windows[bit].pen_col = 0;
                    }
                }
            }
            Dtvcc708Command::DisplayWindows(mask) => {
                for bit in 0..NUM_WINDOWS {
                    if mask & (1 << bit) != 0 {
                        self.windows[bit].visible = true;
                    }
                }
            }
            Dtvcc708Command::HideWindows(mask) => {
                for bit in 0..NUM_WINDOWS {
                    if mask & (1 << bit) != 0 {
                        self.windows[bit].visible = false;
                    }
                }
            }
            Dtvcc708Command::ClearWindows(mask) => {
                for bit in 0..NUM_WINDOWS {
                    if mask & (1 << bit) != 0 {
                        self.windows[bit].text.clear();
                        self.windows[bit].pen_row = 0;
                        self.windows[bit].pen_col = 0;
                    }
                }
            }
            Dtvcc708Command::SetWindowAttributes => {
                // Parameters already consumed during parse; nothing to apply here
            }
            Dtvcc708Command::SetPenAttributes => {
                // Parameters already consumed during parse; nothing to apply here
            }
            Dtvcc708Command::SetPenColor => {
                // Parameters already consumed during parse; nothing to apply here
            }
            Dtvcc708Command::SetPenLocation { row, col } => {
                self.windows[self.current_window].pen_row = *row;
                self.windows[self.current_window].pen_col = *col;
            }
            Dtvcc708Command::Text(text) => {
                self.windows[self.current_window].text.push_str(text);
            }
            Dtvcc708Command::Backspace => {
                let win = &mut self.windows[self.current_window];
                if !win.text.is_empty() {
                    // Remove the last Unicode character
                    let mut chars: Vec<char> = win.text.chars().collect();
                    chars.pop();
                    win.text = chars.into_iter().collect();
                }
            }
            Dtvcc708Command::FormFeed => {
                let win = &mut self.windows[self.current_window];
                win.text.clear();
                win.pen_row = 0;
                win.pen_col = 0;
            }
            Dtvcc708Command::CarriageReturn => {
                let win = &mut self.windows[self.current_window];
                win.text.push('\n');
                win.pen_row = win.pen_row.saturating_add(1);
                win.pen_col = 0;
            }
            Dtvcc708Command::Unknown(_) => {
                // Ignore unknown commands
            }
        }
    }
}

impl Default for Dtvcc708Decoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(seq: u8, data: &[u8]) -> Dtvcc708Packet {
        Dtvcc708Packet::new(seq, data.to_vec())
    }

    #[test]
    fn test_new_decoder_has_eight_windows() {
        let decoder = Dtvcc708Decoder::new();
        assert_eq!(decoder.windows().len(), 8);
    }

    #[test]
    fn test_new_decoder_all_windows_invisible() {
        let decoder = Dtvcc708Decoder::new();
        for win in decoder.windows().iter() {
            assert!(
                !win.visible,
                "window {} should be invisible by default",
                win.id
            );
        }
    }

    #[test]
    fn test_new_decoder_all_windows_empty_text() {
        let decoder = Dtvcc708Decoder::new();
        for win in decoder.windows().iter() {
            assert!(win.text.is_empty());
        }
    }

    #[test]
    fn test_decode_empty_packet_returns_eight_windows() {
        let mut decoder = Dtvcc708Decoder::new();
        let windows = decoder.decode(&packet(0, &[]));
        assert_eq!(windows.len(), 8);
    }

    #[test]
    fn test_decode_printable_text_appends_to_current_window() {
        let mut decoder = Dtvcc708Decoder::new();
        let windows = decoder.decode(&packet(0, b"Hello"));
        assert_eq!(windows[0].text, "Hello");
    }

    #[test]
    fn test_set_current_window_switches_active() {
        let mut decoder = Dtvcc708Decoder::new();
        // 0x82 = SetCurrentWindow(2)
        let windows = decoder.decode(&packet(0, &[0x82, b'X']));
        assert_eq!(windows[2].text, "X");
        assert!(windows[0].text.is_empty());
    }

    #[test]
    fn test_clear_windows_by_bitmask() {
        let mut decoder = Dtvcc708Decoder::new();
        // Write "Hello" to window 0
        decoder.decode(&packet(0, b"Hello"));
        // ClearWindows(0x88) with mask=0x01 (window 0)
        let windows = decoder.decode(&packet(1, &[0x88, 0x01]));
        assert!(windows[0].text.is_empty());
    }

    #[test]
    fn test_display_windows_makes_window_visible() {
        let mut decoder = Dtvcc708Decoder::new();
        // DisplayWindows(0x89) with mask=0x01 (window 0)
        let windows = decoder.decode(&packet(0, &[0x89, 0x01]));
        assert!(windows[0].visible);
        assert!(!windows[1].visible);
    }

    #[test]
    fn test_delete_windows_clears_text_and_visibility() {
        let mut decoder = Dtvcc708Decoder::new();
        decoder.decode(&packet(0, b"Test"));
        decoder.decode(&packet(1, &[0x89, 0x01])); // make visible
        let windows = decoder.decode(&packet(2, &[0x8C, 0x01])); // delete window 0
        assert!(windows[0].text.is_empty());
        assert!(!windows[0].visible);
    }

    #[test]
    fn test_set_pen_location() {
        let mut decoder = Dtvcc708Decoder::new();
        // SetPenLocation with row=3, col=5
        let windows = decoder.decode(&packet(0, &[0x9B, 0x03, 0x05]));
        assert_eq!(windows[0].pen_row, 3);
        assert_eq!(windows[0].pen_col, 5);
    }

    #[test]
    fn test_backspace_removes_last_char() {
        let mut decoder = Dtvcc708Decoder::new();
        decoder.decode(&packet(0, b"Hello"));
        let windows = decoder.decode(&packet(1, &[0x08]));
        assert_eq!(windows[0].text, "Hell");
    }

    #[test]
    fn test_form_feed_clears_current_window() {
        let mut decoder = Dtvcc708Decoder::new();
        decoder.decode(&packet(0, b"Some text"));
        let windows = decoder.decode(&packet(1, &[0x0C]));
        assert!(windows[0].text.is_empty());
    }

    #[test]
    fn test_sequence_number_stored() {
        let pkt = Dtvcc708Packet::new(3, vec![0x20]);
        assert_eq!(pkt.sequence, 3);
    }

    #[test]
    fn test_window_id_bounds() {
        let mut decoder = Dtvcc708Decoder::new();
        // 0x87 = SetCurrentWindow(7) — highest valid id
        decoder.decode(&packet(0, &[0x87, b'Z']));
        assert_eq!(decoder.current_window(), 7);
        assert_eq!(decoder.windows()[7].text, "Z");
    }

    #[test]
    fn test_text_accumulates_across_multiple_decodes() {
        let mut decoder = Dtvcc708Decoder::new();
        decoder.decode(&packet(0, b"Foo"));
        decoder.decode(&packet(1, b"Bar"));
        let windows = decoder.decode(&packet(2, b"!"));
        assert_eq!(windows[0].text, "FooBar!");
    }

    #[test]
    fn test_mixed_control_and_text() {
        let mut decoder = Dtvcc708Decoder::new();
        // Write to window 1, then switch back to window 0
        // 0x81 = SetCurrentWindow(1), then "Hi", then 0x80 = SetCurrentWindow(0), then "Bye"
        let data = [0x81, b'H', b'i', 0x80, b'B', b'y', b'e'];
        let windows = decoder.decode(&packet(0, &data));
        assert_eq!(windows[1].text, "Hi");
        assert_eq!(windows[0].text, "Bye");
    }

    #[test]
    fn test_carriage_return_appends_newline() {
        let mut decoder = Dtvcc708Decoder::new();
        decoder.decode(&packet(0, b"Line1"));
        let windows = decoder.decode(&packet(1, &[0x0D]));
        assert_eq!(windows[0].text, "Line1\n");
    }

    #[test]
    fn test_hide_windows_makes_invisible() {
        let mut decoder = Dtvcc708Decoder::new();
        // Display window 0
        decoder.decode(&packet(0, &[0x89, 0x01]));
        // Hide window 0
        let windows = decoder.decode(&packet(1, &[0x8A, 0x01]));
        assert!(!windows[0].visible);
    }
}
