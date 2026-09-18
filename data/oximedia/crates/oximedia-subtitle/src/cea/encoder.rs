//! CEA-608 and CEA-708 closed caption encoder implementation.
//!
//! This module provides full encoding support for both CEA-608 (analog) and
//! CEA-708 (digital) closed caption standards with ATSC A/53 compliance.

use crate::{SubtitleError, SubtitleResult};

// ============================================================================
// CEA-608 Encoder
// ============================================================================

/// CEA-608 channel selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608Channel {
    /// CC1 - Primary caption channel.
    CC1,
    /// CC2 - Secondary caption channel.
    CC2,
    /// CC3 - Third caption channel.
    CC3,
    /// CC4 - Fourth caption channel.
    CC4,
}

impl Cea608Channel {
    /// Get the base control code for this channel.
    #[must_use]
    pub const fn base_code(&self) -> u8 {
        match self {
            Self::CC1 | Self::CC3 => 0x14,
            Self::CC2 | Self::CC4 => 0x1C,
        }
    }

    /// Check if this is field 2 (CC3/CC4).
    #[must_use]
    pub const fn is_field2(&self) -> bool {
        matches!(self, Self::CC3 | Self::CC4)
    }
}

/// CEA-608 display mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608Mode {
    /// Pop-on mode - captions appear all at once.
    PopOn,
    /// Roll-up 2 lines mode.
    RollUp2,
    /// Roll-up 3 lines mode.
    RollUp3,
    /// Roll-up 4 lines mode.
    RollUp4,
    /// Paint-on mode - characters appear as received.
    PaintOn,
    /// Text mode (T1-T4).
    Text,
}

/// CEA-608 text style/color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608Color {
    /// White text.
    White,
    /// Green text.
    Green,
    /// Blue text.
    Blue,
    /// Cyan text.
    Cyan,
    /// Red text.
    Red,
    /// Yellow text.
    Yellow,
    /// Magenta text.
    Magenta,
}

impl Cea608Color {
    /// Get the style code for this color (used in PAC).
    #[must_use]
    pub const fn style_code(&self) -> u8 {
        match self {
            Self::White => 0,
            Self::Green => 1,
            Self::Blue => 2,
            Self::Cyan => 3,
            Self::Red => 4,
            Self::Yellow => 5,
            Self::Magenta => 6,
        }
    }
}

/// CEA-608 attribute flags.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cea608Attributes {
    /// Italic text.
    pub italic: bool,
    /// Underline text.
    pub underline: bool,
    /// Flash/blink text.
    pub flash: bool,
}

/// Preamble Address Code (PAC) - sets row, column, style.
#[derive(Clone, Copy, Debug)]
pub struct Cea608Pac {
    /// Row position (1-15).
    pub row: u8,
    /// Column position (0-31).
    pub column: u8,
    /// Text color.
    pub color: Cea608Color,
    /// Text attributes.
    pub attributes: Cea608Attributes,
}

impl Cea608Pac {
    /// Encode PAC to two bytes.
    #[must_use]
    pub fn encode(&self, channel: Cea608Channel) -> (u8, u8) {
        // Row encoding uses specific bit patterns
        let row_code = match self.row {
            1 => 0x11,
            2 => 0x11,
            3 => 0x12,
            4 => 0x12,
            5 => 0x15,
            6 => 0x15,
            7 => 0x16,
            8 => 0x16,
            9 => 0x17,
            10 => 0x17,
            11 => 0x13,
            12 => 0x13,
            13 => 0x14,
            14 => 0x14,
            15 => 0x15,
            _ => 0x14, // Default to row 15
        };

        let row_offset = if self.row % 2 == 0 { 0x20 } else { 0x00 };

        let mut byte1 = row_code;
        if channel.is_field2() {
            byte1 |= 0x08;
        }

        // Column and style encoding
        let mut byte2 = row_offset;

        if self.column > 0 {
            // Column offset (indent)
            byte2 |= 0x10 | ((self.column / 4) & 0x0F);
        } else {
            // Color/style code
            byte2 |= self.color.style_code() << 1;
            if self.attributes.underline {
                byte2 |= 0x01;
            }
        }

        (byte1, byte2)
    }
}

/// Mid-row code - changes style within a row.
#[derive(Clone, Copy, Debug)]
pub struct Cea608MidRowCode {
    /// Text color (None = no change).
    pub color: Option<Cea608Color>,
    /// Italic attribute.
    pub italic: bool,
    /// Underline attribute.
    pub underline: bool,
}

impl Cea608MidRowCode {
    /// Encode mid-row code to two bytes.
    #[must_use]
    pub fn encode(&self, channel: Cea608Channel) -> (u8, u8) {
        let byte1 = if channel.is_field2() { 0x19 } else { 0x11 };
        let mut byte2 = 0x20;

        if let Some(color) = self.color {
            byte2 |= color.style_code() << 1;
        }

        if self.underline {
            byte2 |= 0x01;
        }

        (byte1, byte2)
    }
}

/// CEA-608 control command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608Command {
    /// Resume Caption Loading (pop-on mode).
    ResumeCaption,
    /// Backspace - delete previous character.
    Backspace,
    /// Delete to End of Row.
    DeleteToEndOfRow,
    /// Roll-Up Captions (2 lines).
    RollUpCaptions2,
    /// Roll-Up Captions (3 lines).
    RollUpCaptions3,
    /// Roll-Up Captions (4 lines).
    RollUpCaptions4,
    /// Flash On - make captions visible immediately.
    FlashOn,
    /// Resume Direct Captioning (paint-on).
    ResumeDirect,
    /// Text Restart.
    TextRestart,
    /// Resume Text Display.
    ResumeText,
    /// Erase Displayed Memory.
    EraseDisplayedMemory,
    /// Carriage Return - roll up.
    CarriageReturn,
    /// Erase Non-Displayed Memory.
    EraseNonDisplayed,
    /// End Of Caption (flip buffers).
    EndOfCaption,
    /// Tab Offset (1-3 columns).
    TabOffset(u8),
}

impl Cea608Command {
    /// Encode command to two bytes.
    #[must_use]
    pub fn encode(&self, channel: Cea608Channel) -> (u8, u8) {
        let base = channel.base_code();

        let byte2 = match self {
            Self::ResumeCaption => 0x20,
            Self::Backspace => 0x21,
            Self::DeleteToEndOfRow => 0x24,
            Self::RollUpCaptions2 => 0x25,
            Self::RollUpCaptions3 => 0x26,
            Self::RollUpCaptions4 => 0x27,
            Self::FlashOn => 0x28,
            Self::ResumeDirect => 0x29,
            Self::TextRestart => 0x2A,
            Self::ResumeText => 0x2B,
            Self::EraseDisplayedMemory => 0x2C,
            Self::CarriageReturn => 0x2D,
            Self::EraseNonDisplayed => 0x2E,
            Self::EndOfCaption => 0x2F,
            Self::TabOffset(offset) => 0x21 + (*offset).min(3),
        };

        (base, byte2)
    }
}

/// Special character codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608SpecialChar {
    /// Registered mark ®.
    RegisteredMark,
    /// Degree sign °.
    Degree,
    /// 1/2 fraction.
    OneHalf,
    /// Inverted question mark ¿.
    InvertedQuestion,
    /// Trademark ™.
    Trademark,
    /// Cents sign ¢.
    Cents,
    /// Pounds sterling £.
    PoundSterling,
    /// Music note ♪.
    MusicNote,
    /// À.
    AGrave,
    /// Transparent space.
    TransparentSpace,
    /// È.
    EGrave,
    /// Â.
    ACircumflex,
    /// Ê.
    ECircumflex,
    /// Î.
    ICircumflex,
    /// Ô.
    OCircumflex,
    /// Û.
    UCircumflex,
}

impl Cea608SpecialChar {
    /// Encode special character to two bytes.
    #[must_use]
    pub fn encode(&self, channel: Cea608Channel) -> (u8, u8) {
        let byte1 = if channel.is_field2() { 0x1B } else { 0x13 };

        let byte2 = match self {
            Self::RegisteredMark => 0x30,
            Self::Degree => 0x31,
            Self::OneHalf => 0x32,
            Self::InvertedQuestion => 0x33,
            Self::Trademark => 0x34,
            Self::Cents => 0x35,
            Self::PoundSterling => 0x36,
            Self::MusicNote => 0x37,
            Self::AGrave => 0x38,
            Self::TransparentSpace => 0x39,
            Self::EGrave => 0x3A,
            Self::ACircumflex => 0x3B,
            Self::ECircumflex => 0x3C,
            Self::ICircumflex => 0x3D,
            Self::OCircumflex => 0x3E,
            Self::UCircumflex => 0x3F,
        };

        (byte1, byte2)
    }
}

/// Extended character set codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608ExtendedChar {
    /// Á.
    AAcute,
    /// É.
    EAcute,
    /// Ó.
    OAcute,
    /// Ú.
    UAcute,
    /// Ü.
    UDiaeresis,
    /// ü.
    UDiaeresisLower,
    /// '.
    Apostrophe,
    /// ¡.
    InvertedExclamation,
    /// *.
    Asterisk,
    /// '.
    LeftQuote,
    /// —.
    EmDash,
    /// ©.
    Copyright,
    /// ℠.
    ServiceMark,
    /// •.
    Bullet,
    /// ".
    LeftDoubleQuote,
    /// ".
    RightDoubleQuote,
}

impl Cea608ExtendedChar {
    /// Encode extended character to two bytes (includes backspace).
    #[must_use]
    pub fn encode(&self, channel: Cea608Channel) -> (u8, u8) {
        let byte1 = if channel.is_field2() { 0x1B } else { 0x12 };

        let byte2 = match self {
            Self::AAcute => 0x20,
            Self::EAcute => 0x21,
            Self::OAcute => 0x22,
            Self::UAcute => 0x23,
            Self::UDiaeresis => 0x24,
            Self::UDiaeresisLower => 0x25,
            Self::Apostrophe => 0x26,
            Self::InvertedExclamation => 0x27,
            Self::Asterisk => 0x28,
            Self::LeftQuote => 0x29,
            Self::EmDash => 0x2A,
            Self::Copyright => 0x2B,
            Self::ServiceMark => 0x2C,
            Self::Bullet => 0x2D,
            Self::LeftDoubleQuote => 0x2E,
            Self::RightDoubleQuote => 0x2F,
        };

        (byte1, byte2)
    }
}

/// CEA-608 encoder state and output.
pub struct Cea608Encoder {
    channel: Cea608Channel,
    mode: Cea608Mode,
    current_row: u8,
    current_column: u8,
    current_color: Cea608Color,
    current_attributes: Cea608Attributes,
    output_buffer: Vec<(u8, u8)>,
    frame_count: u32,
}

impl Cea608Encoder {
    /// Create a new CEA-608 encoder.
    #[must_use]
    pub fn new(channel: Cea608Channel) -> Self {
        Self {
            channel,
            mode: Cea608Mode::PopOn,
            current_row: 15,
            current_column: 0,
            current_color: Cea608Color::White,
            current_attributes: Cea608Attributes::default(),
            output_buffer: Vec::new(),
            frame_count: 0,
        }
    }

    /// Set the display mode.
    pub fn set_mode(&mut self, mode: Cea608Mode) {
        self.mode = mode;

        // Send appropriate mode command
        let command = match mode {
            Cea608Mode::PopOn => Cea608Command::ResumeCaption,
            Cea608Mode::RollUp2 => Cea608Command::RollUpCaptions2,
            Cea608Mode::RollUp3 => Cea608Command::RollUpCaptions3,
            Cea608Mode::RollUp4 => Cea608Command::RollUpCaptions4,
            Cea608Mode::PaintOn => Cea608Command::ResumeDirect,
            Cea608Mode::Text => Cea608Command::ResumeText,
        };

        self.send_command(command);
    }

    /// Send a control command.
    pub fn send_command(&mut self, command: Cea608Command) {
        let (b1, b2) = command.encode(self.channel);
        self.output_buffer.push((b1, b2));
        // Duplicate command for reliability
        self.output_buffer.push((b1, b2));
    }

    /// Set cursor position with PAC.
    pub fn set_position(&mut self, row: u8, column: u8) {
        self.current_row = row.clamp(1, 15);
        self.current_column = column.clamp(0, 31);

        let pac = Cea608Pac {
            row: self.current_row,
            column: self.current_column,
            color: self.current_color,
            attributes: self.current_attributes,
        };

        let (b1, b2) = pac.encode(self.channel);
        self.output_buffer.push((b1, b2));
    }

    /// Set text style/color (mid-row code).
    pub fn set_style(&mut self, color: Cea608Color, italic: bool, underline: bool) {
        self.current_color = color;
        self.current_attributes.italic = italic;
        self.current_attributes.underline = underline;

        let mid_row = Cea608MidRowCode {
            color: Some(color),
            italic,
            underline,
        };

        let (b1, b2) = mid_row.encode(self.channel);
        self.output_buffer.push((b1, b2));
    }

    /// Add text to the output buffer.
    ///
    /// # Errors
    ///
    /// Returns error if text contains unsupported characters.
    pub fn add_text(&mut self, text: &str) -> SubtitleResult<()> {
        for c in text.chars() {
            self.add_char(c)?;
        }
        Ok(())
    }

    /// Add a single character.
    ///
    /// # Errors
    ///
    /// Returns error if character is not supported.
    pub fn add_char(&mut self, c: char) -> SubtitleResult<()> {
        match c {
            // Basic ASCII printable characters
            ' '..='~' => {
                let b = c as u8;
                // CEA-608 uses odd parity, but we'll add it later
                self.output_buffer.push((b & 0x7F, 0x80)); // Null padding
                self.current_column += 1;
            }

            // Special characters
            '®' => {
                let (b1, b2) = Cea608SpecialChar::RegisteredMark.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '°' => {
                let (b1, b2) = Cea608SpecialChar::Degree.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '½' => {
                let (b1, b2) = Cea608SpecialChar::OneHalf.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '¿' => {
                let (b1, b2) = Cea608SpecialChar::InvertedQuestion.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '™' => {
                let (b1, b2) = Cea608SpecialChar::Trademark.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '¢' => {
                let (b1, b2) = Cea608SpecialChar::Cents.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '£' => {
                let (b1, b2) = Cea608SpecialChar::PoundSterling.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '♪' => {
                let (b1, b2) = Cea608SpecialChar::MusicNote.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'À' => {
                let (b1, b2) = Cea608SpecialChar::AGrave.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'È' => {
                let (b1, b2) = Cea608SpecialChar::EGrave.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }

            // Extended characters
            'Á' => {
                let (b1, b2) = Cea608ExtendedChar::AAcute.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'É' => {
                let (b1, b2) = Cea608ExtendedChar::EAcute.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'Ó' => {
                let (b1, b2) = Cea608ExtendedChar::OAcute.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'Ú' => {
                let (b1, b2) = Cea608ExtendedChar::UAcute.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'Ü' => {
                let (b1, b2) = Cea608ExtendedChar::UDiaeresis.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            'ü' => {
                let (b1, b2) = Cea608ExtendedChar::UDiaeresisLower.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '¡' => {
                let (b1, b2) = Cea608ExtendedChar::InvertedExclamation.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '©' => {
                let (b1, b2) = Cea608ExtendedChar::Copyright.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '•' => {
                let (b1, b2) = Cea608ExtendedChar::Bullet.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }
            '—' => {
                let (b1, b2) = Cea608ExtendedChar::EmDash.encode(self.channel);
                self.output_buffer.push((b1, b2));
            }

            '\n' => {
                // For roll-up mode, send carriage return
                if matches!(
                    self.mode,
                    Cea608Mode::RollUp2 | Cea608Mode::RollUp3 | Cea608Mode::RollUp4
                ) {
                    self.send_command(Cea608Command::CarriageReturn);
                }
            }

            _ => {
                return Err(SubtitleError::ParseError(format!(
                    "Unsupported character in CEA-608: {c}"
                )));
            }
        }

        Ok(())
    }

    /// Clear displayed memory.
    pub fn clear_display(&mut self) {
        self.send_command(Cea608Command::EraseDisplayedMemory);
    }

    /// Clear non-displayed memory (buffer).
    pub fn clear_buffer(&mut self) {
        self.send_command(Cea608Command::EraseNonDisplayed);
    }

    /// End caption and flip buffers (pop-on mode).
    pub fn end_caption(&mut self) {
        self.send_command(Cea608Command::EndOfCaption);
    }

    /// Add parity bit to byte.
    #[must_use]
    pub fn add_parity(byte: u8) -> u8 {
        let mut b = byte & 0x7F;
        let mut parity = 0u8;
        let mut temp = b;

        while temp > 0 {
            parity ^= temp & 1;
            temp >>= 1;
        }

        if parity == 0 {
            b |= 0x80;
        }

        b
    }

    /// Get output buffer and clear it.
    #[must_use]
    pub fn take_output(&mut self) -> Vec<(u8, u8)> {
        let mut output = Vec::new();
        std::mem::swap(&mut output, &mut self.output_buffer);

        // Add parity bits
        output
            .into_iter()
            .map(|(b1, b2)| {
                if b2 == 0x80 {
                    // Single character (b2 is padding)
                    (Self::add_parity(b1), 0x80)
                } else {
                    (Self::add_parity(b1), Self::add_parity(b2))
                }
            })
            .collect()
    }

    /// Get output for a specific frame (field-based timing).
    #[must_use]
    pub fn get_frame_output(&mut self, frame_rate: f64) -> Option<(u8, u8)> {
        // CEA-608 typically sends 2 bytes per frame
        if self.output_buffer.is_empty() {
            return Some((0x80, 0x80)); // Null padding
        }

        self.frame_count += 1;

        // Extract next pair
        if !self.output_buffer.is_empty() {
            let (b1, b2) = self.output_buffer.remove(0);
            Some((Self::add_parity(b1), Self::add_parity(b2)))
        } else {
            Some((0x80, 0x80))
        }
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.current_row = 15;
        self.current_column = 0;
        self.current_color = Cea608Color::White;
        self.current_attributes = Cea608Attributes::default();
        self.output_buffer.clear();
        self.frame_count = 0;
    }
}

// ============================================================================
// CEA-708 Encoder
// ============================================================================

/// CEA-708 service number (1-63).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cea708ServiceNumber(u8);

impl Cea708ServiceNumber {
    /// Create a new service number.
    ///
    /// # Errors
    ///
    /// Returns error if service number is not in range 1-63.
    pub fn new(number: u8) -> SubtitleResult<Self> {
        if number == 0 || number > 63 {
            return Err(SubtitleError::InvalidFormat(format!(
                "Invalid CEA-708 service number: {number}"
            )));
        }
        Ok(Self(number))
    }

    /// Get the service number value.
    #[must_use]
    pub const fn value(&self) -> u8 {
        self.0
    }
}

/// CEA-708 window ID (0-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cea708WindowId(u8);

impl Cea708WindowId {
    /// Create a new window ID.
    ///
    /// # Errors
    ///
    /// Returns error if window ID is not in range 0-7.
    pub fn new(id: u8) -> SubtitleResult<Self> {
        if id > 7 {
            return Err(SubtitleError::InvalidFormat(format!(
                "Invalid CEA-708 window ID: {id}"
            )));
        }
        Ok(Self(id))
    }

    /// Get the window ID value.
    #[must_use]
    pub const fn value(&self) -> u8 {
        self.0
    }
}

/// CEA-708 pen size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea708PenSize {
    /// Small text.
    Small,
    /// Standard text.
    Standard,
    /// Large text.
    Large,
}

impl Cea708PenSize {
    /// Get the pen size code.
    #[must_use]
    pub const fn code(&self) -> u8 {
        match self {
            Self::Small => 0,
            Self::Standard => 1,
            Self::Large => 2,
        }
    }
}

/// CEA-708 pen font style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea708FontStyle {
    /// Default font.
    Default,
    /// Monospaced with serifs.
    MonospacedSerif,
    /// Proportional with serifs.
    ProportionalSerif,
    /// Monospaced without serifs.
    MonospacedSansSerif,
    /// Proportional without serifs.
    ProportionalSansSerif,
    /// Casual font.
    Casual,
    /// Cursive font.
    Cursive,
    /// Small capitals.
    SmallCaps,
}

impl Cea708FontStyle {
    /// Get the font style code.
    #[must_use]
    pub const fn code(&self) -> u8 {
        match self {
            Self::Default => 0,
            Self::MonospacedSerif => 1,
            Self::ProportionalSerif => 2,
            Self::MonospacedSansSerif => 3,
            Self::ProportionalSansSerif => 4,
            Self::Casual => 5,
            Self::Cursive => 6,
            Self::SmallCaps => 7,
        }
    }
}

/// CEA-708 pen attributes.
#[derive(Clone, Copy, Debug)]
pub struct Cea708PenAttributes {
    /// Pen size.
    pub pen_size: Cea708PenSize,
    /// Font style.
    pub font_style: Cea708FontStyle,
    /// Text offset (subscript/superscript).
    pub text_offset: u8,
    /// Italics.
    pub italic: bool,
    /// Underline.
    pub underline: bool,
    /// Edge type (0=none, 1=raised, 2=depressed, 3=uniform, 4=drop shadow).
    pub edge_type: u8,
}

impl Default for Cea708PenAttributes {
    fn default() -> Self {
        Self {
            pen_size: Cea708PenSize::Standard,
            font_style: Cea708FontStyle::Default,
            text_offset: 0,
            italic: false,
            underline: false,
            edge_type: 0,
        }
    }
}

/// CEA-708 color with opacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cea708Color {
    /// Red component (0-3, 2-bit).
    pub r: u8,
    /// Green component (0-3, 2-bit).
    pub g: u8,
    /// Blue component (0-3, 2-bit).
    pub b: u8,
}

impl Cea708Color {
    /// Create a new CEA-708 color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r & 0x03,
            g: g & 0x03,
            b: b & 0x03,
        }
    }

    /// White color.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(3, 3, 3)
    }

    /// Black color.
    #[must_use]
    pub const fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Red color.
    #[must_use]
    pub const fn red() -> Self {
        Self::new(3, 0, 0)
    }

    /// Green color.
    #[must_use]
    pub const fn green() -> Self {
        Self::new(0, 3, 0)
    }

    /// Blue color.
    #[must_use]
    pub const fn blue() -> Self {
        Self::new(0, 0, 3)
    }

    /// Yellow color.
    #[must_use]
    pub const fn yellow() -> Self {
        Self::new(3, 3, 0)
    }

    /// Magenta color.
    #[must_use]
    pub const fn magenta() -> Self {
        Self::new(3, 0, 3)
    }

    /// Cyan color.
    #[must_use]
    pub const fn cyan() -> Self {
        Self::new(0, 3, 3)
    }

    /// Encode color to 6-bit value (RGB 2-2-2).
    #[must_use]
    pub const fn encode(&self) -> u8 {
        (self.r << 4) | (self.g << 2) | self.b
    }
}

/// CEA-708 opacity level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea708Opacity {
    /// Solid (opaque).
    Solid,
    /// Flashing.
    Flash,
    /// Translucent.
    Translucent,
    /// Transparent.
    Transparent,
}

impl Cea708Opacity {
    /// Get the opacity code.
    #[must_use]
    pub const fn code(&self) -> u8 {
        match self {
            Self::Solid => 0,
            Self::Flash => 1,
            Self::Translucent => 2,
            Self::Transparent => 3,
        }
    }
}

/// CEA-708 pen color (foreground/background/edge).
#[derive(Clone, Copy, Debug)]
pub struct Cea708PenColor {
    /// Foreground color.
    pub foreground: Cea708Color,
    /// Foreground opacity.
    pub foreground_opacity: Cea708Opacity,
    /// Background color.
    pub background: Cea708Color,
    /// Background opacity.
    pub background_opacity: Cea708Opacity,
    /// Edge color.
    pub edge: Cea708Color,
}

impl Default for Cea708PenColor {
    fn default() -> Self {
        Self {
            foreground: Cea708Color::white(),
            foreground_opacity: Cea708Opacity::Solid,
            background: Cea708Color::black(),
            background_opacity: Cea708Opacity::Solid,
            edge: Cea708Color::black(),
        }
    }
}

/// CEA-708 window anchor position.
#[derive(Clone, Copy, Debug)]
pub struct Cea708WindowAnchor {
    /// Anchor point (0=top-left, 1=top-center, etc.).
    pub anchor_point: u8,
    /// Vertical anchor (0-99, percentage).
    pub vertical: u8,
    /// Horizontal anchor (0-99, percentage).
    pub horizontal: u8,
}

impl Default for Cea708WindowAnchor {
    fn default() -> Self {
        Self {
            anchor_point: 0,
            vertical: 80,
            horizontal: 50,
        }
    }
}

/// CEA-708 window attributes.
#[derive(Clone, Copy, Debug)]
pub struct Cea708WindowAttributes {
    /// Window priority (0-7).
    pub priority: u8,
    /// Relative positioning.
    pub relative_positioning: bool,
    /// Anchor point.
    pub anchor: Cea708WindowAnchor,
    /// Row count (1-15).
    pub row_count: u8,
    /// Column count (1-42).
    pub column_count: u8,
    /// Row lock (prevent scrolling).
    pub row_lock: bool,
    /// Column lock.
    pub column_lock: bool,
    /// Visible flag.
    pub visible: bool,
}

impl Default for Cea708WindowAttributes {
    fn default() -> Self {
        Self {
            priority: 0,
            relative_positioning: false,
            anchor: Cea708WindowAnchor::default(),
            row_count: 4,
            column_count: 32,
            row_lock: false,
            column_lock: false,
            visible: true,
        }
    }
}

/// CEA-708 command codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Cea708Command {
    /// Set Current Window (0-7).
    SetCurrentWindow(u8),
    /// Clear Windows (bitmap).
    ClearWindows(u8),
    /// Display Windows (bitmap).
    DisplayWindows(u8),
    /// Hide Windows (bitmap).
    HideWindows(u8),
    /// Toggle Windows (bitmap).
    ToggleWindows(u8),
    /// Delete Windows (bitmap).
    DeleteWindows(u8),
    /// Delay (1-255 tenths of second).
    Delay(u8),
    /// Delay Cancel.
    DelayCancel,
    /// Reset.
    Reset,
    /// Set Pen Attributes.
    SetPenAttributes,
    /// Set Pen Color.
    SetPenColor,
    /// Set Pen Location (row, column).
    SetPenLocation(u8, u8),
    /// Set Window Attributes.
    SetWindowAttributes,
    /// Define Window.
    DefineWindow(u8),
}

impl Cea708Command {
    /// Encode command to byte(s).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::SetCurrentWindow(win) => vec![0x80 | (win & 0x07)],
            Self::ClearWindows(bitmap) => vec![0x88, *bitmap],
            Self::DisplayWindows(bitmap) => vec![0x89, *bitmap],
            Self::HideWindows(bitmap) => vec![0x8A, *bitmap],
            Self::ToggleWindows(bitmap) => vec![0x8B, *bitmap],
            Self::DeleteWindows(bitmap) => vec![0x8C, *bitmap],
            Self::Delay(tenths) => vec![0x8D, *tenths],
            Self::DelayCancel => vec![0x8E],
            Self::Reset => vec![0x8F],
            Self::SetPenAttributes => vec![0x90],
            Self::SetPenColor => vec![0x91],
            Self::SetPenLocation(row, col) => vec![0x92, *row, *col],
            Self::SetWindowAttributes => vec![0x97],
            Self::DefineWindow(win) => vec![0x98 | (win & 0x07)],
        }
    }
}

/// CEA-708 service block encoder.
pub struct Cea708Encoder {
    service_number: Cea708ServiceNumber,
    current_window: Cea708WindowId,
    windows: [Option<Cea708WindowAttributes>; 8],
    pen_attributes: Cea708PenAttributes,
    pen_color: Cea708PenColor,
    output_buffer: Vec<u8>,
    sequence_number: u8,
}

impl Cea708Encoder {
    /// Create a new CEA-708 encoder.
    #[must_use]
    pub fn new(service_number: Cea708ServiceNumber) -> Self {
        Self {
            service_number,
            current_window: Cea708WindowId(0),
            windows: [None; 8],
            pen_attributes: Cea708PenAttributes::default(),
            pen_color: Cea708PenColor::default(),
            output_buffer: Vec::new(),
            sequence_number: 0,
        }
    }

    /// Define a window with attributes.
    ///
    /// # Errors
    ///
    /// Returns error if window ID is invalid.
    pub fn define_window(
        &mut self,
        window_id: Cea708WindowId,
        attributes: Cea708WindowAttributes,
    ) -> SubtitleResult<()> {
        let id = window_id.value() as usize;
        if id >= 8 {
            return Err(SubtitleError::InvalidFormat(
                "Window ID must be 0-7".to_string(),
            ));
        }

        // DefineWindow command
        self.output_buffer.push(0x98 | (window_id.value() & 0x07));

        // Window attributes (6 bytes)
        let mut attr = 0u8;
        if attributes.visible {
            attr |= 0x20;
        }
        if attributes.row_lock {
            attr |= 0x10;
        }
        if attributes.column_lock {
            attr |= 0x08;
        }
        attr |= attributes.priority & 0x07;
        self.output_buffer.push(attr);

        // Anchor
        self.output_buffer.push(attributes.anchor.vertical);
        self.output_buffer.push(attributes.anchor.horizontal);
        self.output_buffer.push(attributes.anchor.anchor_point);

        // Row and column count
        self.output_buffer.push(attributes.row_count.clamp(1, 15));
        self.output_buffer
            .push(attributes.column_count.clamp(1, 42));

        self.windows[id] = Some(attributes);
        Ok(())
    }

    /// Set current window.
    pub fn set_current_window(&mut self, window_id: Cea708WindowId) {
        self.current_window = window_id;
        self.output_buffer.push(0x80 | (window_id.value() & 0x07));
    }

    /// Clear windows (bitmap of window IDs).
    pub fn clear_windows(&mut self, window_bitmap: u8) {
        self.output_buffer.push(0x88);
        self.output_buffer.push(window_bitmap);
    }

    /// Display windows (make visible).
    pub fn display_windows(&mut self, window_bitmap: u8) {
        self.output_buffer.push(0x89);
        self.output_buffer.push(window_bitmap);
    }

    /// Hide windows.
    pub fn hide_windows(&mut self, window_bitmap: u8) {
        self.output_buffer.push(0x8A);
        self.output_buffer.push(window_bitmap);
    }

    /// Delete windows.
    pub fn delete_windows(&mut self, window_bitmap: u8) {
        self.output_buffer.push(0x8C);
        self.output_buffer.push(window_bitmap);

        // Clear window state
        for i in 0..8 {
            if (window_bitmap & (1 << i)) != 0 {
                self.windows[i] = None;
            }
        }
    }

    /// Set pen attributes.
    pub fn set_pen_attributes(&mut self, attributes: Cea708PenAttributes) {
        self.pen_attributes = attributes;

        self.output_buffer.push(0x90);

        // Pen size and font style (2 bytes)
        let mut byte1 = attributes.pen_size.code() << 6;
        byte1 |= (attributes.text_offset & 0x03) << 4;
        byte1 |= attributes.font_style.code() & 0x07;
        self.output_buffer.push(byte1);

        let mut byte2 = (attributes.edge_type & 0x07) << 5;
        if attributes.underline {
            byte2 |= 0x01;
        }
        if attributes.italic {
            byte2 |= 0x02;
        }
        self.output_buffer.push(byte2);
    }

    /// Set pen color.
    pub fn set_pen_color(&mut self, color: Cea708PenColor) {
        self.pen_color = color;

        self.output_buffer.push(0x91);

        // Foreground (color + opacity)
        self.output_buffer
            .push((color.foreground_opacity.code() << 6) | color.foreground.encode());

        // Background
        self.output_buffer
            .push((color.background_opacity.code() << 6) | color.background.encode());

        // Edge color
        self.output_buffer.push(color.edge.encode());
    }

    /// Set pen location (row, column).
    pub fn set_pen_location(&mut self, row: u8, column: u8) {
        self.output_buffer.push(0x92);
        self.output_buffer.push(row.clamp(0, 14));
        self.output_buffer.push(column.clamp(0, 41));
    }

    /// Add text to current window.
    ///
    /// # Errors
    ///
    /// Returns error if text encoding fails.
    pub fn add_text(&mut self, text: &str) -> SubtitleResult<()> {
        for c in text.chars() {
            self.add_char(c)?;
        }
        Ok(())
    }

    /// Add a single character.
    ///
    /// # Errors
    ///
    /// Returns error if character cannot be encoded.
    pub fn add_char(&mut self, c: char) -> SubtitleResult<()> {
        let code = c as u32;

        if code <= 0x1F {
            // Control codes (most are not used directly)
            return Ok(());
        }

        if code <= 0x7F {
            // Basic ASCII
            self.output_buffer.push(code as u8);
        } else if code <= 0xFF {
            // Latin-1 supplement (0x80-0xFF)
            // Use extended character set
            self.output_buffer.push(0x10); // Extended set 1
            self.output_buffer.push(code as u8);
        } else if code <= 0xFFFF {
            // Unicode BMP - use UTF-16 encoding for CEA-708
            // P16 command for 16-bit characters
            self.output_buffer.push(0x10);
            self.output_buffer.push(0x00); // P16 code
            self.output_buffer.push((code >> 8) as u8);
            self.output_buffer.push((code & 0xFF) as u8);
        } else {
            // Characters beyond BMP not supported in CEA-708
            return Err(SubtitleError::ParseError(format!(
                "Character not supported in CEA-708: {c}"
            )));
        }

        Ok(())
    }

    /// Reset encoder (send Reset command).
    pub fn reset(&mut self) {
        self.output_buffer.push(0x8F);
        for window in &mut self.windows {
            *window = None;
        }
        self.pen_attributes = Cea708PenAttributes::default();
        self.pen_color = Cea708PenColor::default();
    }

    /// Build a service block.
    #[must_use]
    pub fn build_service_block(&mut self) -> Vec<u8> {
        let mut block = Vec::new();

        // Service header
        let service_number = self.service_number.value();
        let block_size = (self.output_buffer.len() + 1).min(31) as u8;

        block.push((block_size << 5) | service_number);

        // Add buffered commands/text
        let data_len = block_size.saturating_sub(1) as usize;
        if data_len > 0 {
            let drain_len = data_len.min(self.output_buffer.len());
            block.extend_from_slice(&self.output_buffer[..drain_len]);
            self.output_buffer.drain(..drain_len);
        }

        // Pad to block size
        while block.len() < block_size as usize {
            block.push(0x00); // Null padding
        }

        block
    }

    /// Build a Caption Distribution Packet (CDP) for ATSC A/53.
    #[must_use]
    pub fn build_cdp(&mut self, framerate_code: u8, timecode: Option<u32>) -> Vec<u8> {
        let mut cdp = Vec::new();

        // CDP header
        cdp.push(0x96); // CDP identifier

        // CDP length placeholder (will be filled later)
        let length_pos = cdp.len();
        cdp.push(0x00);

        // Frame rate
        cdp.push(0x40 | (framerate_code & 0x0F));

        // Flags (service info, caption service active)
        cdp.push(0x43);

        // Sequence counter
        self.sequence_number = (self.sequence_number + 1) & 0x03;
        cdp.push(self.sequence_number << 6);

        // Timecode (optional)
        if let Some(tc) = timecode {
            cdp.push(0x71); // Timecode section
            cdp.push(0x04); // Length
            cdp.push((tc >> 24) as u8);
            cdp.push((tc >> 16) as u8);
            cdp.push((tc >> 8) as u8);
            cdp.push(tc as u8);
        }

        // Service info section
        cdp.push(0x72); // Service info
        cdp.push(0x02); // Length
        cdp.push(0x01); // Service count
        cdp.push(self.service_number.value());

        // CC data section
        cdp.push(0x73); // CC data

        // Build service block
        let service_block = self.build_service_block();

        // CC data length
        let cc_count = service_block.len().div_ceil(3) as u8;
        cdp.push(0xE0 | cc_count);

        // Add service block as cc_data triples
        for chunk in service_block.chunks(2) {
            cdp.push(0xFC); // cc_valid=1, cc_type=11 (708)
            cdp.push(*chunk.first().unwrap_or(&0x00));
            cdp.push(*chunk.get(1).unwrap_or(&0x00));
        }

        // Footer
        cdp.push(0x74); // CDP footer
        cdp.push(self.sequence_number << 6);

        // Calculate checksum (sum of all bytes should be 0)
        let mut checksum = 0u8;
        for &byte in &cdp {
            checksum = checksum.wrapping_add(byte);
        }
        cdp.push(checksum.wrapping_neg());

        // Fill in length
        cdp[length_pos] = (cdp.len() - 2) as u8;

        cdp
    }

    /// Get remaining buffered data.
    #[must_use]
    pub fn take_buffer(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output_buffer)
    }

    /// Check if buffer has data.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.output_buffer.is_empty()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate frame rate code for CEA-708 CDP.
#[must_use]
pub fn get_framerate_code(fps: f64) -> u8 {
    // Frame rate codes per SMPTE 12M
    if (fps - 23.976).abs() < 0.01 {
        0x01 // 23.976 fps
    } else if (fps - 24.0).abs() < 0.01 {
        0x02 // 24 fps
    } else if (fps - 25.0).abs() < 0.01 {
        0x03 // 25 fps
    } else if (fps - 29.97).abs() < 0.01 {
        0x04 // 29.97 fps
    } else if (fps - 30.0).abs() < 0.01 {
        0x05 // 30 fps
    } else if (fps - 50.0).abs() < 0.01 {
        0x06 // 50 fps
    } else if (fps - 59.94).abs() < 0.01 {
        0x07 // 59.94 fps
    } else if (fps - 60.0).abs() < 0.01 {
        0x08 // 60 fps
    } else {
        0x04 // Default to 29.97
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cea608_basic_text() {
        let mut encoder = Cea608Encoder::new(Cea608Channel::CC1);
        encoder.set_mode(Cea608Mode::PopOn);
        encoder.set_position(15, 0);
        encoder.add_text("Hello").expect("should succeed in test");
        encoder.end_caption();

        let output = encoder.take_output();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_cea708_service_block() {
        let service = Cea708ServiceNumber::new(1).expect("should succeed in test");
        let mut encoder = Cea708Encoder::new(service);

        let window = Cea708WindowId::new(0).expect("should succeed in test");
        encoder.set_current_window(window);
        encoder.add_text("Test").expect("should succeed in test");

        let block = encoder.build_service_block();
        assert!(!block.is_empty());
    }

    #[test]
    fn test_parity_calculation() {
        let byte = 0x20;
        let with_parity = Cea608Encoder::add_parity(byte);
        // Verify odd parity is maintained
        let bit_count = (with_parity & 0x7F).count_ones() + ((with_parity >> 7) & 1) as u32;
        assert!(
            bit_count % 2 == 1,
            "Parity should be odd, got {} 1-bits",
            bit_count
        );
    }

    #[test]
    fn test_framerate_codes() {
        assert_eq!(get_framerate_code(29.97), 0x04);
        assert_eq!(get_framerate_code(30.0), 0x05);
        assert_eq!(get_framerate_code(23.976), 0x01);
    }
}
