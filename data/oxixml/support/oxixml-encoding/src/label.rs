//! The [`Encoding`] enum and IANA/WHATWG label resolution.

use crate::decoder::Decoder;

/// A character encoding recognized by this crate.
///
/// The original core set is exactly what an XML processor needs to deal with in practice:
/// `UTF-8` and `UTF-16` (in both byte orders) are the two encodings XML 1.0 requires every
/// conformant processor to support; [`Encoding::Latin1`] and [`Encoding::Windows1252`] cover the
/// overwhelming majority of legacy (`encoding="..."`) documents seen in the wild, and
/// [`Encoding::Ascii`] is useful as a strict validation target.
///
/// Beyond that core, this crate also implements the rest of the
/// [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/)'s legacy single-byte encodings
/// (the full `ISO-8859-*`, `windows-125x`, `KOI8-*`, `IBM866`, `macintosh` and `x-mac-cyrillic`
/// families), its three legacy Japanese multi-byte encodings (Shift_JIS, EUC-JP, and
/// ISO-2022-JP), and its legacy CJK multi-byte encodings (GBK, gb18030, Big5, EUC-KR),
/// prioritized for COOLJAPAN. See [`Encoding::ALL`] for the complete list.
///
/// Unlike the WHATWG Encoding Standard, which folds `ISO-8859-1` into `windows-1252` for web
/// compatibility, this crate keeps them distinct: XML tooling (libxml2, Xerces, .NET, the JDK,
/// ...) has always treated `ISO-8859-1` as the "pure" Latin-1 repertoire (bytes `0x00..=0xFF` map
/// to `U+0000..=U+00FF` unchanged) and `windows-1252` as the Microsoft superset that additionally
/// assigns the `0x80..=0x9F` range to punctuation such as smart quotes and the euro sign. The
/// same reasoning applies to [`Encoding::Iso8859_9`] versus [`Encoding::Windows1254`]: the WHATWG
/// standard folds the `"iso-8859-9"`/`"latin5"` labels into windows-1254 too, but this crate keeps
/// pure ISO-8859-9 (ISO/IEC 8859-9:1999, a.k.a. Latin-5) as its own encoding for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// UTF-8: the ASCII-compatible variable-width encoding of Unicode, and the default assumed
    /// by XML 1.0 in the absence of a byte-order mark or `encoding` declaration.
    Utf8,
    /// UTF-16, little-endian byte order.
    Utf16Le,
    /// UTF-16, big-endian byte order.
    Utf16Be,
    /// ISO-8859-1 (Latin-1): each byte `b` decodes to the scalar value `U+00xx` with `xx == b`.
    Latin1,
    /// windows-1252 (a.k.a. `cp1252`): ISO-8859-1 with the C1 control range `0x80..=0x9F`
    /// reassigned to printable characters (the five positions Microsoft left unassigned keep
    /// their Latin-1 fallback, per the [WHATWG index table][idx]).
    ///
    /// [idx]: https://encoding.spec.whatwg.org/index-windows-1252.txt
    Windows1252,
    /// US-ASCII: the 7-bit repertoire `U+0000..=U+007F`. Bytes `0x80..=0xFF` are not valid.
    Ascii,

    // --- ISO-8859 family (beyond -1, above) -------------------------------------------------
    /// ISO-8859-2 (Latin-2): Central/Eastern European.
    Iso8859_2,
    /// ISO-8859-3 (Latin-3): South European (Maltese, Esperanto).
    Iso8859_3,
    /// ISO-8859-4 (Latin-4): North European (Baltic languages, pre-ISO-8859-13).
    Iso8859_4,
    /// ISO-8859-5: Cyrillic.
    Iso8859_5,
    /// ISO-8859-6: Arabic.
    Iso8859_6,
    /// ISO-8859-7: Greek.
    Iso8859_7,
    /// ISO-8859-8: Hebrew (visual order; this crate makes no distinction from the "logical
    /// order" `ISO-8859-8-I` variant, since directionality is outside this crate's scope and
    /// both share the same byte/code-point repertoire).
    Iso8859_8,
    /// ISO-8859-9 (Latin-5): identical to ISO-8859-1 except six positions replaced with the
    /// Turkish letters `Ğ`/`ğ`, `İ`/`ı`, `Ş`/`ş`. See the type-level docs for why this is kept
    /// distinct from [`Encoding::Windows1254`].
    Iso8859_9,
    /// ISO-8859-10 (Latin-6): Nordic (an alternative to -4 with full Icelandic coverage).
    Iso8859_10,
    /// ISO-8859-13 (Latin-7): Baltic.
    Iso8859_13,
    /// ISO-8859-14 (Latin-8): Celtic.
    Iso8859_14,
    /// ISO-8859-15 (Latin-9): Latin-1 with the euro sign and a handful of French/Finnish letters.
    Iso8859_15,
    /// ISO-8859-16 (Latin-10): South-Eastern European.
    Iso8859_16,

    // --- windows-125x family (beyond -1252, above) ------------------------------------------
    /// windows-1250: Central/Eastern European.
    Windows1250,
    /// windows-1251: Cyrillic.
    Windows1251,
    /// windows-1253: Greek.
    Windows1253,
    /// windows-1254 (Turkish): the Microsoft superset of [`Encoding::Iso8859_9`].
    Windows1254,
    /// windows-1255: Hebrew.
    Windows1255,
    /// windows-1256: Arabic.
    Windows1256,
    /// windows-1257: Baltic.
    Windows1257,
    /// windows-1258: Vietnamese.
    Windows1258,

    // --- Other single-byte encodings ---------------------------------------------------------
    /// KOI8-R: Russian Cyrillic.
    Koi8R,
    /// KOI8-U: Ukrainian Cyrillic (a KOI8-R variant with a handful of Ukrainian letters).
    Koi8U,
    /// IBM866 (code page 866): Cyrillic, MS-DOS.
    Ibm866,
    /// macintosh (a.k.a. "Mac OS Roman"): the classic Mac OS Western European encoding.
    Macintosh,
    /// x-mac-cyrillic (a.k.a. "Mac OS Cyrillic"): the classic Mac OS Cyrillic encoding.
    XMacCyrillic,

    // --- Legacy multi-byte Japanese encodings ------------------------------------------------
    /// Shift_JIS (a.k.a. `MS932`/`Windows-31J`): the JIS X 0201 + JIS X 0208 encoding most
    /// commonly used on Japanese Windows and in Japanese web content.
    ShiftJis,
    /// EUC-JP: the Unix-family Japanese encoding covering JIS X 0201, JIS X 0208, and (via its
    /// `0x8F`-prefixed `SS3` sequences) JIS X 0212.
    EucJp,
    /// ISO-2022-JP: the 7-bit, escape-sequence-switched Japanese encoding historically used for
    /// Japanese email (`Content-Transfer-Encoding: 7bit`-safe).
    Iso2022Jp,

    // --- Legacy multi-byte CJK encodings ------------------------------------------------------
    /// GBK: the 2-byte-only Chinese (Simplified) encoding, a.k.a. `CP936`. Decodes identically to
    /// [`Encoding::Gb18030`] (per the WHATWG Encoding Standard, "GBK's decoder is gb18030's
    /// decoder"); its encoder is the more conservative one, refusing to emit the 4-byte sequences
    /// [`Encoding::Gb18030`]'s encoder falls back to for code points outside GBK's repertoire.
    Gbk,
    /// gb18030: the Chinese national standard encoding. A superset of [`Encoding::Gbk`] with
    /// 1/2/4-byte sequences whose *encoder* is total over Unicode -- every scalar value except
    /// `U+E5E5` has a representation (see [`crate::encode_from_utf8`]). This does not quite mean
    /// every scalar value round-trips through encode-then-decode: per the WHATWG gb18030 encoder
    /// algorithm, 18 Private Use Area code points are deliberately encoded to a fixed legacy byte
    /// pair (for compatibility with the GB18030-2005 standard) that decodes back to a *different*
    /// code point under the current index -- see `crate::cjk::gb18030::encode`'s own docs for the
    /// full list.
    Gb18030,
    /// Big5: the Traditional Chinese encoding historically used in Taiwan and Hong Kong, including
    /// the Hong Kong Supplementary Character Set (HKSCS) extensions per the WHATWG index.
    Big5,
    /// EUC-KR: the Korean encoding covering KS X 1001 plus the Unified Hangul Code (a.k.a.
    /// Windows-949) extension, per the WHATWG index.
    EucKr,
}

impl Encoding {
    /// Every encoding this crate supports, in declaration order. Useful for exhaustively testing
    /// or listing supported encodings without having to enumerate the (growing) variant list by
    /// hand.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::Encoding;
    ///
    /// assert!(Encoding::ALL.contains(&Encoding::Utf8));
    /// assert!(Encoding::ALL.contains(&Encoding::ShiftJis));
    /// // Every entry's canonical name resolves back to itself.
    /// for &encoding in Encoding::ALL {
    ///     assert_eq!(Encoding::for_label(encoding.name()), Some(encoding));
    /// }
    /// ```
    pub const ALL: &'static [Encoding] = &[
        Encoding::Utf8,
        Encoding::Utf16Le,
        Encoding::Utf16Be,
        Encoding::Latin1,
        Encoding::Windows1252,
        Encoding::Ascii,
        Encoding::Iso8859_2,
        Encoding::Iso8859_3,
        Encoding::Iso8859_4,
        Encoding::Iso8859_5,
        Encoding::Iso8859_6,
        Encoding::Iso8859_7,
        Encoding::Iso8859_8,
        Encoding::Iso8859_9,
        Encoding::Iso8859_10,
        Encoding::Iso8859_13,
        Encoding::Iso8859_14,
        Encoding::Iso8859_15,
        Encoding::Iso8859_16,
        Encoding::Windows1250,
        Encoding::Windows1251,
        Encoding::Windows1253,
        Encoding::Windows1254,
        Encoding::Windows1255,
        Encoding::Windows1256,
        Encoding::Windows1257,
        Encoding::Windows1258,
        Encoding::Koi8R,
        Encoding::Koi8U,
        Encoding::Ibm866,
        Encoding::Macintosh,
        Encoding::XMacCyrillic,
        Encoding::ShiftJis,
        Encoding::EucJp,
        Encoding::Iso2022Jp,
        Encoding::Gbk,
        Encoding::Gb18030,
        Encoding::Big5,
        Encoding::EucKr,
    ];

    /// Resolve an IANA/WHATWG charset label to an [`Encoding`], case-insensitively and after
    /// trimming ASCII whitespace, mirroring the label tables in the
    /// [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/#names-and-labels) — with the
    /// [`Encoding::Iso8859_9`]-versus-[`Encoding::Windows1254`] exception documented on
    /// [`Encoding`] itself.
    ///
    /// Returns [`None`] for labels this crate does not resolve to a supported encoding (for
    /// example `"utf-32"`, `"windows-874"` or `"ebcdic-cp-us"`).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::Encoding;
    ///
    /// assert_eq!(Encoding::for_label("UTF-8"), Some(Encoding::Utf8));
    /// assert_eq!(Encoding::for_label(" utf8 \t"), Some(Encoding::Utf8));
    /// assert_eq!(Encoding::for_label("Windows-1252"), Some(Encoding::Windows1252));
    /// assert_eq!(Encoding::for_label("iso-8859-1"), Some(Encoding::Latin1));
    /// assert_eq!(Encoding::for_label("Shift_JIS"), Some(Encoding::ShiftJis));
    /// assert_eq!(Encoding::for_label("EUC-JP"), Some(Encoding::EucJp));
    /// assert_eq!(Encoding::for_label("gb18030"), Some(Encoding::Gb18030));
    /// assert_eq!(Encoding::for_label("windows-874"), None);
    /// ```
    #[must_use]
    pub fn for_label(label: &str) -> Option<Encoding> {
        let trimmed = label.trim_matches(is_whatwg_ascii_whitespace);
        if trimmed.is_empty() {
            return None;
        }
        // WHATWG label matching is defined over the ASCII-lowercased label; avoid allocating in
        // the common case where the caller already passed a canonical lowercase label.
        if trimmed.bytes().all(|b| !b.is_ascii_uppercase()) {
            return encoding_for_lowercase_label(trimmed);
        }
        let lowered = trimmed.to_ascii_lowercase();
        encoding_for_lowercase_label(&lowered)
    }

    /// The canonical name of this encoding, as it should appear in generated `encoding="..."`
    /// declarations or diagnostics.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::Encoding;
    ///
    /// assert_eq!(Encoding::Utf8.name(), "UTF-8");
    /// assert_eq!(Encoding::Windows1252.name(), "windows-1252");
    /// assert_eq!(Encoding::ShiftJis.name(), "Shift_JIS");
    /// ```
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Encoding::Utf8 => "UTF-8",
            Encoding::Utf16Le => "UTF-16LE",
            Encoding::Utf16Be => "UTF-16BE",
            Encoding::Latin1 => "ISO-8859-1",
            Encoding::Windows1252 => "windows-1252",
            Encoding::Ascii => "US-ASCII",
            Encoding::Iso8859_2 => "ISO-8859-2",
            Encoding::Iso8859_3 => "ISO-8859-3",
            Encoding::Iso8859_4 => "ISO-8859-4",
            Encoding::Iso8859_5 => "ISO-8859-5",
            Encoding::Iso8859_6 => "ISO-8859-6",
            Encoding::Iso8859_7 => "ISO-8859-7",
            Encoding::Iso8859_8 => "ISO-8859-8",
            Encoding::Iso8859_9 => "ISO-8859-9",
            Encoding::Iso8859_10 => "ISO-8859-10",
            Encoding::Iso8859_13 => "ISO-8859-13",
            Encoding::Iso8859_14 => "ISO-8859-14",
            Encoding::Iso8859_15 => "ISO-8859-15",
            Encoding::Iso8859_16 => "ISO-8859-16",
            Encoding::Windows1250 => "windows-1250",
            Encoding::Windows1251 => "windows-1251",
            Encoding::Windows1253 => "windows-1253",
            Encoding::Windows1254 => "windows-1254",
            Encoding::Windows1255 => "windows-1255",
            Encoding::Windows1256 => "windows-1256",
            Encoding::Windows1257 => "windows-1257",
            Encoding::Windows1258 => "windows-1258",
            Encoding::Koi8R => "KOI8-R",
            Encoding::Koi8U => "KOI8-U",
            Encoding::Ibm866 => "IBM866",
            Encoding::Macintosh => "macintosh",
            Encoding::XMacCyrillic => "x-mac-cyrillic",
            Encoding::ShiftJis => "Shift_JIS",
            Encoding::EucJp => "EUC-JP",
            Encoding::Iso2022Jp => "ISO-2022-JP",
            Encoding::Gbk => "GBK",
            Encoding::Gb18030 => "gb18030",
            Encoding::Big5 => "Big5",
            Encoding::EucKr => "EUC-KR",
        }
    }

    /// Whether this encoding is ASCII-compatible in the sense relevant to XML prologue
    /// scanning: every byte in `0x00..=0x7F` decodes to the identically-valued scalar, so an
    /// ASCII-only substring (such as an `<?xml ... ?>` declaration) can be located by scanning
    /// raw bytes without first performing a full decode.
    ///
    /// This holds for every single-byte encoding this crate supports (all of them agree with
    /// ASCII below `0x80`, by construction — see [`Encoding::ALL`]), not only the original four.
    /// It does *not* hold for the seven multi-byte encodings (the three legacy Japanese ones and
    /// the four legacy CJK ones): even though each of them also encodes `0x00..=0x7F` as ASCII at
    /// the start of the document (which is enough to make [`crate::sniff_xml_encoding`]'s
    /// prologue scan work, since an `<?xml ...?>` declaration is required to be the first thing
    /// in the entity and is itself pure ASCII), a byte elsewhere in the document is not reliably
    /// ASCII-interpretable without tracking decoder state (ISO-2022-JP's shift state, or --
    /// for the other six -- whether the byte is a lead or trail byte of a multi-byte sequence),
    /// so this method intentionally answers `false` for all seven.
    #[must_use]
    pub const fn is_ascii_compatible_single_byte(&self) -> bool {
        matches!(
            self,
            Encoding::Utf8
                | Encoding::Latin1
                | Encoding::Windows1252
                | Encoding::Ascii
                | Encoding::Iso8859_2
                | Encoding::Iso8859_3
                | Encoding::Iso8859_4
                | Encoding::Iso8859_5
                | Encoding::Iso8859_6
                | Encoding::Iso8859_7
                | Encoding::Iso8859_8
                | Encoding::Iso8859_9
                | Encoding::Iso8859_10
                | Encoding::Iso8859_13
                | Encoding::Iso8859_14
                | Encoding::Iso8859_15
                | Encoding::Iso8859_16
                | Encoding::Windows1250
                | Encoding::Windows1251
                | Encoding::Windows1253
                | Encoding::Windows1254
                | Encoding::Windows1255
                | Encoding::Windows1256
                | Encoding::Windows1257
                | Encoding::Windows1258
                | Encoding::Koi8R
                | Encoding::Koi8U
                | Encoding::Ibm866
                | Encoding::Macintosh
                | Encoding::XMacCyrillic
        )
    }

    /// Whether this encoding is one of the 26 legacy single-byte encodings backed by a generated
    /// 128-entry upper-half table in `crate::tables::legacy_single_byte` (every [`Encoding`]
    /// variant except UTF-8/UTF-16LE/UTF-16BE, the original [`Encoding::Latin1`]/
    /// [`Encoding::Windows1252`]/[`Encoding::Ascii`] trio, and the seven multi-byte encodings —
    /// the three legacy Japanese ones and the four legacy CJK ones).
    #[must_use]
    pub(crate) const fn is_legacy_single_byte_table(&self) -> bool {
        matches!(
            self,
            Encoding::Iso8859_2
                | Encoding::Iso8859_3
                | Encoding::Iso8859_4
                | Encoding::Iso8859_5
                | Encoding::Iso8859_6
                | Encoding::Iso8859_7
                | Encoding::Iso8859_8
                | Encoding::Iso8859_9
                | Encoding::Iso8859_10
                | Encoding::Iso8859_13
                | Encoding::Iso8859_14
                | Encoding::Iso8859_15
                | Encoding::Iso8859_16
                | Encoding::Windows1250
                | Encoding::Windows1251
                | Encoding::Windows1253
                | Encoding::Windows1254
                | Encoding::Windows1255
                | Encoding::Windows1256
                | Encoding::Windows1257
                | Encoding::Windows1258
                | Encoding::Koi8R
                | Encoding::Koi8U
                | Encoding::Ibm866
                | Encoding::Macintosh
                | Encoding::XMacCyrillic
        )
    }

    /// Create a new incremental [`Decoder`] for this encoding.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_encoding::Encoding;
    ///
    /// let mut decoder = Encoding::Utf8.new_decoder();
    /// let mut out = String::new();
    /// decoder.decode_to_string(b"hello", &mut out, true).expect("valid utf-8");
    /// assert_eq!(out, "hello");
    /// ```
    #[must_use]
    pub const fn new_decoder(self) -> Decoder {
        Decoder::new(self)
    }
}

impl std::fmt::Display for Encoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// WHATWG's definition of "ASCII whitespace" used when trimming labels: tab, line feed, form
/// feed, carriage return and space.
const fn is_whatwg_ascii_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\x0C' | '\r' | ' ')
}

/// Resolve an already-trimmed, already-lowercased label. Kept as a `match` over `&str` (rather
/// than a hash map) since the label set is small, fixed at compile time, and this keeps the
/// crate free of any external dependency.
///
/// Label lists are transcribed from the WHATWG Encoding Standard
/// ([`§4.2 "get an encoding"`](https://encoding.spec.whatwg.org/#names-and-labels), data dated
/// 2024-09-18), with one deliberate deviation: the `"iso-8859-9"`/`"latin5"`-family labels
/// resolve to [`Encoding::Iso8859_9`] here rather than [`Encoding::Windows1254`] (see the
/// type-level docs on [`Encoding`]).
fn encoding_for_lowercase_label(label: &str) -> Option<Encoding> {
    match label {
        // --- UTF-8 --------------------------------------------------------------------------
        "unicode-1-1-utf-8" | "unicode11utf8" | "unicode20utf8" | "utf-8" | "utf8"
        | "x-unicode20utf8" => Some(Encoding::Utf8),

        // --- UTF-16, unspecified/little-endian ----------------------------------------------
        // Per the WHATWG Encoding Standard, a bare "utf-16" label (no endianness suffix) is
        // resolved as UTF-16LE for compatibility with the (much more common) little-endian
        // content that historically omitted the suffix.
        "utf-16" | "utf-16le" | "utf16" | "utf16le" | "ucs-2" | "unicode" | "iso-10646-ucs-2"
        | "csunicode" => Some(Encoding::Utf16Le),

        // --- UTF-16, big-endian ----------------------------------------------------------------
        "utf-16be" | "utf16be" | "unicodefffe" => Some(Encoding::Utf16Be),

        // --- ISO-8859-1 / Latin-1 --------------------------------------------------------------
        "iso-8859-1" | "iso-ir-100" | "iso8859-1" | "iso88591" | "iso_8859-1"
        | "iso_8859-1:1987" | "l1" | "latin1" | "csisolatin1" | "8859-1" | "88591" => {
            Some(Encoding::Latin1)
        }

        // --- windows-1252 ------------------------------------------------------------------------
        "windows-1252" | "cp1252" | "x-cp1252" => Some(Encoding::Windows1252),

        // --- US-ASCII --------------------------------------------------------------------------
        "us-ascii" | "ascii" | "ansi_x3.4-1968" | "ansi_x3.4-1986" | "iso646-us" | "iso-ir-6"
        | "us" | "ibm367" | "cp367" | "csascii" => Some(Encoding::Ascii),

        // --- ISO-8859-2 --------------------------------------------------------------------------
        "csisolatin2" | "iso-8859-2" | "iso-ir-101" | "iso8859-2" | "iso88592" | "iso_8859-2"
        | "iso_8859-2:1987" | "l2" | "latin2" => Some(Encoding::Iso8859_2),

        // --- ISO-8859-3 --------------------------------------------------------------------------
        "csisolatin3" | "iso-8859-3" | "iso-ir-109" | "iso8859-3" | "iso88593" | "iso_8859-3"
        | "iso_8859-3:1988" | "l3" | "latin3" => Some(Encoding::Iso8859_3),

        // --- ISO-8859-4 --------------------------------------------------------------------------
        "csisolatin4" | "iso-8859-4" | "iso-ir-110" | "iso8859-4" | "iso88594" | "iso_8859-4"
        | "iso_8859-4:1988" | "l4" | "latin4" => Some(Encoding::Iso8859_4),

        // --- ISO-8859-5 --------------------------------------------------------------------------
        "csisolatincyrillic" | "cyrillic" | "iso-8859-5" | "iso-ir-144" | "iso8859-5"
        | "iso88595" | "iso_8859-5" | "iso_8859-5:1988" => Some(Encoding::Iso8859_5),

        // --- ISO-8859-6 --------------------------------------------------------------------------
        "arabic" | "asmo-708" | "csiso88596e" | "csiso88596i" | "csisolatinarabic" | "ecma-114"
        | "iso-8859-6" | "iso-8859-6-e" | "iso-8859-6-i" | "iso-ir-127" | "iso8859-6"
        | "iso88596" | "iso_8859-6" | "iso_8859-6:1987" => Some(Encoding::Iso8859_6),

        // --- ISO-8859-7 --------------------------------------------------------------------------
        "csisolatingreek" | "ecma-118" | "elot_928" | "greek" | "greek8" | "iso-8859-7"
        | "iso-ir-126" | "iso8859-7" | "iso88597" | "iso_8859-7" | "iso_8859-7:1987"
        | "sun_eu_greek" => Some(Encoding::Iso8859_7),

        // --- ISO-8859-8 (and ISO-8859-8-I: same repertoire, direction only, out of scope) -------
        "csiso88598e" | "csisolatinhebrew" | "hebrew" | "iso-8859-8" | "iso-8859-8-e"
        | "iso-ir-138" | "iso8859-8" | "iso88598" | "iso_8859-8" | "iso_8859-8:1988" | "visual"
        | "csiso88598i" | "iso-8859-8-i" | "logical" => Some(Encoding::Iso8859_8),

        // --- ISO-8859-9 (kept distinct from windows-1254; see `Encoding` docs) -----------------
        "csisolatin5" | "iso-8859-9" | "iso-ir-148" | "iso8859-9" | "iso88599" | "iso_8859-9"
        | "iso_8859-9:1989" | "l5" | "latin5" => Some(Encoding::Iso8859_9),

        // --- ISO-8859-10 -------------------------------------------------------------------------
        "csisolatin6" | "iso-8859-10" | "iso-ir-157" | "iso8859-10" | "iso885910" | "l6"
        | "latin6" => Some(Encoding::Iso8859_10),

        // --- ISO-8859-13 -------------------------------------------------------------------------
        "iso-8859-13" | "iso8859-13" | "iso885913" => Some(Encoding::Iso8859_13),

        // --- ISO-8859-14 -------------------------------------------------------------------------
        "iso-8859-14" | "iso8859-14" | "iso885914" => Some(Encoding::Iso8859_14),

        // --- ISO-8859-15 -------------------------------------------------------------------------
        "csisolatin9" | "iso-8859-15" | "iso8859-15" | "iso885915" | "iso_8859-15" | "l9" => {
            Some(Encoding::Iso8859_15)
        }

        // --- ISO-8859-16 -------------------------------------------------------------------------
        "iso-8859-16" => Some(Encoding::Iso8859_16),

        // --- windows-1250 ------------------------------------------------------------------------
        "cp1250" | "windows-1250" | "x-cp1250" => Some(Encoding::Windows1250),

        // --- windows-1251 ------------------------------------------------------------------------
        "cp1251" | "windows-1251" | "x-cp1251" => Some(Encoding::Windows1251),

        // --- windows-1253 ------------------------------------------------------------------------
        "cp1253" | "windows-1253" | "x-cp1253" => Some(Encoding::Windows1253),

        // --- windows-1254 ------------------------------------------------------------------------
        "cp1254" | "windows-1254" | "x-cp1254" => Some(Encoding::Windows1254),

        // --- windows-1255 ------------------------------------------------------------------------
        "cp1255" | "windows-1255" | "x-cp1255" => Some(Encoding::Windows1255),

        // --- windows-1256 ------------------------------------------------------------------------
        "cp1256" | "windows-1256" | "x-cp1256" => Some(Encoding::Windows1256),

        // --- windows-1257 ------------------------------------------------------------------------
        "cp1257" | "windows-1257" | "x-cp1257" => Some(Encoding::Windows1257),

        // --- windows-1258 ------------------------------------------------------------------------
        "cp1258" | "windows-1258" | "x-cp1258" => Some(Encoding::Windows1258),

        // --- KOI8-R ------------------------------------------------------------------------------
        "cskoi8r" | "koi" | "koi8" | "koi8-r" | "koi8_r" => Some(Encoding::Koi8R),

        // --- KOI8-U ------------------------------------------------------------------------------
        "koi8-ru" | "koi8-u" => Some(Encoding::Koi8U),

        // --- IBM866 ------------------------------------------------------------------------------
        "866" | "cp866" | "csibm866" | "ibm866" => Some(Encoding::Ibm866),

        // --- macintosh ---------------------------------------------------------------------------
        "csmacintosh" | "mac" | "macintosh" | "x-mac-roman" => Some(Encoding::Macintosh),

        // --- x-mac-cyrillic ----------------------------------------------------------------------
        "x-mac-cyrillic" | "x-mac-ukrainian" => Some(Encoding::XMacCyrillic),

        // --- EUC-JP ------------------------------------------------------------------------------
        "cseucpkdfmtjapanese" | "euc-jp" | "x-euc-jp" => Some(Encoding::EucJp),

        // --- ISO-2022-JP -------------------------------------------------------------------------
        "csiso2022jp" | "iso-2022-jp" => Some(Encoding::Iso2022Jp),

        // --- Shift_JIS ---------------------------------------------------------------------------
        "csshiftjis" | "ms932" | "ms_kanji" | "shift-jis" | "shift_jis" | "sjis"
        | "windows-31j" | "x-sjis" => Some(Encoding::ShiftJis),

        // --- GBK ---------------------------------------------------------------------------------
        "chinese" | "csgb2312" | "csiso58gb231280" | "gb2312" | "gb_2312" | "gb_2312-80"
        | "gbk" | "iso-ir-58" | "x-gbk" => Some(Encoding::Gbk),

        // --- gb18030 -----------------------------------------------------------------------------
        "gb18030" => Some(Encoding::Gb18030),

        // --- Big5 --------------------------------------------------------------------------------
        "big5" | "big5-hkscs" | "cn-big5" | "csbig5" | "x-x-big5" => Some(Encoding::Big5),

        // --- EUC-KR ------------------------------------------------------------------------------
        "cseuckr" | "csksc56011987" | "euc-kr" | "iso-ir-149" | "korean" | "ks_c_5601-1987"
        | "ks_c_5601-1989" | "ksc5601" | "ksc_5601" | "windows-949" => Some(Encoding::EucKr),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_label_is_case_insensitive() {
        assert_eq!(Encoding::for_label("UTF-8"), Some(Encoding::Utf8));
        assert_eq!(Encoding::for_label("Utf-8"), Some(Encoding::Utf8));
        assert_eq!(Encoding::for_label("uTf8"), Some(Encoding::Utf8));
        assert_eq!(
            Encoding::for_label("WINDOWS-1252"),
            Some(Encoding::Windows1252)
        );
        assert_eq!(Encoding::for_label("Iso-8859-1"), Some(Encoding::Latin1));
        assert_eq!(Encoding::for_label("US-ASCII"), Some(Encoding::Ascii));
        assert_eq!(Encoding::for_label("Shift_JIS"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("SHIFT_JIS"), Some(Encoding::ShiftJis));
    }

    #[test]
    fn for_label_trims_whatwg_whitespace() {
        assert_eq!(Encoding::for_label("  utf-8\t\n"), Some(Encoding::Utf8));
        assert_eq!(Encoding::for_label("\r\nlatin1 "), Some(Encoding::Latin1));
        assert_eq!(
            Encoding::for_label(" shift_jis \n"),
            Some(Encoding::ShiftJis)
        );
    }

    #[test]
    fn for_label_unknown_returns_none() {
        assert_eq!(Encoding::for_label("utf-32"), None);
        assert_eq!(Encoding::for_label(""), None);
        assert_eq!(Encoding::for_label("   "), None);
        // Deliberately unsupported: Thai (windows-874/iso-8859-11) is outside this crate's
        // current scope.
        assert_eq!(Encoding::for_label("windows-874"), None);
        assert_eq!(Encoding::for_label("iso-8859-11"), None);
    }

    #[test]
    fn for_label_cjk_multi_byte_labels() {
        assert_eq!(Encoding::for_label("gbk"), Some(Encoding::Gbk));
        assert_eq!(Encoding::for_label("GBK"), Some(Encoding::Gbk));
        assert_eq!(Encoding::for_label("gb2312"), Some(Encoding::Gbk));
        assert_eq!(Encoding::for_label("x-gbk"), Some(Encoding::Gbk));
        assert_eq!(Encoding::for_label("chinese"), Some(Encoding::Gbk));
        assert_eq!(Encoding::for_label("gb18030"), Some(Encoding::Gb18030));
        assert_eq!(Encoding::for_label("GB18030"), Some(Encoding::Gb18030));
        assert_eq!(Encoding::for_label("big5"), Some(Encoding::Big5));
        assert_eq!(Encoding::for_label("Big5"), Some(Encoding::Big5));
        assert_eq!(Encoding::for_label("big5-hkscs"), Some(Encoding::Big5));
        assert_eq!(Encoding::for_label("cn-big5"), Some(Encoding::Big5));
        assert_eq!(Encoding::for_label("euc-kr"), Some(Encoding::EucKr));
        assert_eq!(Encoding::for_label("EUC-KR"), Some(Encoding::EucKr));
        assert_eq!(Encoding::for_label("windows-949"), Some(Encoding::EucKr));
        assert_eq!(Encoding::for_label("ks_c_5601-1987"), Some(Encoding::EucKr));
        assert_eq!(Encoding::for_label("korean"), Some(Encoding::EucKr));
    }

    #[test]
    fn for_label_utf16_variants() {
        assert_eq!(Encoding::for_label("utf-16"), Some(Encoding::Utf16Le));
        assert_eq!(Encoding::for_label("utf-16le"), Some(Encoding::Utf16Le));
        assert_eq!(Encoding::for_label("utf-16be"), Some(Encoding::Utf16Be));
        assert_eq!(Encoding::for_label("unicodeFFFE"), Some(Encoding::Utf16Be));
    }

    #[test]
    fn for_label_iso_8859_9_is_kept_distinct_from_windows_1254() {
        assert_eq!(Encoding::for_label("iso-8859-9"), Some(Encoding::Iso8859_9));
        assert_eq!(Encoding::for_label("latin5"), Some(Encoding::Iso8859_9));
        assert_eq!(
            Encoding::for_label("windows-1254"),
            Some(Encoding::Windows1254)
        );
        assert_eq!(Encoding::for_label("cp1254"), Some(Encoding::Windows1254));
        assert_ne!(Encoding::Iso8859_9, Encoding::Windows1254);
    }

    #[test]
    fn for_label_japanese_multi_byte_labels() {
        assert_eq!(Encoding::for_label("shift_jis"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("sjis"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("ms932"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("windows-31j"), Some(Encoding::ShiftJis));
        assert_eq!(Encoding::for_label("euc-jp"), Some(Encoding::EucJp));
        assert_eq!(Encoding::for_label("x-euc-jp"), Some(Encoding::EucJp));
        assert_eq!(
            Encoding::for_label("iso-2022-jp"),
            Some(Encoding::Iso2022Jp)
        );
        assert_eq!(
            Encoding::for_label("csiso2022jp"),
            Some(Encoding::Iso2022Jp)
        );
    }

    #[test]
    fn name_round_trips_through_for_label_for_every_supported_encoding() {
        for &enc in Encoding::ALL {
            assert_eq!(
                Encoding::for_label(enc.name()),
                Some(enc),
                "name = {}",
                enc.name()
            );
            // Case-insensitivity holds for every encoding's canonical name too.
            assert_eq!(
                Encoding::for_label(&enc.name().to_lowercase()),
                Some(enc),
                "lowercased name = {}",
                enc.name()
            );
            assert_eq!(
                Encoding::for_label(&enc.name().to_uppercase()),
                Some(enc),
                "uppercased name = {}",
                enc.name()
            );
        }
    }

    #[test]
    fn all_is_free_of_duplicates() {
        for (i, &a) in Encoding::ALL.iter().enumerate() {
            for &b in &Encoding::ALL[i + 1..] {
                assert_ne!(a, b, "Encoding::ALL contains a duplicate: {a}");
            }
        }
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(Encoding::Utf8.to_string(), "UTF-8");
        assert_eq!(Encoding::Windows1252.to_string(), "windows-1252");
        assert_eq!(Encoding::ShiftJis.to_string(), "Shift_JIS");
    }

    #[test]
    fn is_ascii_compatible_single_byte_matches_expectation() {
        assert!(Encoding::Utf8.is_ascii_compatible_single_byte());
        assert!(Encoding::Latin1.is_ascii_compatible_single_byte());
        assert!(Encoding::Windows1252.is_ascii_compatible_single_byte());
        assert!(Encoding::Ascii.is_ascii_compatible_single_byte());
        assert!(Encoding::Iso8859_2.is_ascii_compatible_single_byte());
        assert!(Encoding::Windows1251.is_ascii_compatible_single_byte());
        assert!(Encoding::Koi8R.is_ascii_compatible_single_byte());
        assert!(!Encoding::Utf16Le.is_ascii_compatible_single_byte());
        assert!(!Encoding::Utf16Be.is_ascii_compatible_single_byte());
        assert!(!Encoding::ShiftJis.is_ascii_compatible_single_byte());
        assert!(!Encoding::EucJp.is_ascii_compatible_single_byte());
        assert!(!Encoding::Iso2022Jp.is_ascii_compatible_single_byte());
        assert!(!Encoding::Gbk.is_ascii_compatible_single_byte());
        assert!(!Encoding::Gb18030.is_ascii_compatible_single_byte());
        assert!(!Encoding::Big5.is_ascii_compatible_single_byte());
        assert!(!Encoding::EucKr.is_ascii_compatible_single_byte());
    }

    #[test]
    fn is_legacy_single_byte_table_matches_expectation() {
        assert!(Encoding::Iso8859_2.is_legacy_single_byte_table());
        assert!(Encoding::Windows1250.is_legacy_single_byte_table());
        assert!(Encoding::Koi8R.is_legacy_single_byte_table());
        assert!(Encoding::Macintosh.is_legacy_single_byte_table());
        assert!(!Encoding::Latin1.is_legacy_single_byte_table());
        assert!(!Encoding::Windows1252.is_legacy_single_byte_table());
        assert!(!Encoding::Ascii.is_legacy_single_byte_table());
        assert!(!Encoding::Utf8.is_legacy_single_byte_table());
        assert!(!Encoding::ShiftJis.is_legacy_single_byte_table());
        assert!(!Encoding::EucJp.is_legacy_single_byte_table());
        assert!(!Encoding::Iso2022Jp.is_legacy_single_byte_table());
        assert!(!Encoding::Gbk.is_legacy_single_byte_table());
        assert!(!Encoding::Gb18030.is_legacy_single_byte_table());
        assert!(!Encoding::Big5.is_legacy_single_byte_table());
        assert!(!Encoding::EucKr.is_legacy_single_byte_table());
    }

    #[test]
    fn all_contains_every_variant_exactly_once_via_new_decoder_smoke_test() {
        // Not exhaustive by construction (nothing enforces that `ALL` is kept in sync with the
        // enum at compile time), but constructing a decoder for every entry at least catches an
        // entry that was added to `ALL` without the corresponding `Decoder::new` support (which
        // would panic here rather than silently mis-decode).
        for &encoding in Encoding::ALL {
            let _ = encoding.new_decoder();
        }
    }
}
