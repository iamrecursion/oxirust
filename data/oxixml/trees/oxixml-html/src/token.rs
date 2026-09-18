//! The tokens the HTML tokenizer produces.
//!
//! These mirror the token types of
//! [HTML § 13.2.5](https://html.spec.whatwg.org/multipage/parsing.html#tokenization)
//! one-for-one, with a single deliberate difference: the standard emits one
//! *character token* per character, while [`Token::Characters`] carries a run of
//! them. Runs are only ever coalesced when the standard would have emitted the
//! characters consecutively from the same tokenizer state, so a tree builder
//! that iterates the characters of the run sees exactly the sequence the
//! standard describes.

use std::fmt;

/// An attribute of a start or end tag, after tokenization.
///
/// The name has already been lower-cased (HTML tag and attribute names are
/// ASCII case-insensitive) and the value has already had its character
/// references expanded.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Attribute {
    /// The attribute name, ASCII-lower-cased.
    pub name: String,
    /// The attribute value with character references expanded.
    pub value: String,
}

impl Attribute {
    /// Creates an attribute from a name and a value.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A start or end tag token.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Tag {
    /// The tag name, ASCII-lower-cased.
    pub name: String,
    /// The attributes, in source order, with duplicates already dropped.
    pub attributes: Vec<Attribute>,
    /// Whether the tag ended with `/>`.
    ///
    /// The tree construction stage acknowledges this flag only for void and
    /// foreign elements; everywhere else a self-closing start tag is a parse
    /// error and the slash is ignored.
    pub self_closing: bool,
}

impl Tag {
    /// Returns the value of the attribute with the given name, if present.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
            .map(|attribute| attribute.value.as_str())
    }

    /// Returns `true` when an attribute with the given name is present.
    #[must_use]
    pub fn has_attribute(&self, name: &str) -> bool {
        self.attributes.iter().any(|a| a.name == name)
    }
}

/// A `<!DOCTYPE …>` token.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Doctype {
    /// The name, ASCII-lower-cased, or `None` when the DOCTYPE had none.
    pub name: Option<String>,
    /// The public identifier, or `None` when the DOCTYPE had none.
    pub public_id: Option<String>,
    /// The system identifier, or `None` when the DOCTYPE had none.
    pub system_id: Option<String>,
    /// The "force-quirks" flag, set by any malformed DOCTYPE.
    pub force_quirks: bool,
}

/// A `<?target data?>` processing instruction token.
///
/// The HTML tokenizer grew processing instruction states in 2025; before that
/// `<?…>` was a bogus comment. A target that is an ASCII case-insensitive match
/// for `xml` or `xml-stylesheet` is still a comment, as are targets containing
/// characters outside `[A-Za-z0-9_-]`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProcessingInstruction {
    /// The target: the name immediately after `<?`.
    pub target: String,
    /// The data: everything after the target and the white space separating
    /// it, without the closing `?>` or `>`.
    pub data: String,
}

/// One token of the HTML tokenizer output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// A `<!DOCTYPE …>` declaration.
    Doctype(Doctype),
    /// A start tag such as `<p class="x">`.
    StartTag(Tag),
    /// An end tag such as `</p>`.
    EndTag(Tag),
    /// A comment; the data is everything between `<!--` and `-->`.
    Comment(String),
    /// A processing instruction.
    ProcessingInstruction(ProcessingInstruction),
    /// A run of character tokens.
    Characters(String),
    /// The end of the input stream.
    Eof,
}

impl Token {
    /// Returns the tag of a start or end tag token.
    #[must_use]
    pub const fn tag(&self) -> Option<&Tag> {
        match self {
            Self::StartTag(tag) | Self::EndTag(tag) => Some(tag),
            _ => None,
        }
    }

    /// Returns the name of a start or end tag token.
    #[must_use]
    pub fn tag_name(&self) -> Option<&str> {
        self.tag().map(|tag| tag.name.as_str())
    }

    /// Returns `true` when this is a start tag with the given name.
    #[must_use]
    pub fn is_start(&self, name: &str) -> bool {
        matches!(self, Self::StartTag(tag) if tag.name == name)
    }

    /// Returns `true` when this is an end tag with the given name.
    #[must_use]
    pub fn is_end(&self, name: &str) -> bool {
        matches!(self, Self::EndTag(tag) if tag.name == name)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Doctype(doctype) => {
                write!(f, "<!DOCTYPE {}>", doctype.name.as_deref().unwrap_or(""))
            }
            Self::StartTag(tag) => write!(f, "<{}>", tag.name),
            Self::EndTag(tag) => write!(f, "</{}>", tag.name),
            Self::Comment(data) => write!(f, "<!--{data}-->"),
            Self::ProcessingInstruction(pi) => write!(f, "<?{} {}?>", pi.target, pi.data),
            Self::Characters(text) => write!(f, "{text:?}"),
            Self::Eof => f.write_str("<eof>"),
        }
    }
}
