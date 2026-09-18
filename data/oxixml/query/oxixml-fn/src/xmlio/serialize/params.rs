//! The serialization parameters of
//! [Serialization 3.1 §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam) — the
//! [`SerializationParameters`] set itself, and the reader that builds one from an
//! `output:serialization-parameters` element.

use super::xml::element_bindings;
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{ExpandedName, Item, NodeKind, NodeRef, Sequence, XdmNode};

/// The output method `fn:serialize` applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMethod {
    /// `method="xml"`, the default.
    #[default]
    Xml,
    /// `method="text"`.
    Text,
    /// `method="adaptive"`.
    Adaptive,
    /// `method="json"`.
    Json,
    /// `method="html"`.
    Html,
    /// `method="xhtml"`.
    Xhtml,
}

impl OutputMethod {
    /// Parses a `method` serialization parameter.
    ///
    /// [Serialization 3.1 §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam) types
    /// the value as a `QName`: a name in no namespace selects one of the standard methods, and a
    /// name in a namespace selects an implementation-defined one, of which there are none here.
    /// XQuery 3.0's `EQName` spelling writes the first as `Q{}xml`, which names the same method as
    /// `xml` (QT3 `method-xml/K2-Serialization-29`).
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Forg0006`] for a method this implementation does not provide.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_fn::xmlio::OutputMethod;
    ///
    /// assert_eq!(OutputMethod::parse("html")?, OutputMethod::Html);
    /// assert_eq!(OutputMethod::parse("Q{}html")?, OutputMethod::Html);
    /// assert!(OutputMethod::parse("Q{urn:example}mine").is_err());
    /// # Ok::<(), oxixml_fn::FnError>(())
    /// ```
    pub fn parse(name: &str) -> FnResult<Self> {
        let unsupported = || {
            FnError::new(
                FnErrorCode::Forg0006,
                format!("the serialization method `{name}` is not supported"),
            )
        };
        let local = match name.strip_prefix("Q{") {
            None => name,
            Some(rest) => match rest.split_once('}') {
                Some(("", local)) => local,
                _ => return Err(unsupported()),
            },
        };
        match local {
            "xml" | "" => Ok(Self::Xml),
            "text" => Ok(Self::Text),
            "adaptive" => Ok(Self::Adaptive),
            "json" => Ok(Self::Json),
            "html" => Ok(Self::Html),
            "xhtml" => Ok(Self::Xhtml),
            _ => Err(unsupported()),
        }
    }

    /// The method's name as the `method` parameter spells it, for diagnostics.
    #[must_use]
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Xml => "xml",
            Self::Text => "text",
            Self::Adaptive => "adaptive",
            Self::Json => "json",
            Self::Html => "html",
            Self::Xhtml => "xhtml",
        }
    }
}

/// The serialization parameters `fn:serialize` honours.
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::SerializationParameters;
///
/// let parameters = SerializationParameters::default()
///     .with_indent(true)
///     .with_omit_xml_declaration(false);
/// assert!(parameters.indent());
/// assert!(!parameters.omit_xml_declaration());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializationParameters {
    method: OutputMethod,
    indent: bool,
    omit_xml_declaration: bool,
    omit_xml_declaration_specified: bool,
    version: String,
    version_specified: bool,
    encoding: String,
    standalone: Option<bool>,
    item_separator: Option<String>,
    doctype_system: Option<String>,
    doctype_public: Option<String>,
    character_map: Vec<(char, String)>,
    cdata_section_elements: Vec<ExpandedName>,
    allow_duplicate_names: bool,
    include_content_type: bool,
    json_node_output_method: OutputMethod,
    html_version: String,
    html_version_specified: bool,
    escape_uri_attributes: bool,
    byte_order_mark: bool,
    undeclare_prefixes: bool,
    normalization_form: String,
    suppress_indentation: Vec<ExpandedName>,
    max_depth: usize,
}

impl SerializationParameters {
    /// The default value of [`SerializationParameters::max_depth`]: 256 levels of nesting.
    ///
    /// The serializer descends a tree, a map and an array recursively, and the value it is
    /// handed is the caller's, so a value nested past this ceiling is reported as
    /// [`FnErrorCode::Xpdy0130`] rather than allowed to exhaust the thread's stack — which
    /// would abort the process instead of raising an error a host could catch.
    pub const DEFAULT_MAX_DEPTH: usize = 256;

    /// The `use-character-maps` parameter: characters that are replaced by arbitrary text on
    /// output, *after* markup escaping and without being escaped themselves.
    #[must_use]
    pub fn character_map(&self) -> &[(char, String)] {
        &self.character_map
    }

    /// Replaces the `use-character-maps` parameter.
    #[must_use]
    pub fn with_character_map(mut self, mappings: Vec<(char, String)>) -> Self {
        self.character_map = mappings;
        self
    }

    /// The `cdata-section-elements` parameter: the elements whose text-node children the `xml`
    /// and `html` output methods write as CDATA sections rather than as escaped text.
    #[must_use]
    pub fn cdata_section_elements(&self) -> &[ExpandedName] {
        &self.cdata_section_elements
    }

    /// Replaces the `cdata-section-elements` parameter.
    #[must_use]
    pub fn with_cdata_section_elements(mut self, names: Vec<ExpandedName>) -> Self {
        self.cdata_section_elements = names;
        self
    }

    /// Whether `element`'s text-node children belong in a CDATA section.
    pub(super) fn writes_cdata_sections(&self, element: &NodeRef, case_blind: bool) -> bool {
        names_contain(&self.cdata_section_elements, element, case_blind)
    }

    /// The `suppress-indentation` parameter: the elements inside which no indentation is added,
    /// however deeply the content nests.
    #[must_use]
    pub fn suppress_indentation(&self) -> &[ExpandedName] {
        &self.suppress_indentation
    }

    /// Replaces the `suppress-indentation` parameter.
    #[must_use]
    pub fn with_suppress_indentation(mut self, names: Vec<ExpandedName>) -> Self {
        self.suppress_indentation = names;
        self
    }

    /// Whether `element` is one of the `suppress-indentation` elements, so that neither it nor
    /// anything it contains is indented.
    pub(super) fn suppresses_indentation(&self, element: &NodeRef, case_blind: bool) -> bool {
        names_contain(&self.suppress_indentation, element, case_blind)
    }

    /// The HTML version the `html` and `xhtml` output methods work to.
    ///
    /// [Serialization 3.1 §7](https://www.w3.org/TR/xslt-xquery-serialization-31/#html-output)
    /// gives `html-version` a default of 5, and says that in its absence the `version` parameter
    /// identifies the version of HTML — which is how the QT3 cases spell it
    /// (`declare option output:version "4.0"`). Only a value the caller actually set counts, so
    /// the `version` of `1.0` that the `xml` method carries by default never decides an HTML
    /// question.
    ///
    /// The fallback belongs to the `html` method alone. XHTML output is XML, so under
    /// `method="xhtml"` the `version` parameter names the *XML* version that goes into the XML
    /// declaration and says nothing about HTML — `html-version` is the only spelling that does
    /// (QT3 `method-xhtml/Serialization-xhtml-65`…`-67`, which set both).
    #[must_use]
    pub(super) fn effective_html_version(&self) -> &str {
        if self.html_version_specified || self.method == OutputMethod::Xhtml {
            &self.html_version
        } else if self.version_specified {
            &self.version
        } else {
            &self.html_version
        }
    }

    /// Whether the effective HTML version is 5 or later, which is what decides the document type
    /// declaration the `html` and `xhtml` output methods write, whether the `html` method treats
    /// an element in the XHTML namespace as an HTML element, and which vocabulary of empty
    /// elements the `xhtml` method minimizes.
    pub(super) fn is_html5(&self) -> bool {
        self.effective_html_version()
            .trim()
            .parse::<f64>()
            .is_ok_and(|version| version >= 5.0)
    }

    /// Whether the `version` parameter asks for XML 1.1, which decides both which characters the
    /// `xml` and `xhtml` methods may write at all and which of them have to be written as a
    /// character reference.
    ///
    /// Only the `xml` and `xhtml` methods read it as an XML version; for the `html` method the
    /// same parameter names a version of HTML (see [`Self::effective_html_version`]), so a
    /// `version` of `1.1` under `method="html"` says nothing about XML.
    pub(super) fn is_xml_11(&self) -> bool {
        matches!(self.method, OutputMethod::Xml | OutputMethod::Xhtml)
            && self.version.trim() == "1.1"
    }

    /// The replacement `character` maps to, if any.
    pub(super) fn mapped(&self, character: char) -> Option<&str> {
        self.character_map
            .iter()
            .find(|(from, _)| *from == character)
            .map(|(_, to)| to.as_str())
    }
}

impl SerializationParameters {
    /// The parameter set [Serialization 3.1
    /// §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam) defines, which is the
    /// one a *host language* serializes its own result with.
    ///
    /// It differs from [`SerializationParameters::default`] in exactly one place:
    /// `omit-xml-declaration` is `no`, so the `xml` method writes an XML declaration. `Default`
    /// is `fn:serialize`'s parameter set, and F&O 3.1 §14.7.1 gives that function `yes` instead
    /// so that a serialized fragment can be embedded — QT3's `fn/serialize.xml` pins the
    /// difference ("when params supplied as map, omit-xml-declaration default is true").
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_fn::xmlio::SerializationParameters;
    ///
    /// assert!(SerializationParameters::default().omit_xml_declaration());
    /// assert!(!SerializationParameters::serialization_defaults().omit_xml_declaration());
    /// ```
    #[must_use]
    pub fn serialization_defaults() -> Self {
        Self {
            omit_xml_declaration: false,
            ..Self::default()
        }
    }
}

impl Default for SerializationParameters {
    /// The Serialization 3.1 defaults, except that the XML declaration is omitted, which is what
    /// `fn:serialize` specifies for its own default parameter set — see
    /// [`SerializationParameters::serialization_defaults`] for the unmodified set.
    fn default() -> Self {
        Self {
            method: OutputMethod::Xml,
            indent: false,
            omit_xml_declaration: true,
            omit_xml_declaration_specified: false,
            version: "1.0".to_owned(),
            version_specified: false,
            encoding: "UTF-8".to_owned(),
            standalone: None,
            item_separator: None,
            doctype_system: None,
            doctype_public: None,
            character_map: Vec::new(),
            cdata_section_elements: Vec::new(),
            allow_duplicate_names: false,
            include_content_type: true,
            json_node_output_method: OutputMethod::Xml,
            html_version: "5".to_owned(),
            html_version_specified: false,
            // Serialization 3.1 §3 gives `escape-uri-attributes` the default `yes`; the other
            // three default to `no`, `no` and `none`.
            escape_uri_attributes: true,
            byte_order_mark: false,
            undeclare_prefixes: false,
            normalization_form: String::new(),
            suppress_indentation: Vec::new(),
            max_depth: Self::DEFAULT_MAX_DEPTH,
        }
    }
}

macro_rules! parameters {
    ($( $field:ident : $ty:ty, $with:ident ),+ $(,)?) => {
        impl SerializationParameters {
            $(
                #[doc = concat!("The `", stringify!($field), "` serialization parameter.")]
                #[must_use]
                pub fn $field(&self) -> $ty {
                    self.$field.clone()
                }

                #[doc = concat!("Replaces the `", stringify!($field), "` serialization parameter.")]
                #[must_use]
                pub fn $with(mut self, value: impl Into<$ty>) -> Self {
                    self.$field = value.into();
                    self
                }
            )+
        }
    };
}

parameters! {
    method: OutputMethod, with_method,
    indent: bool, with_indent,
    encoding: String, with_encoding,
    standalone: Option<bool>, with_standalone,
    item_separator: Option<String>, with_item_separator,
    doctype_system: Option<String>, with_doctype_system,
    doctype_public: Option<String>, with_doctype_public,
    allow_duplicate_names: bool, with_allow_duplicate_names,
    include_content_type: bool, with_include_content_type,
    json_node_output_method: OutputMethod, with_json_node_output_method,
    escape_uri_attributes: bool, with_escape_uri_attributes,
    byte_order_mark: bool, with_byte_order_mark,
    undeclare_prefixes: bool, with_undeclare_prefixes,
    normalization_form: String, with_normalization_form,
}

impl SerializationParameters {
    /// The `omit-xml-declaration` serialization parameter.
    #[must_use]
    pub const fn omit_xml_declaration(&self) -> bool {
        self.omit_xml_declaration
    }

    /// Replaces the `omit-xml-declaration` serialization parameter.
    #[must_use]
    pub const fn with_omit_xml_declaration(mut self, value: bool) -> Self {
        self.omit_xml_declaration = value;
        self.omit_xml_declaration_specified = true;
        self
    }

    /// Whether a caller actually asked for `omit-xml-declaration`, rather than inheriting the
    /// default of the parameter set it started from.
    ///
    /// `SEPM0009` is a contradiction between two parameters the caller *chose*, so a default that
    /// happens to omit the declaration — which is `fn:serialize`'s — does not raise it.
    pub(super) const fn omit_xml_declaration_specified(&self) -> bool {
        self.omit_xml_declaration_specified
    }

    /// The `version` serialization parameter: the XML version for the `xml` method, and — in the
    /// absence of `html-version` — the HTML version for the `html` method.
    #[must_use]
    pub fn version(&self) -> String {
        self.version.clone()
    }

    /// Replaces the `version` serialization parameter.
    #[must_use]
    pub fn with_version(mut self, value: impl Into<String>) -> Self {
        self.version = value.into();
        self.version_specified = true;
        self
    }

    /// The `html-version` serialization parameter, `5` unless the caller set another value.
    #[must_use]
    pub fn html_version(&self) -> String {
        self.html_version.clone()
    }

    /// Replaces the `html-version` serialization parameter.
    #[must_use]
    pub fn with_html_version(mut self, value: impl Into<String>) -> Self {
        self.html_version = value.into();
        self.html_version_specified = true;
        self
    }

    /// How deeply the serializer may descend into the value it is writing.
    ///
    /// This is not one of Serialization 3.1's parameters — it is this implementation's guard
    /// against unbounded recursion; see [`SerializationParameters::DEFAULT_MAX_DEPTH`].
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Replaces the nesting ceiling — see [`SerializationParameters::max_depth`].
    #[must_use]
    pub const fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

/// Whether a name-set serialization parameter names `element`.
///
/// `case_blind` compares the local names without regard to case, which is what the `html` output
/// method does throughout — HTML element names are case-insensitive, and QT3 spells the point out
/// with `suppress-indentation="TABLE"` against a `table` element (`Serialization-html-56`) and
/// `"math:MROW"` against an `mrow` (`-57`). The namespace name is compared exactly either way.
fn names_contain(names: &[ExpandedName], element: &NodeRef, case_blind: bool) -> bool {
    if names.is_empty() {
        return false;
    }
    let Some(name) = element.node_name() else {
        return false;
    };
    names.iter().any(|candidate| {
        if !case_blind {
            return *candidate == name;
        }
        candidate.namespace() == name.namespace()
            && candidate
                .local_name()
                .eq_ignore_ascii_case(name.local_name())
    })
}

/// The namespace of the `output:serialization-parameters` vocabulary,
/// `http://www.w3.org/2010/xslt-xquery-serialization`.
pub const SERIALIZATION_NAMESPACE: &str = "http://www.w3.org/2010/xslt-xquery-serialization";

/// The parameters `fn:serialize` accepts as an *element*, which
/// [Serialization 3.1 §3.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparams-in-xdm)
/// defines as an alternative to the option map.
///
/// The element is `output:serialization-parameters`, and each of its children names one
/// parameter and carries its value in a `value` attribute; `output:use-character-maps` instead
/// holds `output:character-map` children with `character` and `map-string`.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when `element` is not an `output:serialization-parameters` element
/// at all, because the second argument of `fn:serialize` is then simply of the wrong type;
/// [`FnErrorCode::Sepm0017`] for every fault *within* a well-named parameter element — an
/// unknown parameter, an unexpected attribute, a child in the wrong namespace, a missing or
/// inadmissible `value`; `SEPM0018` for a character mapped twice; and
/// [`FnErrorCode::Sepm0019`] for a parameter given twice.
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::{parameters_from_element, SERIALIZATION_NAMESPACE};
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str(&format!(
///     "<serialization-parameters xmlns=\"{SERIALIZATION_NAMESPACE}\"><indent value=\"yes\"/></serialization-parameters>"
/// ))?;
/// let element = tree.document_element().expect("a document element");
/// assert!(parameters_from_element(&element)?.indent());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parameters_from_element(element: &NodeRef) -> FnResult<SerializationParameters> {
    parameters_from_element_with_defaults(element, &SerializationParameters::default())
}

/// [`parameters_from_element`], reading the parameters over a caller-supplied set of defaults
/// rather than over `fn:serialize`'s.
///
/// A host language serializing its own result starts from
/// [`SerializationParameters::serialization_defaults`] — the values Serialization 3.1 §3 gives —
/// so that a parameter the document does not mention keeps that language's default rather than
/// the function's. Every parameter the document *does* carry overrides the corresponding default.
///
/// # Errors
///
/// The same as [`parameters_from_element`].
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::{parameters_from_element_with_defaults, SerializationParameters, SERIALIZATION_NAMESPACE};
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str(&format!(
///     "<serialization-parameters xmlns=\"{SERIALIZATION_NAMESPACE}\"><standalone value=\"yes\"/></serialization-parameters>"
/// ))?;
/// let element = tree.document_element().expect("a document element");
/// let base = SerializationParameters::serialization_defaults();
/// let parameters = parameters_from_element_with_defaults(&element, &base)?;
/// assert!(!parameters.omit_xml_declaration());
/// assert_eq!(parameters.standalone(), Some(true));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parameters_from_element_with_defaults(
    element: &NodeRef,
    defaults: &SerializationParameters,
) -> FnResult<SerializationParameters> {
    parameters_from_element_in_context(element, defaults, None)
}

/// [`parameters_from_element_with_defaults`], resolving an unprefixed element name in
/// `cdata-section-elements` and `suppress-indentation` against the host language's *default
/// element namespace*.
///
/// Those two parameters carry lexical `QName`s, and a name without a prefix denotes a name in the
/// default element namespace of the context the parameter was written in — not a name in no
/// namespace. In XQuery that context is the module's `declare default element namespace`, which
/// no part of a parameter document can carry, so a host that has one passes it here: QT3
/// `method-xml/K2-Serialization-28` writes `suppress-indentation "p para"` under a default
/// element namespace and requires the no-namespace `p` in its input *not* to be suppressed, and
/// `K2-Serialization-31` writes `cdata-section-elements "b"` and requires the `b` in that
/// namespace to be the one wrapped.
///
/// `None` — which is what [`parameters_from_element`] and
/// [`parameters_from_element_with_defaults`] pass — means there is no default element namespace,
/// so an unprefixed name is in no namespace, as `K2-Serialization-26`, `-27`, `-29`, `-30` and
/// `-32` require of a module that declares none.
///
/// # Errors
///
/// The same as [`parameters_from_element`].
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::{parameters_from_element_in_context, SerializationParameters, SERIALIZATION_NAMESPACE};
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str(&format!(
///     "<serialization-parameters xmlns=\"{SERIALIZATION_NAMESPACE}\"><cdata-section-elements value=\"b\"/></serialization-parameters>"
/// ))?;
/// let element = tree.document_element().expect("a document element");
/// let defaults = SerializationParameters::default();
///
/// let in_no_namespace = parameters_from_element_in_context(&element, &defaults, None)?;
/// assert_eq!(in_no_namespace.cdata_section_elements()[0].namespace(), None);
///
/// let in_context =
///     parameters_from_element_in_context(&element, &defaults, Some("urn:example"))?;
/// assert_eq!(in_context.cdata_section_elements()[0].namespace(), Some("urn:example"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn parameters_from_element_in_context(
    element: &NodeRef,
    defaults: &SerializationParameters,
    default_element_namespace: Option<&str>,
) -> FnResult<SerializationParameters> {
    let mut parameters = defaults.clone();
    if !is_serialization_element(element, "serialization-parameters") {
        // F&O 3.1 §14.7.1 admits exactly two spellings for `fn:serialize`'s `$options`: a map,
        // or an `output:serialization-parameters` element. A node that is neither does not fail
        // *as* a parameter document — it never was one — so the mismatch is the ordinary type
        // error of an argument that does not match the function's signature
        // (`serialize-xml-017b` passes an element of the same local name in another namespace,
        // `-018b` one of another local name).
        return Err(FnError::new(
            FnErrorCode::Xpty0004,
            "the serialization parameters must be an output:serialization-parameters element",
        ));
    }
    reject_foreign_attributes(element)?;
    let mut seen: Vec<ExpandedName> = Vec::new();
    for child in element.children() {
        if child.node_kind() != NodeKind::Element {
            continue;
        }
        let Some(name) = child.node_name() else {
            continue;
        };
        if seen.contains(&name) {
            return Err(FnError::new(
                FnErrorCode::Sepm0019,
                format!("the serialization parameter `{name}` is specified more than once"),
            ));
        }
        seen.push(name.clone());
        // A parameter in a foreign namespace is a processor extension, and an unknown one is
        // simply ignored — but a repeat of it is still `SEPM0019`, which the check above made.
        match name.namespace() {
            Some(SERIALIZATION_NAMESPACE) => {}
            Some(_) => continue,
            None => {
                return Err(invalid_parameters(format!(
                    "`{}` is not in the serialization parameter namespace",
                    name.local_name()
                )));
            }
        }
        parameters = apply_parameter(
            parameters,
            &child,
            name.local_name(),
            default_element_namespace,
        )?;
    }
    Ok(parameters)
}

fn apply_parameter(
    parameters: SerializationParameters,
    element: &NodeRef,
    local: &str,
    default_element_namespace: Option<&str>,
) -> FnResult<SerializationParameters> {
    if local == "use-character-maps" {
        return character_maps(parameters, element);
    }
    reject_foreign_attributes(element)?;
    let value = element
        .attributes()
        .into_iter()
        .find(|attribute| {
            attribute
                .node_name()
                .is_some_and(|name| name.namespace().is_none() && name.local_name() == "value")
        })
        .map(|attribute| attribute.string_value())
        .ok_or_else(|| invalid_parameters(format!("`{local}` needs a `value` attribute")))?;
    let trimmed = value.trim();
    let flag = |text: &str| match text {
        "yes" | "true" | "1" => Ok(true),
        "no" | "false" | "0" => Ok(false),
        other => Err(invalid_parameters(format!(
            "`{other}` is not a yes/no value for `{local}`"
        ))),
    };
    Ok(match local {
        "method" => parameters.with_method(OutputMethod::parse(trimmed).map_err(|_| {
            invalid_parameters(format!("`{trimmed}` is not a supported output method"))
        })?),
        "indent" => parameters.with_indent(flag(trimmed)?),
        "omit-xml-declaration" => parameters.with_omit_xml_declaration(flag(trimmed)?),
        "version" => parameters.with_version(trimmed.to_owned()),
        "encoding" => parameters.with_encoding(trimmed.to_owned()),
        "standalone" => {
            if trimmed == "omit" {
                parameters.with_standalone(None)
            } else {
                parameters.with_standalone(Some(flag(trimmed)?))
            }
        }
        "allow-duplicate-names" => parameters.with_allow_duplicate_names(flag(trimmed)?),
        "include-content-type" => parameters.with_include_content_type(flag(trimmed)?),
        "json-node-output-method" => {
            parameters.with_json_node_output_method(OutputMethod::parse(trimmed).map_err(|_| {
                invalid_parameters(format!("`{trimmed}` is not a supported output method"))
            })?)
        }
        "item-separator" => parameters.with_item_separator(Some(value)),
        "doctype-system" => parameters.with_doctype_system(Some(value)),
        "doctype-public" => parameters.with_doctype_public(Some(value)),
        "html-version" => parameters.with_html_version(trimmed.to_owned()),
        "cdata-section-elements" => parameters.with_cdata_section_elements(qname_list(
            element,
            trimmed,
            local,
            default_element_namespace,
        )?),
        "suppress-indentation" => parameters.with_suppress_indentation(qname_list(
            element,
            trimmed,
            local,
            default_element_namespace,
        )?),
        "escape-uri-attributes" => parameters.with_escape_uri_attributes(flag(trimmed)?),
        "undeclare-prefixes" => parameters.with_undeclare_prefixes(flag(trimmed)?),
        "byte-order-mark" => parameters.with_byte_order_mark(flag(trimmed)?),
        // An unsupported normalization form is `SESU0011`, which
        // [Serialization 3.1 §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam)
        // makes a *serialization* error rather than a fault in the parameter document — so the
        // name is carried through and rejected when the value is serialized, not here.
        "normalization-form" => parameters.with_normalization_form(trimmed.to_owned()),
        // Parameters this serializer accepts and does not act on. `media-type` is metadata for
        // the caller rather than something that changes the text, and `parameter-document` names
        // parameters held somewhere only the caller can reach.
        "media-type" | "parameter-document" => parameters,
        other => {
            return Err(invalid_parameters(format!(
                "`{other}` is not a serialization parameter"
            )));
        }
    })
}

/// Reads a parameter whose value is a whitespace-separated list of element names —
/// `cdata-section-elements` and `suppress-indentation`.
///
/// [Serialization 3.1 §3.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparams-in-xdm)
/// resolves each lexical `QName` against the in-scope namespaces of the element that carries it,
/// so `<output:cdata-section-elements value="p:a c"/>` names `{…}a` when `p` is bound on any
/// ancestor. A name may also be written in XQuery 3.0's `EQName` form, `Q{namespace}local`, which
/// carries its namespace with it and needs no binding (QT3 `method-html/Serialization-html-16a`,
/// `method-xml/K2-Serialization-32`).
///
/// A name with no prefix at all is in `default_element_namespace` — the host language's, not the
/// parameter document's, because the document's own default namespace is the serialization
/// vocabulary and naming an element of *that* is not what a caller means. `None` leaves such a
/// name in no namespace.
fn qname_list(
    element: &NodeRef,
    value: &str,
    parameter: &str,
    default_element_namespace: Option<&str>,
) -> FnResult<Vec<ExpandedName>> {
    let bindings = element_bindings(element);
    let mut names = Vec::new();
    // The value is a list, and a list separates on XML white space alone: any other character
    // Unicode calls white space stays inside its token, where it makes the token no `QName` and
    // so an error rather than a silent second name.
    for token in value
        .split(oxixml_text::ws::is_xml_ws)
        .filter(|token| !token.is_empty())
    {
        if let Some(rest) = token.strip_prefix("Q{") {
            let Some((namespace, local_name)) = rest.split_once('}') else {
                return Err(invalid_parameters(format!(
                    "`{token}` is not an EQName in the value of `{parameter}`"
                )));
            };
            if local_name.is_empty() {
                return Err(invalid_parameters(format!(
                    "`{token}` is not an EQName in the value of `{parameter}`"
                )));
            }
            names.push(ExpandedName::new(
                (!namespace.is_empty()).then_some(namespace),
                local_name,
            ));
            continue;
        }
        let (prefix, name) = match token.split_once(':') {
            Some((prefix, name)) => (Some(prefix), name),
            None => (None, token),
        };
        if name.is_empty() {
            return Err(invalid_parameters(format!(
                "`{token}` is not a QName in the value of `{parameter}`"
            )));
        }
        let namespace = match prefix {
            // An unprefixed name is in the host language's default element namespace, which is
            // nothing a parameter document can carry — the caller supplies it.
            None => default_element_namespace.map(str::to_owned),
            Some(prefix) => {
                let Some((_, namespace)) = bindings
                    .iter()
                    .find(|(bound, _)| bound.as_deref() == Some(prefix))
                else {
                    return Err(invalid_parameters(format!(
                        "the prefix `{prefix}` in the value of `{parameter}` is not bound"
                    )));
                };
                Some(namespace.clone())
            }
        };
        names.push(ExpandedName::new(namespace, name));
    }
    Ok(names)
}

fn character_maps(
    parameters: SerializationParameters,
    element: &NodeRef,
) -> FnResult<SerializationParameters> {
    if !element.attributes().is_empty() {
        return Err(invalid_parameters(
            "output:use-character-maps takes no attributes",
        ));
    }
    let mut mappings: Vec<(char, String)> = Vec::new();
    for child in element.children() {
        if child.node_kind() != NodeKind::Element {
            continue;
        }
        if !is_serialization_element(&child, "character-map") {
            return Err(invalid_parameters(
                "output:use-character-maps may only contain output:character-map",
            ));
        }
        let mut character = None;
        let mut mapped = None;
        for attribute in child.attributes() {
            let Some(name) = attribute.node_name() else {
                continue;
            };
            if name.namespace().is_some() {
                continue;
            }
            match name.local_name() {
                "character" => character = Some(attribute.string_value()),
                "map-string" => mapped = Some(attribute.string_value()),
                other => {
                    return Err(invalid_parameters(format!(
                        "`{other}` is not an attribute of output:character-map"
                    )));
                }
            }
        }
        let (Some(character), Some(mapped)) = (character, mapped) else {
            return Err(invalid_parameters(
                "output:character-map needs both `character` and `map-string`",
            ));
        };
        let mut characters = character.chars();
        let (Some(single), None) = (characters.next(), characters.next()) else {
            return Err(invalid_parameters(
                "the `character` attribute must be exactly one character",
            ));
        };
        if mappings.iter().any(|(seen, _)| *seen == single) {
            return Err(duplicate_character_mapping(single));
        }
        mappings.push((single, mapped));
    }
    Ok(parameters.with_character_map(mappings))
}

/// Reads a serialization parameter whose value is a set of element names from `fn:serialize`'s
/// *option map* form — `cdata-section-elements` and `suppress-indentation`.
///
/// [Serialization 3.1 §3.2](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparams-in-map)
/// types the entry as `xs:QName*`; the option parameter conventions of F&O 3.1 §2.1.1 also let
/// an array stand in for the sequence, so `[QName('','a')]` and `QName('','a')` mean the same
/// thing (QT3 `serialize-xml-106`, `-106a`, `-110`).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when an item of the value is not an `xs:QName`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::qnames_from_option;
/// use oxixml_infoset::{AtomicValue, QNameValue, Sequence};
///
/// let value = Sequence::from(AtomicValue::qname(QNameValue::new(
///     None::<&str>,
///     Some("urn:example"),
///     "b",
/// )));
/// let names = qnames_from_option(&value, "cdata-section-elements")?;
/// assert_eq!(names.len(), 1);
/// assert_eq!(names[0].namespace(), Some("urn:example"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn qnames_from_option(value: &Sequence, parameter: &str) -> FnResult<Vec<ExpandedName>> {
    let mut names = Vec::new();
    let mut collect = |item: &Item| -> FnResult<()> {
        match item {
            Item::Atomic(atomic) => match atomic.as_qname() {
                Some(qname) => {
                    names.push(qname.expanded_name());
                    Ok(())
                }
                None => Err(FnError::new(
                    FnErrorCode::Xpty0004,
                    format!("the serialization parameter `{parameter}` needs xs:QName values"),
                )),
            },
            _ => Err(FnError::new(
                FnErrorCode::Xpty0004,
                format!("the serialization parameter `{parameter}` needs xs:QName values"),
            )),
        }
    };
    for item in value.iter() {
        match item {
            Item::Array(array) => {
                for member in array.members() {
                    for inner in member.iter() {
                        collect(inner)?;
                    }
                }
            }
            other => collect(other)?,
        }
    }
    Ok(names)
}

fn is_serialization_element(element: &NodeRef, local: &str) -> bool {
    element.node_kind() == NodeKind::Element
        && element.node_name().is_some_and(|name| {
            name.namespace() == Some(SERIALIZATION_NAMESPACE) && name.local_name() == local
        })
}

/// Rejects any attribute in no namespace other than `value`, which is `SEPM0017`.
fn reject_foreign_attributes(element: &NodeRef) -> FnResult<()> {
    for attribute in element.attributes() {
        let Some(name) = attribute.node_name() else {
            continue;
        };
        if name.namespace().is_none() && name.local_name() != "value" {
            return Err(invalid_parameters(format!(
                "`{}` is not an attribute of a serialization parameter",
                name.local_name()
            )));
        }
    }
    Ok(())
}

/// Any fault in the `output:serialization-parameters` element: an attribute that does not
/// belong, a parameter no serializer defines, a missing `value` — or a `value` outside the
/// domain the parameter defines.
///
/// [Serialization 3.1 §3.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparams-in-xdm)
/// grades the *parameter document* as a whole: everything that makes it not a valid one is
/// `SEPM0017`, including a bad value, which is why `indent="maybe"` is `SEPM0017` here and
/// `SEPM0016` when the same parameter arrives in an option map (`serialize-xml-009`/`-009b`
/// against `serialize-xml-123`). Two spellings, two vocabularies, two codes.
fn invalid_parameters(what: impl Into<String>) -> FnError {
    FnError::new(FnErrorCode::Sepm0017, what.into())
}

/// `SEPM0018` — two `output:character-map` children of one `output:use-character-maps` name the
/// same character, so the mapping to apply is ambiguous.
///
/// This is a narrower rule than the `SEPM0019` of a parameter given twice, and Serialization 3.1
/// gives it its own code.
fn duplicate_character_mapping(character: char) -> FnError {
    FnError::new(
        FnErrorCode::Sepm0018,
        format!("the character `{character}` is mapped more than once"),
    )
}

#[cfg(test)]
mod tests {
    use super::super::serialize;
    use super::*;
    use oxixml_infoset::{AtomicValue, Tree};

    fn document(xml: &str) -> Result<Sequence, Box<dyn std::error::Error>> {
        let tree = Tree::parse_str(xml)?;
        Ok(Sequence::from(
            tree.document_element().ok_or("no document element")?,
        ))
    }

    fn parameters_element(body: &str) -> Result<NodeRef, Box<dyn std::error::Error>> {
        let xml = format!(
            "<serialization-parameters xmlns=\"{SERIALIZATION_NAMESPACE}\">{body}</serialization-parameters>"
        );
        let tree = Tree::parse_str(&xml)?;
        Ok(tree.document_element().ok_or("no document element")?)
    }

    #[test]
    fn output_methods_are_parsed() -> FnResult<()> {
        assert_eq!(OutputMethod::parse("xml")?, OutputMethod::Xml);
        assert_eq!(OutputMethod::parse("")?, OutputMethod::Xml);
        assert_eq!(OutputMethod::parse("text")?, OutputMethod::Text);
        assert_eq!(OutputMethod::parse("adaptive")?, OutputMethod::Adaptive);
        assert_eq!(OutputMethod::parse("json")?, OutputMethod::Json);
        assert_eq!(OutputMethod::parse("html")?, OutputMethod::Html);
        assert_eq!(OutputMethod::parse("xhtml")?, OutputMethod::Xhtml);
        // The `EQName` spelling of a name in no namespace selects the same method; one in a
        // namespace would select an implementation-defined method, of which there are none.
        assert_eq!(OutputMethod::parse("Q{}xml")?, OutputMethod::Xml);
        assert_eq!(OutputMethod::parse("Q{}html")?, OutputMethod::Html);
        assert_eq!(OutputMethod::parse("Q{}xhtml")?, OutputMethod::Xhtml);
        assert!(OutputMethod::parse("Q{urn:example}mine").is_err());
        assert!(OutputMethod::parse("Q{urn:example").is_err());
        Ok(())
    }

    /// Serialization 3.1 §3.1 grades the parameter *document*: whatever makes it not a valid one
    /// is `SEPM0017`, a bad `value` included. `SEPM0016` belongs to the option-map spelling
    /// instead (QT3 `serialize-xml-009`/`-009b`, and `-014`/`-015`/`-016`/`-020` for the
    /// structural faults).
    #[test]
    fn every_fault_in_a_parameter_document_is_sepm0017() -> Result<(), Box<dyn std::error::Error>> {
        for body in [
            // Inadmissible values.
            r#"<method value="bogus"/>"#,
            r#"<json-node-output-method value="bogus"/>"#,
            r#"<indent value="maybe"/>"#,
            r#"<omit-xml-declaration value="perhaps"/>"#,
            r#"<standalone value="perhaps"/>"#,
            r#"<allow-duplicate-names value="perhaps"/>"#,
            r#"<byte-order-mark value="perhaps"/>"#,
            r#"<undeclare-prefixes value="perhaps"/>"#,
            // Structural faults.
            r#"<nonesuch value="1"/>"#,
            r#"<indent/>"#,
            r#"<indent value="yes" extra="1"/>"#,
            r#"<cdata-section-elements value="nosuch:a"/>"#,
        ] {
            let element = parameters_element(body)?;
            let error = parameters_from_element(&element).expect_err(body);
            assert!(error.is(FnErrorCode::Sepm0017), "{body} gave {error}");
        }
        let repeated = parameters_element(r#"<indent value="yes"/><indent value="no"/>"#)?;
        assert!(
            parameters_from_element(&repeated)
                .expect_err("a repeated parameter")
                .is(FnErrorCode::Sepm0019)
        );
        Ok(())
    }

    /// A node that is not an `output:serialization-parameters` element was never a parameter
    /// document, so it is the argument that is wrongly typed — QT3 `serialize-xml-017b` (right
    /// local name, wrong namespace) and `-018b` (right namespace, wrong local name).
    #[test]
    fn a_foreign_parameters_element_is_a_type_error() -> Result<(), Box<dyn std::error::Error>> {
        for xml in [
            r#"<serialization-parameters xmlns="http://www.w3.org/2010/xslt-xquery-serialization/wrong"><indent value="yes"/></serialization-parameters>"#,
            r#"<serialization-parameters-wrong xmlns="http://www.w3.org/2010/xslt-xquery-serialization"><indent value="yes"/></serialization-parameters-wrong>"#,
        ] {
            let tree = Tree::parse_str(xml)?;
            let element = tree.document_element().ok_or("no document element")?;
            let error = parameters_from_element(&element).expect_err(xml);
            assert!(error.is(FnErrorCode::Xpty0004), "{xml} gave {error}");
        }
        Ok(())
    }

    /// Two `output:character-map` children naming the same character have their own code,
    /// `SEPM0018` — narrower than the `SEPM0019` of a repeated parameter (QT3
    /// `serialize-xml-024`/`-024b`).
    #[test]
    fn a_character_mapped_twice_is_sepm0018() -> Result<(), Box<dyn std::error::Error>> {
        let element = parameters_element(
            r#"<use-character-maps><character-map character="$" map-string="&#xA3;"/><character-map character="$" map-string="&#x20AC;"/></use-character-maps>"#,
        )?;
        let error = parameters_from_element(&element).expect_err("a character mapped twice");
        assert!(error.is(FnErrorCode::Sepm0018));
        assert_eq!(error.name().local_name(), "SEPM0018");
        assert_eq!(
            error.name().namespace(),
            Some(crate::error::ERROR_NAMESPACE)
        );
        Ok(())
    }

    /// `cdata-section-elements` wraps the text-node children of the named elements, and only
    /// those, whichever spelling named them (QT3 `serialize-xml-110`, `-006`).
    #[test]
    fn cdata_section_elements_wrap_their_text() -> Result<(), Box<dyn std::error::Error>> {
        let value = document(
            r#"<chapter xmlns:p="http://www.example.org/ns/p"><section><para><b>bold</b><i>italic</i><p:b>BOLD</p:b><p:i>ITALIC</p:i></para></section></chapter>"#,
        )?;
        let parameters = SerializationParameters::default().with_cdata_section_elements(vec![
            ExpandedName::new(None::<&str>, "b"),
            ExpandedName::new(Some("http://www.example.org/ns/p"), "b"),
        ]);
        let written = serialize(&value, &parameters)?;
        assert!(written.contains("CDATA[bold]"), "{written}");
        assert!(written.contains("CDATA[BOLD]"), "{written}");
        assert!(!written.contains("CDATA[italic]"), "{written}");
        assert!(!written.contains("CDATA[ITALIC]"), "{written}");

        // The same names read off a parameter document, where a prefix is resolved against the
        // in-scope namespaces of the element that carries it.
        let element = parameters_element(
            r#"<cdata-section-elements xmlns:q="http://www.example.org/ns/p" value="b q:b"/>"#,
        )?;
        assert_eq!(
            serialize(&value, &parameters_from_element(&element)?)?,
            written
        );
        Ok(())
    }

    /// An unprefixed name in `cdata-section-elements` or `suppress-indentation` is in the *host
    /// language's* default element namespace, which only the host can supply (QT3
    /// `method-xml/K2-Serialization-28`, `-31`, against `-26`, `-27`, `-29`, `-30`, `-32`, whose
    /// modules declare none).
    #[test]
    fn an_unprefixed_name_takes_the_default_element_namespace()
    -> Result<(), Box<dyn std::error::Error>> {
        // `K2-Serialization-31`'s own document and parameters: the `b` that must be wrapped is
        // the one in `http://www.example.org/ns/p`, not the one in the document's own namespace.
        let value = document(
            r#"<chapter xmlns="http://www.example.com/ns/c"><section><para><b>bold</b><i>italic</i></para><para xmlns:p="http://www.example.org/ns/p"><p:b>BOLD</p:b><p:i>ITALIC</p:i></para></section></chapter>"#,
        )?;
        let element = parameters_element(r#"<cdata-section-elements value="b"/>"#)?;
        let defaults = SerializationParameters::default();

        let in_context = parameters_from_element_in_context(
            &element,
            &defaults,
            Some("http://www.example.org/ns/p"),
        )?;
        assert_eq!(
            in_context.cdata_section_elements(),
            [ExpandedName::new(Some("http://www.example.org/ns/p"), "b")]
        );
        let written = serialize(&value, &in_context)?;
        assert!(written.contains("CDATA[BOLD]"), "{written}");
        assert!(!written.contains("CDATA[bold]"), "{written}");
        assert!(!written.contains("CDATA[ITALIC]"), "{written}");

        // With no default element namespace the name is in no namespace, which is what the two
        // entry points that do not take one pass — and neither `b` is then wrapped.
        for parameters in [
            parameters_from_element_in_context(&element, &defaults, None)?,
            parameters_from_element_with_defaults(&element, &defaults)?,
            parameters_from_element(&element)?,
        ] {
            assert_eq!(
                parameters.cdata_section_elements(),
                [ExpandedName::new(None::<&str>, "b")]
            );
            let written = serialize(&value, &parameters)?;
            assert!(!written.contains("CDATA["), "{written}");
        }

        // A prefixed name and an `EQName` carry their own namespace and are untouched by it.
        let spelled = parameters_element(
            r#"<cdata-section-elements xmlns:q="http://www.example.org/ns/p" value="q:b Q{http://www.example.com/ns/c}i"/>"#,
        )?;
        assert_eq!(
            parameters_from_element_in_context(&spelled, &defaults, Some("urn:example"))?
                .cdata_section_elements(),
            [
                ExpandedName::new(Some("http://www.example.org/ns/p"), "b"),
                ExpandedName::new(Some("http://www.example.com/ns/c"), "i"),
            ]
        );
        Ok(())
    }

    /// A name-set parameter also takes XQuery 3.0's `EQName` spelling, which carries its
    /// namespace with it and needs no prefix binding (QT3 `method-html/Serialization-html-16a`,
    /// `method-xml/K2-Serialization-32`).
    #[test]
    fn a_name_set_parameter_takes_eqnames() -> Result<(), Box<dyn std::error::Error>> {
        let element = parameters_element(
            r#"<cdata-section-elements value="Q{http://www.example.org/ns/p}b Q{}c d"/>"#,
        )?;
        let parameters = parameters_from_element(&element)?;
        assert_eq!(
            parameters.cdata_section_elements(),
            [
                ExpandedName::new(Some("http://www.example.org/ns/p"), "b"),
                ExpandedName::new(None::<&str>, "c"),
                ExpandedName::new(None::<&str>, "d"),
            ]
        );
        let suppressed = parameters_element(
            r#"<suppress-indentation value="Q{http://www.w3.org/1999/xhtml}td"/>"#,
        )?;
        assert_eq!(
            parameters_from_element(&suppressed)?.suppress_indentation(),
            [ExpandedName::new(
                Some("http://www.w3.org/1999/xhtml"),
                "td"
            )]
        );
        // An `EQName` with no local part is as invalid as any other bad value.
        for body in [
            r#"<cdata-section-elements value="Q{urn:x}"/>"#,
            r#"<cdata-section-elements value="Q{urn:x"/>"#,
        ] {
            let element = parameters_element(body)?;
            assert!(
                parameters_from_element(&element)
                    .expect_err(body)
                    .is(FnErrorCode::Sepm0017)
            );
        }
        Ok(())
    }

    /// The option-map spelling of a name-set parameter takes a sequence or an array of
    /// `xs:QName` (QT3 `serialize-xml-106`, `-106a`, `-110`).
    #[test]
    fn name_set_options_accept_a_sequence_or_an_array() -> FnResult<()> {
        let name = |namespace: Option<&str>, local: &str| {
            AtomicValue::qname(oxixml_infoset::QNameValue::new(
                None::<&str>,
                namespace,
                local,
            ))
        };
        let sequence = Sequence::from_iter([name(None, "a"), name(Some("urn:example"), "b")]);
        let names = qnames_from_option(&sequence, "cdata-section-elements")?;
        assert_eq!(
            names,
            vec![
                ExpandedName::new(None::<&str>, "a"),
                ExpandedName::new(Some("urn:example"), "b"),
            ]
        );

        let array = Sequence::singleton(Item::from(oxixml_infoset::XdmArray::from_iter([
            name(None, "a"),
            name(Some("urn:example"), "b"),
        ])));
        assert_eq!(qnames_from_option(&array, "cdata-section-elements")?, names);

        let wrong = Sequence::from(AtomicValue::string("a"));
        assert!(
            qnames_from_option(&wrong, "cdata-section-elements")
                .expect_err("a string is not a QName")
                .is(FnErrorCode::Xpty0004)
        );
        Ok(())
    }

    /// Serialization 3.1 §7: `html-version` decides the HTML version, and in its absence the
    /// `version` parameter does — which is the spelling QT3's `ser/method-html.xml` uses
    /// throughout (`declare option output:version "4.0"`). The `version` of `1.0` the `xml`
    /// method carries by default must not be read as an HTML version.
    #[test]
    fn the_html_version_falls_back_to_the_version_parameter() {
        let defaults = SerializationParameters::default();
        assert_eq!(defaults.effective_html_version(), "5");
        assert!(defaults.is_html5());

        let four = SerializationParameters::default().with_version("4.0");
        assert_eq!(four.effective_html_version(), "4.0");
        assert!(!four.is_html5());

        // `html-version` wins over `version` when both are set.
        let both = SerializationParameters::default()
            .with_version("4.0")
            .with_html_version("5.0");
        assert!(both.is_html5());

        // An unparsable version is not HTML 5.
        assert!(
            !SerializationParameters::default()
                .with_html_version("nonesuch")
                .is_html5()
        );
    }
}
