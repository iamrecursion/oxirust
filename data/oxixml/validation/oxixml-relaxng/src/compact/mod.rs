//! The [RELAX NG compact syntax][rnc].
//!
//! The compact syntax is *not* a separate schema language: the specification
//! defines it purely as a translation into the RELAX NG data model, production by
//! production. This module implements that translation, so a compact-syntax
//! schema goes through exactly the same simplification, restriction checking and
//! compilation as one written in XML.
//!
//! [rnc]: https://relaxng.org/compact-20021121.html

mod lex;
mod parse;

use crate::error::SchemaError;
use crate::tree::Element;

/// Parses a compact-syntax schema into the RELAX NG data model.
///
/// `base` is the base URI of the schema, used to resolve the `href` of `include`
/// and `external`.
pub(crate) fn parse(source: &str, base: Option<&str>) -> Result<Element, SchemaError> {
    parse::parse(source, base)
}

#[cfg(test)]
mod tests {
    use crate::RelaxNg;
    use crate::error::SchemaError;

    #[test]
    fn elements_attributes_and_repetition() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"element book {
                 attribute isbn { text },
                 element title { text },
                 element author { text }*
               }"#,
        )?;
        assert!(
            schema
                .validate(
                    r#"<book isbn="1"><title>t</title><author>a</author><author>b</author></book>"#
                )
                .is_ok()
        );
        assert!(schema.validate(r#"<book isbn="1"/>"#).is_err());
        Ok(())
    }

    #[test]
    fn namespaces_and_default_namespace() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"default namespace = "urn:example"
               namespace other = "urn:other"
               element root { element other:child { empty } }"#,
        )?;
        assert!(
            schema
                .validate(r#"<root xmlns="urn:example"><child xmlns="urn:other"/></root>"#)
                .is_ok()
        );
        assert!(schema.validate("<root><child/></root>").is_err());
        Ok(())
    }

    #[test]
    fn grammars_with_combine() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"start = element r { content }
               content |= element a { empty }
               content |= element b { empty }"#,
        )?;
        assert!(schema.validate("<r><a/></r>").is_ok());
        assert!(schema.validate("<r><b/></r>").is_ok());
        assert!(schema.validate("<r><c/></r>").is_err());
        Ok(())
    }

    #[test]
    fn name_classes() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"namespace ex = "urn:example"
               element root { element * - ex:* { empty }* }"#,
        )?;
        assert!(schema.validate("<root><anything/></root>").is_ok());
        assert!(
            schema
                .validate(r#"<root><x xmlns="urn:example"/></root>"#)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn datatypes_with_parameters() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"element n { xsd:int { minInclusive = "1" maxInclusive = "10" } }"#,
        )?;
        assert!(schema.validate("<n>5</n>").is_ok());
        assert!(schema.validate("<n>50</n>").is_err());
        Ok(())
    }

    #[test]
    fn data_except_and_values() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(r#"element n { xsd:token - "forbidden" }"#)?;
        assert!(schema.validate("<n>allowed</n>").is_ok());
        assert!(schema.validate("<n>forbidden</n>").is_err());
        Ok(())
    }

    #[test]
    fn list_and_mixed() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"element root { element nums { list { xsd:int+ } }, element p { mixed { element b { text }* } } }"#,
        )?;
        assert!(
            schema
                .validate("<root><nums>1 2 3</nums><p>a<b>c</b>d</p></root>")
                .is_ok()
        );
        assert!(
            schema
                .validate("<root><nums>1 x 3</nums><p/></root>")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn annotations_are_ignored() -> Result<(), SchemaError> {
        let schema = RelaxNg::from_compact(
            r#"namespace a = "urn:annotation"
               ## documentation
               [ a:note = "hello" ]
               start = element root { empty } >> a:after [ "text" ]"#,
        )?;
        assert!(schema.validate("<root/>").is_ok());
        Ok(())
    }

    #[test]
    fn annotation_constraints_are_enforced() {
        // A top-level pattern must translate to a single element.
        assert!(RelaxNg::from_compact("element root { empty } >> x []").is_err());
        // An annotation attribute must be namespace-qualified and unique.
        assert!(RelaxNg::from_compact("[ note = \"x\" ] element root { empty }").is_err());
        assert!(
            RelaxNg::from_compact(
                "namespace a = \"urn:a\"\n[ a:n = \"x\" a:n = \"y\" ] element root { empty }"
            )
            .is_err()
        );
        // An annotation must not use the RELAX NG namespace.
        assert!(
            RelaxNg::from_compact(
                "namespace rng = \"http://relaxng.org/ns/structure/1.0\"\n\
                 [ rng:foo = \"x\" ] element root { empty }"
            )
            .is_err()
        );
    }

    #[test]
    fn an_escaped_carriage_return_is_not_a_separator() {
        // `#xD` is not in the `separator` production, so an escape that produces
        // one cannot stand between two tokens.
        assert!(RelaxNg::from_compact("element \\x{D}foo { empty }").is_err());
        // …but a `#xA` produced by an escape is a character inside a literal.
        assert!(RelaxNg::from_compact("element foo { \"z\\x{A}\" }").is_ok());
    }

    #[test]
    fn syntax_errors_are_reported() {
        assert!(RelaxNg::from_compact("element { empty }").is_err());
        assert!(RelaxNg::from_compact("element a { b | c, d }").is_err());
        assert!(RelaxNg::from_compact("datatypes xsd = \"urn:wrong\"").is_err());
        assert!(RelaxNg::from_compact("namespace xmlns = \"urn:x\"").is_err());
    }
}
