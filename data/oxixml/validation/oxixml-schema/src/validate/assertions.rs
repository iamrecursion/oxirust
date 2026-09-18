//! Assessment-time evaluation of XSD 1.1 assertions and conditional type
//! assignment.
//!
//! [`Validator::conditional_type`] implements §3.3.4 *cvc-type*'s appeal to the
//! element declaration's {type table}: the alternatives are tried in order and
//! the first whose `test` holds selects the governing type. A default
//! alternative — one written without a `test` — always holds, and when nothing
//! holds the *declared* type governs, as §3.3.2 says.
//!
//! [`Validator::check_assertions`] implements *cvc-assertion* (§3.4.4.2): every
//! assertion of the governing complex type must hold of the element, and the
//! first that does not is reported. The assertions of a derived type include
//! its base's, which the compiler has already merged in.

use oxixml_dom::NodeId;

use crate::components::{ElementId, TypeId};
use crate::xdm::{Shape, XdmView};

use super::Validator;

impl Validator<'_> {
    /// The XDM view of `node` for an assertion or a type alternative.
    fn view(&self) -> XdmView<'_> {
        XdmView {
            doc: self.doc,
            schema: self.schema,
            elements: &self.elements,
            attributes: &self.attributes,
        }
    }

    /// The governing type of `node` under conditional type assignment.
    pub(crate) fn conditional_type(
        &mut self,
        node: NodeId,
        declaration: ElementId,
        declared: TypeId,
    ) -> TypeId {
        let alternatives = self
            .schema
            .element_declaration(declaration)
            .alternatives
            .clone();
        // §3.3.2: the expression sees the element with its attributes and no
        // children, and nothing in it is typed — assessment has not happened
        // yet.
        let tree = self.view().build(node, Shape::Alternative);
        for alternative in &alternatives {
            match &alternative.test {
                None => return alternative.type_id,
                Some(test) => {
                    let Some(tree) = &tree else { continue };
                    if crate::assertion::evaluate(test, tree, None, false) {
                        return alternative.type_id;
                    }
                }
            }
        }
        declared
    }

    /// Checks every `xs:assert` of the governing type (*cvc-assertion*).
    pub(crate) fn check_assertions(&mut self, node: NodeId, type_id: TypeId, path: &str) {
        let Some(complex) = self.schema.complex_type(type_id) else {
            return;
        };
        if complex.assertions.is_empty() {
            return;
        }
        let assertions = complex.assertions.clone();
        let complex_content = match &complex.content {
            crate::components::ContentType::Simple(inner) => Some(*inner),
            _ => None,
        };
        let Some(tree) = self.view().build(node, Shape::Assertion) else {
            self.error(
                "cvc-assertion",
                path,
                "the element cannot be viewed as an XDM instance, so its assertions cannot be \
                 evaluated",
            );
            return;
        };
        // §3.13.1: `$value` is in scope in an `xs:assert` whose type has
        // simple content, bound to the element's typed value.
        let value = self
            .elements
            .get(&node)
            .and_then(|psvi| psvi.normalized_value.clone())
            .and_then(|lexical| match &complex_content {
                Some(inner) => crate::xdm::typed_value(self.schema, *inner, &lexical),
                None => None,
            });
        for assertion in &assertions {
            if !crate::assertion::evaluate(assertion, &tree, value.clone(), true) {
                self.error(
                    "cvc-assertion",
                    path,
                    format!("the assertion `{}` is not satisfied", assertion.test),
                );
                // One report per element is enough: the remaining assertions
                // would only repeat that the element is invalid.
                return;
            }
        }
    }
}
