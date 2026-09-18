//! Turning a [`SimpleGrammar`](crate::flatten::SimpleGrammar) into the compiled
//! [`Schema`].
//!
//! By the time this runs the schema matches the grammar of
//! [§5](https://relaxng.org/spec-20011203.html#simple-syntax) exactly, so the
//! translation is a direct structural mapping — the only work left is resolving
//! `ref` names to definition indices, compiling the datatypes with their
//! parameters, and capturing the namespace context that `value` patterns need.

use std::collections::HashMap;
use std::rc::Rc;

use crate::datatype::{Datatype, DatatypeContext, params_of};
use crate::error::{SchemaError, datatype as datatype_error};
use crate::flatten::{SimpleGrammar, malformed};
use crate::pattern::{
    Arena, ContextId, DatatypeId, DefineId, EMPTY, NOT_ALLOWED, NameClass, NameClassId,
    NameClasses, PatId, Pattern, TEXT,
};
use crate::tree::Element;

/// A compiled RELAX NG schema, ready to validate against.
#[derive(Debug)]
pub(crate) struct Schema {
    /// The interned patterns.
    pub arena: Arena,
    /// The interned name classes.
    pub name_classes: NameClasses,
    /// The compiled datatypes.
    pub datatypes: Vec<Datatype>,
    /// The contexts that `value` patterns are interpreted in.
    pub contexts: Vec<DatatypeContext>,
    /// The definitions; every entry is an [`Pattern::Element`].
    pub defines: Vec<PatId>,
    /// The `start` pattern.
    pub start: PatId,
}

impl Schema {
    /// Resolves a definition to the `element` pattern it holds.
    pub(crate) fn define(&self, id: DefineId) -> PatId {
        self.defines
            .get(id as usize)
            .copied()
            .unwrap_or(NOT_ALLOWED)
    }
}

/// Compiles a simplified grammar.
pub(crate) fn compile(grammar: &SimpleGrammar) -> Result<Schema, SchemaError> {
    let mut indices = HashMap::new();
    for (index, (name, _)) in grammar.defines.iter().enumerate() {
        indices.insert(name.clone(), u32::try_from(index).unwrap_or(u32::MAX));
    }
    let mut compiler = Compiler {
        arena: Arena::default(),
        name_classes: NameClasses::default(),
        datatypes: Vec::new(),
        contexts: Vec::new(),
        indices,
    };

    let start = compiler.pattern(&grammar.start)?;
    let mut defines = Vec::with_capacity(grammar.defines.len());
    for (_, body) in &grammar.defines {
        defines.push(compiler.pattern(body)?);
    }

    Ok(Schema {
        arena: compiler.arena,
        name_classes: compiler.name_classes,
        datatypes: compiler.datatypes,
        contexts: compiler.contexts,
        defines,
        start,
    })
}

struct Compiler {
    arena: Arena,
    name_classes: NameClasses,
    datatypes: Vec<Datatype>,
    contexts: Vec<DatatypeContext>,
    indices: HashMap<String, DefineId>,
}

impl Compiler {
    fn pattern(&mut self, element: &Element) -> Result<PatId, SchemaError> {
        match element.name.as_str() {
            "empty" => Ok(EMPTY),
            "notAllowed" => Ok(NOT_ALLOWED),
            "text" => Ok(TEXT),
            "ref" => {
                let name = element.attribute("name").unwrap_or_default();
                let index = self
                    .indices
                    .get(name)
                    .copied()
                    .ok_or_else(|| malformed(element, "a resolvable ref"))?;
                Ok(self.arena.intern(Pattern::Ref(index)))
            }
            "choice" | "group" | "interleave" => {
                let (left, right) = self.two_children(element)?;
                Ok(match element.name.as_str() {
                    "choice" => self.arena.choice(left, right),
                    "group" => self.arena.group(left, right),
                    _ => self.arena.interleave(left, right),
                })
            }
            "oneOrMore" => {
                let inner = self.one_child(element)?;
                Ok(self.arena.one_or_more(inner))
            }
            "list" => {
                let inner = self.one_child(element)?;
                Ok(self.arena.intern(Pattern::List(inner)))
            }
            "attribute" | "element" => {
                let mut children = element.child_elements();
                let name_class = children
                    .next()
                    .ok_or_else(|| malformed(element, "a name class"))?;
                let body = children
                    .next()
                    .ok_or_else(|| malformed(element, "a content pattern"))?;
                let name_class = self.name_class(name_class)?;
                let body = self.pattern(body)?;
                Ok(if element.name == "attribute" {
                    self.arena.intern(Pattern::Attribute(name_class, body))
                } else {
                    self.arena.intern(Pattern::Element(name_class, body))
                })
            }
            "data" => {
                let datatype = self.datatype(element, &params_of(element))?;
                let except = element
                    .child_elements()
                    .find(|child| child.name == "except")
                    .map(|except| {
                        except
                            .child_elements()
                            .next()
                            .ok_or_else(|| malformed(except, "a pattern"))
                            .and_then(|inner| self.pattern(inner))
                    })
                    .transpose()?;
                Ok(self.arena.intern(Pattern::Data { datatype, except }))
            }
            "value" => {
                let datatype = self.datatype(element, &[])?;
                let namespace = element.attribute("ns").unwrap_or_default();
                let context = self.context(element, namespace);
                Ok(self.arena.intern(Pattern::Value {
                    datatype,
                    value: element.text(),
                    context,
                }))
            }
            _ => Err(malformed(element, "a pattern")),
        }
    }

    fn one_child(&mut self, element: &Element) -> Result<PatId, SchemaError> {
        let child = element
            .child_elements()
            .next()
            .ok_or_else(|| malformed(element, "one child pattern"))?;
        self.pattern(child)
    }

    fn two_children(&mut self, element: &Element) -> Result<(PatId, PatId), SchemaError> {
        let mut children = element.child_elements();
        let left = children
            .next()
            .ok_or_else(|| malformed(element, "two child patterns"))?;
        let right = children
            .next()
            .ok_or_else(|| malformed(element, "two child patterns"))?;
        let left = self.pattern(left)?;
        let right = self.pattern(right)?;
        Ok((left, right))
    }

    fn name_class(&mut self, element: &Element) -> Result<NameClassId, SchemaError> {
        let class = match element.name.as_str() {
            "name" => NameClass::Name(
                element.attribute("ns").unwrap_or_default().to_owned(),
                element.text(),
            ),
            "anyName" => match element.child_elements().next() {
                Some(except) => {
                    let inner = except
                        .child_elements()
                        .next()
                        .ok_or_else(|| malformed(except, "a name class"))?;
                    NameClass::AnyNameExcept(self.name_class(inner)?)
                }
                None => NameClass::AnyName,
            },
            "nsName" => {
                let namespace = element.attribute("ns").unwrap_or_default().to_owned();
                match element.child_elements().next() {
                    Some(except) => {
                        let inner = except
                            .child_elements()
                            .next()
                            .ok_or_else(|| malformed(except, "a name class"))?;
                        NameClass::NsNameExcept(namespace, self.name_class(inner)?)
                    }
                    None => NameClass::NsName(namespace),
                }
            }
            "choice" => {
                let mut children = element.child_elements();
                let left = children
                    .next()
                    .ok_or_else(|| malformed(element, "two name classes"))?;
                let right = children
                    .next()
                    .ok_or_else(|| malformed(element, "two name classes"))?;
                let left = self.name_class(left)?;
                let right = self.name_class(right)?;
                NameClass::Choice(left, right)
            }
            _ => return Err(malformed(element, "a name class")),
        };
        Ok(self.name_classes.intern(class))
    }

    fn datatype(
        &mut self,
        element: &Element,
        params: &[(String, String)],
    ) -> Result<DatatypeId, SchemaError> {
        let library = element.attribute("datatypeLibrary").unwrap_or_default();
        let name = element.attribute("type").unwrap_or_default();
        let context = DatatypeContext {
            namespaces: Rc::clone(&element.context.namespaces),
        };
        let datatype = Datatype::compile(library, name, params, &context)
            .map_err(|error| datatype_error("6.2.8", error.to_string()).at(element.position))?;
        let id = u32::try_from(self.datatypes.len()).unwrap_or(u32::MAX);
        self.datatypes.push(datatype);
        Ok(id)
    }

    fn context(&mut self, element: &Element, namespace: &str) -> ContextId {
        let context = DatatypeContext {
            namespaces: Rc::clone(&element.context.namespaces),
        }
        .with_default_namespace(namespace);
        let id = u32::try_from(self.contexts.len()).unwrap_or(u32::MAX);
        self.contexts.push(context);
        id
    }
}
