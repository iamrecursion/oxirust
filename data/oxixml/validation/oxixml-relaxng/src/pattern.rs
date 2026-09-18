//! The compiled representation of a schema in the simple syntax: a hash-consed
//! pattern arena plus a name-class table.
//!
//! Every pattern is interned, so structurally identical patterns share one
//! identifier. That matters for the derivative algorithm in two ways: the
//! `choice p p = p` simplification becomes a cheap identifier comparison, and the
//! set of patterns reachable by differentiation stays finite, which is what makes
//! validation run in bounded memory.

use std::collections::HashMap;

/// An index into the pattern arena.
pub(crate) type PatId = u32;

/// An index into the name-class table.
pub(crate) type NameClassId = u32;

/// An index into the datatype table.
pub(crate) type DatatypeId = u32;

/// An index into the datatype-context table.
pub(crate) type ContextId = u32;

/// An index into the list of definitions.
pub(crate) type DefineId = u32;

/// A name class of [§7](https://relaxng.org/spec-20011203.html#IDA3EAS), in the
/// simple syntax where `except` and `choice` are binary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum NameClass {
    /// `<anyName/>`.
    AnyName,
    /// `<anyName><except> nc </except></anyName>`.
    AnyNameExcept(NameClassId),
    /// `<nsName ns="u"/>`.
    NsName(String),
    /// `<nsName ns="u"><except> nc </except></nsName>`.
    NsNameExcept(String, NameClassId),
    /// `<name ns="u"> ln </name>`; an empty namespace means "no namespace".
    Name(String, String),
    /// `<choice> nc1 nc2 </choice>`.
    Choice(NameClassId, NameClassId),
}

/// A pattern in the simple syntax, plus the `After` pattern that only the
/// derivative algorithm creates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Pattern {
    /// Matches the empty sequence.
    Empty,
    /// Matches nothing at all.
    NotAllowed,
    /// Matches any string.
    Text,
    /// `<choice> p1 p2 </choice>`.
    Choice(PatId, PatId),
    /// `<interleave> p1 p2 </interleave>`.
    Interleave(PatId, PatId),
    /// `<group> p1 p2 </group>`.
    Group(PatId, PatId),
    /// `<oneOrMore> p </oneOrMore>`.
    OneOrMore(PatId),
    /// `<list> p </list>`.
    List(PatId),
    /// `<data type="…"> params [except] </data>`.
    Data {
        /// The compiled datatype, with its parameters.
        datatype: DatatypeId,
        /// The pattern of the `except` child, when there is one.
        except: Option<PatId>,
    },
    /// `<value type="…" ns="…"> s </value>`.
    Value {
        /// The compiled (parameterless) datatype.
        datatype: DatatypeId,
        /// The literal string.
        value: String,
        /// The context the literal is interpreted in.
        context: ContextId,
    },
    /// `<attribute> nc p </attribute>`.
    Attribute(NameClassId, PatId),
    /// `<element> nc p </element>`; only ever the body of a definition.
    Element(NameClassId, PatId),
    /// `<ref name="…"/>`, resolved to a definition index.
    Ref(DefineId),
    /// The `after` pattern of the derivative algorithm: match `p1`, then, once
    /// the enclosing element ends, continue with `p2`.
    After(PatId, PatId),
}

/// The interning pattern arena.
///
/// Identifiers 0, 1 and 2 are always [`Pattern::Empty`], [`Pattern::NotAllowed`]
/// and [`Pattern::Text`].
#[derive(Debug, Clone)]
pub(crate) struct Arena {
    patterns: Vec<Pattern>,
    nullable: Vec<bool>,
    intern: HashMap<Pattern, PatId>,
}

/// The identifier of [`Pattern::Empty`].
pub(crate) const EMPTY: PatId = 0;
/// The identifier of [`Pattern::NotAllowed`].
pub(crate) const NOT_ALLOWED: PatId = 1;
/// The identifier of [`Pattern::Text`].
pub(crate) const TEXT: PatId = 2;

impl Default for Arena {
    fn default() -> Self {
        let mut arena = Self {
            patterns: Vec::new(),
            nullable: Vec::new(),
            intern: HashMap::new(),
        };
        let _ = arena.intern(Pattern::Empty);
        let _ = arena.intern(Pattern::NotAllowed);
        let _ = arena.intern(Pattern::Text);
        arena
    }
}

impl Arena {
    /// Returns the pattern behind an identifier.
    pub(crate) fn get(&self, id: PatId) -> &Pattern {
        self.patterns
            .get(id as usize)
            .unwrap_or(&Pattern::NotAllowed)
    }

    /// Returns whether the pattern matches the empty sequence.
    pub(crate) fn nullable(&self, id: PatId) -> bool {
        self.nullable.get(id as usize).copied().unwrap_or(false)
    }

    /// Interns a pattern, computing its nullability once.
    pub(crate) fn intern(&mut self, pattern: Pattern) -> PatId {
        if let Some(existing) = self.intern.get(&pattern) {
            return *existing;
        }
        let nullable = match &pattern {
            Pattern::Empty | Pattern::Text => true,
            Pattern::Choice(left, right) => self.nullable(*left) || self.nullable(*right),
            Pattern::Group(left, right) | Pattern::Interleave(left, right) => {
                self.nullable(*left) && self.nullable(*right)
            }
            Pattern::OneOrMore(inner) => self.nullable(*inner),
            _ => false,
        };
        let id = u32::try_from(self.patterns.len()).unwrap_or(u32::MAX);
        self.patterns.push(pattern.clone());
        self.nullable.push(nullable);
        self.intern.insert(pattern, id);
        id
    }

    /// The `choice` constructor, with the simplifications of Clark's algorithm.
    pub(crate) fn choice(&mut self, left: PatId, right: PatId) -> PatId {
        if left == NOT_ALLOWED {
            return right;
        }
        if right == NOT_ALLOWED || left == right {
            return left;
        }
        self.intern(Pattern::Choice(left, right))
    }

    /// The `group` constructor.
    pub(crate) fn group(&mut self, left: PatId, right: PatId) -> PatId {
        if left == NOT_ALLOWED || right == NOT_ALLOWED {
            return NOT_ALLOWED;
        }
        if left == EMPTY {
            return right;
        }
        if right == EMPTY {
            return left;
        }
        self.intern(Pattern::Group(left, right))
    }

    /// The `interleave` constructor.
    pub(crate) fn interleave(&mut self, left: PatId, right: PatId) -> PatId {
        if left == NOT_ALLOWED || right == NOT_ALLOWED {
            return NOT_ALLOWED;
        }
        if left == EMPTY {
            return right;
        }
        if right == EMPTY {
            return left;
        }
        self.intern(Pattern::Interleave(left, right))
    }

    /// The `after` constructor.
    pub(crate) fn after(&mut self, left: PatId, right: PatId) -> PatId {
        if left == NOT_ALLOWED || right == NOT_ALLOWED {
            return NOT_ALLOWED;
        }
        self.intern(Pattern::After(left, right))
    }

    /// The `oneOrMore` constructor.
    pub(crate) fn one_or_more(&mut self, inner: PatId) -> PatId {
        if inner == NOT_ALLOWED {
            return NOT_ALLOWED;
        }
        self.intern(Pattern::OneOrMore(inner))
    }

    /// The number of interned patterns.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.patterns.len()
    }
}

/// The table of name classes, with `contains` and an exact overlap test.
#[derive(Debug, Clone, Default)]
pub(crate) struct NameClasses {
    classes: Vec<NameClass>,
    intern: HashMap<NameClass, NameClassId>,
}

impl NameClasses {
    /// Interns a name class.
    pub(crate) fn intern(&mut self, class: NameClass) -> NameClassId {
        if let Some(existing) = self.intern.get(&class) {
            return *existing;
        }
        let id = u32::try_from(self.classes.len()).unwrap_or(u32::MAX);
        self.classes.push(class.clone());
        self.intern.insert(class, id);
        id
    }

    /// Returns the name class behind an identifier.
    pub(crate) fn get(&self, id: NameClassId) -> &NameClass {
        self.classes.get(id as usize).unwrap_or(&NameClass::AnyName)
    }

    /// The `contains` relation of §7: does the name class accept this name?
    ///
    /// A namespace of `""` means "no namespace".
    pub(crate) fn contains(&self, id: NameClassId, namespace: &str, local: &str) -> bool {
        match self.get(id) {
            NameClass::AnyName => true,
            NameClass::AnyNameExcept(except) => !self.contains(*except, namespace, local),
            NameClass::NsName(ns) => ns == namespace,
            NameClass::NsNameExcept(ns, except) => {
                ns == namespace && !self.contains(*except, namespace, local)
            }
            NameClass::Name(ns, ln) => ns == namespace && ln == local,
            NameClass::Choice(left, right) => {
                self.contains(*left, namespace, local) || self.contains(*right, namespace, local)
            }
        }
    }

    /// Decides whether two name classes accept a name in common.
    ///
    /// The test is exact. A name class can only tell names apart by the finitely
    /// many namespaces and local names it mentions, so trying every mentioned
    /// namespace and local name — plus one of each that appears nowhere — covers
    /// the whole (infinite) space of names.
    pub(crate) fn overlap(&self, left: NameClassId, right: NameClassId) -> bool {
        let mut namespaces = Vec::new();
        let mut locals = Vec::new();
        self.collect_names(left, &mut namespaces, &mut locals);
        self.collect_names(right, &mut namespaces, &mut locals);
        namespaces.push(fresh(&namespaces));
        locals.push(fresh(&locals));
        namespaces.iter().any(|namespace| {
            locals.iter().any(|local| {
                self.contains(left, namespace, local) && self.contains(right, namespace, local)
            })
        })
    }

    /// Returns `true` when the name class mentions `anyName` or `nsName`
    /// anywhere, i.e. when it may accept infinitely many names (§7.3).
    #[cfg(test)]
    pub(crate) fn is_infinite(&self, id: NameClassId) -> bool {
        match self.get(id) {
            NameClass::AnyName | NameClass::NsName(_) => true,
            NameClass::AnyNameExcept(_) | NameClass::NsNameExcept(_, _) => true,
            NameClass::Name(_, _) => false,
            NameClass::Choice(left, right) => self.is_infinite(*left) || self.is_infinite(*right),
        }
    }

    /// Renders a name class the way a diagnostic should show it.
    pub(crate) fn describe(&self, id: NameClassId) -> String {
        match self.get(id) {
            NameClass::AnyName => "*".to_owned(),
            NameClass::AnyNameExcept(except) => format!("* - {}", self.describe(*except)),
            NameClass::NsName(ns) => format!("{{{ns}}}*"),
            NameClass::NsNameExcept(ns, except) => {
                format!("{{{ns}}}* - {}", self.describe(*except))
            }
            NameClass::Name(ns, local) if ns.is_empty() => local.clone(),
            NameClass::Name(ns, local) => format!("{{{ns}}}{local}"),
            NameClass::Choice(left, right) => {
                format!("{} | {}", self.describe(*left), self.describe(*right))
            }
        }
    }

    fn collect_names(
        &self,
        id: NameClassId,
        namespaces: &mut Vec<String>,
        locals: &mut Vec<String>,
    ) {
        match self.get(id) {
            NameClass::AnyName => {}
            NameClass::AnyNameExcept(except) => self.collect_names(*except, namespaces, locals),
            NameClass::NsName(ns) => push_unique(namespaces, ns),
            NameClass::NsNameExcept(ns, except) => {
                push_unique(namespaces, ns);
                self.collect_names(*except, namespaces, locals);
            }
            NameClass::Name(ns, local) => {
                push_unique(namespaces, ns);
                push_unique(locals, local);
            }
            NameClass::Choice(left, right) => {
                self.collect_names(*left, namespaces, locals);
                self.collect_names(*right, namespaces, locals);
            }
        }
    }
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_owned());
    }
}

/// Builds a string that is not in `values`.
fn fresh(values: &[String]) -> String {
    let mut candidate = String::from("\u{1}");
    while values.contains(&candidate) {
        candidate.push('\u{1}');
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_have_fixed_identifiers() {
        let arena = Arena::default();
        assert_eq!(arena.get(EMPTY), &Pattern::Empty);
        assert_eq!(arena.get(NOT_ALLOWED), &Pattern::NotAllowed);
        assert_eq!(arena.get(TEXT), &Pattern::Text);
        assert!(arena.nullable(EMPTY));
        assert!(arena.nullable(TEXT));
        assert!(!arena.nullable(NOT_ALLOWED));
    }

    #[test]
    fn constructors_simplify() {
        let mut arena = Arena::default();
        assert_eq!(arena.choice(TEXT, NOT_ALLOWED), TEXT);
        assert_eq!(arena.choice(NOT_ALLOWED, TEXT), TEXT);
        assert_eq!(arena.choice(TEXT, TEXT), TEXT);
        assert_eq!(arena.group(TEXT, EMPTY), TEXT);
        assert_eq!(arena.group(TEXT, NOT_ALLOWED), NOT_ALLOWED);
        assert_eq!(arena.interleave(EMPTY, TEXT), TEXT);
        assert_eq!(arena.after(TEXT, NOT_ALLOWED), NOT_ALLOWED);
        assert_eq!(arena.one_or_more(NOT_ALLOWED), NOT_ALLOWED);
    }

    #[test]
    fn interning_shares_identifiers() {
        let mut arena = Arena::default();
        let inner = arena.one_or_more(TEXT);
        let first = arena.group(TEXT, inner);
        let inner_again = arena.one_or_more(TEXT);
        assert_eq!(inner, inner_again);
        let second = arena.group(TEXT, inner_again);
        assert_eq!(first, second);
        assert_eq!(arena.len(), 5);
    }

    #[test]
    fn name_class_contains() {
        let mut classes = NameClasses::default();
        let foo = classes.intern(NameClass::Name(String::new(), "foo".to_owned()));
        assert!(classes.contains(foo, "", "foo"));
        assert!(!classes.contains(foo, "urn:x", "foo"));

        let any = classes.intern(NameClass::AnyName);
        let any_but_foo = classes.intern(NameClass::AnyNameExcept(foo));
        assert!(classes.contains(any, "urn:x", "bar"));
        assert!(!classes.contains(any_but_foo, "", "foo"));
        assert!(classes.contains(any_but_foo, "", "bar"));
    }

    #[test]
    fn overlap_is_exact() {
        let mut classes = NameClasses::default();
        let foo = classes.intern(NameClass::Name(String::new(), "foo".to_owned()));
        let bar = classes.intern(NameClass::Name(String::new(), "bar".to_owned()));
        assert!(!classes.overlap(foo, bar));
        assert!(classes.overlap(foo, foo));

        let any = classes.intern(NameClass::AnyName);
        assert!(classes.overlap(any, foo));

        let any_but_foo = classes.intern(NameClass::AnyNameExcept(foo));
        assert!(!classes.overlap(any_but_foo, foo));
        assert!(classes.overlap(any_but_foo, bar));

        let ns = classes.intern(NameClass::NsName("urn:x".to_owned()));
        let other_ns = classes.intern(NameClass::NsName("urn:y".to_owned()));
        assert!(!classes.overlap(ns, other_ns));
        assert!(classes.overlap(ns, any));

        let ns_foo = classes.intern(NameClass::Name("urn:x".to_owned(), "foo".to_owned()));
        let ns_but_foo = classes.intern(NameClass::NsNameExcept("urn:x".to_owned(), ns_foo));
        assert!(classes.overlap(ns_but_foo, ns));
        assert!(!classes.overlap(ns_but_foo, ns_foo));
    }

    #[test]
    fn infinite_name_classes_are_detected() {
        let mut classes = NameClasses::default();
        let foo = classes.intern(NameClass::Name(String::new(), "foo".to_owned()));
        let any = classes.intern(NameClass::AnyName);
        let both = classes.intern(NameClass::Choice(foo, any));
        assert!(!classes.is_infinite(foo));
        assert!(classes.is_infinite(any));
        assert!(classes.is_infinite(both));
    }
}
