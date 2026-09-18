use super::*;
use serde::de::{Deserializer, EnumAccess, VariantAccess, Visitor};
use std::fmt::{self, Formatter};

// NOTE: Derive could be possible once https://github.com/serde-rs/serde/issues/2126 is resolved
macro_rules! impl_deserialize_choice {
        ($name:ident : $(($field:ident, $field_name:literal)),*) => {
            impl<'de> Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: Deserializer<'de>,
                {
                    #[derive(Deserialize)]
                    #[serde(field_identifier)]
                    #[serde(rename_all = "kebab-case")]
                    enum Tag {
                        $($field,)*
                        Other(String),
                    }

                    struct EnumVisitor;
                    impl<'de> Visitor<'de> for EnumVisitor {
                        type Value = $name;

                        fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                            f.write_str("enum ")?;
                            f.write_str(stringify!($name))
                        }

                        fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
                        where
                            A: EnumAccess<'de>,
                        {
                            match data.variant()? {
                                $(
                                    (Tag::$field, variant) => variant.unit_variant().map(|_| $name::$field),
                                )*
                                (Tag::Other(t), v) => v.unit_variant().map(|_| $name::Other(t)),
                            }
                        }
                    }

                    const VARIANTS: &'static [&'static str] = &[
                        $($field_name,)*
                        "<any other tag>"
                    ];
                    deserializer.deserialize_enum(stringify!($name), VARIANTS, EnumVisitor)
                }
            }
        };
    }

/// Type that can be deserialized from `<one>`, `<two>`, or any other element
#[derive(Debug, PartialEq)]
enum Choice {
    One,
    Two,
    /// Any other tag name except `One` or `Two`, name of tag stored inside variant
    Other(String),
}
impl_deserialize_choice!(Choice: (One, "one"), (Two, "two"));

/// Type that can be deserialized from `<first>`, `<second>`, or any other element
///
/// Only referenced by the commented-out tests that upstream quick-xml keeps
/// for the unsupported `[Choice2; N]` / `Vec<Choice2>` cases.
#[allow(dead_code)]
#[derive(Debug, PartialEq)]
enum Choice2 {
    First,
    Second,
    /// Any other tag name except `First` or `Second`, name of tag stored inside variant
    Other(String),
}
impl_deserialize_choice!(Choice2: (First, "first"), (Second, "second"));

/// Type that can be deserialized from `<one>`, `<two>`, or any other element.
/// Used for `primitives` tests
#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Choice3 {
    One(usize),
    Two(String),
    #[serde(other)]
    Other,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Choice4 {
    One {
        inner: [(); 1],
    },
    Two {
        inner: [(); 1],
    },
    #[serde(other)]
    Other,
}

/// This module contains tests where size of the list have a compile-time size
#[path = "variable_name/fixed_size.rs"]
mod fixed_size;

/// This module contains tests where size of the list have an unspecified size
#[path = "variable_name/variable_size.rs"]
mod variable_size;
