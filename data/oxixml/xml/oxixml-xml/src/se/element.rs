//! Contains serializer for an XML element

use crate::de::{TEXT_KEY, VALUE_KEY};
use crate::se::content::ContentSerializer;
use crate::se::key::QNameSerializer;
use crate::se::simple_type::{QuoteTarget, SimpleSeq, SimpleTypeSerializer};
use crate::se::text::TextSerializer;
use crate::se::{EmptyElementHandling, SeError, WriteResult, XmlName};
use serde::ser::{
    Impossible, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, Serializer,
};
use std::fmt::Write;

/// Writes simple type content between [`ElementSerializer::key`] tags.
macro_rules! write_primitive {
    ($method:ident ( $ty:ty )) => {
        fn $method(self, value: $ty) -> Result<Self::Ok, Self::Error> {
            self.ser.write_wrapped(self.key, |ser| ser.$method(value))
        }
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A serializer used to serialize element with specified name. Unlike the [`ContentSerializer`],
/// this serializer never uses variant names of enum variants, and because of that
/// it is unable to serialize any enum values, except unit variants.
///
/// Returns the classification of the last written type.
///
/// This serializer is used for an ordinary fields in structs, which are not special
/// fields named `$text` ([`TEXT_KEY`]) or `$value` ([`VALUE_KEY`]). `$text` field
/// should be serialized using [`SimpleTypeSerializer`] and `$value` field should be
/// serialized using [`ContentSerializer`].
///
/// This serializer does the following:
/// - numbers converted to a decimal representation and serialized as `<key>value</key>`;
/// - booleans serialized ether as `<key>true</key>` or `<key>false</key>`;
/// - strings and characters are serialized as `<key>value</key>`. In particular,
///   an empty string is serialized as `<key/>`;
/// - `None` is serialized as `<key/>`;
/// - `Some` and newtypes are serialized as an inner type using the same serializer;
/// - units (`()`) and unit structs are serialized as `<key/>`;
/// - sequences, tuples and tuple structs are serialized as repeated `<key>` tag.
///   In particular, empty sequence is serialized to nothing;
/// - structs are serialized as a sequence of fields wrapped in a `<key>` tag. Each
///   field is serialized recursively using either `ElementSerializer`, [`ContentSerializer`]
///   (`$value` fields), or [`SimpleTypeSerializer`] (`$text` fields).
///   In particular, the empty struct is serialized as `<key/>`;
/// - maps are serialized as a sequence of entries wrapped in a `<key>` tag. If key is
///   serialized to a special name, the same rules as for struct fields are applied.
///   In particular, the empty map is serialized as `<key/>`;
/// - enums:
///   - unit variants are serialized as `<key>variant</key>`;
///   - other variants are not supported ([`SeError::Unsupported`] is returned);
///
/// Usage of empty tags depends on the [`ContentSerializer::empty_element_handling`] setting.
pub struct ElementSerializer<'w, 'k, W: Write> {
    /// The inner serializer that contains the settings and mostly do the actual work
    pub ser: ContentSerializer<'w, 'k, W>,
    /// Tag name used to wrap serialized types except enum variants which uses the variant name
    pub(super) key: XmlName<'k>,
}

impl<'w, 'k, W: Write> Serializer for ElementSerializer<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Impossible<Self::Ok, Self::Error>;
    type SerializeMap = Map<'w, 'k, W>;
    type SerializeStruct = Struct<'w, 'k, W>;
    type SerializeStructVariant = Struct<'w, 'k, W>;

    write_primitive!(serialize_bool(bool));

    write_primitive!(serialize_i8(i8));
    write_primitive!(serialize_i16(i16));
    write_primitive!(serialize_i32(i32));
    write_primitive!(serialize_i64(i64));

    write_primitive!(serialize_u8(u8));
    write_primitive!(serialize_u16(u16));
    write_primitive!(serialize_u32(u32));
    write_primitive!(serialize_u64(u64));

    write_primitive!(serialize_i128(i128));
    write_primitive!(serialize_u128(u128));

    write_primitive!(serialize_f32(f32));
    write_primitive!(serialize_f64(f64));

    write_primitive!(serialize_char(char));
    write_primitive!(serialize_bytes(&[u8]));

    fn serialize_str(self, value: &str) -> Result<Self::Ok, Self::Error> {
        if value.is_empty() {
            self.ser.write_empty(self.key)
        } else {
            self.ser
                .write_wrapped(self.key, |ser| ser.serialize_str(value))
        }
    }

    /// By serde contract we should serialize key of [`None`] values. If someone
    /// wants to skip the field entirely, he should use
    /// `#[serde(skip_serializing_if = "Option::is_none")]`.
    ///
    /// In XML when we serialize field, we write field name as:
    /// - element name, or
    /// - attribute name
    ///
    /// and field value as
    /// - content of the element, or
    /// - attribute value
    ///
    /// So serialization of `None` works the same as [serialization of `()`](#method.serialize_unit)
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.serialize_unit()
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Self::Ok, Self::Error> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.ser.write_empty(self.key)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.ser.write_empty(self.key)
    }

    /// Writes a tag with name [`Self::key`] and content of unit variant inside.
    /// If variant is a special `$text` value, then empty tag `<key/>` is written.
    /// Otherwise a `<key>variant</key>` is written.
    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        if variant == TEXT_KEY {
            self.ser.write_empty(self.key)
        } else {
            self.ser.write_wrapped(self.key, |ser| {
                ser.serialize_unit_variant(name, variant_index, variant)
            })
        }
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        value.serialize(self)
    }

    /// Always returns [`SeError::Unsupported`]. Newtype variants can be serialized
    /// only in `$value` fields, which is serialized using [`ContentSerializer`].
    #[inline]
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        Err(SeError::Unsupported(
            format!(
                "cannot serialize enum newtype variant `{}::{}`",
                name, variant
            )
            .into(),
        ))
    }

    #[inline]
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(self)
    }

    #[inline]
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.serialize_seq(Some(len))
    }

    #[inline]
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.serialize_tuple(len)
    }

    /// Always returns [`SeError::Unsupported`]. Tuple variants can be serialized
    /// only in `$value` fields, which is serialized using [`ContentSerializer`].
    #[inline]
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(SeError::Unsupported(
            format!(
                "cannot serialize enum tuple variant `{}::{}`",
                name, variant
            )
            .into(),
        ))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Ok(Map {
            ser: self.serialize_struct("", 0)?,
            key: None,
        })
    }

    #[inline]
    fn serialize_struct(
        mut self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.ser.write_indent()?;
        self.ser.indent.increase();

        self.ser.writer.write_char('<')?;
        self.ser.writer.write_str(self.key.0)?;
        Ok(Struct {
            ser: self,
            children: String::new(),
            write_indent: true,
        })
    }

    /// Always returns [`SeError::Unsupported`]. Struct variants can be serialized
    /// only in `$value` fields, which is serialized using [`ContentSerializer`].
    #[inline]
    fn serialize_struct_variant(
        self,
        name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(SeError::Unsupported(
            format!(
                "cannot serialize enum struct variant `{}::{}`",
                name, variant
            )
            .into(),
        ))
    }
}

impl<'w, 'k, W: Write> SerializeSeq for ElementSerializer<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(ElementSerializer {
            ser: self.ser.new_seq_element_serializer(true),
            key: self.key,
        })?;
        // Write indent for the next element
        self.ser.write_indent = true;
        Ok(())
    }

    #[inline]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(WriteResult::Element)
    }
}

impl<'w, 'k, W: Write> SerializeTuple for ElementSerializer<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    #[inline]
    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        SerializeSeq::serialize_element(self, value)
    }

    #[inline]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        SerializeSeq::end(self)
    }
}

impl<'w, 'k, W: Write> SerializeTupleStruct for ElementSerializer<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    #[inline]
    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        SerializeSeq::serialize_element(self, value)
    }

    #[inline]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        SerializeSeq::end(self)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A serializer for tuple variants. Tuples can be serialized in two modes:
/// - wrapping each tuple field into a tag
/// - without wrapping, fields are delimited by a space
pub enum Tuple<'w, 'k, W: Write> {
    /// Serialize each tuple field as an element
    Element(ElementSerializer<'w, 'k, W>),
    /// Serialize tuple as an `xs:list`: space-delimited content of fields
    Text(SimpleSeq<&'w mut W>),
}

impl<'w, 'k, W: Write> SerializeTupleVariant for Tuple<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    #[inline]
    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        match self {
            Self::Element(ser) => SerializeTuple::serialize_element(ser, value),
            Self::Text(ser) => SerializeTuple::serialize_element(ser, value),
        }
    }

    #[inline]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        match self {
            Self::Element(ser) => SerializeTuple::end(ser),
            // Do not write indent after `$text` fields because it may be interpreted as
            // part of content when deserialize
            Self::Text(ser) => SerializeTuple::end(ser).map(|_| WriteResult::SensitiveText),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A serializer for struct variants, which serializes the struct contents inside
/// of wrapping tags (`<${tag}>...</${tag}>`).
///
/// Returns the classification of the last written type.
///
/// Serialization of each field depends on it representation:
/// - attributes written directly to the higher serializer
/// - elements buffered into internal buffer and at the end written into higher
///   serializer
pub struct Struct<'w, 'k, W: Write> {
    ser: ElementSerializer<'w, 'k, W>,
    /// Buffer to store serialized elements
    // Buffering is required because all attributes of an element must be
    // written before any child element, but serde hands the fields over in
    // declaration order.
    children: String,
    /// Whether need to write indent after the last written field
    write_indent: bool,
}

impl<'w, 'k, W: Write> Struct<'w, 'k, W> {
    #[inline]
    fn write_field<T>(&mut self, key: &str, value: &T) -> Result<(), SeError>
    where
        T: ?Sized + Serialize,
    {
        // Fields whose (possibly renamed) name starts with `@` become attributes.
        if let Some(key) = key.strip_prefix('@') {
            let key = XmlName::try_from(key)?;
            self.write_attribute(key, value)
        } else {
            self.write_element(key, value)
        }
    }

    /// Writes `value` as an attribute
    #[inline]
    fn write_attribute<T>(&mut self, key: XmlName, value: &T) -> Result<(), SeError>
    where
        T: ?Sized + Serialize,
    {
        // Attributes are always written on the same line as the tag name.
        self.ser.ser.writer.write_char(' ')?;
        self.ser.ser.writer.write_str(key.0)?;
        self.ser.ser.writer.write_char('=')?;

        // Attribute values are always double-quoted.
        self.ser.ser.writer.write_char('"')?;
        value.serialize(SimpleTypeSerializer {
            writer: &mut self.ser.ser.writer,
            target: QuoteTarget::DoubleQAttr,
            level: self.ser.ser.level,
        })?;
        self.ser.ser.writer.write_char('"')?;

        Ok(())
    }

    /// Writes `value` either as a text content, or as an element.
    ///
    /// If `key` has a magic value [`TEXT_KEY`], then `value` serialized as a
    /// [simple type].
    ///
    /// If `key` has a magic value [`VALUE_KEY`], then `value` serialized as a
    /// [content] without wrapping in tags, otherwise it is wrapped in
    /// `<${key}>...</${key}>`.
    ///
    /// [simple type]: SimpleTypeSerializer
    /// [content]: ContentSerializer
    fn write_element<T>(&mut self, key: &str, value: &T) -> Result<(), SeError>
    where
        T: ?Sized + Serialize,
    {
        let ser = ContentSerializer {
            writer: &mut self.children,
            level: self.ser.ser.level,
            indent: self.ser.ser.indent.borrow(),
            // If previous field does not require indent, do not write it
            write_indent: self.write_indent,
            text_format: self.ser.ser.text_format,
            allow_primitive: true,
            empty_element_handling: self.ser.ser.empty_element_handling,
        };

        if key == TEXT_KEY {
            value.serialize(TextSerializer(ser.into_simple_type_serializer()?))?;
            // Text was written so we don't need to indent next field
            self.write_indent = false;
        } else if key == VALUE_KEY {
            // If element was written then we need to indent next field unless it is a text field
            self.write_indent = value.serialize(ser)?.allow_indent();
        } else {
            value.serialize(ElementSerializer {
                key: XmlName::try_from(key)?,
                ser,
            })?;
            // Element was written so we need to indent next field unless it is a text field
            self.write_indent = true;
        }
        Ok(())
    }
}

impl<'w, 'k, W: Write> SerializeStruct for Struct<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.write_field(key, value)
    }

    fn end(mut self) -> Result<Self::Ok, Self::Error> {
        self.ser.ser.indent.decrease();

        if self.children.is_empty() {
            match self.ser.ser.empty_element_handling {
                EmptyElementHandling::SelfClosed => {
                    self.ser.ser.writer.write_str("/>")?;
                }
                EmptyElementHandling::SelfClosedWithSpace => {
                    self.ser.ser.writer.write_str(" />")?;
                }
                EmptyElementHandling::Expanded => {
                    self.ser.ser.writer.write_str("></")?;
                    self.ser.ser.writer.write_str(self.ser.key.0)?;
                    self.ser.ser.writer.write_char('>')?;
                }
            }
        } else {
            self.ser.ser.writer.write_char('>')?;
            self.ser.ser.writer.write_str(&self.children)?;

            if self.write_indent {
                self.ser.ser.indent.write_indent(&mut self.ser.ser.writer)?;
            }

            self.ser.ser.writer.write_str("</")?;
            self.ser.ser.writer.write_str(self.ser.key.0)?;
            self.ser.ser.writer.write_char('>')?;
        }
        Ok(WriteResult::Element)
    }
}

impl<'w, 'k, W: Write> SerializeStructVariant for Struct<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    #[inline]
    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        SerializeStruct::serialize_field(self, key, value)
    }

    #[inline]
    fn end(self) -> Result<Self::Ok, Self::Error> {
        SerializeStruct::end(self)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Map<'w, 'k, W: Write> {
    ser: Struct<'w, 'k, W>,
    /// Key, serialized by `QNameSerializer` if consumer uses `serialize_key` +
    /// `serialize_value` calls instead of `serialize_entry`
    key: Option<String>,
}

impl<'w, 'k, W: Write> Map<'w, 'k, W> {
    fn make_key<T>(&mut self, key: &T) -> Result<String, SeError>
    where
        T: ?Sized + Serialize,
    {
        key.serialize(QNameSerializer {
            writer: String::new(),
        })
    }
}

impl<'w, 'k, W: Write> SerializeMap for Map<'w, 'k, W> {
    type Ok = WriteResult;
    type Error = SeError;

    fn serialize_key<T>(&mut self, key: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        if self.key.take().is_some() {
            return Err(SeError::Custom(
                "calling `serialize_key` twice without `serialize_value`".to_string(),
            ));
        }
        self.key = Some(self.make_key(key)?);
        Ok(())
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        if let Some(key) = self.key.take() {
            return self.ser.write_field(&key, value);
        }
        Err(SeError::Custom(
            "calling `serialize_value` without call of `serialize_key`".to_string(),
        ))
    }

    fn serialize_entry<K, V>(&mut self, key: &K, value: &V) -> Result<(), Self::Error>
    where
        K: ?Sized + Serialize,
        V: ?Sized + Serialize,
    {
        let key = self.make_key(key)?;
        self.ser.write_field(&key, value)
    }

    fn end(mut self) -> Result<Self::Ok, Self::Error> {
        if let Some(key) = self.key.take() {
            return Err(SeError::Custom(format!(
                "calling `end` without call of `serialize_value` for key `{key}`"
            )));
        }
        SerializeStruct::end(self.ser)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests;
