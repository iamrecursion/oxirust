//! Macros shared by the deserializer submodules.
//!
//! Mirrors the macros defined in the `de` module of quick-xml 0.41.

macro_rules! forward_to_simple_type {
    ($deserialize:ident, $($mut:tt)?) => {
        #[inline]
        fn $deserialize<V>($($mut)? self, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            SimpleTypeDeserializer::from_text(self.read_string()?).$deserialize(visitor)
        }
    };
}

/// Implement deserialization methods for scalar types, such as numbers, strings,
/// byte arrays, booleans and identifiers.
macro_rules! deserialize_primitives {
    ($($mut:tt)?) => {
        forward_to_simple_type!(deserialize_i8, $($mut)?);
        forward_to_simple_type!(deserialize_i16, $($mut)?);
        forward_to_simple_type!(deserialize_i32, $($mut)?);
        forward_to_simple_type!(deserialize_i64, $($mut)?);

        forward_to_simple_type!(deserialize_u8, $($mut)?);
        forward_to_simple_type!(deserialize_u16, $($mut)?);
        forward_to_simple_type!(deserialize_u32, $($mut)?);
        forward_to_simple_type!(deserialize_u64, $($mut)?);

        forward_to_simple_type!(deserialize_i128, $($mut)?);
        forward_to_simple_type!(deserialize_u128, $($mut)?);

        forward_to_simple_type!(deserialize_f32, $($mut)?);
        forward_to_simple_type!(deserialize_f64, $($mut)?);

        forward_to_simple_type!(deserialize_bool, $($mut)?);
        forward_to_simple_type!(deserialize_char, $($mut)?);

        forward_to_simple_type!(deserialize_str, $($mut)?);
        forward_to_simple_type!(deserialize_string, $($mut)?);

        /// Forwards deserialization to the [`deserialize_any`](#method.deserialize_any).
        #[inline]
        fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_any(visitor)
        }

        /// Forwards deserialization to the [`deserialize_bytes`](#method.deserialize_bytes).
        #[inline]
        fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_bytes(visitor)
        }

        /// Representation of the named units the same as [unnamed units](#method.deserialize_unit).
        #[inline]
        fn deserialize_unit_struct<V>(
            self,
            _name: &'static str,
            visitor: V,
        ) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_unit(visitor)
        }

        /// Representation of tuples the same as [sequences](#method.deserialize_seq).
        #[inline]
        fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_seq(visitor)
        }

        /// Representation of named tuples the same as [unnamed tuples](#method.deserialize_tuple).
        #[inline]
        fn deserialize_tuple_struct<V>(
            self,
            _name: &'static str,
            len: usize,
            visitor: V,
        ) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_tuple(len, visitor)
        }

        /// Forwards deserialization to the [`deserialize_struct`](#method.deserialize_struct)
        /// with empty name and fields.
        #[inline]
        fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_struct("", &[], visitor)
        }

        /// Identifiers represented as [strings](#method.deserialize_str).
        #[inline]
        fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_str(visitor)
        }

        /// Forwards deserialization to the [`deserialize_unit`](#method.deserialize_unit).
        #[inline]
        fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, DeError>
        where
            V: Visitor<'de>,
        {
            self.deserialize_unit(visitor)
        }
    };
}
