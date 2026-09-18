//! Encode implementations for tuples (1-16 elements)

use super::{Encode, Encoder};
use crate::error::Error;

// Implement Encode for tuples up to 16 elements
// Following bincode's pattern of direct implementations (not macros)

impl<T0: Encode> Encode for (T0,) {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)
    }
}

impl<T0: Encode, T1: Encode> Encode for (T0, T1) {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)
    }
}

impl<T0: Encode, T1: Encode, T2: Encode> Encode for (T0, T1, T2) {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)
    }
}

impl<T0: Encode, T1: Encode, T2: Encode, T3: Encode> Encode for (T0, T1, T2, T3) {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)
    }
}

impl<T0: Encode, T1: Encode, T2: Encode, T3: Encode, T4: Encode> Encode for (T0, T1, T2, T3, T4) {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)
    }
}

impl<T0: Encode, T1: Encode, T2: Encode, T3: Encode, T4: Encode, T5: Encode> Encode
    for (T0, T1, T2, T3, T4, T5)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)
    }
}

impl<T0: Encode, T1: Encode, T2: Encode, T3: Encode, T4: Encode, T5: Encode, T6: Encode> Encode
    for (T0, T1, T2, T3, T4, T5, T6)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7, T8)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
        T10: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)?;
        self.10.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
        T10: Encode,
        T11: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)?;
        self.10.encode(encoder)?;
        self.11.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
        T10: Encode,
        T11: Encode,
        T12: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)?;
        self.10.encode(encoder)?;
        self.11.encode(encoder)?;
        self.12.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
        T10: Encode,
        T11: Encode,
        T12: Encode,
        T13: Encode,
    > Encode for (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)?;
        self.10.encode(encoder)?;
        self.11.encode(encoder)?;
        self.12.encode(encoder)?;
        self.13.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
        T10: Encode,
        T11: Encode,
        T12: Encode,
        T13: Encode,
        T14: Encode,
    > Encode
    for (
        T0,
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        T8,
        T9,
        T10,
        T11,
        T12,
        T13,
        T14,
    )
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)?;
        self.10.encode(encoder)?;
        self.11.encode(encoder)?;
        self.12.encode(encoder)?;
        self.13.encode(encoder)?;
        self.14.encode(encoder)
    }
}

impl<
        T0: Encode,
        T1: Encode,
        T2: Encode,
        T3: Encode,
        T4: Encode,
        T5: Encode,
        T6: Encode,
        T7: Encode,
        T8: Encode,
        T9: Encode,
        T10: Encode,
        T11: Encode,
        T12: Encode,
        T13: Encode,
        T14: Encode,
        T15: Encode,
    > Encode
    for (
        T0,
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        T8,
        T9,
        T10,
        T11,
        T12,
        T13,
        T14,
        T15,
    )
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), Error> {
        self.0.encode(encoder)?;
        self.1.encode(encoder)?;
        self.2.encode(encoder)?;
        self.3.encode(encoder)?;
        self.4.encode(encoder)?;
        self.5.encode(encoder)?;
        self.6.encode(encoder)?;
        self.7.encode(encoder)?;
        self.8.encode(encoder)?;
        self.9.encode(encoder)?;
        self.10.encode(encoder)?;
        self.11.encode(encoder)?;
        self.12.encode(encoder)?;
        self.13.encode(encoder)?;
        self.14.encode(encoder)?;
        self.15.encode(encoder)
    }
}
