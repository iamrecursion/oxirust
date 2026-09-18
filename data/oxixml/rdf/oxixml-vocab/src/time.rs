//! [OWL-Time](https://www.w3.org/TR/owl-time/) — an OWL 2 ontology of temporal concepts, for describing the temporal properties of resources.
//!
//! Source: <https://www.w3.org/2006/time#>

use oxixml_model::NamedNodeRef;

/// The `time` namespace IRI (`http://www.w3.org/2006/time#`).
pub const NAMESPACE: &str = "http://www.w3.org/2006/time#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/2006/time#", $local));
    };
}

term!(
    DATE_TIME_DESCRIPTION,
    "DateTimeDescription",
    "Description of date and time structured with separate values for the various elements of a calendar-clock system. The temporal reference system is fixed to Gregorian Calendar, and the range of year, month, day properties restricted to corresponding XML Schema types xsd:gYear, xsd:gMonth and xsd:gDay, respectively."
);
term!(
    DATE_TIME_INTERVAL,
    "DateTimeInterval",
    "DateTimeInterval is a subclass of ProperInterval, defined using the multi-element DateTimeDescription."
);
term!(DAY_OF_WEEK_CLASS, "DayOfWeek", "The day of week");
term!(
    DURATION,
    "Duration",
    "Duration of a temporal extent expressed as a number scaled by a temporal unit"
);
term!(
    DURATION_DESCRIPTION,
    "DurationDescription",
    "Description of temporal extent structured with separate values for the various elements of a calendar-clock system. The temporal reference system is fixed to Gregorian Calendar, and the range of each of the numeric properties is restricted to xsd:decimal"
);
term!(FRIDAY, "Friday", "Friday");
term!(
    GENERAL_DATE_TIME_DESCRIPTION,
    "GeneralDateTimeDescription",
    "Description of date and time structured with separate values for the various elements of a calendar-clock system"
);
term!(
    GENERAL_DURATION_DESCRIPTION,
    "GeneralDurationDescription",
    "Description of temporal extent structured with separate values for the various elements of a calendar-clock system."
);
term!(
    INSTANT,
    "Instant",
    "A temporal entity with zero extent or duration"
);
term!(
    INTERVAL,
    "Interval",
    "A temporal entity with an extent or duration"
);
term!(JANUARY, "January", "January");
term!(MONDAY, "Monday", "Monday");
term!(MONTH_OF_YEAR_CLASS, "MonthOfYear", "The month of the year");
term!(
    PROPER_INTERVAL,
    "ProperInterval",
    "A temporal entity with non-zero extent or duration, i.e. for which the value of the beginning and end are different"
);
term!(SATURDAY, "Saturday", "Saturday");
term!(SUNDAY, "Sunday", "Sunday");
term!(
    TRS,
    "TRS",
    "Un sistema de referencia temporal, tal como un sistema de coordenadas temporales (con un origen, una dirección y una escala), una combinación calendario-reloj, o un sistema ordinal (posiblemente jerárquico). Esta clase comodín representa el conjunto de todos los sistemas de referencia temporal.\"\"\"@es ; rdfs:label \"sistema de referencia temporal\"@es ; skos:definition"
);
term!(
    TEMPORAL_DURATION,
    "TemporalDuration",
    "Time extent; duration of a time interval separate from its particular start position"
);
term!(
    TEMPORAL_ENTITY,
    "TemporalEntity",
    "A temporal interval or instant."
);
term!(
    TEMPORAL_POSITION,
    "TemporalPosition",
    "A position on a time-line"
);
term!(
    TEMPORAL_UNIT,
    "TemporalUnit",
    "A standard duration, which provides a scale factor for a time extent, or the granularity or precision for a time position."
);
term!(THURSDAY, "Thursday", "Thursday");
term!(
    TIME_POSITION,
    "TimePosition",
    "A temporal position described using either a (nominal) value from an ordinal reference system, or a (numeric) value in a temporal coordinate system."
);
term!(
    TIME_ZONE_CLASS,
    "TimeZone",
    "Un huso horario especifica la cantidad en que la hora local está desplazada con respecto a UTC. Un huso horario normalmente se denota geográficamente (p.ej. el horario de verano del este de Australia), con un valor constante en una región dada. La región donde aplica y el desplazamiento desde UTC las especifica una autoridad gubernamental localmente reconocida.\"\"\"@es ; rdfs:label \"huso horario\"@es ; skos:definition"
);
term!(TUESDAY, "Tuesday", "Tuesday");
term!(WEDNESDAY, "Wednesday", "Wednesday");
term!(YEAR_CLASS, "Year", "Year duration");
term!(
    AFTER,
    "after",
    "Gives directionality to time. If a temporal entity T1 is after another temporal entity T2, then the beginning of T1 is after the end of T2."
);
term!(
    BEFORE,
    "before",
    "Gives directionality to time. If a temporal entity T1 is before another temporal entity T2, then the end of T1 is before the beginning of T2. Thus, \\"
);
term!(DAY, "day", "day");
term!(
    DAY_OF_WEEK,
    "dayOfWeek",
    "The day of week, whose value is a member of the class time:DayOfWeek"
);
term!(
    DAY_OF_YEAR,
    "dayOfYear",
    "The number of the day within the year"
);
term!(
    DAYS,
    "days",
    "length of, or element of the length of, a temporal extent expressed in days"
);
term!(
    GENERAL_DAY,
    "generalDay",
    "Día del mes - formulado como una cadena de texto con una restricción patrón para reproducir la misma forma léxica que gDay, excepto que se permiten valores hasta el 99, con el propósito de proporcionar soporte a calendarios con meses con más de 31 días. Nótese que el espacio de valores no está definido, por tanto, un procesador genérico de OWL2 no puede computar relaciones de orden de valores de este tipo.\"\"\"@es ; rdfs:label \"Día generalizado\"@es ; skos:definition"
);
term!(
    GENERAL_MONTH,
    "generalMonth",
    "Mes del año - formulado como una cadena de texto con una restricción patrón para reproducir la misma forma léxica que gMonth, excepto que se permiten valores hasta el 20, con el propósito de proporcionar soporte a calendarios con años con más de 12 meses. Nótese que el espacio de valores no está definido, por tanto, un procesador genérico de OWL2 no puede computar relaciones de orden de valores de este tipo.\"\"\"@es ; rdfs:label \"Mes generalizado\"@es ; skos:definition"
);
term!(
    GENERAL_YEAR,
    "generalYear",
    "Número de año - formulado como una cadena de texto con una restricción patrón para reproducir la misma forma léxica que gYear, aunque no está restringido a valores del calendario gregoriano. Nótese que el espacio de valores no está definido, por tanto, un procesador genérico de OWL2 no puede computar relaciones de orden de valores de este tipo.\"\"\"@es ; rdfs:label \"Año generalizado\"@es ; skos:definition"
);
term!(
    HAS_BEGINNING,
    "hasBeginning",
    "Beginning of a temporal entity"
);
term!(
    HAS_DATE_TIME_DESCRIPTION,
    "hasDateTimeDescription",
    "Value of DateTimeInterval expressed as a structured value. The beginning and end of the interval coincide with the limits of the shortest element in the description."
);
term!(
    HAS_DURATION,
    "hasDuration",
    "Duration of a temporal entity, expressed as a scaled value or nominal value"
);
term!(
    HAS_DURATION_DESCRIPTION,
    "hasDurationDescription",
    "Duration of a temporal entity, expressed using a structured description"
);
term!(HAS_END, "hasEnd", "End of a temporal entity.");
term!(
    HAS_TRS,
    "hasTRS",
    "The temporal reference system used by a temporal position or extent description."
);
term!(
    HAS_TEMPORAL_DURATION,
    "hasTemporalDuration",
    "Duration of a temporal entity."
);
term!(
    HAS_TIME,
    "hasTime",
    "Supports the association of a temporal entity (instant or interval) to any thing"
);
term!(
    HAS_XSD_DURATION,
    "hasXSDDuration",
    "Extent of a temporal entity, expressed using xsd:duration"
);
term!(HOUR, "hour", "Hour position in a calendar-clock system.");
term!(
    HOURS,
    "hours",
    "length of, or element of the length of, a temporal extent expressed in hours"
);
term!(
    IN_DATE_TIME,
    "inDateTime",
    "Position of an instant, expressed using a structured description"
);
term!(
    IN_TEMPORAL_POSITION,
    "inTemporalPosition",
    "Position of a time instant"
);
term!(
    IN_TIME_POSITION,
    "inTimePosition",
    "Position of an instant, expressed as a temporal coordinate or nominal value"
);
term!(
    IN_XSD_DATE,
    "inXSDDate",
    "Position of an instant, expressed using xsd:date"
);
term!(
    IN_XSD_DATE_TIME,
    "inXSDDateTime",
    "Position of an instant, expressed using xsd:dateTime"
);
term!(
    IN_XSD_DATE_TIME_STAMP,
    "inXSDDateTimeStamp",
    "Position of an instant, expressed using xsd:dateTimeStamp"
);
term!(
    IN_XSD_G_YEAR,
    "inXSDgYear",
    "Position of an instant, expressed using xsd:gYear"
);
term!(
    IN_XSD_G_YEAR_MONTH,
    "inXSDgYearMonth",
    "Position of an instant, expressed using xsd:gYearMonth"
);
term!(
    INSIDE,
    "inside",
    "An instant that falls inside the interval. It is not intended to include beginnings and ends of intervals."
);
term!(
    INTERVAL_AFTER,
    "intervalAfter",
    "If a proper interval T1 is intervalAfter another proper interval T2, then the beginning of T1 is after the end of T2."
);
term!(
    INTERVAL_BEFORE,
    "intervalBefore",
    "If a proper interval T1 is intervalBefore another proper interval T2, then the end of T1 is before the beginning of T2."
);
term!(
    INTERVAL_CONTAINS,
    "intervalContains",
    "If a proper interval T1 is intervalContains another proper interval T2, then the beginning of T1 is before the beginning of T2, and the end of T1 is after the end of T2."
);
term!(
    INTERVAL_DISJOINT,
    "intervalDisjoint",
    "If a proper interval T1 is intervalDisjoint another proper interval T2, then the beginning of T1 is after the end of T2, or the end of T1 is before the beginning of T2, i.e. the intervals do not overlap in any way, but their ordering relationship is not known."
);
term!(
    INTERVAL_DURING,
    "intervalDuring",
    "If a proper interval T1 is intervalDuring another proper interval T2, then the beginning of T1 is after the beginning of T2, and the end of T1 is before the end of T2."
);
term!(
    INTERVAL_EQUALS,
    "intervalEquals",
    "If a proper interval T1 is intervalEquals another proper interval T2, then the beginning of T1 is coincident with the beginning of T2, and the end of T1 is coincident with the end of T2."
);
term!(
    INTERVAL_FINISHED_BY,
    "intervalFinishedBy",
    "If a proper interval T1 is intervalFinishedBy another proper interval T2, then the beginning of T1 is before the beginning of T2, and the end of T1 is coincident with the end of T2."
);
term!(
    INTERVAL_FINISHES,
    "intervalFinishes",
    "If a proper interval T1 is intervalFinishes another proper interval T2, then the beginning of T1 is after the beginning of T2, and the end of T1 is coincident with the end of T2."
);
term!(
    INTERVAL_IN,
    "intervalIn",
    "If a proper interval T1 is intervalIn another proper interval T2, then the beginning of T1 is after the beginning of T2 or is coincident with the beginning of T2, and the end of T1 is before the end of T2, or is coincident with the end of T2, except that end of T1 may not be coincident with the end of T2 if the beginning of T1 is coincident with the beginning of T2."
);
term!(
    INTERVAL_MEETS,
    "intervalMeets",
    "If a proper interval T1 is intervalMeets another proper interval T2, then the end of T1 is coincident with the beginning of T2."
);
term!(
    INTERVAL_MET_BY,
    "intervalMetBy",
    "If a proper interval T1 is intervalMetBy another proper interval T2, then the beginning of T1 is coincident with the end of T2."
);
term!(
    INTERVAL_OVERLAPPED_BY,
    "intervalOverlappedBy",
    "If a proper interval T1 is intervalOverlappedBy another proper interval T2, then the beginning of T1 is after the beginning of T2, the beginning of T1 is before the end of T2, and the end of T1 is after the end of T2."
);
term!(
    INTERVAL_OVERLAPS,
    "intervalOverlaps",
    "If a proper interval T1 is intervalOverlaps another proper interval T2, then the beginning of T1 is before the beginning of T2, the end of T1 is after the beginning of T2, and the end of T1 is before the end of T2."
);
term!(
    INTERVAL_STARTED_BY,
    "intervalStartedBy",
    "If a proper interval T1 is intervalStarted another proper interval T2, then the beginning of T1 is coincident with the beginning of T2, and the end of T1 is after the end of T2."
);
term!(
    INTERVAL_STARTS,
    "intervalStarts",
    "If a proper interval T1 is intervalStarts another proper interval T2, then the beginning of T1 is coincident with the beginning of T2, and the end of T1 is before the end of T2."
);
term!(
    MINUTE,
    "minute",
    "Minute position in a calendar-clock system."
);
term!(
    MINUTES,
    "minutes",
    "length, or element of, a temporal extent expressed in minutes"
);
term!(
    MONTH,
    "month",
    "Posición de mes en un sistema calendario-reloj. El rango de esta propiedad no está especificado, por tanto, se puede reemplazar por cualquier representación específica de un mes de calendario de un calendario cualquiera.\"\"\"@es ; rdfs:label \"mes\"@es ; skos:definition"
);
term!(
    MONTH_OF_YEAR,
    "monthOfYear",
    "The month of the year, whose value is a member of the class time:MonthOfYear"
);
term!(
    MONTHS,
    "months",
    "length of, or element of the length of, a temporal extent expressed in months"
);
term!(
    NOMINAL_POSITION,
    "nominalPosition",
    "The (nominal) value indicating temporal position in an ordinal reference system"
);
term!(
    NUMERIC_DURATION,
    "numericDuration",
    "Value of a temporal extent expressed as a decimal number scaled by a temporal unit"
);
term!(
    NUMERIC_POSITION,
    "numericPosition",
    "The (numeric) value indicating position within a temporal coordinate system"
);
term!(
    SECOND,
    "second",
    "Second position in a calendar-clock system."
);
term!(
    SECONDS,
    "seconds",
    "length of, or element of the length of, a temporal extent expressed in seconds"
);
term!(
    TIME_ZONE,
    "timeZone",
    "The time zone for clock elements in the temporal position"
);
term!(UNIT_DAY, "unitDay", "Day (unit of temporal duration)");
term!(UNIT_HOUR, "unitHour", "Hour (unit of temporal duration)");
term!(
    UNIT_MINUTE,
    "unitMinute",
    "Minute (unit of temporal duration)"
);
term!(UNIT_MONTH, "unitMonth", "Month (unit of temporal duration)");
term!(
    UNIT_SECOND,
    "unitSecond",
    "Second (unit of temporal duration)"
);
term!(
    UNIT_TYPE,
    "unitType",
    "The temporal unit which provides the precision of a date-time value or scale of a temporal extent"
);
term!(UNIT_WEEK, "unitWeek", "Week (unit of temporal duration)");
term!(UNIT_YEAR, "unitYear", "Year (unit of temporal duration)");
term!(WEEK, "week", "Week number within the year.");
term!(
    WEEKS,
    "weeks",
    "length of, or element of the length of, a temporal extent expressed in weeks"
);
term!(
    XSD_DATE_TIME,
    "xsdDateTime",
    "Value of DateTimeInterval expressed as a compact value."
);
term!(YEAR, "year", "year");
term!(
    YEARS,
    "years",
    "length of, or element of the length of, a temporal extent expressed in years"
);

/// All 95 terms defined in the [`time`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    DATE_TIME_DESCRIPTION,
    DATE_TIME_INTERVAL,
    DAY_OF_WEEK_CLASS,
    DURATION,
    DURATION_DESCRIPTION,
    FRIDAY,
    GENERAL_DATE_TIME_DESCRIPTION,
    GENERAL_DURATION_DESCRIPTION,
    INSTANT,
    INTERVAL,
    JANUARY,
    MONDAY,
    MONTH_OF_YEAR_CLASS,
    PROPER_INTERVAL,
    SATURDAY,
    SUNDAY,
    TRS,
    TEMPORAL_DURATION,
    TEMPORAL_ENTITY,
    TEMPORAL_POSITION,
    TEMPORAL_UNIT,
    THURSDAY,
    TIME_POSITION,
    TIME_ZONE_CLASS,
    TUESDAY,
    WEDNESDAY,
    YEAR_CLASS,
    AFTER,
    BEFORE,
    DAY,
    DAY_OF_WEEK,
    DAY_OF_YEAR,
    DAYS,
    GENERAL_DAY,
    GENERAL_MONTH,
    GENERAL_YEAR,
    HAS_BEGINNING,
    HAS_DATE_TIME_DESCRIPTION,
    HAS_DURATION,
    HAS_DURATION_DESCRIPTION,
    HAS_END,
    HAS_TRS,
    HAS_TEMPORAL_DURATION,
    HAS_TIME,
    HAS_XSD_DURATION,
    HOUR,
    HOURS,
    IN_DATE_TIME,
    IN_TEMPORAL_POSITION,
    IN_TIME_POSITION,
    IN_XSD_DATE,
    IN_XSD_DATE_TIME,
    IN_XSD_DATE_TIME_STAMP,
    IN_XSD_G_YEAR,
    IN_XSD_G_YEAR_MONTH,
    INSIDE,
    INTERVAL_AFTER,
    INTERVAL_BEFORE,
    INTERVAL_CONTAINS,
    INTERVAL_DISJOINT,
    INTERVAL_DURING,
    INTERVAL_EQUALS,
    INTERVAL_FINISHED_BY,
    INTERVAL_FINISHES,
    INTERVAL_IN,
    INTERVAL_MEETS,
    INTERVAL_MET_BY,
    INTERVAL_OVERLAPPED_BY,
    INTERVAL_OVERLAPS,
    INTERVAL_STARTED_BY,
    INTERVAL_STARTS,
    MINUTE,
    MINUTES,
    MONTH,
    MONTH_OF_YEAR,
    MONTHS,
    NOMINAL_POSITION,
    NUMERIC_DURATION,
    NUMERIC_POSITION,
    SECOND,
    SECONDS,
    TIME_ZONE,
    UNIT_DAY,
    UNIT_HOUR,
    UNIT_MINUTE,
    UNIT_MONTH,
    UNIT_SECOND,
    UNIT_TYPE,
    UNIT_WEEK,
    UNIT_YEAR,
    WEEK,
    WEEKS,
    XSD_DATE_TIME,
    YEAR,
    YEARS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn namespace_ends_with_separator() {
        assert!(NAMESPACE.ends_with('#') || NAMESPACE.ends_with('/'));
    }

    #[test]
    fn terms_are_namespace_prefixed() {
        for term in ALL {
            assert!(
                term.as_str().starts_with(NAMESPACE),
                "{} is not prefixed by {}",
                term.as_str(),
                NAMESPACE
            );
            assert!(term.as_str().len() > NAMESPACE.len());
        }
    }

    #[test]
    fn no_duplicate_terms() {
        let mut seen = HashSet::new();
        for term in ALL {
            assert!(
                seen.insert(term.as_str()),
                "duplicate IRI: {}",
                term.as_str()
            );
        }
    }

    #[test]
    fn expected_term_count() {
        assert_eq!(ALL.len(), 95);
    }
}
