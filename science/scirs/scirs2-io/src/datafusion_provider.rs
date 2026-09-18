//! DataFusion-compatible table provider interface.
//!
//! Provides a pure-Rust table abstraction that mirrors the Apache DataFusion
//! `TableProvider` trait without pulling in the `datafusion` or `arrow` crates.
//!
//! # Components
//!
//! - `DataType` / `ColumnDef` / `TableSchema` — schema definitions
//! - `ColumnData` / `RecordBatch` — columnar in-memory batches
//! - `Expr` / `BinaryOperator` / `LiteralValue` — filter expression trees
//! - `TableProvider` trait — uniform scan interface
//! - `MemTableProvider` — simple in-memory implementation

use std::sync::Arc;

use scirs2_core::ndarray::Array2;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors returned from table provider operations.
#[derive(Debug, thiserror::Error)]
pub enum TableProviderError {
    /// The requested column was not found in the schema.
    #[error("Column not found: {0}")]
    ColumnNotFound(std::string::String),
    /// A type mismatch was encountered during a schema or value operation.
    #[error("Type error: {0}")]
    TypeError(std::string::String),
    /// An error occurred during a table scan.
    #[error("Scan error: {0}")]
    ScanError(std::string::String),
}

// ──────────────────────────────────────────────────────────────────────────────
// Schema types
// ──────────────────────────────────────────────────────────────────────────────

/// Supported column data types.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit IEEE 754 floating-point number.
    Float32,
    /// 64-bit IEEE 754 floating-point number.
    Float64,
    /// Boolean.
    Boolean,
    /// UTF-8 string.
    Utf8,
    /// Opaque binary data.
    Binary,
    /// Variable-length list.
    List(Box<DataType>),
}

/// A named column descriptor.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Column name.
    pub name: std::string::String,
    /// Column data type.
    pub data_type: DataType,
    /// Whether the column may contain null values.
    pub nullable: bool,
}

/// An ordered collection of `ColumnDef` descriptors forming a table schema.
#[derive(Debug, Clone)]
pub struct TableSchema {
    /// Ordered column descriptors.
    pub columns: Vec<ColumnDef>,
}

impl TableSchema {
    /// Create a new schema from a list of column definitions.
    pub fn new(columns: Vec<ColumnDef>) -> Self {
        Self { columns }
    }

    /// Find a column by name (case-sensitive).
    pub fn find_column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Return the zero-based index of a column by name, or `None` if not found.
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Columnar data
// ──────────────────────────────────────────────────────────────────────────────

/// The typed contents of one column inside a `RecordBatch`.
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// 32-bit signed integers (non-nullable).
    Int32(Vec<i32>),
    /// 64-bit signed integers (non-nullable).
    Int64(Vec<i64>),
    /// 32-bit floats (non-nullable).
    Float32(Vec<f32>),
    /// 64-bit floats (non-nullable).
    Float64(Vec<f64>),
    /// Booleans (non-nullable).
    Boolean(Vec<bool>),
    /// UTF-8 strings (non-nullable).
    Utf8(Vec<std::string::String>),
    /// All-null column of the given length.
    Null(usize),
}

impl ColumnData {
    /// Return the number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
            ColumnData::Null(n) => *n,
        }
    }

    /// Return `true` when the column contains no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Filter rows by a boolean mask of the same length.
    pub fn filter_by_mask(&self, mask: &[bool]) -> ColumnData {
        match self {
            ColumnData::Int32(v) => ColumnData::Int32(
                v.iter()
                    .zip(mask)
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect(),
            ),
            ColumnData::Int64(v) => ColumnData::Int64(
                v.iter()
                    .zip(mask)
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect(),
            ),
            ColumnData::Float32(v) => ColumnData::Float32(
                v.iter()
                    .zip(mask)
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect(),
            ),
            ColumnData::Float64(v) => ColumnData::Float64(
                v.iter()
                    .zip(mask)
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect(),
            ),
            ColumnData::Boolean(v) => ColumnData::Boolean(
                v.iter()
                    .zip(mask)
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect(),
            ),
            ColumnData::Utf8(v) => ColumnData::Utf8(
                v.iter()
                    .zip(mask)
                    .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
                    .collect(),
            ),
            ColumnData::Null(_) => {
                let count = mask.iter().filter(|&&m| m).count();
                ColumnData::Null(count)
            }
        }
    }

    /// Select rows by a vec of indices.
    pub fn select_rows(&self, indices: &[usize]) -> ColumnData {
        match self {
            ColumnData::Int32(v) => {
                ColumnData::Int32(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::Int64(v) => {
                ColumnData::Int64(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::Float32(v) => {
                ColumnData::Float32(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::Float64(v) => {
                ColumnData::Float64(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::Boolean(v) => {
                ColumnData::Boolean(indices.iter().filter_map(|&i| v.get(i).copied()).collect())
            }
            ColumnData::Utf8(v) => {
                ColumnData::Utf8(indices.iter().filter_map(|&i| v.get(i).cloned()).collect())
            }
            ColumnData::Null(_) => ColumnData::Null(indices.len()),
        }
    }
}

/// A batch of rows stored in columnar format.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    /// Schema describing the columns.
    pub schema: Arc<TableSchema>,
    /// One `ColumnData` entry per column in the schema.
    pub columns: Vec<ColumnData>,
    /// Number of rows represented in this batch.
    pub num_rows: usize,
}

impl RecordBatch {
    /// Create a new `RecordBatch` from a schema and column data.
    pub fn new(schema: Arc<TableSchema>, columns: Vec<ColumnData>) -> Self {
        let num_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        Self {
            schema,
            columns,
            num_rows,
        }
    }

    /// Return a reference to the column at `index`.
    pub fn column(&self, index: usize) -> Option<&ColumnData> {
        self.columns.get(index)
    }

    /// Return a reference to the column with the given name.
    pub fn column_by_name(&self, name: &str) -> Option<&ColumnData> {
        self.schema
            .field_index(name)
            .and_then(|i| self.columns.get(i))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Expression tree
// ──────────────────────────────────────────────────────────────────────────────

/// A scalar literal value used in filter expressions.
#[derive(Debug, Clone)]
pub enum LiteralValue {
    /// 64-bit integer.
    Int64(i64),
    /// 64-bit float.
    Float64(f64),
    /// Boolean.
    Boolean(bool),
    /// UTF-8 string.
    Utf8(std::string::String),
    /// SQL NULL.
    Null,
}

/// Binary arithmetic or comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    /// Equality (`=`).
    Eq,
    /// Inequality (`!=`).
    NotEq,
    /// Less-than (`<`).
    Lt,
    /// Less-than-or-equal (`<=`).
    LtEq,
    /// Greater-than (`>`).
    Gt,
    /// Greater-than-or-equal (`>=`).
    GtEq,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Arithmetic addition.
    Plus,
    /// Arithmetic subtraction.
    Minus,
    /// Arithmetic multiplication.
    Multiply,
    /// Arithmetic division.
    Divide,
}

/// A filter or projection expression.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Reference to a column by name.
    Column(std::string::String),
    /// A scalar literal constant.
    Literal(LiteralValue),
    /// Binary operation on two sub-expressions.
    BinaryOp {
        /// Left-hand operand.
        left: Box<Expr>,
        /// Operator.
        op: BinaryOperator,
        /// Right-hand operand.
        right: Box<Expr>,
    },
    /// Test whether a column value is SQL NULL.
    IsNull(Box<Expr>),
    /// Test whether a column value is not SQL NULL.
    IsNotNull(Box<Expr>),
    /// Logical negation.
    Not(Box<Expr>),
}

// ──────────────────────────────────────────────────────────────────────────────
// TableProvider trait
// ──────────────────────────────────────────────────────────────────────────────

/// Uniform table scan interface compatible with DataFusion's `TableProvider`.
pub trait TableProvider: Send + Sync {
    /// Return the table schema.
    fn schema(&self) -> Arc<TableSchema>;

    /// Scan the table.
    ///
    /// # Parameters
    /// - `projection`: Optional list of column indices to return. `None` returns all columns.
    /// - `filters`: Filter expressions to push down (best-effort; implementations may ignore).
    /// - `limit`: Optional maximum number of rows to return.
    fn scan(
        &self,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Vec<RecordBatch>, TableProviderError>;
}

// ──────────────────────────────────────────────────────────────────────────────
// In-memory table provider
// ──────────────────────────────────────────────────────────────────────────────

/// An in-memory table that stores data as a vector of `RecordBatch` slices.
pub struct MemTableProvider {
    schema: Arc<TableSchema>,
    batches: Vec<RecordBatch>,
}

impl MemTableProvider {
    /// Create a new provider from a schema and a pre-built list of batches.
    pub fn new(schema: TableSchema, batches: Vec<RecordBatch>) -> Self {
        Self {
            schema: Arc::new(schema),
            batches,
        }
    }

    /// Construct a `MemTableProvider` from a 2-D `f64` matrix.
    ///
    /// Each column in `matrix` becomes a `Float64` column. `column_names` must
    /// have exactly as many entries as there are columns in `matrix`.
    pub fn from_f64_matrix(
        matrix: &Array2<f64>,
        column_names: &[&str],
    ) -> Result<Self, TableProviderError> {
        let ncols = matrix.ncols();
        if column_names.len() != ncols {
            return Err(TableProviderError::TypeError(format!(
                "matrix has {ncols} columns but {} names were supplied",
                column_names.len()
            )));
        }

        let columns_def: Vec<ColumnDef> = column_names
            .iter()
            .map(|&name| ColumnDef {
                name: name.to_string(),
                data_type: DataType::Float64,
                nullable: false,
            })
            .collect();
        let schema = Arc::new(TableSchema::new(columns_def));

        let columns: Vec<ColumnData> = (0..ncols)
            .map(|col_idx| {
                let col_vec: Vec<f64> = matrix.column(col_idx).iter().copied().collect();
                ColumnData::Float64(col_vec)
            })
            .collect();

        let num_rows = matrix.nrows();
        let batch = RecordBatch {
            schema: Arc::clone(&schema),
            columns,
            num_rows,
        };

        Ok(Self {
            schema,
            batches: vec![batch],
        })
    }
}

impl TableProvider for MemTableProvider {
    fn schema(&self) -> Arc<TableSchema> {
        Arc::clone(&self.schema)
    }

    fn scan(
        &self,
        projection: Option<&[usize]>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Vec<RecordBatch>, TableProviderError> {
        let mut result_batches: Vec<RecordBatch> = Vec::new();
        let mut rows_remaining = limit;

        for batch in &self.batches {
            // Determine how many rows to take from this batch.
            let take_rows = match rows_remaining {
                None => batch.num_rows,
                Some(0) => break,
                Some(rem) => rem.min(batch.num_rows),
            };

            let projected_schema: Arc<TableSchema>;
            let projected_cols: Vec<ColumnData>;

            match projection {
                None => {
                    // Return all columns, sliced to `take_rows`.
                    projected_schema = Arc::clone(&batch.schema);
                    projected_cols = batch
                        .columns
                        .iter()
                        .map(|c| slice_column(c, 0, take_rows))
                        .collect();
                }
                Some(indices) => {
                    // Return only the projected columns.
                    let proj_defs: Vec<ColumnDef> = indices
                        .iter()
                        .map(|&i| {
                            batch.schema.columns.get(i).cloned().ok_or_else(|| {
                                TableProviderError::ColumnNotFound(format!(
                                    "projection index {i} out of range"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    projected_schema = Arc::new(TableSchema::new(proj_defs));

                    projected_cols = indices
                        .iter()
                        .map(|&i| {
                            batch
                                .columns
                                .get(i)
                                .map(|c| slice_column(c, 0, take_rows))
                                .ok_or_else(|| {
                                    TableProviderError::ColumnNotFound(format!(
                                        "projection index {i} out of range"
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
            }

            result_batches.push(RecordBatch {
                schema: projected_schema,
                columns: projected_cols,
                num_rows: take_rows,
            });

            if let Some(ref mut rem) = rows_remaining {
                *rem -= take_rows;
            }
        }

        Ok(result_batches)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Slice `col` to `[offset, offset + len)`.
pub(crate) fn slice_column(col: &ColumnData, offset: usize, len: usize) -> ColumnData {
    let end = (offset + len).min(col.len());
    match col {
        ColumnData::Int32(v) => ColumnData::Int32(v[offset..end].to_vec()),
        ColumnData::Int64(v) => ColumnData::Int64(v[offset..end].to_vec()),
        ColumnData::Float32(v) => ColumnData::Float32(v[offset..end].to_vec()),
        ColumnData::Float64(v) => ColumnData::Float64(v[offset..end].to_vec()),
        ColumnData::Boolean(v) => ColumnData::Boolean(v[offset..end].to_vec()),
        ColumnData::Utf8(v) => ColumnData::Utf8(v[offset..end].to_vec()),
        ColumnData::Null(n) => ColumnData::Null((end - offset).min(*n)),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(TableSchema::new(vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::Int32,
                nullable: false,
            },
            ColumnDef {
                name: "score".to_string(),
                data_type: DataType::Float64,
                nullable: false,
            },
            ColumnDef {
                name: "label".to_string(),
                data_type: DataType::Utf8,
                nullable: true,
            },
        ]));
        let columns = vec![
            ColumnData::Int32(vec![1, 2, 3, 4, 5]),
            ColumnData::Float64(vec![1.1, 2.2, 3.3, 4.4, 5.5]),
            ColumnData::Utf8(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
            ]),
        ];
        RecordBatch::new(schema, columns)
    }

    #[test]
    fn test_mem_table_scan_all() {
        let batch = make_batch();
        let schema = (*batch.schema).clone();
        let provider = MemTableProvider::new(schema, vec![batch]);

        let result = provider.scan(None, &[], None).expect("scan failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows, 5);
        assert_eq!(result[0].columns.len(), 3);
    }

    #[test]
    fn test_mem_table_projection() {
        let batch = make_batch();
        let schema = (*batch.schema).clone();
        let provider = MemTableProvider::new(schema, vec![batch]);

        // Project only column 0 (id) and column 2 (label).
        let result = provider
            .scan(Some(&[0, 2]), &[], None)
            .expect("scan failed");
        assert_eq!(result.len(), 1);
        let rb = &result[0];
        assert_eq!(rb.columns.len(), 2);
        assert_eq!(rb.schema.columns[0].name, "id");
        assert_eq!(rb.schema.columns[1].name, "label");
    }

    #[test]
    fn test_mem_table_from_matrix() {
        let mat = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let provider =
            MemTableProvider::from_f64_matrix(&mat, &["x", "y"]).expect("from_f64_matrix failed");

        let result = provider.scan(None, &[], None).expect("scan failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows, 3);

        if let ColumnData::Float64(vals) = &result[0].columns[0] {
            assert!((vals[0] - 1.0).abs() < 1e-12);
            assert!((vals[2] - 5.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64 column");
        }
    }

    #[test]
    fn test_table_schema_find() {
        let batch = make_batch();
        let schema = (*batch.schema).clone();

        let col = schema.find_column("score");
        assert!(col.is_some());
        assert_eq!(col.unwrap().data_type, DataType::Float64);

        let missing = schema.find_column("nonexistent");
        assert!(missing.is_none());

        let idx = schema.field_index("label");
        assert_eq!(idx, Some(2));
    }
}
