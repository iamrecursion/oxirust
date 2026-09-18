//! Join algorithms for cross-format dataset merging.
//!
//! Provides three join strategies over `RecordBatch` values:
//! - `hash_join`: equi-join using an in-memory hash table (build/probe model)
//! - `merge_join`: sort-merge equi-join for pre-sorted integer key columns
//! - `nested_loop_join`: general predicate-based join (O(n*m) fallback)

use std::collections::HashMap;
use std::sync::Arc;

use crate::datafusion_provider::{
    ColumnData, ColumnDef, DataType, RecordBatch, TableProviderError, TableSchema,
};

// ──────────────────────────────────────────────────────────────────────────────
// Join type
// ──────────────────────────────────────────────────────────────────────────────

/// Specifies the semantics of a join operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Return only rows present in both left and right.
    Inner,
    /// Return all rows from left; fill right columns with null when no match exists.
    LeftOuter,
    /// Return all rows from right; fill left columns with null when no match exists.
    RightOuter,
    /// Return all rows from both sides; fill the opposite side with null when no match.
    FullOuter,
}

// ──────────────────────────────────────────────────────────────────────────────
// Hash join
// ──────────────────────────────────────────────────────────────────────────────

/// Equi-join using a hash table built from the left batch.
///
/// For `Inner` / `LeftOuter` the left batch is the build side; for `RightOuter`
/// and `FullOuter` the semantics are symmetric.
pub fn hash_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_key: &str,
    right_key: &str,
    join_type: JoinType,
) -> Result<RecordBatch, TableProviderError> {
    let left_key_idx = left.schema.field_index(left_key).ok_or_else(|| {
        TableProviderError::ColumnNotFound(format!("Left key column '{left_key}' not found"))
    })?;
    let right_key_idx = right.schema.field_index(right_key).ok_or_else(|| {
        TableProviderError::ColumnNotFound(format!("Right key column '{right_key}' not found"))
    })?;

    // Build hash table: key string representation -> Vec<left row index>
    let mut build_table: HashMap<std::string::String, Vec<usize>> = HashMap::new();
    for row_idx in 0..left.num_rows {
        let key = extract_key(&left.columns[left_key_idx], row_idx);
        build_table.entry(key).or_default().push(row_idx);
    }

    // Probe phase: for each right row, find matching left rows.
    let mut left_indices: Vec<Option<usize>> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();

    for right_row in 0..right.num_rows {
        let key = extract_key(&right.columns[right_key_idx], right_row);
        if let Some(left_rows) = build_table.get(&key) {
            for &left_row in left_rows {
                left_indices.push(Some(left_row));
                right_indices.push(Some(right_row));
            }
        } else {
            // No match on the right side.
            match join_type {
                JoinType::RightOuter | JoinType::FullOuter => {
                    left_indices.push(None);
                    right_indices.push(Some(right_row));
                }
                _ => {}
            }
        }
    }

    // Add unmatched left rows for LeftOuter / FullOuter.
    if matches!(join_type, JoinType::LeftOuter | JoinType::FullOuter) {
        // Collect which left rows were matched.
        let matched_left: std::collections::HashSet<usize> =
            left_indices.iter().filter_map(|o| *o).collect();
        for left_row in 0..left.num_rows {
            if !matched_left.contains(&left_row) {
                left_indices.push(Some(left_row));
                right_indices.push(None);
            }
        }
    }

    assemble_join_result(left, right, &left_indices, &right_indices)
}

// ──────────────────────────────────────────────────────────────────────────────
// Merge join (sort-merge on integer key column)
// ──────────────────────────────────────────────────────────────────────────────

/// Sort-merge equi-join on a pre-sorted integer key column.
///
/// Both batches **must** be sorted ascending on their respective key columns.
/// Non-integer key columns are also supported (comparison is done via
/// `extract_key` string representation).
pub fn merge_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_key: &str,
    right_key: &str,
    join_type: JoinType,
) -> Result<RecordBatch, TableProviderError> {
    let left_key_idx = left.schema.field_index(left_key).ok_or_else(|| {
        TableProviderError::ColumnNotFound(format!("Left key column '{left_key}' not found"))
    })?;
    let right_key_idx = right.schema.field_index(right_key).ok_or_else(|| {
        TableProviderError::ColumnNotFound(format!("Right key column '{right_key}' not found"))
    })?;

    let mut left_indices: Vec<Option<usize>> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();

    let mut li = 0usize;
    let mut ri = 0usize;

    while li < left.num_rows && ri < right.num_rows {
        let lk = extract_key(&left.columns[left_key_idx], li);
        let rk = extract_key(&right.columns[right_key_idx], ri);

        match lk.cmp(&rk) {
            std::cmp::Ordering::Less => {
                if matches!(join_type, JoinType::LeftOuter | JoinType::FullOuter) {
                    left_indices.push(Some(li));
                    right_indices.push(None);
                }
                li += 1;
            }
            std::cmp::Ordering::Greater => {
                if matches!(join_type, JoinType::RightOuter | JoinType::FullOuter) {
                    left_indices.push(None);
                    right_indices.push(Some(ri));
                }
                ri += 1;
            }
            std::cmp::Ordering::Equal => {
                // Emit all matching pairs (handle duplicates by scanning forward).
                let lk_start = li;
                let rk_start = ri;

                // Find the extent of the left run.
                let mut l_end = li + 1;
                while l_end < left.num_rows && extract_key(&left.columns[left_key_idx], l_end) == lk
                {
                    l_end += 1;
                }

                // Find the extent of the right run.
                let mut r_end = ri + 1;
                while r_end < right.num_rows
                    && extract_key(&right.columns[right_key_idx], r_end) == rk
                {
                    r_end += 1;
                }

                // Cross product of the two runs.
                for l_i in lk_start..l_end {
                    for r_i in rk_start..r_end {
                        left_indices.push(Some(l_i));
                        right_indices.push(Some(r_i));
                    }
                }

                li = l_end;
                ri = r_end;
            }
        }
    }

    // Emit any remaining left rows for LeftOuter / FullOuter.
    if matches!(join_type, JoinType::LeftOuter | JoinType::FullOuter) {
        while li < left.num_rows {
            left_indices.push(Some(li));
            right_indices.push(None);
            li += 1;
        }
    }

    // Emit any remaining right rows for RightOuter / FullOuter.
    if matches!(join_type, JoinType::RightOuter | JoinType::FullOuter) {
        while ri < right.num_rows {
            left_indices.push(None);
            right_indices.push(Some(ri));
            ri += 1;
        }
    }

    assemble_join_result(left, right, &left_indices, &right_indices)
}

// ──────────────────────────────────────────────────────────────────────────────
// Nested loop join
// ──────────────────────────────────────────────────────────────────────────────

/// General-purpose O(n*m) join: produce all pairs `(left_row, right_row)` for
/// which `predicate(left_row, right_row)` returns `true`.
///
/// Unlike `hash_join` and `merge_join`, this function is unconditionally inner-join
/// semantics: only rows satisfying the predicate appear in the output.
pub fn nested_loop_join(
    left: &RecordBatch,
    right: &RecordBatch,
    predicate: impl Fn(usize, usize) -> bool,
) -> Result<RecordBatch, TableProviderError> {
    let mut left_indices: Vec<Option<usize>> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();

    for li in 0..left.num_rows {
        for ri in 0..right.num_rows {
            if predicate(li, ri) {
                left_indices.push(Some(li));
                right_indices.push(Some(ri));
            }
        }
    }

    assemble_join_result(left, right, &left_indices, &right_indices)
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extract a sort-key string from a single row of a column.
fn extract_key(col: &ColumnData, row: usize) -> std::string::String {
    match col {
        ColumnData::Int32(v) => v.get(row).map(|x| format!("{:020}", x)).unwrap_or_default(),
        ColumnData::Int64(v) => v.get(row).map(|x| format!("{:020}", x)).unwrap_or_default(),
        ColumnData::Float32(v) => v
            .get(row)
            .map(|x| format!("{:030.15}", x))
            .unwrap_or_default(),
        ColumnData::Float64(v) => v
            .get(row)
            .map(|x| format!("{:030.15}", x))
            .unwrap_or_default(),
        ColumnData::Boolean(v) => v
            .get(row)
            .map(|x| if *x { "1" } else { "0" }.to_string())
            .unwrap_or_default(),
        ColumnData::Utf8(v) => v.get(row).cloned().unwrap_or_default(),
        ColumnData::Null(_) => std::string::String::new(),
    }
}

/// Assemble the final `RecordBatch` from index pairs.
///
/// Output schema = all left columns followed by all right columns. Columns from
/// the "missing" side (index `None`) are filled with typed null/zero sentinel
/// values appropriate for their type. In a production implementation these would
/// be nullable columns; here we use a `ColumnData::Null(n)` sentinel to keep the
/// code simple and allocation-free for the null case.
fn assemble_join_result(
    left: &RecordBatch,
    right: &RecordBatch,
    left_indices: &[Option<usize>],
    right_indices: &[Option<usize>],
) -> Result<RecordBatch, TableProviderError> {
    let num_rows = left_indices.len();

    // Build output schema: left columns + right columns.
    let mut schema_cols: Vec<ColumnDef> = left
        .schema
        .columns
        .iter()
        .map(|c| ColumnDef {
            name: c.name.clone(),
            data_type: c.data_type.clone(),
            nullable: true,
        })
        .collect();
    for right_col in &right.schema.columns {
        schema_cols.push(ColumnDef {
            name: right_col.name.clone(),
            data_type: right_col.data_type.clone(),
            nullable: true,
        });
    }
    let schema = Arc::new(TableSchema::new(schema_cols));

    // Materialise left columns.
    let mut out_cols: Vec<ColumnData> = left
        .columns
        .iter()
        .map(|col| materialise_nullable(col, left_indices, num_rows))
        .collect();

    // Materialise right columns.
    for col in &right.columns {
        out_cols.push(materialise_nullable(col, right_indices, num_rows));
    }

    Ok(RecordBatch {
        schema,
        columns: out_cols,
        num_rows,
    })
}

/// Materialise a column selecting rows by `indices`, using the type-appropriate
/// zero/null value when `indices[i] == None`.
fn materialise_nullable(
    col: &ColumnData,
    indices: &[Option<usize>],
    num_rows: usize,
) -> ColumnData {
    // If all indices are Some we can take the fast path.
    let all_some = indices.iter().all(|o| o.is_some());

    match col {
        ColumnData::Int32(v) => {
            if all_some {
                ColumnData::Int32(
                    indices
                        .iter()
                        .map(|o| o.map_or(0, |i| v.get(i).copied().unwrap_or(0)))
                        .collect(),
                )
            } else {
                ColumnData::Int32(
                    indices
                        .iter()
                        .map(|o| o.and_then(|i| v.get(i).copied()).unwrap_or(0))
                        .collect(),
                )
            }
        }
        ColumnData::Int64(v) => ColumnData::Int64(
            indices
                .iter()
                .map(|o| o.and_then(|i| v.get(i).copied()).unwrap_or(0))
                .collect(),
        ),
        ColumnData::Float32(v) => ColumnData::Float32(
            indices
                .iter()
                .map(|o| o.and_then(|i| v.get(i).copied()).unwrap_or(f32::NAN))
                .collect(),
        ),
        ColumnData::Float64(v) => ColumnData::Float64(
            indices
                .iter()
                .map(|o| o.and_then(|i| v.get(i).copied()).unwrap_or(f64::NAN))
                .collect(),
        ),
        ColumnData::Boolean(v) => ColumnData::Boolean(
            indices
                .iter()
                .map(|o| o.and_then(|i| v.get(i).copied()).unwrap_or(false))
                .collect(),
        ),
        ColumnData::Utf8(v) => ColumnData::Utf8(
            indices
                .iter()
                .map(|o| o.and_then(|i| v.get(i).cloned()).unwrap_or_default())
                .collect(),
        ),
        ColumnData::Null(_) => ColumnData::Null(num_rows),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafusion_provider::{ColumnDef, DataType, TableSchema};

    fn make_left_batch() -> RecordBatch {
        // id: [1, 2, 3, 4], name: ["a","b","c","d"]
        let schema = Arc::new(TableSchema::new(vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::Int32,
                nullable: false,
            },
            ColumnDef {
                name: "name".to_string(),
                data_type: DataType::Utf8,
                nullable: false,
            },
        ]));
        RecordBatch::new(
            schema,
            vec![
                ColumnData::Int32(vec![1, 2, 3, 4]),
                ColumnData::Utf8(vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                ]),
            ],
        )
    }

    fn make_right_batch() -> RecordBatch {
        // id: [2, 3, 5], score: [10.0, 20.0, 30.0]
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
        ]));
        RecordBatch::new(
            schema,
            vec![
                ColumnData::Int32(vec![2, 3, 5]),
                ColumnData::Float64(vec![10.0, 20.0, 30.0]),
            ],
        )
    }

    #[test]
    fn test_hash_join_inner() {
        let left = make_left_batch();
        let right = make_right_batch();
        let result =
            hash_join(&left, &right, "id", "id", JoinType::Inner).expect("hash_join inner failed");

        // Only rows with id=2 and id=3 match.
        assert_eq!(result.num_rows, 2);

        // The first column should be the left "id" values 2 and 3 (order may vary).
        let ids: Vec<i32> = if let ColumnData::Int32(v) = &result.columns[0] {
            v.clone()
        } else {
            panic!("Expected Int32 id column");
        };
        let mut ids_sorted = ids.clone();
        ids_sorted.sort();
        assert_eq!(ids_sorted, vec![2, 3]);
    }

    #[test]
    fn test_hash_join_left_outer() {
        let left = make_left_batch();
        let right = make_right_batch();
        let result = hash_join(&left, &right, "id", "id", JoinType::LeftOuter)
            .expect("hash_join left_outer failed");

        // All 4 left rows must appear (id=1,2,3,4).
        assert_eq!(result.num_rows, 4);

        let ids: Vec<i32> = if let ColumnData::Int32(v) = &result.columns[0] {
            v.clone()
        } else {
            panic!("Expected Int32 id column");
        };
        let mut ids_sorted = ids.clone();
        ids_sorted.sort();
        assert_eq!(ids_sorted, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_join_inner() {
        // Both batches pre-sorted by id.
        let left = make_left_batch();
        let right = make_right_batch();
        let result = merge_join(&left, &right, "id", "id", JoinType::Inner)
            .expect("merge_join inner failed");

        assert_eq!(result.num_rows, 2);

        let ids: Vec<i32> = if let ColumnData::Int32(v) = &result.columns[0] {
            v.clone()
        } else {
            panic!("Expected Int32 id column");
        };
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn test_nested_loop_join() {
        let left = make_left_batch();
        let right = make_right_batch();

        // Predicate: left id == right id  (equi-join via nested loop).
        let predicate = |li: usize, ri: usize| {
            let left_ids = match &left.columns[0] {
                ColumnData::Int32(v) => v.clone(),
                _ => vec![],
            };
            let right_ids = match &right.columns[0] {
                ColumnData::Int32(v) => v.clone(),
                _ => vec![],
            };
            left_ids.get(li) == right_ids.get(ri)
        };

        let result = nested_loop_join(&left, &right, predicate).expect("nested_loop_join failed");

        assert_eq!(result.num_rows, 2);

        let ids: Vec<i32> = if let ColumnData::Int32(v) = &result.columns[0] {
            v.clone()
        } else {
            panic!("Expected Int32 id column");
        };
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn test_hash_join_right_outer() {
        let left = make_left_batch();
        let right = make_right_batch();
        let result = hash_join(&left, &right, "id", "id", JoinType::RightOuter)
            .expect("hash_join right_outer failed");

        // All 3 right rows must appear (id=2,3,5).
        assert_eq!(result.num_rows, 3);
    }

    #[test]
    fn test_merge_join_left_outer() {
        let left = make_left_batch();
        let right = make_right_batch();
        let result = merge_join(&left, &right, "id", "id", JoinType::LeftOuter)
            .expect("merge_join left_outer failed");

        // All 4 left rows must appear.
        assert_eq!(result.num_rows, 4);
    }
}
