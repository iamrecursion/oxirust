//! Computational tape for reverse-mode automatic differentiation
//!
//! This module implements the tape structure used to record computational
//! operations for later backpropagation in reverse-mode AD.

use crate::error::OptimizeError;
use std::collections::HashMap;

/// Type alias for batch processor function
type BatchProcessor = Box<dyn Fn(&[TapeNode]) -> Result<(), OptimizeError>>;

/// A variable in the computational tape
#[derive(Debug, Clone)]
pub struct Variable {
    /// Unique identifier for this variable
    pub id: usize,
    /// Current value
    pub value: f64,
}

impl Variable {
    /// Create a new variable
    pub fn new(id: usize, value: f64) -> Self {
        Self { id, value }
    }
}

/// Type of unary operation
#[derive(Debug, Clone, Copy)]
pub enum UnaryOpType {
    /// Negation: -x
    Neg,
    /// Natural logarithm: ln(x)
    Ln,
    /// Exponential: exp(x)
    Exp,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Tangent: tan(x)
    Tan,
    /// Square root: sqrt(x)
    Sqrt,
    /// Square: x^2
    Square,
    /// Reciprocal: 1/x
    Reciprocal,
}

/// Type of binary operation
#[derive(Debug, Clone, Copy)]
pub enum BinaryOpType {
    /// Addition: x + y
    Add,
    /// Subtraction: x - y
    Sub,
    /// Multiplication: x * y
    Mul,
    /// Division: x / y
    Div,
    /// Power: x^y
    Pow,
}

/// A node in the computational tape representing an operation
#[derive(Debug, Clone)]
pub enum TapeNode {
    /// Input variable (leaf node)
    Input { var_id: usize },
    /// Constant value
    Constant { value: f64, result: usize },
    /// Unary operation
    UnaryOp {
        op_type: UnaryOpType,
        input: usize,
        result: usize,
        partial: f64, // ∂result/∂input
    },
    /// Binary operation
    BinaryOp {
        op_type: BinaryOpType,
        left: usize,
        right: usize,
        result: usize,
        left_partial: f64,  // ∂result/∂left
        right_partial: f64, // ∂result/∂right
    },
    /// N-ary operation (for efficiency with many inputs)
    NAryOp {
        inputs: Vec<usize>,
        result: usize,
        partials: Vec<f64>, // ∂result/∂inputs[i]
    },
}

/// Computational tape for recording operations
#[derive(Debug)]
pub struct ComputationTape {
    /// Sequence of operations in forward order
    nodes: Vec<TapeNode>,
    /// Input variables
    inputs: Vec<Variable>,
    /// Mapping from variable ID to its position in the tape
    var_positions: HashMap<usize, usize>,
    /// Maximum variable ID used
    max_var_id: usize,
}

impl ComputationTape {
    /// Create a new empty tape
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            var_positions: HashMap::new(),
            max_var_id: 0,
        }
    }

    /// Add an input variable to the tape
    pub fn add_input(&mut self, var: Variable) {
        self.var_positions.insert(var.id, self.nodes.len());
        self.max_var_id = self.max_var_id.max(var.id);

        self.nodes.push(TapeNode::Input { var_id: var.id });
        self.inputs.push(var);
    }

    /// Add a computation node to the tape
    pub fn add_node(&mut self, node: TapeNode) {
        // Update variable positions for result variables
        match &node {
            TapeNode::Constant { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            TapeNode::UnaryOp { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            TapeNode::BinaryOp { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            TapeNode::NAryOp { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            _ => {}
        }

        self.nodes.push(node);
    }

    /// Perform backpropagation to compute gradients
    pub fn backward(&self, gradients: &mut Vec<f64>) -> Result<(), OptimizeError> {
        // Ensure gradients vector is large enough
        if gradients.len() <= self.max_var_id {
            gradients.resize(self.max_var_id + 1, 0.0);
        }

        // Reverse pass through the tape
        for node in self.nodes.iter().rev() {
            match node {
                TapeNode::Input { .. } => {
                    // Input nodes don't propagate gradients backward
                }
                TapeNode::Constant { .. } => {
                    // Constants have zero gradient
                }
                TapeNode::UnaryOp {
                    op_type: _,
                    input,
                    result,
                    partial,
                } => {
                    // Propagate gradient: ∂L/∂input += ∂L/∂result * ∂result/∂input
                    // Skip constants (which have index usize::MAX)
                    if *input != usize::MAX && *input < gradients.len() {
                        gradients[*input] += gradients[*result] * partial;
                    }
                }
                TapeNode::BinaryOp {
                    op_type: _,
                    left,
                    right,
                    result,
                    left_partial,
                    right_partial,
                } => {
                    // Propagate gradients to both inputs
                    // Skip constants (which have index usize::MAX)
                    if *left != usize::MAX && *left < gradients.len() {
                        gradients[*left] += gradients[*result] * left_partial;
                    }
                    if *right != usize::MAX && *right < gradients.len() {
                        gradients[*right] += gradients[*result] * right_partial;
                    }
                }
                TapeNode::NAryOp {
                    inputs,
                    result,
                    partials,
                } => {
                    // Propagate gradient to all inputs
                    // Skip constants (which have index usize::MAX)
                    for (input_id, partial) in inputs.iter().zip(partials.iter()) {
                        if *input_id != usize::MAX && *input_id < gradients.len() {
                            gradients[*input_id] += gradients[*result] * partial;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Forward pass to compute all variable values
    pub fn forward(&self, input_values: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let mut values = vec![0.0; self.max_var_id + 1];

        // Set input values
        for (i, var) in self.inputs.iter().enumerate() {
            if i < input_values.len() {
                values[var.id] = input_values[i];
            } else {
                values[var.id] = var.value; // Use default value
            }
        }

        // Forward pass through the tape
        for node in &self.nodes {
            match node {
                TapeNode::Input { .. } => {
                    // Already handled above
                }
                TapeNode::Constant { value, result } => {
                    // Set constant value
                    values[*result] = *value;
                }
                TapeNode::UnaryOp {
                    op_type,
                    input,
                    result,
                    ..
                } => {
                    // Perform actual unary operation
                    let input_val = values[*input];
                    values[*result] = match op_type {
                        UnaryOpType::Neg => -input_val,
                        UnaryOpType::Ln => input_val.ln(),
                        UnaryOpType::Exp => input_val.exp(),
                        UnaryOpType::Sin => input_val.sin(),
                        UnaryOpType::Cos => input_val.cos(),
                        UnaryOpType::Tan => input_val.tan(),
                        UnaryOpType::Sqrt => input_val.sqrt(),
                        UnaryOpType::Square => input_val * input_val,
                        UnaryOpType::Reciprocal => 1.0 / input_val,
                    };
                }
                TapeNode::BinaryOp {
                    op_type,
                    left,
                    right,
                    result,
                    ..
                } => {
                    // Perform actual binary operation
                    let left_val = values[*left];
                    let right_val = values[*right];
                    values[*result] = match op_type {
                        BinaryOpType::Add => left_val + right_val,
                        BinaryOpType::Sub => left_val - right_val,
                        BinaryOpType::Mul => left_val * right_val,
                        BinaryOpType::Div => left_val / right_val,
                        BinaryOpType::Pow => left_val.powf(right_val),
                    };
                }
                TapeNode::NAryOp { inputs, result, .. } => {
                    // N-ary operations are application-specific
                    // For now, implement as sum (could be extended for other operations)
                    values[*result] = inputs.iter().map(|&id| values[id]).sum();
                }
            }
        }

        Ok(values)
    }

    /// Add a constant to the tape
    pub fn add_constant(&mut self, value: f64) -> usize {
        let result_id = self.max_var_id + 1;
        self.add_node(TapeNode::Constant {
            value,
            result: result_id,
        });
        result_id
    }

    /// Add a unary operation with automatic partial derivative computation
    pub fn add_unary_op(
        &mut self,
        op_type: UnaryOpType,
        input: usize,
        input_values: &[f64],
    ) -> usize {
        let result_id = self.max_var_id + 1;

        // Compute partial derivative based on operation type and current input value
        let input_val = input_values[input];
        let partial = match op_type {
            UnaryOpType::Neg => -1.0,
            UnaryOpType::Ln => 1.0 / input_val,
            UnaryOpType::Exp => input_val.exp(),
            UnaryOpType::Sin => input_val.cos(),
            UnaryOpType::Cos => -input_val.sin(),
            UnaryOpType::Tan => 1.0 + input_val.tan().powi(2), // sec^2(x)
            UnaryOpType::Sqrt => 1.0 / (2.0 * input_val.sqrt()),
            UnaryOpType::Square => 2.0 * input_val,
            UnaryOpType::Reciprocal => -1.0 / (input_val * input_val),
        };

        self.add_node(TapeNode::UnaryOp {
            op_type,
            input,
            result: result_id,
            partial,
        });

        result_id
    }

    /// Add a binary operation with automatic partial derivative computation  
    pub fn add_binary_op(
        &mut self,
        op_type: BinaryOpType,
        left: usize,
        right: usize,
        input_values: &[f64],
    ) -> usize {
        let result_id = self.max_var_id + 1;

        // Compute partial derivatives based on operation type and current input values
        let left_val = input_values[left];
        let right_val = input_values[right];

        let (left_partial, right_partial) = match op_type {
            BinaryOpType::Add => (1.0, 1.0),
            BinaryOpType::Sub => (1.0, -1.0),
            BinaryOpType::Mul => (right_val, left_val),
            BinaryOpType::Div => (1.0 / right_val, -left_val / (right_val * right_val)),
            BinaryOpType::Pow => {
                // d/dx[f^g] = f^(g-1) * (g * f' + f * ln(f) * g')
                // d/dx[x^y] = y * x^(y-1), d/dy[x^y] = x^y * ln(x)
                (
                    right_val * left_val.powf(right_val - 1.0),
                    left_val.powf(right_val) * left_val.ln(),
                )
            }
        };

        self.add_node(TapeNode::BinaryOp {
            op_type,
            left,
            right,
            result: result_id,
            left_partial,
            right_partial,
        });

        result_id
    }

    /// Forward-mode AD: compute function value and derivatives simultaneously
    pub fn forward_ad(
        &self,
        input_values: &[f64],
        seed_derivatives: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), OptimizeError> {
        let mut values = vec![0.0; self.max_var_id + 1];
        let mut derivatives = vec![0.0; self.max_var_id + 1];

        // Set input values and seed derivatives
        for (i, var) in self.inputs.iter().enumerate() {
            if i < input_values.len() {
                values[var.id] = input_values[i];
                if i < seed_derivatives.len() {
                    derivatives[var.id] = seed_derivatives[i];
                }
            } else {
                values[var.id] = var.value;
            }
        }

        // Forward pass through the tape
        for node in &self.nodes {
            match node {
                TapeNode::Input { .. } => {
                    // Already handled above
                }
                TapeNode::Constant { value, result } => {
                    // Constants have zero derivative
                    values[*result] = *value;
                    derivatives[*result] = 0.0;
                }
                TapeNode::UnaryOp {
                    op_type,
                    input,
                    result,
                    ..
                } => {
                    // Forward-mode AD for unary operations
                    let input_val = values[*input];
                    let input_deriv = derivatives[*input];

                    // Compute function value
                    values[*result] = match op_type {
                        UnaryOpType::Neg => -input_val,
                        UnaryOpType::Ln => input_val.ln(),
                        UnaryOpType::Exp => input_val.exp(),
                        UnaryOpType::Sin => input_val.sin(),
                        UnaryOpType::Cos => input_val.cos(),
                        UnaryOpType::Tan => input_val.tan(),
                        UnaryOpType::Sqrt => input_val.sqrt(),
                        UnaryOpType::Square => input_val * input_val,
                        UnaryOpType::Reciprocal => 1.0 / input_val,
                    };

                    // Compute derivative using chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
                    let f_prime = match op_type {
                        UnaryOpType::Neg => -1.0,
                        UnaryOpType::Ln => 1.0 / input_val,
                        UnaryOpType::Exp => input_val.exp(),
                        UnaryOpType::Sin => input_val.cos(),
                        UnaryOpType::Cos => -input_val.sin(),
                        UnaryOpType::Tan => 1.0 + input_val.tan().powi(2),
                        UnaryOpType::Sqrt => 1.0 / (2.0 * input_val.sqrt()),
                        UnaryOpType::Square => 2.0 * input_val,
                        UnaryOpType::Reciprocal => -1.0 / (input_val * input_val),
                    };
                    derivatives[*result] = f_prime * input_deriv;
                }
                TapeNode::BinaryOp {
                    op_type,
                    left,
                    right,
                    result,
                    ..
                } => {
                    // Forward-mode AD for binary operations
                    let left_val = values[*left];
                    let right_val = values[*right];
                    let left_deriv = derivatives[*left];
                    let right_deriv = derivatives[*right];

                    // Compute function value
                    values[*result] = match op_type {
                        BinaryOpType::Add => left_val + right_val,
                        BinaryOpType::Sub => left_val - right_val,
                        BinaryOpType::Mul => left_val * right_val,
                        BinaryOpType::Div => left_val / right_val,
                        BinaryOpType::Pow => left_val.powf(right_val),
                    };

                    // Compute derivative using product rule and chain rule
                    derivatives[*result] = match op_type {
                        BinaryOpType::Add => left_deriv + right_deriv,
                        BinaryOpType::Sub => left_deriv - right_deriv,
                        BinaryOpType::Mul => left_deriv * right_val + left_val * right_deriv,
                        BinaryOpType::Div => {
                            (left_deriv * right_val - left_val * right_deriv)
                                / (right_val * right_val)
                        }
                        BinaryOpType::Pow => {
                            // d/dx[f^g] = f^g * (g' * ln(f) + g * f'/f)
                            let result_val = left_val.powf(right_val);
                            result_val
                                * (right_deriv * left_val.ln() + right_val * left_deriv / left_val)
                        }
                    };
                }
                TapeNode::NAryOp {
                    inputs,
                    result,
                    partials,
                } => {
                    // N-ary operations: sum for now
                    values[*result] = inputs.iter().map(|&id| values[id]).sum();
                    derivatives[*result] = inputs
                        .iter()
                        .enumerate()
                        .map(|(i, &id)| partials.get(i).unwrap_or(&1.0) * derivatives[id])
                        .sum();
                }
            }
        }

        Ok((values, derivatives))
    }

    /// Optimize the tape by removing unnecessary operations
    pub fn optimize(&mut self) {
        // Remove redundant operations, constant folding, etc.
        // This is a placeholder for more sophisticated optimizations

        // Remove nodes that are never used
        let mut used_vars = std::collections::HashSet::new();

        // Mark all variables that are actually used
        for node in &self.nodes {
            match node {
                TapeNode::UnaryOp { input, result, .. } => {
                    used_vars.insert(*input);
                    used_vars.insert(*result);
                }
                TapeNode::BinaryOp {
                    left,
                    right,
                    result,
                    ..
                } => {
                    used_vars.insert(*left);
                    used_vars.insert(*right);
                    used_vars.insert(*result);
                }
                TapeNode::NAryOp { inputs, result, .. } => {
                    for &input_id in inputs {
                        used_vars.insert(input_id);
                    }
                    used_vars.insert(*result);
                }
                TapeNode::Input { var_id } => {
                    used_vars.insert(*var_id);
                }
                _ => {}
            }
        }

        // Could implement more optimizations here
    }

    /// Get the size of the tape
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.inputs.clear();
        self.var_positions.clear();
        self.max_var_id = 0;
    }

    /// Get statistics about the tape
    pub fn get_stats(&self) -> TapeStats {
        let mut unary_ops = 0;
        let mut binary_ops = 0;
        let mut nary_ops = 0;
        let mut constants = 0;

        for node in &self.nodes {
            match node {
                TapeNode::Input { .. } => {}
                TapeNode::Constant { .. } => constants += 1,
                TapeNode::UnaryOp { .. } => unary_ops += 1,
                TapeNode::BinaryOp { .. } => binary_ops += 1,
                TapeNode::NAryOp { .. } => nary_ops += 1,
            }
        }

        TapeStats {
            total_nodes: self.nodes.len(),
            input_vars: self.inputs.len(),
            unary_ops,
            binary_ops,
            nary_ops,
            constants,
            max_var_id: self.max_var_id,
        }
    }
}

impl Default for ComputationTape {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a computation tape
#[derive(Debug, Clone)]
pub struct TapeStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of input variables
    pub input_vars: usize,
    /// Number of unary operations
    pub unary_ops: usize,
    /// Number of binary operations
    pub binary_ops: usize,
    /// Number of n-ary operations
    pub nary_ops: usize,
    /// Number of constants
    pub constants: usize,
    /// Maximum variable ID
    pub max_var_id: usize,
}

/// Tape builder for more convenient tape construction
pub struct TapeBuilder {
    tape: ComputationTape,
    next_var_id: usize,
}

impl TapeBuilder {
    /// Create a new tape builder
    pub fn new() -> Self {
        Self {
            tape: ComputationTape::new(),
            next_var_id: 0,
        }
    }

    /// Add an input variable
    pub fn input(&mut self, value: f64) -> usize {
        let var_id = self.next_var_id;
        self.next_var_id += 1;

        let var = Variable::new(var_id, value);
        self.tape.add_input(var);

        var_id
    }

    /// Add a unary operation
    pub fn unary_op(&mut self, op_type: UnaryOpType, input: usize, partial: f64) -> usize {
        let result_id = self.next_var_id;
        self.next_var_id += 1;

        let node = TapeNode::UnaryOp {
            op_type,
            input,
            result: result_id,
            partial,
        };
        self.tape.add_node(node);

        result_id
    }

    /// Add a binary operation
    pub fn binary_op(
        &mut self,
        op_type: BinaryOpType,
        left: usize,
        right: usize,
        left_partial: f64,
        right_partial: f64,
    ) -> usize {
        let result_id = self.next_var_id;
        self.next_var_id += 1;

        let node = TapeNode::BinaryOp {
            op_type,
            left,
            right,
            result: result_id,
            left_partial,
            right_partial,
        };
        self.tape.add_node(node);

        result_id
    }

    /// Finish building and return the tape
    pub fn build(self) -> ComputationTape {
        self.tape
    }
}

impl Default for TapeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-efficient tape that can handle very large computations
pub struct StreamingTape {
    /// Current batch of operations
    current_batch: Vec<TapeNode>,
    /// Batch size for processing
    batch_size: usize,
    /// Function to process completed batches
    batch_processor: Option<BatchProcessor>,
}

impl StreamingTape {
    /// Create a new streaming tape
    pub fn new(batch_size: usize) -> Self {
        Self {
            current_batch: Vec::with_capacity(batch_size),
            batch_size,
            batch_processor: None,
        }
    }

    /// Set the batch processor
    pub fn set_batch_processor<F>(&mut self, processor: F)
    where
        F: Fn(&[TapeNode]) -> Result<(), OptimizeError> + 'static,
    {
        self.batch_processor = Some(Box::new(processor));
    }

    /// Add a node to the streaming tape
    pub fn add_node(&mut self, node: TapeNode) -> Result<(), OptimizeError> {
        self.current_batch.push(node);

        if self.current_batch.len() >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    /// Flush the current batch
    pub fn flush_batch(&mut self) -> Result<(), OptimizeError> {
        if let Some(ref processor) = self.batch_processor {
            processor(&self.current_batch)?;
        }
        self.current_batch.clear();
        Ok(())
    }

    /// Finalize the streaming tape
    pub fn finalize(&mut self) -> Result<(), OptimizeError> {
        if !self.current_batch.is_empty() {
            self.flush_batch()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_construction() {
        let mut builder = TapeBuilder::new();

        // Build tape for: z = (x + y) * x
        let x = builder.input(2.0);
        let y = builder.input(3.0);
        let sum = builder.binary_op(BinaryOpType::Add, x, y, 1.0, 1.0); // x + y, partials: ∂/∂x=1, ∂/∂y=1
        let _result = builder.binary_op(BinaryOpType::Mul, sum, x, 2.0, 5.0); // sum * x, partials: ∂/∂sum=x=2, ∂/∂x=sum=5

        let tape = builder.build();

        assert_eq!(tape.size(), 4); // 2 inputs + 2 operations

        let stats = tape.get_stats();
        assert_eq!(stats.input_vars, 2);
        assert_eq!(stats.binary_ops, 2);
    }

    #[test]
    fn test_backward_pass() {
        let mut tape = ComputationTape::new();

        // Add inputs: x=2, y=3
        tape.add_input(Variable::new(0, 2.0));
        tape.add_input(Variable::new(1, 3.0));

        // Add operation: z = x + y (result=2)
        tape.add_node(TapeNode::BinaryOp {
            op_type: BinaryOpType::Add,
            left: 0,
            right: 1,
            result: 2,
            left_partial: 1.0,  // ∂z/∂x = 1
            right_partial: 1.0, // ∂z/∂y = 1
        });

        // Initialize gradients: ∂L/∂z = 1 (z is the output)
        let mut gradients = vec![0.0, 0.0, 1.0];

        tape.backward(&mut gradients).expect("Operation failed");

        // Check gradients
        assert_eq!(gradients[0], 1.0); // ∂L/∂x = ∂L/∂z * ∂z/∂x = 1 * 1 = 1
        assert_eq!(gradients[1], 1.0); // ∂L/∂y = ∂L/∂z * ∂z/∂y = 1 * 1 = 1
    }

    #[test]
    fn test_tape_optimization() {
        let mut tape = ComputationTape::new();

        tape.add_input(Variable::new(0, 1.0));
        tape.add_node(TapeNode::UnaryOp {
            op_type: UnaryOpType::Neg,
            input: 0,
            result: 1,
            partial: 1.0,
        });

        let original_size = tape.size();
        tape.optimize();

        // Optimization might not change this simple tape, but it shouldn't break it
        assert!(tape.size() <= original_size);
    }

    #[test]
    fn test_streaming_tape() {
        let mut streaming_tape = StreamingTape::new(2);

        streaming_tape.set_batch_processor(move |_batch| {
            // Just a placeholder processor for testing
            Ok(())
        });

        // Add nodes - should trigger batch processing
        streaming_tape
            .add_node(TapeNode::Input { var_id: 0 })
            .expect("Operation failed");
        streaming_tape
            .add_node(TapeNode::Input { var_id: 1 })
            .expect("Operation failed");

        // This should have triggered one batch
        streaming_tape
            .add_node(TapeNode::UnaryOp {
                op_type: UnaryOpType::Neg,
                input: 0,
                result: 2,
                partial: 1.0,
            })
            .expect("Operation failed");

        streaming_tape.finalize().expect("Operation failed");
    }

    #[test]
    fn test_tape_stats() {
        let mut builder = TapeBuilder::new();

        let x = builder.input(1.0);
        let y = builder.input(2.0);
        builder.binary_op(BinaryOpType::Add, x, y, 1.0, 1.0);
        builder.unary_op(UnaryOpType::Neg, x, 2.0);

        let tape = builder.build();
        let stats = tape.get_stats();

        assert_eq!(stats.input_vars, 2);
        assert_eq!(stats.binary_ops, 1);
        assert_eq!(stats.unary_ops, 1);
        assert_eq!(stats.total_nodes, 4); // 2 inputs + 1 binary + 1 unary
    }
}
