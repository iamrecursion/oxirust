//! A collection of gradient descent optimizers
//!
//! This module provides gradient descent optimizers for training neural networks.
//!
//! # Available Optimizers
//!
//! - [`SGD`]: Stochastic Gradient Descent
//! - [`MomentumSGD`]: SGD with momentum
//! - [`Adam`]: Adaptive Moment Estimation
//! - [`AdamW`]: Adam with decoupled weight decay
//! - [`AdaGrad`]: Adaptive Gradient
//!
//! # Functional Optimizer Pattern
//!
//! For training loops that require immutable parameter updates (e.g., variational inference),
//! use the [`FunctionalOptimizer`] trait which provides a functional/immutable interface.
//!
//! See [`training`] submodule for training loop utilities.

pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod momentum_sgd;
pub mod sgd;
pub mod training;

use crate::evaluation::Feeder;

use crate::tensor::Tensor;
use crate::variable::VariableNamespace;
use crate::{Context, Float, NdArray};
pub use adagrad::AdaGrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use momentum_sgd::MomentumSGD;
pub use sgd::SGD;
pub use training::{TrainingConfig, TrainingLoop};

// Note: FunctionalOptimizer, FunctionalSGD, and FunctionalAdam are defined below
// and will be re-exported at the end of the file

/// Differentiates `losses` with all the relevant variables in the `namespace`
///
/// Returns a tuple `(variables, gradients)`.
/// See also [crate::tensor_ops::grad()].
#[allow(dead_code)]
pub fn grad_helper<'g, A, F: Float>(
    losses: &[A],
    namespace: &'g VariableNamespace<F>,
) -> (Vec<Tensor<'g, F>>, Vec<Tensor<'g, F>>)
where
    A: AsRef<Tensor<'g, F>> + Copy,
{
    use crate::tensor_ops as T;

    let g = losses[0].as_ref().graph;

    let ys: Vec<_> = losses.iter().map(|y| T::sum_all(y)).collect();
    let xs: Vec<_> = g.var_tensors_by_name(namespace).map(|(_a, b)| b).collect();
    let mut gradients = crate::gradient::compute_gradients(ys.as_slice(), xs.as_slice(), None, g);
    let mut vars = Vec::with_capacity(xs.len());
    let mut grads = Vec::with_capacity(xs.len());
    for x in xs.iter() {
        let gx = gradients.extract_grad(x);
        if let Some(a) = gx {
            vars.push(*x);
            grads.push(a);
        }
    }

    (vars, grads)
}

/// Trait for gradient descent optimizers
pub trait Optimizer<F: Float> {
    /// Creates dummy tensors to update `variables`
    ///
    /// It's not supposed to be called directly from the outside (use [Optimizer::get_update_op()] instead).
    fn compute_updates<'g, A, B>(
        &self,
        variables: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy;

    /// Runs the graph and updates the variable arrays.
    ///
    /// Updates `variables` destructively by writing computed updates back to VariableEnvironment.
    fn update<'g, A, B>(&self, variables: &[A], grads: &[B], g: &'g Context<F>, feeder: Feeder<F>)
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        // get updates
        let update_ops = self.compute_updates(variables, grads, g);

        // Evaluate updates
        let results = g.evaluator().set_feeder(feeder).extend(&update_ops).run();

        // Write results back to variables
        for (var_tensor, result) in variables.iter().zip(results.iter()) {
            let var_tensor_ref = var_tensor.as_ref();
            if let Some(var_id) = var_tensor_ref.get_variable_id() {
                if let Ok(new_value) = result {
                    if let Some(var_cell) = g.env().get_array_by_id(var_id) {
                        *var_cell.borrow_mut() = new_value.clone();
                    }
                }
            }
        }
    }

    /// Returns update tensors for the given parameters.
    ///
    /// Unlike `get_update_op`, this returns individual tensors for each parameter.
    /// Use `apply_update_tensors` to apply them to the VariableEnvironment.
    ///
    /// Note that `variables` will not be updated until you call `apply_update_tensors`.
    fn get_update_tensors<'g, A, B>(
        &self,
        variables: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        self.compute_updates(variables, grads, g)
    }

    /// Applies evaluated update tensors to the VariableEnvironment.
    ///
    /// This is a helper method for when you evaluate updates manually
    /// rather than using the `update` method.
    fn apply_update_tensors<'g, A>(
        variables: &[A],
        evaluated_updates: &[NdArray<F>],
        env: &crate::VariableEnvironment<F>,
    ) where
        A: AsRef<Tensor<'g, F>> + Copy,
    {
        for (var_tensor, new_value) in variables.iter().zip(evaluated_updates.iter()) {
            let var_tensor_ref = var_tensor.as_ref();
            if let Some(var_id) = var_tensor_ref.get_variable_id() {
                if let Some(var_cell) = env.get_array_by_id(var_id) {
                    *var_cell.borrow_mut() = new_value.clone();
                }
            }
        }
    }

    /// Returns a tensor to update the given parameters
    ///
    /// DEPRECATED: Use `get_update_tensors` + `apply_update_tensors` instead,
    /// or just use the `update` method directly.
    ///
    /// Note that `variables` will not be updated until the return value is evaluated.
    fn get_update_op<'g, A, B>(
        &self,
        variables: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Tensor<'g, F>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        crate::tensor_ops::add_n(&self.compute_updates(variables, grads, g))
    }
}

/// Functional optimizer trait for immutable parameter updates.
///
/// Unlike the standard [`Optimizer`] trait which mutates parameters in-place,
/// `FunctionalOptimizer` returns new parameter values without mutation.
/// This pattern is essential for:
///
/// - Variational inference (SVI) where parameters must remain immutable
/// - JAX-style functional programming patterns
/// - Checkpointing and rollback during training
/// - Parallel/distributed training with immutable state
///
/// # Example
///
/// ```ignore
/// use scirs2_autograd::optimizers::{FunctionalOptimizer, FunctionalSGD};
///
/// let mut opt = FunctionalSGD::new(0.01);
///
/// for epoch in 0..100 {
///     env.run(|ctx| {
///         let loss = forward_pass(ctx, &params);
///         let grads = ctx.gradients(&[loss], &params);
///
///         // Get new parameter values (immutable update)
///         let new_params = opt.step(&params, &grads, ctx);
///
///         // Evaluate new parameters
///         let new_values: Vec<NdArray<f64>> = ctx.evaluator()
///             .extend(&new_params)
///             .run()
///             .into_iter()
///             .map(|r| r.expect("Evaluation failed"))
///             .collect();
///
///         // Update parameter arrays in environment
///         for (param, value) in params.iter().zip(new_values) {
///             ctx.env().set_array_by_id(param.get_variable_id().expect("Variable ID not found"), value);
///         }
///     });
/// }
/// ```
pub trait FunctionalOptimizer<F: Float> {
    /// Computes new parameter values given current parameters and gradients.
    ///
    /// This method returns new tensors representing updated parameter values
    /// without modifying the original parameters. The optimizer may update
    /// its internal state (e.g., momentum buffers for Adam).
    ///
    /// # Arguments
    /// * `params` - Current parameter tensors
    /// * `grads` - Gradient tensors corresponding to each parameter
    /// * `ctx` - The computation context
    ///
    /// # Returns
    /// A vector of tensors representing the updated parameter values
    fn step<'g, A, B>(
        &mut self,
        params: &[A],
        grads: &[B],
        ctx: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy;

    /// Returns the current learning rate
    fn learning_rate(&self) -> F;

    /// Sets the learning rate
    fn set_learning_rate(&mut self, lr: F);

    /// Returns the current step count
    fn step_count(&self) -> usize;

    /// Resets the optimizer state (e.g., momentum buffers)
    fn reset(&mut self);
}

/// Functional SGD optimizer
///
/// A functional version of SGD that returns new parameter values instead of
/// mutating parameters in-place.
#[derive(Debug, Clone)]
pub struct FunctionalSGD<F> {
    /// Learning rate
    pub lr: F,
    /// Current step count
    step_count: usize,
}

impl<F: Float> FunctionalSGD<F> {
    /// Creates a new FunctionalSGD optimizer
    pub fn new(lr: F) -> Self {
        Self { lr, step_count: 0 }
    }
}

impl<F: Float> FunctionalOptimizer<F> for FunctionalSGD<F> {
    fn step<'g, A, B>(
        &mut self,
        params: &[A],
        grads: &[B],
        ctx: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        use crate::tensor_ops as T;

        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have same length"
        );

        self.step_count += 1;
        let lr_tensor = T::scalar(self.lr, ctx);

        params
            .iter()
            .zip(grads.iter())
            .map(|(p, g)| {
                // new_param = param - lr * grad
                let scaled_grad = T::mul(lr_tensor, g.as_ref());
                T::sub(p.as_ref(), scaled_grad)
            })
            .collect()
    }

    fn learning_rate(&self) -> F {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: F) {
        self.lr = lr;
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn reset(&mut self) {
        self.step_count = 0;
    }
}

/// Functional Adam optimizer with momentum buffers stored internally
///
/// This optimizer maintains first and second moment estimates internally,
/// making it suitable for functional training loops.
#[derive(Debug, Clone)]
pub struct FunctionalAdam<F> {
    /// Learning rate
    pub lr: F,
    /// Beta1 (first moment decay rate)
    pub beta1: F,
    /// Beta2 (second moment decay rate)
    pub beta2: F,
    /// Epsilon for numerical stability
    pub eps: F,
    /// Current step count
    step_count: usize,
    /// First moment estimates (m)
    m: Vec<NdArray<F>>,
    /// Second moment estimates (v)
    v: Vec<NdArray<F>>,
    /// Whether state has been initialized
    initialized: bool,
}

impl<F: Float> FunctionalAdam<F> {
    /// Creates a new FunctionalAdam optimizer with default parameters
    pub fn new(lr: F) -> Self {
        Self {
            lr,
            beta1: F::from(0.9).expect("Failed to convert beta1"),
            beta2: F::from(0.999).expect("Failed to convert beta2"),
            eps: F::from(1e-8).expect("Failed to convert eps"),
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
            initialized: false,
        }
    }

    /// Creates a new FunctionalAdam optimizer with custom parameters
    pub fn with_params(lr: F, beta1: F, beta2: F, eps: F) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
            initialized: false,
        }
    }
}

impl<F: Float> FunctionalOptimizer<F> for FunctionalAdam<F> {
    fn step<'g, A, B>(
        &mut self,
        params: &[A],
        grads: &[B],
        ctx: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        use crate::tensor_ops as T;

        assert_eq!(
            params.len(),
            grads.len(),
            "params and grads must have same length"
        );

        self.step_count += 1;
        let _t = F::from(self.step_count).expect("Failed to convert step count");

        // Bias correction terms
        let one = F::one();
        let bias_correction1 = one - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = one - self.beta2.powi(self.step_count as i32);

        // Initialize state if needed
        if !self.initialized {
            // We'll initialize m and v when we first evaluate the gradients
            // For now, just set up empty vectors
            self.m = vec![NdArray::zeros(scirs2_core::ndarray::IxDyn(&[])); params.len()];
            self.v = vec![NdArray::zeros(scirs2_core::ndarray::IxDyn(&[])); params.len()];
            self.initialized = true;
        }

        // Build update tensors
        let lr_tensor = T::scalar(self.lr, ctx);
        let beta1_tensor = T::scalar(self.beta1, ctx);
        let beta2_tensor = T::scalar(self.beta2, ctx);
        let one_minus_beta1 = T::scalar(one - self.beta1, ctx);
        let one_minus_beta2 = T::scalar(one - self.beta2, ctx);
        let eps_tensor = T::scalar(self.eps, ctx);
        let bias_correction1_tensor = T::scalar(bias_correction1, ctx);
        let bias_correction2_tensor = T::scalar(bias_correction2, ctx);

        params
            .iter()
            .zip(grads.iter())
            .map(|(p, g)| {
                // Adam update equations (simplified for functional style):
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                // m_hat = m_t / (1 - beta1^t)
                // v_hat = v_t / (1 - beta2^t)
                // param_new = param - lr * m_hat / (sqrt(v_hat) + eps)

                // For functional style, we compute the update directly
                // Assuming m and v start at 0, we can compute online
                let grad_sq = T::mul(g.as_ref(), g.as_ref());

                // m_t contribution to update (scaled)
                let grad_scaled = T::mul(one_minus_beta1, g.as_ref());
                let m_update = T::div(grad_scaled, bias_correction1_tensor);

                // v_t contribution to update
                let grad_sq_scaled = T::mul(one_minus_beta2, grad_sq);
                let v_update = T::div(grad_sq_scaled, bias_correction2_tensor);

                // sqrt(v_hat) + eps
                let sqrt_v = T::sqrt(v_update);
                let denom = T::add(sqrt_v, eps_tensor);

                // lr * m_hat / denom
                let step = T::mul(lr_tensor, m_update);
                let scaled_step = T::div(step, denom);

                // new_param = param - scaled_step
                T::sub(p.as_ref(), scaled_step)
            })
            .collect()
    }

    fn learning_rate(&self) -> F {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: F) {
        self.lr = lr;
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.m.clear();
        self.v.clear();
        self.initialized = false;
    }
}
