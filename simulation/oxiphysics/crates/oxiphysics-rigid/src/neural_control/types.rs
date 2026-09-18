//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::backprop::log_softmax_gradient;
use super::functions::*;
use crate::rng_util::uniform;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;
/// Online running-mean / running-variance normaliser (Welford's algorithm).
#[derive(Debug, Clone)]
pub struct RunningNormaliser {
    /// Number of samples seen.
    pub count: usize,
    /// Running mean.
    pub mean: Vec<f64>,
    /// Running M2 (for variance).
    pub m2: Vec<f64>,
    /// Dimension.
    pub dim: usize,
}
impl RunningNormaliser {
    /// Create a normaliser for `dim`-dimensional inputs.
    pub fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            dim,
        }
    }
    /// Update statistics with a new sample.
    pub fn update(&mut self, x: &[f64]) {
        self.count += 1;
        for (i, (mean, m2)) in self.mean.iter_mut().zip(self.m2.iter_mut()).enumerate() {
            let xi = x[i];
            let delta = xi - *mean;
            *mean += delta / self.count as f64;
            let delta2 = xi - *mean;
            *m2 += delta * delta2;
        }
    }
    /// Normalise a sample: (x - mean) / std.
    pub fn normalise(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let variance = if self.count > 1 {
                    self.m2[i] / (self.count - 1) as f64
                } else {
                    1.0
                };
                (xi - self.mean[i]) / variance.sqrt().max(1e-8)
            })
            .collect()
    }
}
/// Feedforward neural network controller that maps a state vector to an action vector.
///
/// Architecture: `state_dim → [hidden_1, hidden_2?] → action_dim`.
/// Output is passed through `tanh` and scaled by `action_scale` so that
/// actions stay bounded.
#[derive(Debug, Clone)]
pub struct NeuralController {
    /// Hidden layers (weights + biases).
    pub layers: Vec<DenseLayer>,
    /// Output layer.
    pub output_layer: DenseLayer,
    /// Hidden layer activation.
    pub activation: Activation,
    /// Multiplier applied to the tanh output.
    pub action_scale: f64,
    /// Dimensionality of the input state.
    pub state_dim: usize,
    /// Dimensionality of the action output.
    pub action_dim: usize,
}
impl NeuralController {
    /// Build a controller with one or two hidden layers.
    ///
    /// * `state_dim`    — length of the state vector
    /// * `hidden_sizes` — sizes of the hidden layers (1–2 entries)
    /// * `action_dim`   — length of the action vector
    /// * `action_scale` — output multiplier (e.g. max torque in N·m)
    pub fn new(
        state_dim: usize,
        hidden_sizes: &[usize],
        action_dim: usize,
        action_scale: f64,
    ) -> Self {
        let activation = Activation::Tanh;
        let mut layers = Vec::new();
        let mut prev = state_dim;
        for &h in hidden_sizes {
            layers.push(DenseLayer::new(prev, h));
            prev = h;
        }
        let output_layer = DenseLayer::new(prev, action_dim);
        Self {
            layers,
            output_layer,
            activation,
            action_scale,
            state_dim,
            action_dim,
        }
    }
    /// Run the network forward.  Returns bounded actions ∈ (-action_scale, +action_scale).
    pub fn forward(&self, state: &[f64]) -> Vec<f64> {
        let mut x = state.to_vec();
        for layer in &self.layers {
            x = match self.activation {
                Activation::Relu => layer.forward_relu(&x),
                Activation::Tanh => layer.forward_tanh(&x),
                Activation::Sigmoid => {
                    let pre = layer.forward_linear(&x);
                    pre.into_iter().map(sigmoid).collect()
                }
                Activation::Linear => layer.forward_linear(&x),
            };
        }
        let raw = self.output_layer.forward_linear(&x);
        raw.into_iter()
            .map(|v| tanh_act(v) * self.action_scale)
            .collect()
    }
    /// Collect all parameters (weights + biases) as a flat vector.
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for layer in &self.layers {
            for row in &layer.weights {
                params.extend_from_slice(row);
            }
            params.extend_from_slice(&layer.biases);
        }
        for row in &self.output_layer.weights {
            params.extend_from_slice(row);
        }
        params.extend_from_slice(&self.output_layer.biases);
        params
    }
    /// Set all parameters from a flat vector (same order as `get_parameters`).
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;
        for layer in &mut self.layers {
            for row in &mut layer.weights {
                for w in row.iter_mut() {
                    *w = params[idx];
                    idx += 1;
                }
            }
            for b in &mut layer.biases {
                *b = params[idx];
                idx += 1;
            }
        }
        for row in &mut self.output_layer.weights {
            for w in row.iter_mut() {
                *w = params[idx];
                idx += 1;
            }
        }
        for b in &mut self.output_layer.biases {
            *b = params[idx];
            idx += 1;
        }
    }
    /// Count total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .chain(std::iter::once(&self.output_layer))
            .map(|layer| layer.weights.iter().map(Vec::len).sum::<usize>() + layer.biases.len())
            .sum()
    }
    /// Apply a gradient step (SGD) given a flat gradient vector and learning rate.
    pub fn sgd_step(&mut self, grad: &[f64], lr: f64) {
        let mut params = self.get_parameters();
        for (p, g) in params.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
        self.set_parameters(&params);
    }
}
/// Policy gradient (REINFORCE) and actor-critic reinforcement learning.
#[derive(Debug, Clone)]
pub struct ReinforcementLearning {
    /// Actor network (policy).
    pub actor: NeuralController,
    /// Critic network (value function).
    pub critic: NeuralController,
    /// Discount factor γ.
    pub gamma: f64,
    /// Actor learning rate.
    pub actor_lr: f64,
    /// Critic learning rate.
    pub critic_lr: f64,
    /// Episode trajectory storage: (state, action_index, log_prob, reward).
    pub trajectory: Vec<(Vec<f64>, usize, f64, f64)>,
}
impl ReinforcementLearning {
    /// Construct an RL agent with actor and critic networks.
    pub fn new(
        state_dim: usize,
        hidden: &[usize],
        action_dim: usize,
        gamma: f64,
        actor_lr: f64,
        critic_lr: f64,
    ) -> Self {
        let actor = NeuralController::new(state_dim, hidden, action_dim, 1.0);
        let critic = NeuralController::new(state_dim, hidden, 1, 1.0);
        Self {
            actor,
            critic,
            gamma,
            actor_lr,
            critic_lr,
            trajectory: Vec::new(),
        }
    }
    /// Sample an action from the actor's softmax policy.
    /// Returns (action index, log probability).
    pub fn sample_action(&self, state: &[f64]) -> (usize, f64) {
        let mut rng = rand::rng();
        self.sample_action_with_rng(state, &mut rng)
    }
    /// Like [`sample_action`](Self::sample_action), but draws from `rng`; a
    /// seeded generator makes the sampled actions reproducible.
    pub fn sample_action_with_rng<R: rand::Rng + ?Sized>(
        &self,
        state: &[f64],
        rng: &mut R,
    ) -> (usize, f64) {
        let logits = self.actor.forward(state);
        let probs = softmax(&logits);
        let sample: f64 = rng.random_range(0.0..1.0);
        let mut cumsum = 0.0;
        let mut action = probs.len().saturating_sub(1);
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                action = i;
                break;
            }
        }
        let log_prob = (probs.get(action).copied().unwrap_or(0.0) + 1e-8).ln();
        (action, log_prob)
    }
    /// Store a step in the current episode trajectory.
    pub fn record_step(&mut self, state: Vec<f64>, action: usize, log_prob: f64, reward: f64) {
        self.trajectory.push((state, action, log_prob, reward));
    }
    /// Compute discounted returns for the stored trajectory.
    pub fn compute_returns(&self) -> Vec<f64> {
        let n = self.trajectory.len();
        let mut returns = vec![0.0; n];
        let mut running = 0.0;
        for i in (0..n).rev() {
            running = self.trajectory[i].3 + self.gamma * running;
            returns[i] = running;
        }
        returns
    }
    /// REINFORCE policy gradient update step.
    ///
    /// The advantage of a step is its normalised discounted return.  The actor
    /// descends `-(1/T) · Σₜ Aₜ ∇θ log π(aₜ | sₜ)`, where the gradient of the
    /// log-probability of the softmax policy is computed exactly by
    /// back-propagation (`∂ log πₐ/∂logits = onehot(a) − π`), at the cost of
    /// one forward and one backward pass per step.  Steps whose action index
    /// is not a valid action of the actor are ignored.
    /// Clears the trajectory after updating.
    pub fn reinforce_update(&mut self) {
        let returns = self.compute_returns();
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let std_ret: f64 = {
            let var: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                / returns.len().max(1) as f64;
            var.sqrt().max(1e-8)
        };
        let n_params = self.actor.num_parameters();
        let mut grad = vec![0.0_f64; n_params];
        for ((state, action_idx, _log_prob, _reward), ret) in
            self.trajectory.iter().zip(returns.iter())
        {
            let cache = self.actor.forward_cached(state);
            let Some(d_logits) = log_softmax_gradient(&cache.output, *action_idx) else {
                continue;
            };
            let advantage = (ret - mean_ret) / std_ret;
            let step_grad = self.actor.backward_cached(&cache, &d_logits);
            for (total, g) in grad.iter_mut().zip(step_grad.iter()) {
                *total -= advantage * g;
            }
        }
        let scale = 1.0 / self.trajectory.len().max(1) as f64;
        let grad_scaled: Vec<f64> = grad.iter().map(|g| g * scale).collect();
        self.actor.sgd_step(&grad_scaled, self.actor_lr);
        self.trajectory.clear();
    }
    /// Advantage actor-critic (A2C) update for a single transition.
    ///
    /// The critic descends the squared TD error `(r + γ V(s') − V(s))²` with the
    /// target held constant, `∇θ = −2 · advantage · ∇θ V(s)`; the actor descends
    /// `−advantage · log π(a | s)`.  Both gradients are exact (back-propagation,
    /// one forward and one backward pass per network).  The actor is not
    /// updated when `action` is not a valid action of the actor.
    pub fn a2c_update(
        &mut self,
        state: &[f64],
        action: usize,
        reward: f64,
        next_state: &[f64],
        done: bool,
    ) {
        let critic_cache = self.critic.forward_cached(state);
        let value = critic_cache.output.first().copied().unwrap_or(0.0);
        let next_value = if done {
            0.0
        } else {
            self.critic
                .forward(next_state)
                .first()
                .copied()
                .unwrap_or(0.0)
        };
        let td_target = reward + self.gamma * next_value;
        let advantage = td_target - value;
        let c_grad: Vec<f64> = self
            .critic
            .backward_cached(&critic_cache, &[1.0])
            .into_iter()
            .map(|dv_dw| -2.0 * advantage * dv_dw)
            .collect();
        self.critic.sgd_step(&c_grad, self.critic_lr);
        let actor_cache = self.actor.forward_cached(state);
        if let Some(d_logits) = log_softmax_gradient(&actor_cache.output, action) {
            let a_grad: Vec<f64> = self
                .actor
                .backward_cached(&actor_cache, &d_logits)
                .into_iter()
                .map(|dlp| -advantage * dlp)
                .collect();
            self.actor.sgd_step(&a_grad, self.actor_lr);
        }
    }
}
/// A single (s, a, r, s') transition.
#[derive(Debug, Clone)]
pub struct Transition {
    /// State before action.
    pub state: Vec<f64>,
    /// Action taken.
    pub action: usize,
    /// Reward received.
    pub reward: f64,
    /// Next state.
    pub next_state: Vec<f64>,
    /// Terminal flag.
    pub done: bool,
}
/// A single fully-connected layer with bias and activation.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weight matrix (out_size × in_size).
    pub weights: Vec<Vec<f64>>,
    /// Bias vector (out_size).
    pub biases: Vec<f64>,
}
impl DenseLayer {
    /// Construct a layer with Xavier-uniform initialisation.
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let limit = (6.0 / (in_size + out_size) as f64).sqrt();
        let mut rng = rand::rng();
        let weights = (0..out_size)
            .map(|_| {
                (0..in_size)
                    .map(|_| uniform(&mut rng, -limit, limit))
                    .collect()
            })
            .collect();
        let biases = vec![0.0; out_size];
        Self { weights, biases }
    }
    /// Forward pass with ReLU activation.
    pub fn forward_relu(&self, input: &[f64]) -> Vec<f64> {
        let pre: Vec<f64> = matvec(&self.weights, input)
            .into_iter()
            .zip(self.biases.iter())
            .map(|(z, b)| relu(z + b))
            .collect();
        pre
    }
    /// Forward pass with tanh activation.
    pub fn forward_tanh(&self, input: &[f64]) -> Vec<f64> {
        matvec(&self.weights, input)
            .into_iter()
            .zip(self.biases.iter())
            .map(|(z, b)| tanh_act(z + b))
            .collect()
    }
    /// Forward pass with linear (identity) activation.
    pub fn forward_linear(&self, input: &[f64]) -> Vec<f64> {
        matvec(&self.weights, input)
            .into_iter()
            .zip(self.biases.iter())
            .map(|(z, b)| z + b)
            .collect()
    }
    /// Number of output neurons.
    pub fn out_size(&self) -> usize {
        self.biases.len()
    }
    /// Number of input neurons.
    pub fn in_size(&self) -> usize {
        if self.weights.is_empty() {
            0
        } else {
            self.weights[0].len()
        }
    }
}
/// Behavioural cloning, DAgger, and inverse reinforcement learning.
#[derive(Debug, Clone)]
pub struct ImitationLearning {
    /// Policy network being trained.
    pub policy: NeuralController,
    /// Dataset of demonstrations.
    pub dataset: Vec<Demonstration>,
    /// Learning rate for supervised updates.
    pub lr: f64,
    /// Iteration counter (used for DAgger mixing schedule).
    pub iter: usize,
    /// Reward weight vector for IRL (linear reward model: w · φ(s)).
    pub reward_weights: Vec<f64>,
    /// Feature dimension for IRL.
    pub feature_dim: usize,
    /// Random source of the mini-batch sampling in [`bc_step`](Self::bc_step);
    /// seeded from the operating system by [`new`](Self::new), reproducible
    /// after [`with_seed`](Self::with_seed).
    ///
    /// A [`SmallRng`] rather than a `StdRng`: this struct is `Clone`, and the
    /// standard generator of `rand` 0.10 deliberately is not.
    pub rng: SmallRng,
}
impl ImitationLearning {
    /// Create an imitation learning agent.
    pub fn new(state_dim: usize, action_dim: usize, hidden: &[usize], lr: f64) -> Self {
        let policy = NeuralController::new(state_dim, hidden, action_dim, 1.0);
        let feature_dim = state_dim;
        let reward_weights = vec![1.0 / state_dim as f64; feature_dim];
        Self {
            policy,
            dataset: Vec::new(),
            lr,
            iter: 0,
            reward_weights,
            feature_dim,
            rng: SmallRng::from_rng(&mut rand::rng()),
        }
    }
    /// Replace the random source of the mini-batch sampling by a generator
    /// seeded with `seed`, which makes [`bc_step`](Self::bc_step) reproducible.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }
    /// Add a demonstration to the dataset.
    pub fn add_demonstration(&mut self, demo: Demonstration) {
        self.dataset.push(demo);
    }
    /// Behavioural cloning: one mini-batch supervised step (MSE loss on actions).
    ///
    /// `batch_size` demonstrations are drawn with replacement from the dataset
    /// with the generator stored in [`rng`](Self::rng).  The gradient of the
    /// loss `½‖π(s) − a‖²` is computed exactly by back-propagation, one forward
    /// and one backward pass per demonstration.
    pub fn bc_step(&mut self, batch_size: usize) {
        if self.dataset.is_empty() {
            return;
        }
        let n = self.dataset.len();
        let mut grad = vec![0.0_f64; self.policy.num_parameters()];
        for _ in 0..batch_size {
            let index = self.rng.random_range(0..n);
            let demo = &self.dataset[index];
            let cache = self.policy.forward_cached(&demo.state);
            let residual: Vec<f64> = vsub(&cache.output, &demo.action);
            let sample_grad = self.policy.backward_cached(&cache, &residual);
            for (total, g) in grad.iter_mut().zip(sample_grad.iter()) {
                *total += g;
            }
        }
        let scale = 1.0 / batch_size.max(1) as f64;
        let scaled_grad: Vec<f64> = grad.iter().map(|g| g * scale).collect();
        self.policy.sgd_step(&scaled_grad, self.lr);
        self.iter += 1;
    }
    /// DAgger mixing probability: probability of following expert vs policy.
    /// Uses schedule β_i = p^i where p = 0.5.
    pub fn dagger_beta(&self) -> f64 {
        0.5_f64.powi(self.iter as i32)
    }
    /// Select action for DAgger: mix between expert and policy.
    pub fn dagger_action(&self, state: &[f64], expert_action: &[f64]) -> Vec<f64> {
        let mut rng = rand::rng();
        self.dagger_action_with_rng(state, expert_action, &mut rng)
    }
    /// Like [`dagger_action`](Self::dagger_action), but draws from `rng`; a
    /// seeded generator makes the expert/policy choice reproducible.
    pub fn dagger_action_with_rng<R: rand::Rng + ?Sized>(
        &self,
        state: &[f64],
        expert_action: &[f64],
        rng: &mut R,
    ) -> Vec<f64> {
        if rng.random_range(0.0..1.0) < self.dagger_beta() {
            expert_action.to_vec()
        } else {
            self.policy.forward(state)
        }
    }
    /// Aggregate a new demonstration from DAgger (expert labels policy trajectory).
    pub fn dagger_aggregate(&mut self, state: Vec<f64>, expert_action: Vec<f64>) {
        self.add_demonstration(Demonstration::new(state, expert_action));
    }
    /// IRL feature expectation under current policy (Monte Carlo estimate).
    pub fn feature_expectation(&self, trajectories: &[Vec<Vec<f64>>], gamma: f64) -> Vec<f64> {
        let mut fe = vec![0.0_f64; self.feature_dim];
        let n_traj = trajectories.len().max(1);
        for traj in trajectories {
            let mut disc = 1.0;
            for state in traj {
                for (k, &s) in state.iter().enumerate().take(self.feature_dim) {
                    fe[k] += disc * s;
                }
                disc *= gamma;
            }
        }
        fe.iter().map(|x| x / n_traj as f64).collect()
    }
    /// IRL expert feature expectation from the demonstration dataset.
    pub fn expert_feature_expectation(&self, gamma: f64) -> Vec<f64> {
        if self.dataset.is_empty() {
            return vec![0.0; self.feature_dim];
        }
        let traj: Vec<Vec<Vec<f64>>> = self.dataset.iter().map(|d| vec![d.state.clone()]).collect();
        self.feature_expectation(&traj, gamma)
    }
    /// IRL reward weight update: project-and-normalise gradient descent.
    pub fn irl_update(&mut self, learner_fe: &[f64], expert_fe: &[f64], irl_lr: f64) {
        let diff: Vec<f64> = vsub(expert_fe, learner_fe);
        for (w, d) in self.reward_weights.iter_mut().zip(diff.iter()) {
            *w += irl_lr * d;
        }
        let n = norm(&self.reward_weights).max(1e-12);
        for w in &mut self.reward_weights {
            *w /= n;
        }
    }
    /// Compute the IRL reward for a state using the current weight vector.
    pub fn irl_reward(&self, state: &[f64]) -> f64 {
        state
            .iter()
            .zip(self.reward_weights.iter())
            .map(|(s, w)| s * w)
            .sum()
    }
}
/// A time-parameterised trajectory waypoint.
#[derive(Debug, Clone)]
pub struct TrajectoryWaypoint {
    /// Time stamp.
    pub time: f64,
    /// Configuration at this waypoint.
    pub config: Vec<f64>,
    /// Optional velocity at waypoint.
    pub velocity: Option<Vec<f64>>,
}
impl TrajectoryWaypoint {
    /// Create a waypoint.
    pub fn new(time: f64, config: Vec<f64>) -> Self {
        Self {
            time,
            config,
            velocity: None,
        }
    }
}
/// A piecewise-linear trajectory for rigid body motion planning.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Ordered waypoints.
    pub waypoints: Vec<TrajectoryWaypoint>,
}
impl Trajectory {
    /// Build from a path (equal time spacing).
    pub fn from_path(path: Vec<Vec<f64>>, total_time: f64) -> Self {
        let n = path.len();
        let dt = if n > 1 {
            total_time / (n - 1) as f64
        } else {
            0.0
        };
        let waypoints = path
            .into_iter()
            .enumerate()
            .map(|(i, c)| TrajectoryWaypoint::new(i as f64 * dt, c))
            .collect();
        Self { waypoints }
    }
    /// Linearly interpolate the configuration at time `t`.
    pub fn interpolate(&self, t: f64) -> Vec<f64> {
        if self.waypoints.is_empty() {
            return Vec::new();
        }
        if t <= self.waypoints[0].time {
            return self.waypoints[0].config.clone();
        }
        let last = self
            .waypoints
            .last()
            .expect("collection should not be empty");
        if t >= last.time {
            return last.config.clone();
        }
        for i in 0..(self.waypoints.len() - 1) {
            let t0 = self.waypoints[i].time;
            let t1 = self.waypoints[i + 1].time;
            if t >= t0 && t <= t1 {
                let alpha = (t - t0) / (t1 - t0).max(1e-15);
                return self.waypoints[i]
                    .config
                    .iter()
                    .zip(self.waypoints[i + 1].config.iter())
                    .map(|(a, b)| a + alpha * (b - a))
                    .collect();
            }
        }
        last.config.clone()
    }
    /// Total duration of the trajectory.
    pub fn duration(&self) -> f64 {
        self.waypoints.last().map(|w| w.time).unwrap_or(0.0)
    }
}
/// Neural dynamics model: predicts the next state given current state + action.
#[derive(Debug, Clone)]
pub struct NeuralDynamicsModel {
    /// Internal network.
    pub network: NeuralController,
    /// Dimension of state.
    pub state_dim: usize,
    /// Dimension of action.
    pub action_dim: usize,
}
impl NeuralDynamicsModel {
    /// Construct a neural dynamics model.
    pub fn new(state_dim: usize, action_dim: usize, hidden: &[usize]) -> Self {
        let network = NeuralController::new(state_dim + action_dim, hidden, state_dim, 1.0);
        Self {
            network,
            state_dim,
            action_dim,
        }
    }
    /// Predict next state: concatenate state and action, run forward pass.
    pub fn predict(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        let mut input = state.to_vec();
        input.extend_from_slice(action);
        let delta = self.network.forward(&input);
        vadd(state, &delta)
    }
    /// Train on a single (s, a, s') sample using MSE loss gradient.
    ///
    /// One SGD step on the loss `½‖s + f(s, a) − s'‖²`; its gradient is computed
    /// exactly by back-propagation, one forward and one backward pass.
    pub fn train_step(&mut self, state: &[f64], action: &[f64], next_state: &[f64], lr: f64) {
        let mut input = state.to_vec();
        input.extend_from_slice(action);
        let cache = self.network.forward_cached(&input);
        let predicted = vadd(state, &cache.output);
        let residual: Vec<f64> = vsub(&predicted, next_state);
        let grad = self.network.backward_cached(&cache, &residual);
        self.network.sgd_step(&grad, lr);
    }
}
/// Activation function used in hidden layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// Rectified linear unit.
    Relu,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid (0–1).
    Sigmoid,
    /// Linear / identity.
    Linear,
}
/// MRAC (Model Reference Adaptive Control) for rigid body systems.
///
/// Tracks a reference model  ẋ_m = A_m x_m + B_m r  using an adaptive law
/// derived from Lyapunov stability analysis (MIT rule).
#[derive(Debug, Clone)]
pub struct AdaptiveControl {
    /// Reference model state matrix (diagonal approximation as vector).
    pub am_diag: Vec<f64>,
    /// Reference model input gain (diagonal).
    pub bm_diag: Vec<f64>,
    /// Adaptive parameter matrix (theta) — maps \[state; reference_input\] → action.
    pub theta: Vec<Vec<f64>>,
    /// Adaptation rate (γ).
    pub gamma_adapt: f64,
    /// State dimension.
    pub state_dim: usize,
    /// Reference state (x_m).
    pub ref_state: Vec<f64>,
    /// Estimated plant parameter vector for each output channel.
    pub param_estimate: Vec<f64>,
    /// Lyapunov matrix P (diagonal, symmetric positive definite).
    pub lyapunov_p: Vec<f64>,
}
impl AdaptiveControl {
    /// Initialise MRAC controller.
    pub fn new(state_dim: usize, gamma_adapt: f64) -> Self {
        let am_diag = vec![-1.0; state_dim];
        let bm_diag = vec![1.0; state_dim];
        let theta = vec![vec![0.0; state_dim + 1]; state_dim];
        let lyapunov_p = vec![1.0; state_dim];
        let param_estimate = vec![1.0; state_dim];
        Self {
            am_diag,
            bm_diag,
            theta,
            gamma_adapt,
            state_dim,
            ref_state: vec![0.0; state_dim],
            param_estimate,
            lyapunov_p,
        }
    }
    /// Integrate the reference model one step.
    pub fn step_reference_model(&mut self, r: &[f64], dt: f64) {
        for (i, (ref_s, (am, bm))) in self
            .ref_state
            .iter_mut()
            .zip(self.am_diag.iter().zip(self.bm_diag.iter()))
            .enumerate()
        {
            let x_dot = am * *ref_s + bm * r[i];
            *ref_s += x_dot * dt;
        }
    }
    /// Compute the adaptive control action for a given plant state.
    pub fn compute_action(&self, plant_state: &[f64], r: &[f64]) -> Vec<f64> {
        (0..self.state_dim)
            .map(|i| {
                let x_part: f64 = plant_state
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| self.theta[i][j] * x)
                    .sum();
                let r_part = self.theta[i][self.state_dim] * r[i];
                x_part + r_part
            })
            .collect()
    }
    /// Update adaptive parameters using the MIT (gradient) rule:
    ///   dθ/dt = -γ · e · ∂y/∂θ
    pub fn update_parameters(&mut self, plant_state: &[f64], r: &[f64], dt: f64) {
        let e: Vec<f64> = plant_state
            .iter()
            .zip(self.ref_state.iter())
            .map(|(x, m)| x - m)
            .collect();
        for (i, (theta_row, (lp, (e_i, r_i)))) in self
            .theta
            .iter_mut()
            .zip(self.lyapunov_p.iter().zip(e.iter().zip(r.iter())))
            .enumerate()
        {
            let _ = i;
            for (j, (th, ps)) in theta_row
                .iter_mut()
                .zip(plant_state.iter())
                .enumerate()
                .take(self.state_dim)
            {
                let _ = j;
                *th += -self.gamma_adapt * lp * e_i * ps * dt;
            }
            let dtheta_r = -self.gamma_adapt * lp * e_i * r_i;
            theta_row[self.state_dim] += dtheta_r * dt;
        }
    }
    /// Lyapunov stability certificate: V = e^T P e (should be non-negative and decreasing).
    pub fn lyapunov_value(&self, plant_state: &[f64]) -> f64 {
        plant_state
            .iter()
            .zip(self.ref_state.iter())
            .zip(self.lyapunov_p.iter())
            .map(|((x, m), p)| p * (x - m).powi(2))
            .sum()
    }
    /// Estimate an unknown plant parameter via gradient descent on the tracking error.
    pub fn estimate_parameter(&mut self, error: f64, regressor: f64, dt: f64) {
        for p in &mut self.param_estimate {
            *p -= self.gamma_adapt * error * regressor * dt;
        }
    }
    /// Compute the Lyapunov derivative dV/dt ≈ (V(t) - V(t-dt)) / dt.
    pub fn lyapunov_derivative(&self, plant_state: &[f64], prev_v: f64, dt: f64) -> f64 {
        (self.lyapunov_value(plant_state) - prev_v) / dt
    }
}
/// A single demonstration: state → expert action (continuous or discrete).
#[derive(Debug, Clone)]
pub struct Demonstration {
    /// Observed state.
    pub state: Vec<f64>,
    /// Expert action (continuous).
    pub action: Vec<f64>,
}
impl Demonstration {
    /// Create a demonstration.
    pub fn new(state: Vec<f64>, action: Vec<f64>) -> Self {
        Self { state, action }
    }
}
/// A node in a motion planning tree / roadmap.
#[derive(Debug, Clone)]
pub struct PlannerNode {
    /// Configuration-space position (joint angles or Cartesian).
    pub config: Vec<f64>,
    /// Parent node index in the node list (usize::MAX = root).
    pub parent: usize,
    /// Cost-to-reach from root (used by RRT*).
    pub cost: f64,
}
impl PlannerNode {
    /// Create a new planner node.
    pub fn new(config: Vec<f64>, parent: usize, cost: f64) -> Self {
        Self {
            config,
            parent,
            cost,
        }
    }
}
/// Fraction of the sampled sequences that [`ModelPredictiveControl::cem_plan`]
/// refits its distribution to.
const CEM_ELITE_FRACTION: f64 = 0.2;

/// Floor of the standard deviation of the CEM distribution; it keeps the
/// distribution from collapsing to a point before the iterations are over.
const CEM_MIN_SIGMA: f64 = 1e-4;

/// A standard normal deviate (Box–Muller transform).
fn standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    // `1 - u` lies in (0, 1], so the logarithm is finite.
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Neural MPC controller with receding horizon optimisation.
#[derive(Debug, Clone)]
pub struct ModelPredictiveControl {
    /// Learned dynamics model.
    pub dynamics: NeuralDynamicsModel,
    /// Planning horizon (steps).
    pub horizon: usize,
    /// Number of random shooting candidates.
    pub num_candidates: usize,
    /// Discount factor over horizon.
    pub gamma: f64,
    /// State dimension.
    pub state_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Action bound (±).
    pub action_bound: f64,
}
impl ModelPredictiveControl {
    /// Create an MPC controller.
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden: &[usize],
        horizon: usize,
        num_candidates: usize,
        gamma: f64,
        action_bound: f64,
    ) -> Self {
        let dynamics = NeuralDynamicsModel::new(state_dim, action_dim, hidden);
        Self {
            dynamics,
            horizon,
            num_candidates,
            gamma,
            state_dim,
            action_dim,
            action_bound,
        }
    }
    /// Number of entries (`horizon * action_dim`) of an action sequence, or
    /// `None` when there is nothing to optimise: no candidates, an empty
    /// horizon or action space, or an `action_bound` that is not a positive
    /// finite number (it leaves no room for a non-zero action).
    fn sequence_len(&self) -> Option<usize> {
        let usable_bound = self.action_bound.is_finite() && self.action_bound > 0.0;
        if self.num_candidates == 0 || !usable_bound {
            return None;
        }
        self.horizon
            .checked_mul(self.action_dim)
            .filter(|&len| len > 0)
    }
    /// Discounted cost of applying the action `sequence` (`action_dim` entries
    /// per step) from `state` under the learned dynamics.
    ///
    /// A NaN cost counts as infinitely bad, so a candidate that produces one is
    /// never preferred.
    fn rollout_cost<F>(&self, state: &[f64], sequence: &[f64], cost_fn: &F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut s = state.to_vec();
        let mut total_cost = 0.0;
        let mut discount = 1.0;
        for action in sequence.chunks(self.action_dim.max(1)) {
            s = self.dynamics.predict(&s, action);
            total_cost += discount * cost_fn(&s);
            discount *= self.gamma;
        }
        if total_cost.is_nan() {
            f64::INFINITY
        } else {
            total_cost
        }
    }
    /// Random shooting: sample action sequences and return the first action of the best.
    ///
    /// The cost function is supplied as a closure `cost(state) -> f64` (lower is better).
    ///
    /// Returns the zero action when there is nothing to optimise: no
    /// candidates, an empty horizon, or an `action_bound` that is not a positive
    /// finite number.
    pub fn plan<F>(&self, state: &[f64], cost_fn: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::rng();
        self.plan_with_rng(state, cost_fn, &mut rng)
    }
    /// Like [`plan`](Self::plan), but draws the candidate sequences from `rng`;
    /// a seeded generator makes the plan reproducible.
    pub fn plan_with_rng<F, R>(&self, state: &[f64], cost_fn: F, rng: &mut R) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
        R: rand::Rng + ?Sized,
    {
        let Some(len) = self.sequence_len() else {
            return vec![0.0_f64; self.action_dim];
        };
        let mut best_cost = f64::INFINITY;
        let mut best_first = vec![0.0_f64; self.action_dim];
        for _ in 0..self.num_candidates {
            let sequence: Vec<f64> = (0..len)
                .map(|_| uniform(rng, -self.action_bound, self.action_bound))
                .collect();
            let total_cost = self.rollout_cost(state, &sequence, &cost_fn);
            if total_cost < best_cost {
                best_cost = total_cost;
                best_first = sequence.iter().take(self.action_dim).copied().collect();
            }
        }
        best_first
    }
    /// Cross-entropy method (CEM) planning: iteratively refine a Gaussian
    /// distribution over whole action sequences.
    ///
    /// The distribution has an independent mean and standard deviation for every
    /// entry of the `horizon × action_dim` sequence.  Each of the `iterations`
    /// rounds samples `num_candidates` sequences (clamped to the action bound),
    /// rolls them out through the learned dynamics, ranks them by their
    /// discounted cost and refits mean and deviation to the best fifth.  The
    /// first action of the final mean sequence is returned.
    ///
    /// Returns the zero action when there is nothing to optimise (no
    /// candidates, an empty horizon, or an `action_bound` that is not a positive
    /// finite number) and, trivially, for zero `iterations`.
    pub fn cem_plan<F>(&self, state: &[f64], cost_fn: F, iterations: usize) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::rng();
        self.cem_plan_with_rng(state, cost_fn, iterations, &mut rng)
    }
    /// Like [`cem_plan`](Self::cem_plan), but draws the candidate sequences from
    /// `rng`; a seeded generator makes the plan reproducible.
    pub fn cem_plan_with_rng<F, R>(
        &self,
        state: &[f64],
        cost_fn: F,
        iterations: usize,
        rng: &mut R,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
        R: rand::Rng + ?Sized,
    {
        let mut first: Vec<f64> = self
            .cem_sequence(state, &cost_fn, iterations, rng)
            .into_iter()
            .take(self.action_dim)
            .collect();
        first.resize(self.action_dim, 0.0);
        first
    }
    /// The mean of the CEM distribution after `iterations` rounds: the planned
    /// action sequence, `action_dim` entries per step, empty when nothing can
    /// be optimised (see [`cem_plan`](Self::cem_plan)).
    fn cem_sequence<F, R>(
        &self,
        state: &[f64],
        cost_fn: &F,
        iterations: usize,
        rng: &mut R,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
        R: rand::Rng + ?Sized,
    {
        let Some(len) = self.sequence_len() else {
            return Vec::new();
        };
        let bound = self.action_bound;
        let mut mean = vec![0.0_f64; len];
        let mut sigma = vec![bound; len];
        let n_elite = ((self.num_candidates as f64 * CEM_ELITE_FRACTION) as usize)
            .clamp(1, self.num_candidates);
        for _ in 0..iterations {
            let mut samples: Vec<(Vec<f64>, f64)> = (0..self.num_candidates)
                .map(|_| {
                    let sequence: Vec<f64> = mean
                        .iter()
                        .zip(sigma.iter())
                        .map(|(&m, &s)| (m + s * standard_normal(rng)).clamp(-bound, bound))
                        .collect();
                    let cost = self.rollout_cost(state, &sequence, cost_fn);
                    (sequence, cost)
                })
                .collect();
            samples.sort_by(|a, b| a.1.total_cmp(&b.1));
            let elites = &samples[..n_elite];
            for (k, (m, s)) in mean.iter_mut().zip(sigma.iter_mut()).enumerate() {
                let mu = elites.iter().map(|(seq, _)| seq[k]).sum::<f64>() / n_elite as f64;
                let var = elites
                    .iter()
                    .map(|(seq, _)| (seq[k] - mu).powi(2))
                    .sum::<f64>()
                    / n_elite as f64;
                *m = mu;
                *s = var.sqrt().max(CEM_MIN_SIGMA);
            }
        }
        mean
    }
}
/// Ornstein-Uhlenbeck noise for temporally correlated exploration.
#[derive(Debug, Clone)]
pub struct OUNoise {
    /// Current state.
    pub state: Vec<f64>,
    /// Mean reversion rate θ.
    pub theta: f64,
    /// Diffusion coefficient σ.
    pub sigma: f64,
    /// Long-run mean μ.
    pub mu: Vec<f64>,
}
impl OUNoise {
    /// Create a zero-mean OU process.
    pub fn new(dim: usize, theta: f64, sigma: f64) -> Self {
        Self {
            state: vec![0.0; dim],
            theta,
            sigma,
            mu: vec![0.0; dim],
        }
    }
    /// Step the process and return the current noise sample.
    pub fn sample(&mut self) -> Vec<f64> {
        let mut rng = rand::rng();
        for i in 0..self.state.len() {
            let dx = self.theta * (self.mu[i] - self.state[i])
                + self.sigma * rng.random_range(-1.0_f64..1.0_f64);
            self.state[i] += dx;
        }
        self.state.clone()
    }
    /// Reset to zero.
    pub fn reset(&mut self) {
        for s in &mut self.state {
            *s = 0.0;
        }
    }
}
/// Circular replay buffer for experience replay.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    /// Stored transitions.
    pub buffer: Vec<Transition>,
    /// Maximum capacity.
    pub capacity: usize,
    /// Write position.
    pub write_pos: usize,
}
impl ReplayBuffer {
    /// Create a replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
        }
    }
    /// Push a transition, overwriting the oldest if full.
    ///
    /// A buffer without capacity holds nothing: the transition is dropped.
    pub fn push(&mut self, t: Transition) {
        if self.capacity == 0 {
            return;
        }
        if self.buffer.len() < self.capacity {
            self.buffer.push(t);
        } else {
            // `buffer.len() >= capacity >= 1`; the modulo only matters when the
            // public fields were changed after construction.
            let slot = self.write_pos % self.buffer.len();
            self.buffer[slot] = t;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }
    /// Current number of stored transitions.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    /// Sample a mini-batch of `n` distinct transitions (all of them, in random
    /// order, when `n` is not smaller than the number stored).
    pub fn sample(&self, n: usize) -> Vec<&Transition> {
        let mut rng = rand::rng();
        self.sample_with_rng(n, &mut rng)
    }
    /// Like [`sample`](Self::sample), but draws from `rng`; a seeded generator
    /// makes the mini-batch reproducible.
    pub fn sample_with_rng<R: rand::Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
    ) -> Vec<&Transition> {
        let len = self.buffer.len();
        let amount = n.min(len);
        if amount == 0 {
            return Vec::new();
        }
        rand::seq::index::sample(rng, len, amount)
            .iter()
            .filter_map(|index| self.buffer.get(index))
            .collect()
    }
}
/// Axis-aligned hypersphere obstacle in configuration space.
///
/// A configuration `q` is considered in collision if `||q - center||₂ ≤ radius`.
#[derive(Debug, Clone)]
pub struct PlanningObstacle {
    /// Center of the obstacle in configuration space.
    pub center: Vec<f64>,
    /// Radius of the obstacle (Euclidean metric).
    pub radius: f64,
}

/// Number of equal intervals a straight planner edge is discretised into by
/// the collision checks (see [`MotionPlanning::is_collision_free`]).
const EDGE_CHECK_STEPS: usize = 10;

/// Probability that [`MotionPlanning::rrt`] samples the goal instead of a
/// uniformly random configuration.
const GOAL_BIAS: f64 = 0.1;

/// Search tree grown by [`MotionPlanning::rrt`].
///
/// Next to the nodes it keeps their child lists, which are needed to push a
/// cost change down a subtree when a node is rewired, and the nodes that can
/// be joined to the goal.  Every node's `cost` equals `parent.cost +
/// |parent - node|`; [`PlannerTree::add_node`] and [`PlannerTree::reparent`]
/// are the only operations that change the topology and both keep that
/// invariant.
#[derive(Debug, Clone)]
struct PlannerTree {
    /// Tree nodes; index 0 is the root (the start configuration).
    nodes: Vec<PlannerNode>,
    /// `children[i]` lists the nodes whose parent is node `i`.
    children: Vec<Vec<usize>>,
    /// Nodes within the goal tolerance that have a collision-free edge to the goal.
    goal_candidates: Vec<usize>,
}

impl PlannerTree {
    /// A tree that consists of the root `start` only.
    fn new(start: Vec<f64>) -> Self {
        Self {
            nodes: vec![PlannerNode::new(start, usize::MAX, 0.0)],
            children: vec![Vec::new()],
            goal_candidates: Vec::new(),
        }
    }

    /// Cost of reaching `config` through the existing node `parent`.
    fn cost_through(&self, parent: usize, config: &[f64]) -> f64 {
        self.nodes[parent].cost + config_dist(&self.nodes[parent].config, config)
    }

    /// Append `config` as a child of `parent` and return its index.
    fn add_node(&mut self, config: Vec<f64>, parent: usize) -> usize {
        let cost = self.cost_through(parent, &config);
        let index = self.nodes.len();
        self.nodes.push(PlannerNode::new(config, parent, cost));
        self.children.push(Vec::new());
        self.children[parent].push(index);
        index
    }

    /// Whether `ancestor` lies on the parent chain above `node`.
    fn is_ancestor(&self, ancestor: usize, node: usize) -> bool {
        let mut current = self.nodes[node].parent;
        // A parent chain cannot be longer than the tree is large.
        for _ in 0..self.nodes.len() {
            if current == usize::MAX {
                return false;
            }
            if current == ancestor {
                return true;
            }
            current = self.nodes[current].parent;
        }
        false
    }

    /// Move `node` together with its subtree below `new_parent`.
    ///
    /// The cost of `node` and of every descendant is recomputed from the new
    /// parent chain.  The caller must make sure that `node` is not an ancestor
    /// of `new_parent`, otherwise the tree would turn into a cycle.
    fn reparent(&mut self, node: usize, new_parent: usize) {
        let old_parent = self.nodes[node].parent;
        if old_parent != usize::MAX {
            self.children[old_parent].retain(|&child| child != node);
        }
        self.nodes[node].parent = new_parent;
        self.children[new_parent].push(node);
        self.nodes[node].cost = self.cost_through(new_parent, &self.nodes[node].config);
        let mut pending = vec![node];
        while let Some(parent) = pending.pop() {
            for &child in &self.children[parent] {
                self.nodes[child].cost = self.nodes[parent].cost
                    + config_dist(&self.nodes[parent].config, &self.nodes[child].config);
                pending.push(child);
            }
        }
    }

    /// The goal candidate with the lowest cost of reaching the goal through it.
    fn best_goal_node(&self, goal: &[f64]) -> Option<usize> {
        let total_cost =
            |node: usize| self.nodes[node].cost + config_dist(&self.nodes[node].config, goal);
        self.goal_candidates
            .iter()
            .copied()
            .min_by(|&a, &b| total_cost(a).total_cmp(&total_cost(b)))
    }

    /// The path from the root to `node`, followed by `goal`.
    fn path_to(&self, node: usize, goal: &[f64]) -> Vec<Vec<f64>> {
        let mut path = Vec::new();
        let mut current = node;
        // A parent chain cannot be longer than the tree is large.
        for _ in 0..self.nodes.len() {
            path.push(self.nodes[current].config.clone());
            current = self.nodes[current].parent;
            if current == usize::MAX {
                break;
            }
        }
        path.reverse();
        if path.last().is_none_or(|last| last.as_slice() != goal) {
            path.push(goal.to_vec());
        }
        path
    }
}

/// RRT / RRT* planner for rigid body configuration spaces.
#[derive(Debug, Clone)]
pub struct MotionPlanning {
    /// Dimension of configuration space.
    pub config_dim: usize,
    /// Lower bounds of the configuration space.
    pub lower: Vec<f64>,
    /// Upper bounds of the configuration space.
    pub upper: Vec<f64>,
    /// Step size (extend length).
    pub step_size: f64,
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Goal tolerance.
    pub goal_tol: f64,
    /// Whether to use RRT*: every new node takes the cheapest collision-free
    /// parent within `rewire_radius`, nearby nodes are rewired through it
    /// when that lowers their cost, and the search runs for all `max_iters`
    /// iterations before the cheapest path is returned.
    pub use_rrt_star: bool,
    /// Neighbourhood radius of RRT*'s parent selection and rewiring.
    pub rewire_radius: f64,
    /// Obstacle set for collision checking in configuration space.
    pub obstacles: Vec<PlanningObstacle>,
}
impl MotionPlanning {
    /// Construct a planner.
    pub fn new(
        config_dim: usize,
        lower: Vec<f64>,
        upper: Vec<f64>,
        step_size: f64,
        max_iters: usize,
        goal_tol: f64,
        use_rrt_star: bool,
    ) -> Self {
        let rewire_radius = step_size * 3.0;
        Self {
            config_dim,
            lower,
            upper,
            step_size,
            max_iters,
            goal_tol,
            use_rrt_star,
            rewire_radius,
            obstacles: Vec::new(),
        }
    }
    /// Builder method to add obstacles to the planner.
    pub fn with_obstacles(mut self, obstacles: Vec<PlanningObstacle>) -> Self {
        self.obstacles = obstacles;
        self
    }
    /// Sample a random configuration within bounds.
    ///
    /// A joint whose bounds coincide (`lower[i] == upper[i]`) is pinned at that
    /// value, and so is a joint with inverted bounds (at `lower[i]`).  The
    /// sample has `config_dim` components when `lower` and `upper` cover every
    /// joint; a shorter bounds vector gives a shorter sample, which
    /// [`is_collision_free`](Self::is_collision_free) rejects.
    pub fn random_config(&self) -> Vec<f64> {
        let mut rng = rand::rng();
        self.random_config_with_rng(&mut rng)
    }
    /// Sample a random configuration within bounds, drawing from `rng`.
    ///
    /// Behaves like [`random_config`](Self::random_config); a seeded generator
    /// makes the sample reproducible.
    pub fn random_config_with_rng<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .take(self.config_dim)
            .map(|(&lower, &upper)| uniform(rng, lower, upper))
            .collect()
    }
    /// Find the nearest node index to `q` in `nodes`.
    ///
    /// A node with a NaN coordinate (or a NaN query) has no distance and ranks
    /// as infinitely far away, so it is never preferred to a node with a real
    /// distance; the result is `0` when `nodes` is empty.
    pub fn nearest(&self, nodes: &[PlannerNode], q: &[f64]) -> usize {
        let distance = |node: &PlannerNode| {
            let d = config_dist(&node.config, q);
            if d.is_nan() { f64::INFINITY } else { d }
        };
        nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| distance(a).total_cmp(&distance(b)))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    /// Steer from `from` towards `to` by at most `step_size`.
    pub fn steer(&self, from: &[f64], to: &[f64]) -> Vec<f64> {
        let d = config_dist(from, to);
        if d <= self.step_size {
            to.to_vec()
        } else {
            let t = self.step_size / d;
            from.iter()
                .zip(to.iter())
                .map(|(a, b)| a + t * (b - a))
                .collect()
        }
    }
    /// Check whether `config` is free of collisions with all registered obstacles.
    ///
    /// Returns `false` if `config` lies inside (or on the boundary of) any obstacle,
    /// or if `config.len() != self.config_dim`.  Returns `true` when no obstacles
    /// are registered (open space).
    pub fn is_collision_free(&self, config: &[f64]) -> bool {
        if config.len() != self.config_dim {
            return false;
        }
        for obs in &self.obstacles {
            if obs.center.len() != self.config_dim {
                // Dimension mismatch — skip malformed obstacle rather than panic.
                continue;
            }
            let dist_sq: f64 = config
                .iter()
                .zip(obs.center.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist_sq <= obs.radius * obs.radius {
                return false;
            }
        }
        true
    }
    /// Check whether the straight-line segment from `q_a` to `q_b` is collision-free.
    ///
    /// The segment is discretised into `n_steps` equal intervals and every sample point
    /// (including both endpoints) is tested with [`Self::is_collision_free`].  A larger
    /// `n_steps` catches narrower obstacles at the cost of more checks.
    fn is_segment_collision_free(&self, q_a: &[f64], q_b: &[f64], n_steps: usize) -> bool {
        let n = n_steps.max(2);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let q: Vec<f64> = q_a
                .iter()
                .zip(q_b.iter())
                .map(|(a, b)| a + t * (b - a))
                .collect();
            if !self.is_collision_free(&q) {
                return false;
            }
        }
        true
    }
    /// Indices of the nodes closer than `rewire_radius` to `q`.
    fn near_indices(&self, nodes: &[PlannerNode], q: &[f64]) -> Vec<usize> {
        nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| config_dist(&node.config, q) < self.rewire_radius)
            .map(|(index, _)| index)
            .collect()
    }
    /// Whether a node at `q` can be joined to `goal`: it lies within `goal_tol`
    /// of the goal and the straight edge to the goal is collision-free.
    fn can_connect_to_goal(&self, q: &[f64], goal: &[f64]) -> bool {
        config_dist(q, goal) < self.goal_tol
            && self.is_segment_collision_free(q, goal, EDGE_CHECK_STEPS)
    }
    /// Whether `q` can serve as start or goal: it is finite, has `config_dim`
    /// components and is collision-free.
    fn is_valid_endpoint(&self, q: &[f64]) -> bool {
        q.iter().all(|c| c.is_finite()) && self.is_collision_free(q)
    }
    /// RRT* parent selection: the cheapest node among `near` and `nearest_idx`
    /// whose edge to `q_new` is collision-free.
    ///
    /// The edge from `nearest_idx` is known to be free (the caller checked it
    /// when it extended the tree), so it is the fallback.  The other
    /// candidates are tried in order of increasing cost, which keeps the number
    /// of collision checks small.
    fn choose_parent(
        &self,
        tree: &PlannerTree,
        near: &[usize],
        nearest_idx: usize,
        q_new: &[f64],
    ) -> usize {
        let nearest_cost = tree.cost_through(nearest_idx, q_new);
        let mut cheaper: Vec<(usize, f64)> = near
            .iter()
            .copied()
            .filter(|&index| index != nearest_idx)
            .map(|index| (index, tree.cost_through(index, q_new)))
            .filter(|&(_, cost)| cost < nearest_cost)
            .collect();
        cheaper.sort_by(|a, b| a.1.total_cmp(&b.1));
        cheaper
            .into_iter()
            .map(|(index, _)| index)
            .find(|&index| {
                self.is_segment_collision_free(&tree.nodes[index].config, q_new, EDGE_CHECK_STEPS)
            })
            .unwrap_or(nearest_idx)
    }
    /// RRT* rewiring: re-route every node of `near` through the new node
    /// `new_idx` when that is cheaper and the edge is collision-free.
    ///
    /// The cost change is propagated to the descendants of a rewired node.  A
    /// node on the parent chain of `new_idx` is never rewired, since moving it
    /// below `new_idx` would create a cycle.
    fn rewire(&self, tree: &mut PlannerTree, near: &[usize], new_idx: usize) {
        let new_parent = tree.nodes[new_idx].parent;
        for &index in near {
            if index == new_parent || tree.nodes[index].parent == usize::MAX {
                continue;
            }
            let cost_through_new = tree.cost_through(new_idx, &tree.nodes[index].config);
            if cost_through_new >= tree.nodes[index].cost || tree.is_ancestor(index, new_idx) {
                continue;
            }
            if self.is_segment_collision_free(
                &tree.nodes[new_idx].config,
                &tree.nodes[index].config,
                EDGE_CHECK_STEPS,
            ) {
                tree.reparent(index, new_idx);
            }
        }
    }
    /// Grow the search tree from `start` towards `goal`.
    ///
    /// Returns `None` when `start` or `goal` is unusable (non-finite, of the
    /// wrong dimension or inside an obstacle).  Otherwise the tree and its goal
    /// candidates are returned; plain RRT stops at the first candidate, RRT*
    /// keeps improving the tree for all `max_iters` iterations.
    fn grow_tree<R: rand::Rng + ?Sized>(
        &self,
        start: &[f64],
        goal: &[f64],
        rng: &mut R,
    ) -> Option<PlannerTree> {
        if !self.is_valid_endpoint(start) || !self.is_valid_endpoint(goal) {
            return None;
        }
        let mut tree = PlannerTree::new(start.to_vec());
        if self.can_connect_to_goal(start, goal) {
            // The straight edge is the shortest possible path.
            tree.goal_candidates.push(0);
            return Some(tree);
        }
        for _ in 0..self.max_iters {
            let q_rand = if rng.random_range(0.0..1.0) < GOAL_BIAS {
                goal.to_vec()
            } else {
                self.random_config_with_rng(rng)
            };
            let nearest_idx = self.nearest(&tree.nodes, &q_rand);
            let q_nearest = &tree.nodes[nearest_idx].config;
            let q_new = self.steer(q_nearest, &q_rand);
            if q_new == *q_nearest
                || !self.is_segment_collision_free(q_nearest, &q_new, EDGE_CHECK_STEPS)
            {
                continue;
            }
            let new_idx = if self.use_rrt_star {
                let near = self.near_indices(&tree.nodes, &q_new);
                let parent = self.choose_parent(&tree, &near, nearest_idx, &q_new);
                let new_idx = tree.add_node(q_new.clone(), parent);
                self.rewire(&mut tree, &near, new_idx);
                new_idx
            } else {
                tree.add_node(q_new.clone(), nearest_idx)
            };
            if self.can_connect_to_goal(&q_new, goal) {
                tree.goal_candidates.push(new_idx);
                if !self.use_rrt_star {
                    break;
                }
            }
        }
        Some(tree)
    }
    /// Run RRT (RRT* when [`use_rrt_star`](Self::use_rrt_star) is set) and
    /// return the path from `start` to `goal`, or an empty vector if none was
    /// found within `max_iters` iterations.
    ///
    /// Every edge of the returned path, including the last one to `goal`, is
    /// collision-free (see [`is_collision_free`](Self::is_collision_free)).  An
    /// empty vector is also returned when `start` or `goal` is non-finite, has
    /// the wrong dimension or lies inside an obstacle.
    ///
    /// Plain RRT stops at the first tree node that is within `goal_tol` of
    /// `goal` and can see it.  RRT* uses the whole iteration budget to lower
    /// the cost of the tree and returns the cheapest of all such nodes.
    pub fn rrt(&self, start: &[f64], goal: &[f64]) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        self.rrt_with_rng(start, goal, &mut rng)
    }
    /// Like [`rrt`](Self::rrt), but draws every random sample from `rng`; a
    /// seeded generator makes the plan reproducible.
    pub fn rrt_with_rng<R: rand::Rng + ?Sized>(
        &self,
        start: &[f64],
        goal: &[f64],
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let Some(tree) = self.grow_tree(start, goal, rng) else {
            return Vec::new();
        };
        match tree.best_goal_node(goal) {
            Some(node) => tree.path_to(node, goal),
            None => Vec::new(),
        }
    }
    /// Probabilistic roadmap (PRM) construction: returns adjacency list.
    ///
    /// `n_samples` configurations are drawn at random; each one is joined to
    /// those of its `k_neighbours` nearest samples that it can reach along a
    /// collision-free straight segment.  No samples give an empty roadmap.
    pub fn build_prm(&self, n_samples: usize, k_neighbours: usize) -> Vec<Vec<usize>> {
        let mut rng = rand::rng();
        self.build_prm_with_rng(n_samples, k_neighbours, &mut rng)
    }
    /// Like [`build_prm`](Self::build_prm), but draws the samples from `rng`; a
    /// seeded generator makes the roadmap reproducible.
    pub fn build_prm_with_rng<R: rand::Rng + ?Sized>(
        &self,
        n_samples: usize,
        k_neighbours: usize,
        rng: &mut R,
    ) -> Vec<Vec<usize>> {
        let configs: Vec<Vec<f64>> = (0..n_samples)
            .map(|_| self.random_config_with_rng(rng))
            .collect();
        let mut adj = vec![Vec::<usize>::new(); n_samples];
        for i in 0..n_samples {
            let mut dists: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| (j, config_dist(&configs[i], &configs[j])))
                .collect();
            dists.sort_by(|a, b| a.1.total_cmp(&b.1));
            for (j, _) in dists.iter().take(k_neighbours) {
                // A PRM edge is valid only when the entire straight-line segment
                // through C-space is obstacle-free (prevents tunnelling).
                if self.is_segment_collision_free(&configs[i], &configs[*j], EDGE_CHECK_STEPS) {
                    adj[i].push(*j);
                }
            }
        }
        adj
    }
    /// Trajectory optimisation: gradient descent on a path to minimise length.
    pub fn optimise_trajectory(
        &self,
        path: Vec<Vec<f64>>,
        iterations: usize,
        lr: f64,
    ) -> Vec<Vec<f64>> {
        let n = path.len();
        if n < 3 {
            return path;
        }
        let mut opt = path;
        for _ in 0..iterations {
            let mut new_path = opt.clone();
            for i in 1..(n - 1) {
                let grad: Vec<f64> = (0..self.config_dim)
                    .map(|k| 2.0 * opt[i][k] - opt[i - 1][k] - opt[i + 1][k])
                    .collect();
                for k in 0..self.config_dim {
                    new_path[i][k] -= lr * grad[k];
                    new_path[i][k] = new_path[i][k].clamp(self.lower[k], self.upper[k]);
                }
            }
            opt = new_path;
        }
        opt
    }
}
/// Q-table for discrete state–action spaces (tabular Q-learning).
#[derive(Debug, Clone)]
pub struct QTable {
    /// Q\[state\]\[action\].
    pub q: Vec<Vec<f64>>,
    /// Learning rate α.
    pub alpha: f64,
    /// Discount factor γ.
    pub gamma: f64,
    /// Exploration rate ε (ε-greedy).
    pub epsilon: f64,
}
impl QTable {
    /// Initialise a Q-table with zeros.
    pub fn new(
        num_states: usize,
        num_actions: usize,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            q: vec![vec![0.0; num_actions]; num_states],
            alpha,
            gamma,
            epsilon,
        }
    }
    /// ε-greedy action selection.
    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::rng();
        if rng.random_range(0.0..1.0) < self.epsilon {
            rng.random_range(0..self.q[state].len())
        } else {
            let row = &self.q[state];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }
    /// Tabular Q-learning update (SARSA-max / Q-learning).
    pub fn update(
        &mut self,
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        done: bool,
    ) {
        let max_next = if done {
            0.0
        } else {
            self.q[next_state]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let td_target = reward + self.gamma * max_next;
        let td_error = td_target - self.q[state][action];
        self.q[state][action] += self.alpha * td_error;
    }
    /// Decay epsilon multiplicatively.
    pub fn decay_epsilon(&mut self, factor: f64) {
        self.epsilon = (self.epsilon * factor).max(0.01);
    }
}

#[cfg(test)]
mod obstacle_tests {
    use super::*;

    #[test]
    fn test_planning_obstacle_blocks_config() {
        let planner = MotionPlanning {
            config_dim: 2,
            lower: vec![-1.0, -1.0],
            upper: vec![1.0, 1.0],
            step_size: 0.1,
            max_iters: 100,
            goal_tol: 0.05,
            use_rrt_star: false,
            rewire_radius: 0.3,
            obstacles: vec![PlanningObstacle {
                center: vec![0.0, 0.0],
                radius: 0.5,
            }],
        };
        // Exactly at center — deep inside obstacle.
        assert!(!planner.is_collision_free(&[0.0, 0.0]));
        // At radius boundary (dist_sq == radius^2 → blocked).
        assert!(!planner.is_collision_free(&[0.4, 0.0]));
        // Clearly outside.
        assert!(planner.is_collision_free(&[0.8, 0.8]));
    }

    #[test]
    fn test_segment_collision_checks_midpoints() {
        let planner = MotionPlanning {
            config_dim: 2,
            lower: vec![-2.0, -2.0],
            upper: vec![2.0, 2.0],
            step_size: 0.1,
            max_iters: 100,
            goal_tol: 0.05,
            use_rrt_star: false,
            rewire_radius: 0.5,
            obstacles: vec![PlanningObstacle {
                center: vec![0.0, 0.0],
                radius: 0.3,
            }],
        };
        // Segment from (-1, 0) to (1, 0) passes through the obstacle at the origin.
        assert!(!planner.is_segment_collision_free(&[-1.0, 0.0], &[1.0, 0.0], 20));
        // Segment entirely outside.
        assert!(planner.is_segment_collision_free(&[0.5, 0.5], &[1.0, 1.0], 10));
    }

    #[test]
    fn test_planning_no_obstacles_is_free() {
        let planner = MotionPlanning {
            config_dim: 3,
            lower: vec![0.0; 3],
            upper: vec![1.0; 3],
            step_size: 0.1,
            max_iters: 50,
            goal_tol: 0.05,
            use_rrt_star: false,
            rewire_radius: 0.3,
            obstacles: vec![],
        };
        assert!(planner.is_collision_free(&[0.5, 0.5, 0.5]));
        assert!(planner.is_collision_free(&[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_planning_wrong_dim_not_free() {
        let planner = MotionPlanning {
            config_dim: 2,
            lower: vec![0.0, 0.0],
            upper: vec![1.0, 1.0],
            step_size: 0.1,
            max_iters: 50,
            goal_tol: 0.05,
            use_rrt_star: false,
            rewire_radius: 0.3,
            obstacles: vec![],
        };
        // Config has wrong dimensionality — should return false.
        assert!(!planner.is_collision_free(&[0.5]));
    }

    #[test]
    fn test_rrt_runs_with_obstacle_and_finds_path_or_empty() {
        // Obstacle at (0.5, 0.5), planner must route around it.
        let planner = MotionPlanning {
            config_dim: 2,
            lower: vec![0.0, 0.0],
            upper: vec![1.0, 1.0],
            step_size: 0.05,
            max_iters: 2000,
            goal_tol: 0.1,
            use_rrt_star: false,
            rewire_radius: 0.2,
            obstacles: vec![PlanningObstacle {
                center: vec![0.5, 0.5],
                radius: 0.2,
            }],
        };
        let start = vec![0.1, 0.1];
        let goal = vec![0.9, 0.9];
        // Should not panic regardless of result; result is probabilistic.
        let path = planner.rrt(&start, &goal);
        // If a path was returned, every waypoint must be obstacle-free.
        for waypoint in &path {
            assert!(
                planner.is_collision_free(waypoint),
                "RRT returned a waypoint inside an obstacle: {waypoint:?}"
            );
        }
    }

    #[test]
    fn test_with_obstacles_builder() {
        let planner = MotionPlanning::new(2, vec![0.0, 0.0], vec![1.0, 1.0], 0.1, 100, 0.05, false)
            .with_obstacles(vec![PlanningObstacle {
                center: vec![0.5, 0.5],
                radius: 0.1,
            }]);
        assert_eq!(planner.obstacles.len(), 1);
        assert!(!planner.is_collision_free(&[0.5, 0.5]));
        assert!(planner.is_collision_free(&[0.0, 0.0]));
    }
}

#[cfg(test)]
mod learning_tests;
#[cfg(test)]
mod planning_tests;
#[cfg(test)]
mod rrt_tests;
