use super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::{Array1, ScientificNumber};

#[test]
fn test_neuron_models() {
    let lif = NeuronModel::LeakyIntegrateFire {
        tau_m: 20.0,
        v_rest: -70.0,
        v_threshold: -55.0,
        v_reset: -70.0,
    };

    match lif {
        NeuronModel::LeakyIntegrateFire { tau_m, .. } => {
            assert_eq!(tau_m, 20.0);
        }
        _ => panic!("Wrong neuron model"),
    }

    let izhikevich = NeuronModel::Izhikevich {
        a: 0.02,
        b: 0.2,
        c: -65.0,
        d: 8.0,
    };

    match izhikevich {
        NeuronModel::Izhikevich { a, b, c, d } => {
            assert_eq!(a, 0.02);
            assert_eq!(b, 0.2);
            assert_eq!(c, -65.0);
            assert_eq!(d, 8.0);
        }
        _ => panic!("Wrong neuron model"),
    }
}

#[test]
fn test_spiking_neural_network() {
    let layer_sizes = vec![3, 5, 2];
    let neuron_models = vec![
        NeuronModel::LeakyIntegrateFire {
            tau_m: 20.0,
            v_rest: -70.0,
            v_threshold: -55.0,
            v_reset: -70.0,
        },
        NeuronModel::LeakyIntegrateFire {
            tau_m: 20.0,
            v_rest: -70.0,
            v_threshold: -55.0,
            v_reset: -70.0,
        },
        NeuronModel::LeakyIntegrateFire {
            tau_m: 20.0,
            v_rest: -70.0,
            v_threshold: -55.0,
            v_reset: -70.0,
        },
    ];
    let plasticity_rules = vec![
        PlasticityRule::STDP {
            tau_plus: 20.0,
            tau_minus: 20.0,
            a_plus: 0.01,
            a_minus: 0.01,
        },
        PlasticityRule::STDP {
            tau_plus: 20.0,
            tau_minus: 20.0,
            a_plus: 0.01,
            a_minus: 0.01,
        },
    ];

    let snn = SpikingNeuralNetwork::<f64>::new(layer_sizes, neuron_models, plasticity_rules, 0.1);
    assert!(snn.is_ok());

    let mut network = snn.expect("Operation failed");

    // Test spike encoding
    let data = Array1::from_vec(vec![0.1, 0.5, 0.8, 0.3, 0.9]);
    let spikes = network.encode_time_series(&data);
    assert!(!spikes.is_empty());

    // Test network processing
    let result = network.process_spikes(&spikes);
    assert!(result.is_ok());
}

#[test]
fn test_liquid_state_machine() {
    let lsm = LiquidStateMachine::<f64>::new(10, 3, 2, 0.9, 0.1);
    assert!(lsm.is_ok());

    let mut machine = lsm.expect("Operation failed");

    // Test processing
    let data = Array1::from_vec(vec![0.1, 0.2, 0.3]);
    let result = machine.process_time_series(&data);
    assert!(result.is_ok());

    let output = result.expect("Operation failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_memristive_network() {
    let learning_params = MemristiveLearningParams {
        learning_rate: 0.01,
        decay: 0.99,
        noise: 0.01,
        threshold: 0.1,
    };

    let network =
        MemristiveNetwork::<f64>::new(5, NetworkTopology::FullyConnected, learning_params);

    assert_eq!(network.crossbar.dim(), (5, 5));

    // Test output computation
    let input = Array1::from_vec(vec![1.0, 0.5, 0.0, 0.8, 0.3]);
    let output = network.compute_output(&input);
    assert_eq!(output.len(), 5);
}

#[test]
fn test_neuron_state() {
    let state = NeuronState::<f64>::default();

    assert_abs_diff_eq!(
        state.v.to_f64().expect("Operation failed"),
        -70.0,
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(
        state.u.to_f64().expect("Operation failed"),
        0.0,
        epsilon = 1e-10
    );
    assert!(state.last_spike.is_none());
    assert_eq!(state.refractory, 0.0);
}

#[test]
fn test_plasticity_rules() {
    let stdp = PlasticityRule::STDP {
        tau_plus: 20.0,
        tau_minus: 20.0,
        a_plus: 0.01,
        a_minus: 0.01,
    };

    match stdp {
        PlasticityRule::STDP {
            tau_plus,
            tau_minus,
            a_plus,
            a_minus,
        } => {
            assert_eq!(tau_plus, 20.0);
            assert_eq!(tau_minus, 20.0);
            assert_eq!(a_plus, 0.01);
            assert_eq!(a_minus, 0.01);
        }
        _ => panic!("Wrong plasticity rule"),
    }

    let hebbian = PlasticityRule::Hebbian {
        learning_rate: 0.01,
        decay_rate: 0.001,
    };

    match hebbian {
        PlasticityRule::Hebbian {
            learning_rate,
            decay_rate,
        } => {
            assert_eq!(learning_rate, 0.01);
            assert_eq!(decay_rate, 0.001);
        }
        _ => panic!("Wrong plasticity rule"),
    }
}

#[test]
fn test_spike_structure() {
    let spike = Spike {
        time: 10.5,
        neuron_id: 42,
        amplitude: 1.2,
    };

    assert_eq!(spike.time, 10.5);
    assert_eq!(spike.neuron_id, 42);
    assert_eq!(spike.amplitude, 1.2);
}
