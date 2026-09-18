# SciRS2 Neural Network Examples

This directory contains comprehensive examples demonstrating the full capabilities of the `scirs2-neural` crate, showcasing everything from basic neural network operations to advanced deep learning architectures and training techniques.

## 🚀 Ultrathink Mode Examples

The following examples demonstrate the advanced "ultrathink mode" capabilities of scirs2-neural:

### Core Showcase Examples

- **`ultrathink_neural_showcase.rs`** - Comprehensive demonstration of all major neural network features
- **`ultrathink_practical_training.rs`** - End-to-end practical training pipeline with production deployment

### Architecture Examples

#### Transformer Architectures
- **`transformer_example.rs`** - Basic transformer implementation
- **`bert_example.rs`** - BERT model for natural language understanding
- **`gpt_example.rs`** - GPT-style autoregressive language model
- **`vit_example.rs`** - Vision Transformer for image classification
- **`transformer_translation.rs`** - Sequence-to-sequence transformer for translation

#### Convolutional Networks
- **`convolutional_neural_network.rs`** - Standard CNN architectures
- **`resnet_example.rs`** - ResNet with skip connections
- **`efficientnet_example.rs`** - EfficientNet scaled architecture
- **`convnext_example.rs`** - ConvNeXt modern CNN architecture
- **`mobilenet_example.rs`** - MobileNet for mobile deployment

#### Recurrent Networks
- **`recurrent_layers.rs`** - LSTM and GRU implementations
- **`bidirectional_lstm_embedding.rs`** - Bidirectional LSTM with embeddings
- **`seq2seq_example.rs`** - Sequence-to-sequence models
- **`seq2seq_attention.rs`** - Attention mechanisms in seq2seq

### Training and Optimization

#### Basic Training
- **`neural_network_xor.rs`** - Simple XOR problem demonstration
- **`simple_xor_network.rs`** - Minimal neural network example
- **`manual_xor.rs`** - Manual backpropagation implementation
- **`improved_xor.rs`** - Enhanced XOR with modern techniques

#### Advanced Training
- **`advanced_training_example.rs`** - Comprehensive training pipeline
- **`training_loop_example.rs`** - Custom training loop implementation
- **`minibatch_training.rs`** - Efficient minibatch processing
- **`memory_efficient_example.rs`** - Memory optimization techniques

#### Optimization Techniques
- **`advanced_optimizers_example.rs`** - Adam, AdamW, and other optimizers
- **`scheduler_optimizer.rs`** - Learning rate scheduling
- **`gradient_clipping_example.rs`** - Gradient clipping strategies

### Specialized Applications

#### Computer Vision
- **`image_classification_complete.rs`** - End-to-end image classification
- **`object_detection_complete.rs`** - Object detection pipeline
- **`semantic_segmentation_complete.rs`** - Semantic segmentation
- **`model_visualization_cnn.rs`** - CNN visualization techniques

#### Natural Language Processing
- **`text_classification_complete.rs`** - Text classification pipeline
- **`sentiment_analysis_rnn.rs`** - Sentiment analysis with RNNs
- **`text_generation_rnn.rs`** - Text generation models
- **`embedding_example.rs`** - Word embeddings

#### Generative Models
- **`generative_models_complete.rs`** - VAE, GAN implementations
- **`new_features_showcase.rs`** - Latest generative techniques

#### Multi-modal Learning
- **`multimodal_neural_network.rs`** - Vision + Language models

### Neural Network Components

#### Layers and Activations
- **`activations_example.rs`** - Various activation functions
- **`dense_layer_example.rs`** - Dense/fully connected layers
- **`normalization_layers.rs`** - Batch norm, layer norm, etc.
- **`dropout_example.rs`** - Dropout and regularization
- **`batchnorm_example.rs`** - Batch normalization specifics

#### Loss Functions
- **`loss_functions_example.rs`** - Various loss function implementations

#### Attention Mechanisms
- **`attention_example.rs`** - Self-attention and multi-head attention

### Training Infrastructure

#### Callbacks and Monitoring
- **`training_callbacks.rs`** - Training callback system
- **`advanced_callbacks.rs`** - Advanced callback implementations
- **`visualize_training_progress.rs`** - Training visualization

#### Model Management
- **`model_serialization.rs`** - Model saving and loading
- **`improved_model_serialization.rs`** - Enhanced serialization
- **`model_serialization_example.rs`** - Serialization examples
- **`model_config_example.rs`** - Model configuration management

### Evaluation and Visualization

#### Model Evaluation
- **`model_evaluation_example.rs`** - Comprehensive model evaluation
- **`neural_confusion_matrix.rs`** - Confusion matrix analysis
- **`metrics_integration_example.rs`** - Metrics computation

#### Visualization
- **`model_visualization_example.rs`** - Model architecture visualization
- **`model_visualization_simple.rs`** - Simple visualization examples
- **`model_architecture_visualization.rs`** - Architecture diagrams
- **`colored_curve_visualization.rs`** - Training curve visualization
- **`colored_eval_visualization.rs`** - Evaluation visualization
- **`error_pattern_heatmap.rs`** - Error analysis heatmaps

### Advanced Features

#### Performance Optimization
- **`simd_acceleration_example.rs`** - SIMD optimizations
- **`accelerated_neural_ops_example.rs`** - Hardware acceleration
- **`regularization_techniques.rs`** - Advanced regularization

#### Time Series and Forecasting
- **`time_series_forecasting.rs`** - Time series neural networks

#### Neural Architecture Search
- **`neural_advanced_features.rs`** - NAS and AutoML features

#### Specialized Networks
- **`unified_neural_network.rs`** - Multi-task neural networks
- **`general_purpose_nn.rs`** - General-purpose architectures

## 🏃‍♂️ Running the Examples

### Prerequisites

Make sure you have the required dependencies installed:

```bash
# Install SciRS2 neural network crate
cargo add scirs2-neural

# For GPU acceleration (optional)
cargo add scirs2-neural --features="gpu,cuda"

# For visualization examples (optional)
cargo add scirs2-neural --features="visualization"
```

### Basic Usage

Run any example with:

```bash
cargo run --example <example_name>
```

For example:
```bash
# Run the comprehensive ultrathink showcase
cargo run --example ultrathink_neural_showcase

# Run practical training pipeline
cargo run --example ultrathink_practical_training

# Run a simple XOR example
cargo run --example neural_network_xor

# Run image classification
cargo run --example image_classification_complete
```

### With GPU Acceleration

```bash
cargo run --example ultrathink_neural_showcase --features="gpu,cuda"
```

### With Visualization

```bash
cargo run --example model_visualization_example --features="visualization"
```

## 📚 Example Categories

### 🎯 Beginner Examples
Start here if you're new to neural networks:
- `neural_network_xor.rs`
- `simple_xor_network.rs`
- `activations_example.rs`
- `loss_functions_example.rs`

### 🚀 Intermediate Examples
For those with some neural network experience:
- `convolutional_neural_network.rs`
- `recurrent_layers.rs`
- `image_classification_complete.rs`
- `text_classification_complete.rs`

### 🧠 Advanced Examples
For deep learning practitioners:
- `transformer_example.rs`
- `generative_models_complete.rs`
- `multimodal_neural_network.rs`
- `advanced_training_example.rs`

### 🌟 Ultrathink Mode
Cutting-edge features and comprehensive demonstrations:
- `ultrathink_neural_showcase.rs`
- `ultrathink_practical_training.rs`
- `neural_advanced_features.rs`

## 🔧 Features Demonstrated

### Core Neural Network Features
- ✅ Dense/Linear layers
- ✅ Convolutional layers (1D, 2D, 3D)
- ✅ Recurrent layers (LSTM, GRU)
- ✅ Attention mechanisms
- ✅ Transformer architectures
- ✅ Normalization layers
- ✅ Dropout and regularization
- ✅ Various activation functions
- ✅ Multiple loss functions

### Advanced Training
- ✅ Automatic differentiation
- ✅ Advanced optimizers (Adam, AdamW, etc.)
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Mixed precision training
- ✅ Distributed training
- ✅ Memory-efficient training

### Model Architecture
- ✅ Pre-built architectures (ResNet, EfficientNet, etc.)
- ✅ Custom model building
- ✅ Sequential and functional APIs
- ✅ Multi-input/multi-output models
- ✅ Model subclassing

### Evaluation and Visualization
- ✅ Comprehensive metrics
- ✅ Model visualization
- ✅ Training progress monitoring
- ✅ Confusion matrices
- ✅ Learning curves

### Production Features
- ✅ Model serialization/deserialization
- ✅ Model compression and quantization
- ✅ ONNX export
- ✅ Mobile deployment
- ✅ Serving infrastructure

### Hardware Acceleration
- ✅ SIMD optimizations
- ✅ GPU acceleration (CUDA)
- ✅ Multi-threading
- ✅ Memory optimization

## 🎨 Visualization Examples

Many examples include visualization capabilities:

- **Training Progress**: Real-time loss and accuracy plots
- **Model Architecture**: Network structure diagrams
- **Attention Maps**: Visualization of attention patterns
- **Feature Maps**: CNN feature visualization
- **Confusion Matrices**: Classification result analysis
- **Learning Curves**: Training and validation curves

## 🚀 Performance Examples

Examples demonstrating performance optimization:

- **SIMD Acceleration**: Vectorized operations
- **GPU Acceleration**: CUDA-based computations
- **Memory Efficiency**: Reduced memory usage techniques
- **Batch Processing**: Efficient batch operations
- **Distributed Training**: Multi-GPU and multi-node training

## 📖 Documentation

Each example includes:
- Comprehensive code comments
- Usage instructions
- Parameter explanations
- Expected outputs
- Performance benchmarks (where applicable)

## 🤝 Contributing

To add new examples:

1. Create a new `.rs` file in this directory
2. Follow the naming convention: `category_specific_name.rs`
3. Include comprehensive documentation
4. Add the example to this README
5. Test the example thoroughly

## 📄 License

All examples are provided under the same license as the SciRS2 project.

## 🔗 Additional Resources

- [SciRS2 Neural Documentation](../docs/)
- [API Reference](https://docs.rs/scirs2-neural)
- [Performance Benchmarks](../benches/)
- [Integration Tests](../tests/)

---

*These examples demonstrate the full power of SciRS2's neural network capabilities, from basic concepts to cutting-edge research implementations.*