
"""
FactorNet RNA-seq Analysis Tutorial
==================================

This tutorial demonstrates how to use the FactorNet ML utility for RNA-seq data analysis.
The utility provides a complete pipeline for:
1. Loading RNA-seq data from GEO
2. Training FactorNet for transcription factor binding prediction
3. Making predictions and interpreting results

Tutorial Steps:
1. Basic usage example
2. Advanced configuration
3. Custom data loading
4. Model interpretation
5. Saving and loading models
"""

# Import the utility
from rna_seq_factornet_utility import (
    GEODataLoader, 
    FactorNet, 
    RNASeqFactorNetPipeline,
    run_example_analysis
)
import numpy as np
import pandas as pd

def tutorial_step_1_basic_usage():
    """
    Step 1: Basic usage of the FactorNet utility
    """
    print("=" * 60)
    print("STEP 1: Basic Usage Example")
    print("=" * 60)

    # Initialize the complete pipeline
    pipeline = RNASeqFactorNetPipeline()

    # Load data from GEO (this will create example data if GEO is not accessible)
    print("Loading data from GEO...")
    data = pipeline.load_and_process_data("GSE68086", sequence_length=1000)

    print(f"Loaded data:")
    print(f"- Number of sequences: {len(data['sequences'])}")
    print(f"- Number of labels: {len(data['labels'])}")
    print(f"- Expression data shape: {data['expression_data'].shape}")
    print(f"- Sample sequence length: {len(data['sequences'][0])}")

    # Train the model with basic parameters
    print("\nTraining FactorNet model...")
    history = pipeline.train_model(
        num_kernels=32,    # Reduced for faster training
        lstm_units=32,     # Reduced for faster training
        dense_units=64,    # Reduced for faster training
        epochs=5           # Reduced for demonstration
    )

    print("Training completed!")
    return pipeline

def tutorial_step_2_advanced_configuration():
    """
    Step 2: Advanced configuration and custom parameters
    """
    print("\n" + "=" * 60)
    print("STEP 2: Advanced Configuration")
    print("=" * 60)

    # Create FactorNet with custom parameters
    factor_net = FactorNet(
        sequence_length=1200,      # Longer sequences
        num_kernels=128,           # More kernels for complex patterns
        kernel_size=20,            # Smaller kernels
        lstm_units=64,             # More LSTM units
        dense_units=256,           # Larger dense layer
        dropout_rate=0.3           # Lower dropout
    )

    # Build the model to see architecture
    model = factor_net.build_model()

    print("Model Architecture:")
    model.summary()

    return factor_net

def tutorial_step_3_custom_data():
    """
    Step 3: Working with custom data
    """
    print("\n" + "=" * 60)
    print("STEP 3: Custom Data Handling")
    print("=" * 60)

    # Create custom sequences and labels
    print("Creating custom dataset...")

    # Generate custom DNA sequences with known binding patterns
    binding_motif = "TGACTCA"  # Example TF binding motif
    sequences = []
    labels = []

    # Create sequences with binding sites (positive examples)
    for i in range(50):
        # Insert motif at random position
        position = np.random.randint(100, 900)
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], 1000))
        seq = seq[:position] + binding_motif + seq[position+len(binding_motif):]
        sequences.append(seq)
        labels.append(1)

    # Create sequences without binding sites (negative examples)
    for i in range(50):
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], 1000))
        # Make sure no binding motif is present
        while binding_motif in seq:
            seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], 1000))
        sequences.append(seq)
        labels.append(0)

    print(f"Created custom dataset:")
    print(f"- {len(sequences)} sequences")
    print(f"- {sum(labels)} positive examples")
    print(f"- {len(labels) - sum(labels)} negative examples")
    print(f"- Binding motif: {binding_motif}")

    # Train model on custom data
    factor_net = FactorNet()
    factor_net.build_model()

    print("\nTraining on custom data...")
    history = factor_net.train(
        sequences=sequences,
        labels=labels,
        epochs=10,
        batch_size=16
    )

    # Test predictions on new sequences
    test_seq_with_motif = 'A' * 400 + binding_motif + 'T' * 593
    test_seq_without_motif = 'A' * 500 + 'T' * 500

    pred_with = factor_net.predict([test_seq_with_motif])[0]
    pred_without = factor_net.predict([test_seq_without_motif])[0]

    print(f"\nPrediction results:")
    print(f"- Sequence with motif: {pred_with:.3f}")
    print(f"- Sequence without motif: {pred_without:.3f}")

    return factor_net, sequences, labels

def tutorial_step_4_interpretation():
    """
    Step 4: Model interpretation and visualization
    """
    print("\n" + "=" * 60)
    print("STEP 4: Model Interpretation")
    print("=" * 60)

    # Use the trained model from step 3
    factor_net, sequences, labels = tutorial_step_3_custom_data()

    # Interpret some predictions
    test_sequences = sequences[:5]  # First 5 sequences
    interpretation = factor_net.interpret_model(test_sequences)

    print("Interpretation results:")
    print(f"- Saliency scores shape: {interpretation['saliency_scores'].shape}")
    print(f"- Number of predictions: {len(interpretation['predictions'])}")

    # Analyze saliency scores
    for i, (seq, pred, saliency) in enumerate(zip(
        interpretation['sequences'][:3],
        interpretation['predictions'][:3],
        interpretation['saliency_scores'][:3]
    )):
        print(f"\nSequence {i+1}:")
        print(f"- Prediction: {pred:.3f}")
        print(f"- Max saliency position: {np.argmax(np.sum(saliency, axis=1))}")
        print(f"- Sequence around max saliency: {seq[np.argmax(np.sum(saliency, axis=1)):np.argmax(np.sum(saliency, axis=1))+20]}")

    return interpretation

def tutorial_step_5_save_load():
    """
    Step 5: Saving and loading models
    """
    print("\n" + "=" * 60)
    print("STEP 5: Saving and Loading Models")
    print("=" * 60)

    # Create and train a simple model
    pipeline = RNASeqFactorNetPipeline()
    data = pipeline.load_and_process_data("GSE12345")

    history = pipeline.train_model(
        num_kernels=16,
        lstm_units=16,
        dense_units=32,
        epochs=3
    )

    # Save the pipeline
    print("Saving pipeline...")
    pipeline.save_pipeline("./demo_factornet")

    # Create new pipeline and load
    print("Loading pipeline...")
    new_pipeline = RNASeqFactorNetPipeline()
    new_pipeline.load_pipeline("./demo_factornet")

    # Test loaded model
    test_sequences = data['sequences'][:3]
    predictions = new_pipeline.predict_binding(test_sequences)

    print(f"Loaded model predictions:")
    for i, pred in enumerate(predictions):
        print(f"- Sequence {i+1}: {pred:.3f}")

    print("Save/Load completed successfully!")

def create_usage_guide():
    """
    Create a comprehensive usage guide
    """
    guide = """# FactorNet RNA-seq Utility - Usage Guide

## Quick Start

from rna_seq_factornet_utility import RNASeqFactorNetPipeline

# Initialize pipeline
pipeline = RNASeqFactorNetPipeline()

# Load data from GEO
data = pipeline.load_and_process_data("GSE68086")

# Train model
history = pipeline.train_model(epochs=20)

# Make predictions
sequences = ["ATCG" * 250]  # Your DNA sequences
predictions = pipeline.predict_binding(sequences)

## Components

### 1. GEODataLoader
- Downloads RNA-seq data from GEO database
- Caches data locally for reuse
- Handles multiple file formats

### 2. FactorNet
- Custom implementation of FactorNet architecture
- Siamese network design for forward/reverse complement
- CNN + Bidirectional LSTM architecture
- Built-in interpretation methods

### 3. RNASeqFactorNetPipeline
- Complete pipeline combining data loading and modeling
- Easy save/load functionality
- Preprocessing and postprocessing

## Parameters

### FactorNet Parameters:
- sequence_length: Length of input DNA sequences (default: 1000)
- num_kernels: Number of convolutional kernels (default: 128)
- kernel_size: Size of convolutional kernels (default: 26)
- lstm_units: Number of LSTM units (default: 128)
- dense_units: Number of dense layer units (default: 256)
- dropout_rate: Dropout rate (default: 0.5)

### Training Parameters:
- epochs: Number of training epochs
- batch_size: Batch size for training
- validation_split: Fraction for validation

## Advanced Usage

### Custom Data Loading
loader = GEODataLoader()
data = loader.download_gse_data("GSE12345")
processed = loader.preprocess_for_factornet(data)

### Model Interpretation
interpretation = pipeline.interpret_predictions(sequences)
saliency_scores = interpretation['saliency_scores']

### Custom Model Configuration
factor_net = FactorNet(
    sequence_length=1200,
    num_kernels=64,
    lstm_units=32
)

## Tips and Best Practices

1. **Data Quality**: Ensure your RNA-seq data is properly normalized
2. **Sequence Length**: Use appropriate sequence length for your TF of interest
3. **Training**: Monitor validation loss to avoid overfitting
4. **Interpretation**: Use saliency analysis to understand model decisions
5. **Caching**: The utility caches data to speed up repeated analyses

## Troubleshooting

- **Memory Issues**: Reduce batch_size or sequence_length
- **GEO Download Issues**: The utility creates example data if download fails
- **Training Issues**: Try reducing learning rate or increasing epochs
"""

    with open('factornet_usage_guide.md', 'w') as f:
        f.write(guide)

    print("Usage guide created: factornet_usage_guide.md")

def run_complete_tutorial():
    """
    Run the complete tutorial
    """
    print("Starting FactorNet RNA-seq Analysis Tutorial...")
    print("This tutorial will demonstrate all features of the utility.\n")

    # Run all tutorial steps
    try:
        pipeline1 = tutorial_step_1_basic_usage()
        factor_net1 = tutorial_step_2_advanced_configuration()
        interpretation = tutorial_step_4_interpretation()
        tutorial_step_5_save_load()
        create_usage_guide()

        print("\n" + "=" * 60)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou now have:")
        print("- A trained FactorNet model")
        print("- Example of custom data handling")
        print("- Model interpretation results")
        print("- Saved/loaded models")
        print("- Usage guide (factornet_usage_guide.md)")

    except Exception as e:
        print(f"Tutorial encountered an error: {str(e)}")
        print("This is likely due to missing dependencies.")
        print("To use this utility, install: tensorflow, numpy, pandas, requests")

if __name__ == "__main__":
    run_complete_tutorial()
