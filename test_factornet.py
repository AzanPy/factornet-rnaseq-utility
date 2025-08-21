
"""
FactorNet Utility Test Script
============================

Simple test to verify that the FactorNet utility is working correctly.
Run this script to check if all dependencies are installed and the utility functions properly.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not found. Install with: pip install tensorflow")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False

    try:
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
    except ImportError:
        print("✗ Pandas not found. Install with: pip install pandas")
        return False

    try:
        import requests
        print(f"✓ Requests available")
    except ImportError:
        print("✗ Requests not found. Install with: pip install requests")
        return False

    return True

def test_utility_import():
    """Test if the FactorNet utility can be imported"""
    print("\nTesting utility import...")

    try:
        from rna_seq_factornet_utility import (
            GEODataLoader, 
            FactorNet, 
            RNASeqFactorNetPipeline
        )
        print("✓ FactorNet utility imported successfully")
        return True, (GEODataLoader, FactorNet, RNASeqFactorNetPipeline)
    except ImportError as e:
        print(f"✗ Could not import FactorNet utility: {e}")
        print("Make sure rna_seq_factornet_utility.py is in the same directory")
        return False, None

def test_basic_functionality():
    """Test basic functionality of the utility"""
    print("\nTesting basic functionality...")

    success, modules = test_utility_import()
    if not success:
        return False

    GEODataLoader, FactorNet, RNASeqFactorNetPipeline = modules

    try:
        # Test GEODataLoader
        loader = GEODataLoader()
        print("✓ GEODataLoader initialized")

        # Test FactorNet model building
        factor_net = FactorNet(
            sequence_length=100,  # Small for testing
            num_kernels=8,
            lstm_units=8,
            dense_units=16
        )
        model = factor_net.build_model()
        print("✓ FactorNet model built successfully")

        # Test pipeline initialization
        pipeline = RNASeqFactorNetPipeline()
        print("✓ RNASeqFactorNetPipeline initialized")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_simple_training():
    """Test simple training with synthetic data"""
    print("\nTesting simple training...")

    try:
        from rna_seq_factornet_utility import FactorNet

        # Create simple synthetic data
        sequences = []
        labels = []

        # Positive examples (with 'ATCG' pattern)
        for i in range(20):
            seq = 'A' * 25 + 'ATCG' + 'T' * 71
            sequences.append(seq)
            labels.append(1)

        # Negative examples (without pattern)
        for i in range(20):
            seq = 'G' * 50 + 'C' * 50
            sequences.append(seq)
            labels.append(0)

        # Train model
        factor_net = FactorNet(
            sequence_length=100,
            num_kernels=4,
            lstm_units=4,
            dense_units=8
        )

        factor_net.build_model()
        history = factor_net.train(
            sequences=sequences,
            labels=labels,
            epochs=2,  # Very short training
            batch_size=8
        )

        # Test prediction
        test_seq = ['A' * 25 + 'ATCG' + 'T' * 71]
        prediction = factor_net.predict(test_seq)

        print(f"✓ Training completed successfully")
        print(f"✓ Test prediction: {prediction[0]:.3f}")

        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("FactorNet Utility Test Suite")
    print("=" * 60)

    all_passed = True

    # Test 1: Imports
    if not test_imports():
        all_passed = False

    # Test 2: Utility import
    if not test_utility_import()[0]:
        all_passed = False

    # Test 3: Basic functionality
    if not test_basic_functionality():
        all_passed = False

    # Test 4: Simple training
    if not test_simple_training():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The FactorNet utility is ready to use.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above and install missing dependencies.")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    run_all_tests()
