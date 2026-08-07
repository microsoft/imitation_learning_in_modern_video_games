#!/usr/bin/env python
"""Evaluate pretrained models on CS:GO HDF5 data"""

import argparse
import h5py
import json
from pathlib import Path
import numpy as np

def evaluate_pretrained(model_path, data_dir, num_files=5):
    """
    Load pretrained TensorFlow model and evaluate on HDF5 data
    Computes action prediction accuracy
    """
    print(f"Evaluating pretrained model: {model_path}")
    print(f"Data directory: {data_dir}")
    print()

    # Try to import tensorflow
    try:
        import tensorflow as tf
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {model.summary()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: Model is TensorFlow/Keras format, not PyTorch")
        return

    # Load HDF5 data
    hdf5_files = sorted(Path(data_dir).glob("hdf5_dm_july2021_*.hdf5"))[:num_files]
    print(f"\nLoading {len(hdf5_files)} HDF5 files...")

    all_frames = []
    all_actions = []

    for hdf5_file in hdf5_files:
        print(f"  Loading {hdf5_file.name}...")
        with h5py.File(hdf5_file, 'r') as f:
            # Load first 100 frames from each file
            for i in range(100):
                try:
                    frame = f[f'frame_{i}_x'][:]
                    action = f[f'frame_{i}_y'][:]
                    all_frames.append(frame)
                    all_actions.append(action)
                except KeyError:
                    break

    all_frames = np.array(all_frames, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    print(f"\nLoaded {len(all_frames)} frames")
    print(f"Frame shape: {all_frames[0].shape}")
    print(f"Action dim: {all_actions[0].shape}")
    print()

    # Normalize frames (0-1)
    all_frames = all_frames / 255.0

    # Run inference
    print("Running inference...")
    try:
        predictions = model.predict(all_frames[:1000], batch_size=32, verbose=1)
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Compute metrics
    print("\nComputing metrics...")

    # Action prediction accuracy (for button actions)
    # Actions are one-hot encoded, find argmax
    pred_actions = np.argmax(predictions, axis=-1) if predictions.shape[-1] > 1 else predictions
    target_actions = np.argmax(all_actions[:len(predictions)], axis=-1)
    accuracy = np.mean(pred_actions == target_actions)

    # MSE
    mse = np.mean((predictions - all_actions[:len(predictions)]) ** 2)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = np.mean([
        cosine_similarity([predictions[i]], [all_actions[i]])[0, 0]
        for i in range(min(100, len(predictions)))
    ])

    print(f"\nEvaluation Results:")
    print(f"  Action Accuracy: {accuracy:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine Similarity: {cos_sim:.4f}")

    # Save results
    results = {
        "model": str(model_path),
        "num_frames": len(all_frames),
        "accuracy": float(accuracy),
        "mse": float(mse),
        "cosine_similarity": float(cos_sim),
    }

    results_file = Path("eval_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trained_models/ak47_sub_55k_drop_d4_dmexpert_28.h5')
    parser.add_argument('--data-dir', type=str, default=r'C:\csgo_data')
    parser.add_argument('--num-files', type=int, default=5)
    args = parser.parse_args()

    evaluate_pretrained(args.model, args.data_dir, args.num_files)
