#!/usr/bin/env python
"""CS:GO rollout using TensorFlow pretrained model."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def load_model_safe(model_path):
    """Load model with fallbacks for compatibility issues."""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        # Try standard load
        try:
            model = tf.keras.models.load_model(str(model_path))
            print(f"✓ Loaded with tf.keras.models.load_model")
            return model
        except Exception as e1:
            print(f"Standard load failed: {e1}")

            # Try with custom_objects
            try:
                from tensorflow.keras.applications import EfficientNetB0
                model = tf.keras.models.load_model(
                    str(model_path),
                    custom_objects={
                        'Functional': tf.keras.Model,
                    }
                )
                print(f"✓ Loaded with custom_objects fallback")
                return model
            except Exception as e2:
                print(f"Custom objects fallback failed: {e2}")

                # Try h5py direct
                try:
                    import h5py
                    with h5py.File(str(model_path), 'r') as f:
                        print(f"✓ HDF5 file readable")
                        print(f"  Keys: {list(f.keys())}")
                        if 'model_config' in f.attrs:
                            config = f.attrs['model_config']
                            print(f"  Model config present: {len(config)} bytes")
                    raise Exception("Model file corrupt or incompatible - requires native TensorFlow version")
                except Exception as e3:
                    print(f"H5py inspection failed: {e3}")
                    raise

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", default="rollout_videos")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return 1

    print(f"Loading: {model_path}")
    model = load_model_safe(str(model_path))

    if model:
        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"✓ Ready for inference")
        return 0
    else:
        print(f"✗ Could not load model")
        return 1

if __name__ == "__main__":
    sys.exit(main())
