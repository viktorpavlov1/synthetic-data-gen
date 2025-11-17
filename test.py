"""Simple one-command test script for the synthetic data generation pipeline."""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("=" * 60)
print("Synthetic Skin Lesion Generation - Quick Test")
print("=" * 60)
print()

# Step 1: Check dependencies
print("[1/4] Checking dependencies...")
try:
    import torch
    print(f"  [OK] PyTorch {torch.__version__}")
except ImportError:
    print("  [FAIL] PyTorch not found. Run: pip install torch torchvision")
    sys.exit(1)

try:
    import kedro
    print("  [OK] Kedro installed")
except ImportError:
    print("  [FAIL] Kedro not found. Run: pip install kedro")
    sys.exit(1)

# Try to import diffusers - just warn if there are issues, don't fail
try:
    from diffusers import StableDiffusionPipeline
    print("  [OK] Diffusers installed")
except Exception as e:
    print("  [WARNING] Diffusers import check failed, but continuing...")
    print("  (The test will try to run anyway - version conflicts are common)")

try:
    from synthetic_data_gen.pipelines.data_generation.nodes import generate_images, create_prompts
    from synthetic_data_gen.pipelines.classification.nodes import classify_images
    print("  [OK] Package imports working")
except ImportError as e:
    print(f"  ✗ Package import failed: {e}")
    print("  Try running: pip install -e .")
    sys.exit(1)

print()

# Step 2: Generate images
print("[2/4] Generating 2 test images (this may take a few minutes on first run)...")
print("  This will download model weights if not already cached...")
print()

try:
    # Create simple prompts
    prompts = [
        "A high-quality dermoscopic image of a benign skin lesion, medical photography, detailed, professional",
        "A high-quality dermoscopic image of a melanoma skin lesion, medical photography, detailed, professional"
    ]
    
    # Generate images
    generated_images = generate_images(
        model_name="stable_diffusion",
        prompts=prompts,
        num_images=2,
        image_size=256,  # Smaller for faster testing
        batch_size=1,
        seed=42
    )
    
    print(f"  [OK] Generated {len(generated_images)} images")
    
except Exception as e:
    print(f"  [FAIL] Image generation failed: {e}")
    print("  This might be due to:")
    print("    - Missing dependencies (run: pip install -r requirements.txt)")
    print("    - GPU memory issues (try reducing image size)")
    print("    - Network issues (models need to be downloaded)")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 3: Classify images
print("[3/4] Classifying images...")
print()

try:
    results_df = classify_images(
        generated_images=generated_images,
        model_name="resnet50",
        batch_size=2,
        num_classes=2,
        class_names=["benign", "malignant"]
    )
    
    print(f"  [OK] Classified {len(results_df)} images")
    
except Exception as e:
    print(f"  [FAIL] Classification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 4: Show results
print("[4/4] Results:")
print("=" * 60)
print()

if len(results_df) > 0:
    print("Classification Results:")
    print("-" * 60)
    for idx, row in results_df.iterrows():
        print(f"Image {idx + 1}:")
        print(f"  Predicted: {row.get('predicted_class_name', 'unknown')}")
        print(f"  Confidence: {row.get('confidence', 0):.2%}")
        if 'prob_benign' in row:
            print(f"  Probabilities - Benign: {row['prob_benign']:.2%}, Malignant: {row.get('prob_malignant', 0):.2%}")
        print()
    
    # Summary
    if 'confidence' in results_df.columns:
        avg_conf = results_df['confidence'].mean()
        print(f"Average Confidence: {avg_conf:.2%}")
    
    if 'predicted_class_name' in results_df.columns:
        print("\nClass Distribution:")
        print(results_df['predicted_class_name'].value_counts())
else:
    print("No results to display")

print()
print("=" * 60)
print("[SUCCESS] Test completed successfully!")
print()
print("Next steps:")
print("  - Run the full CLI: python -m synthetic_data_gen.interface.cli run --help")
print("  - Generate more images: python -m synthetic_data_gen.interface.cli run --num-images 10")
print("=" * 60)

