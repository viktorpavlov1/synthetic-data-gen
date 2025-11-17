"""Test script to check if images are being generated correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_data_gen.pipelines.data_generation.nodes import generate_images
from PIL import Image
import numpy as np

print("Testing image generation...")
print("=" * 60)

# Generate 1 test image
prompts = ["A high-quality dermoscopic image of a benign skin lesion, medical photography, detailed, professional"]

try:
    generated_images = generate_images(
        model_name="stable_diffusion",
        prompts=prompts,
        num_images=1,
        image_size=256,
        batch_size=1,
        seed=42
    )
    
    print(f"\nGenerated {len(generated_images)} image(s)")
    
    if generated_images:
        img_data = generated_images[0]
        image = img_data.get('image')
        
        print(f"\nImage type: {type(image)}")
        print(f"Is PIL Image: {isinstance(image, Image.Image)}")
        
        if isinstance(image, Image.Image):
            print(f"Image mode: {image.mode}")
            print(f"Image size: {image.size}")
            
            # Check if image is all black
            img_array = np.array(image)
            print(f"Image array shape: {img_array.shape}")
            print(f"Image array min: {img_array.min()}")
            print(f"Image array max: {img_array.max()}")
            print(f"Image array mean: {img_array.mean():.2f}")
            
            if img_array.max() == 0:
                print("\n❌ ERROR: Image is completely black!")
            elif img_array.mean() < 10:
                print("\n⚠️ WARNING: Image appears very dark (mean < 10)")
            else:
                print("\n✅ Image appears to have content")
            
            # Try to save and display
            test_path = "test_generated_image.png"
            image.save(test_path)
            print(f"\n✅ Saved test image to: {test_path}")
            print("   Please check this file to see if the image is visible")
        else:
            print(f"\n❌ ERROR: Image is not a PIL Image, it's: {type(image)}")
    else:
        print("\n❌ ERROR: No images generated!")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

