"""Helper script to prepare dataset for fine-tuning."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_data_gen.utils.dataset_loader import SkinLesionDatasetLoader
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["flat", "by_class", "train_val_test"],
        default="flat",
        help="Dataset structure"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to load (None for all)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of loaded images"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)
    print()
    
    try:
        loader = SkinLesionDatasetLoader(args.dataset_path)
        
        print(f"Loading images from: {args.dataset_path}")
        print(f"Structure: {args.structure}")
        if args.max_images:
            print(f"Max images: {args.max_images}")
        print()
        
        images = loader.load_from_structured_format(
            structure=args.structure,
            max_images=args.max_images
        )
        
        print(f"✅ Successfully loaded {len(images)} images")
        
        if args.preview and images:
            print("\nPreview of first 5 images:")
            for i, img_data in enumerate(images[:5]):
                print(f"  Image {i+1}: {img_data.get('filename', 'unknown')}")
                if 'path' in img_data:
                    print(f"    Path: {img_data['path']}")
        
        print("\n✅ Dataset is ready for fine-tuning!")
        print(f"\nUse this path in the web interface: {args.dataset_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

