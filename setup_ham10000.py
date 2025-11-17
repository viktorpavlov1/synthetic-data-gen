"""One-command setup script for HAM10000 integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("HAM10000 Dataset Integration Setup")
print("=" * 60)
print()

ham10000_path = Path("data/00_external/HAM10000")

if not ham10000_path.exists():
    print(f"[FAIL] HAM10000 dataset not found at: {ham10000_path}")
    print("Please ensure the dataset is extracted to this location.")
    sys.exit(1)

print(f"[OK] HAM10000 dataset found at: {ham10000_path}")
print()

# Load and analyze dataset
try:
    from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
    
    print("Loading HAM10000 dataset...")
    loader = HAM10000Loader(str(ham10000_path))
    
    # Get statistics
    stats = loader.get_statistics()
    
    print("\nDataset Statistics:")
    print("-" * 60)
    print(f"Total Images: {stats.get('total_images', 0)}")
    print(f"\nDiagnosis Distribution:")
    for dx, count in stats.get('diagnosis_distribution', {}).items():
        print(f"  {dx}: {count}")
    
    print(f"\nAge Range: {stats.get('age_range', {}).get('min', 'N/A')} - {stats.get('age_range', {}).get('max', 'N/A')}")
    print(f"Sex Distribution: {stats.get('sex_distribution', {})}")
    
    print("\n[OK] HAM10000 dataset is ready!")
    print("\nThe system will automatically:")
    print("  - Use HAM10000 metadata for prompt generation")
    print("  - Allow fine-tuning on HAM10000 images")
    print("  - Generate more realistic prompts based on actual data")
    
    print("\nNext Steps:")
    print("  1. Run the web interface: python run_interface.py")
    print("  2. The system will automatically detect and use HAM10000")
    print("  3. Go to 'Fine-Tune Generator' tab for quick setup")
    
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

