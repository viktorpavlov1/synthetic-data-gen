"""Simple test script to verify the installation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError as e:
    print(f"[FAIL] PyTorch not found: {e}")
    sys.exit(1)

try:
    import kedro
    print("[OK] Kedro installed")
except ImportError as e:
    print(f"[FAIL] Kedro not found: {e}")
    sys.exit(1)

try:
    from diffusers import StableDiffusionPipeline
    print("[OK] Diffusers installed")
except ImportError as e:
    print(f"[FAIL] Diffusers not found: {e}")
    sys.exit(1)

try:
    from synthetic_data_gen.utils.model_loaders import ENHANCEModelLoader
    print("[OK] Package imports working")
except ImportError as e:
    print(f"[FAIL] Package import failed: {e}")
    print("  Try running: pip install -e .")
    sys.exit(1)

print("\n[SUCCESS] All basic imports successful!")
print("\nYou can now run the CLI with:")
print("  python -m synthetic_data_gen.interface.cli run --generation-model stable_diffusion --num-images 2 --action classify")

