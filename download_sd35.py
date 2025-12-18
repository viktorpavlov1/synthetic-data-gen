"""
Login to Hugging Face and download Stable Diffusion 3.5 Large.

This script will:
1. Ask for your Hugging Face token
2. Login programmatically
3. Download SD 3.5 Large (~10GB)

Get your token from: https://huggingface.co/settings/tokens
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def login_and_download():
    """Login to HuggingFace and download SD 3.5."""
    from huggingface_hub import login
    
    print("=" * 70)
    print("Stable Diffusion 3.5 Large - Login & Download")
    print("=" * 70)
    print()
    print("Before continuing:")
    print("  1. Visit: https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
    print("  2. Sign in and accept the license agreement")
    print("  3. Get your access token from: https://huggingface.co/settings/tokens")
    print("     (Create a 'Read' token if you don't have one)")
    print()
    
    # Ask for token
    token = input("Enter your Hugging Face token (or press Enter to skip login): ").strip()
    
    if token:
        try:
            print()
            print("🔑 Logging in to Hugging Face...")
            login(token=token, add_to_git_credential=True)
            print("✅ Login successful! Token saved for future use.")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            print()
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                return
    else:
        print()
        print("⚠️  Skipping login. This may fail if the model requires authentication.")
    
    print()
    print("📥 Starting download of Stable Diffusion 3.5 Large...")
    print("   Model ID: stabilityai/stable-diffusion-3.5-large")
    print("   Size: ~10GB")
    print("   This may take 10-30 minutes depending on your connection.")
    print()
    print("   Progress will be shown below:")
    print()
    
    try:
        import torch
        from diffusers import StableDiffusion3Pipeline
        
        # This will download the model if not cached
        model_id = "stabilityai/stable-diffusion-3.5-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            use_auth_token=True if token else None
        )
        
        print()
        print("✅ Download complete!")
        print()
        print("Model is now cached and ready to use in your application.")
        print("You won't need to download it again.")
        
        # Clean up
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        print()
        print(f"❌ Error during download: {e}")
        print()
        print("Common issues:")
        print("  1. No internet connection")
        print("  2. Not authenticated (enter valid token)")
        print("  3. License not accepted on model page")
        print("  4. Network timeout (try again)")
        print()
        print("Solutions:")
        print("  1. Check your internet connection")
        print("  2. Make sure you entered the correct token")
        print("  3. Visit: https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
        print("     and accept the license")
        print("  4. Try running this script again")
        sys.exit(1)

if __name__ == "__main__":
    try:
        login_and_download()
    except KeyboardInterrupt:
        print()
        print()
        print("⚠️  Download cancelled by user.")
        sys.exit(0)
