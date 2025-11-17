"""Install dependencies and run test - all in one script."""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n[{description}]")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print("-" * 60)
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=True, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

print("=" * 60)
print("Installing dependencies and running test...")
print("=" * 60)

# Step 1: Upgrade all ML dependencies together
if not run_command(
    [sys.executable, "-m", "pip", "install", "--upgrade", 
     "tokenizers", "transformers", "diffusers", "peft", "huggingface-hub", "accelerate"],
    "Step 1: Upgrading ML dependencies"
):
    print("\n[WARNING] Failed to upgrade packages, but continuing...")

# Step 2: Install requirements
if not run_command(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    "Step 2: Installing requirements"
):
    print("\n[ERROR] Failed to install requirements")
    input("Press Enter to exit...")
    sys.exit(1)

# Step 3: Install package
if not run_command(
    [sys.executable, "-m", "pip", "install", "-e", "."],
    "Step 3: Installing package"
):
    print("\n[WARNING] Failed to install package, but continuing...")

# Step 4: Run test
print("\n" + "=" * 60)
print("Step 4: Running test...")
print("=" * 60)
print()

# Run the test script
try:
    result = subprocess.run([sys.executable, "test.py"], check=False)
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\n\nTest interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Failed to run test: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

