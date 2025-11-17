"""Setup file for synthetic_data_gen package."""

from setuptools import setup, find_packages

setup(
    name="synthetic-data-gen",
    version="0.1.0",
    description="Synthetic Skin Lesion Generation and Classification Pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "kedro>=0.18.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "synthetic-data-gen=synthetic_data_gen.interface.cli:cli",
        ],
    },
)

