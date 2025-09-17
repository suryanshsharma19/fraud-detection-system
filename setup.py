"""
Fraud Detection System - Production-ready ML system for financial fraud detection

This package provides a complete fraud detection solution with:
- Real-time transaction processing API
- Ensemble ML models (Random Forest + XGBoost)
- GPU-accelerated training pipeline  
- Interactive monitoring dashboard
- Docker containerization

Author: Suryansh Sharma
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Parse requirements from requirements.txt, excluding GPU-specific ones for general install
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = []
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#"):
            # Skip GPU-specific packages for base install
            if not any(gpu_lib in line.lower() for gpu_lib in ['torch', 'tensorflow', 'cupy', 'rapids']):
                requirements.append(line)

setup(
    name="fraud-detection-system",
    version="1.0.0",
    author="Suryansh Sharma",
    author_email="contact@suryanshsharma.dev",  # Update with your email
    description="Production-ready fraud detection system with ensemble ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suryanshsharma19/fraud-detection-system",  # Update with your repo URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",  
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "gpu": ["torch>=2.1.0", "torchvision>=0.16.0"],
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0", "black", "isort"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "fraud-api=api.main:main",
            "fraud-train=train_gpu_final:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
