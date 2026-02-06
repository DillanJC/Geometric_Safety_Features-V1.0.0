from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mirrorfield-geometry",
    version="2.0.0",
    author="Dillan John Coghlan",
    author_email="DillanJC91@Gmail.com",
    description="k-NN Geometric Features for AI Safety Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DillanJC/geometric_safety_features",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "plots": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "experiments": ["openai>=1.0.0", "python-dotenv>=1.0.0"],
        "mcp": ["mcp[cli]>=1.20"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    keywords="ai-safety geometric-features embeddings uncertainty-quantification",
)
