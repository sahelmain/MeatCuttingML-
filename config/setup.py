from setuptools import setup, find_packages

setup(
    name="beef_volume_predictor",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0"
    ],
    python_requires=">=3.9",
) 