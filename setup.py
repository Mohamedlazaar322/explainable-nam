from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="explainable_nam",
    version="0.1.0",
    description="A small library for Neural Additive Models (NAM).",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12",
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    python_requires=">=3.9",
)
