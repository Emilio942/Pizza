from setuptools import setup, find_packages

setup(
    name="pizza-detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.65.0",
        "scipy>=1.7.0"
    ],
    author="Emilio",
    description="RP2040-basiertes Pizza-Erkennungssystem",
    python_requires=">=3.8",
)