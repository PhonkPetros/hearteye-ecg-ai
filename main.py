#!/usr/bin/env python3
import subprocess
import sys

# Dictionary mapping pip package names to module names (if different)
required_packages = {
    "pandas": None,
    "numpy": None,
    "wfdb": None,
    "neurokit2": None,
    "matplotlib": None,
    "seaborn": None,
    "scikit-learn": "sklearn",
    "fpdf": None
}

def install_and_import(package, import_name=None):
    try:
        __import__(import_name if import_name else package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Installed {package}")

for package, import_name in required_packages.items():
    install_and_import(package, import_name)

# Now import the libraries after ensuring installation
import os
import ast
import pandas as pd
import numpy as np
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import signal
import sys
from fpdf import FPDF

def main():
    print("All required packages are installed and imported.")
    print("Running main application workflow...")
    # Add your application workflow here

if __name__ == "__main__":
    main()
