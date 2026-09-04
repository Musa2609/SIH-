"""
Top-Level HCV Prediction Tool (Delegates to src/inference/predict_hcv.py)
-----------------------------------------------------------------------
"""

import os
import sys

# Import from src/inference/predict_hcv.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference.predict_hcv import predict_patient_hcv, main

if __name__ == "__main__":
    main()
