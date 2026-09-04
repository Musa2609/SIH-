"""
Top-Level Heart Disease Prediction Tool (Delegates to src/inference/predict_heart.py)
-----------------------------------------------------------------------------------
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference.predict_heart import predict_patient_heart, main

if __name__ == "__main__":
    main()
