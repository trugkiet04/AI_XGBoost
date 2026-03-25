import numpy as np

from ember1390_encoder import extract_feature1390_from_exe

def extract_features_for_model(file_path: str):
    return extract_feature1390_from_exe(file_path)