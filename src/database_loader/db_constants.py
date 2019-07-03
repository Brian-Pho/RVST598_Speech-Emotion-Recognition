"""
This file holds database constants such as paths to the IEMOCAP, CREMA-D, and
RAVDESS database.
"""

import os

# GENERAL DB PATHS
REPO_PATH = r"C:\Users\BT_Lab\Documents\RVST598_Speech-Emotion-Recognition"
DATA_PATH = os.path.join(REPO_PATH, "data")
RAW_DB_PATH = os.path.join(DATA_PATH, "raw")
PROCESS_DB_PATH = os.path.join(DATA_PATH, "processed")

# IEMOCAP DATABASE
IEM_DB_PATH = os.path.join(RAW_DB_PATH, "iemocap")
IEM_SAMPLES_CACHE_PATH = os.path.join(
    IEM_DB_PATH, "iemocap_samples_cache.npy")
IEM_LABELS_CACHE_PATH = os.path.join(
    IEM_DB_PATH, "iemocap_labels_cache.npy")

# CREMA-D DATABASE
CRE_DB_PATH = os.path.join(RAW_DB_PATH, "cremad")
CRE_SAMPLES_CACHE_PATH = os.path.join(
    CRE_DB_PATH, "cremad_samples_cache.npy")
CRE_LABELS_CACHE_PATH = os.path.join(
    CRE_DB_PATH, "cremad_labels_cache.npy")

# RAVDESS DATABASE
RAV_DB_PATH = os.path.join(RAW_DB_PATH, "ravdess")
RAV_SAMPLES_CACHE_PATH = os.path.join(
    RAV_DB_PATH, "ravdess_samples_cache.npy")
RAV_LABELS_CACHE_PATH = os.path.join(
    RAV_DB_PATH, "ravdess_labels_cache.npy")

# MODEL SAVE PATH
MODEL_SAVE_PATH = os.path.join(DATA_PATH, "model.h5")
