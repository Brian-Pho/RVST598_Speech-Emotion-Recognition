import os

# GENERAL DB PATHS
REPO_PATH = r"C:\Users\BT_Lab\Documents\RVST598_Speech-Emotion-Recognition"
DATA_PATH = os.path.join(REPO_PATH, "data")
RAW_DB_PATH = os.path.join(DATA_PATH, "raw")
PROCESS_DB_PATH = os.path.join(DATA_PATH, "processed")

# IEMOCAP DATABASE
IEMOCAP_DB_PATH = os.path.join(RAW_DB_PATH, "iemocap")

# CREMA-D DATABASE
CREMAD_DB_PATH = os.path.join(RAW_DB_PATH, "crema-d")

# RAVDESS DATABASE
RAV_RAW_DB_PATH = os.path.join(RAW_DB_PATH, "ravdess")
RAV_SAMPLES_CACHE_PATH = os.path.join(
    RAV_RAW_DB_PATH, "ravdess_samples_cache.npy")
RAV_LABELS_CACHE_PATH = os.path.join(
    RAV_RAW_DB_PATH, "ravdess_labels_cache.npy")

# PREPROCESSED DATA CACHES
PRE_SAMPLES_CACHE_PATH = os.path.join(PROCESS_DB_PATH, "samples_cache.npy")
PRE_LABELS_CACHE_PATH = os.path.join(PROCESS_DB_PATH, "labels_cache.npy")
