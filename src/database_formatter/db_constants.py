import os

# GENERAL DB PATHS
RAW_DB_PATH = r"C:\Users\BT_Lab\Documents\RVST598_Speech-Emotion-Recognition\data\raw"
PROCESS_DB_PATH = r"C:\Users\BT_Lab\Documents\RVST598_Speech-Emotion-Recognition\data\processed"

# IEMOCAP DATABASE
IEMOCAP_DB_PATH = os.path.join(RAW_DB_PATH, "iemocap")

# CREMA-D DATABASE
CREMAD_DB_PATH = os.path.join(RAW_DB_PATH, "crema-d")

# TESS DATABASE
TESS_DB_PATH = os.path.join(RAW_DB_PATH, "tess")

# RAVDESS DATABASE
RAV_RAW_DB_PATH = os.path.join(RAW_DB_PATH, "ravdess")
RAV_PROCESS_DB_PATH = os.path.join(PROCESS_DB_PATH, "ravdess")
RAV_SAMPLES_CACHE_PATH = os.path.join(
    RAV_RAW_DB_PATH, "ravdess_samples_cache.npy")
RAV_LABELS_CACHE_PATH = os.path.join(
    RAV_RAW_DB_PATH, "ravdess_labels_cache.npy")
RAV_FREQ_CACHE_PATH = os.path.join(
    RAV_PROCESS_DB_PATH, "ravdess_freq_cache.npy")
