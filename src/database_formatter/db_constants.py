import os

DB_PATH = r"C:\Users\BT_Lab\Documents\RVST598_Speech-Emotion-Recognition\data"
CREMAD_DB_PATH = os.path.join(DB_PATH, "crema-d")

IEMOCAP_DB_PATH = os.path.join(DB_PATH, "iemocap")

RAVDESS_DB_PATH = os.path.join(DB_PATH, "ravdess")
RAVDESS_SAMPLES_CACHE_PATH = os.path.join(
    RAVDESS_DB_PATH, "ravdess_samples_cache.npy")
RAVDESS_LABELS_CACHE_PATH = os.path.join(
    RAVDESS_DB_PATH, "ravdess_labels_cache.npy")
