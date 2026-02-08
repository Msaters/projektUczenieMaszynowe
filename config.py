import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RANDOM_STATE = 42
EMBEDEDINGS_FILEPATH_Mini_LML6_v2 = os.path.join(BASE_DIR, "embedings", "embeddings_all-MiniLM-L6-v2.npy")
EMBEDEDINGS_FILEPATH_mpnet_base_v2 = os.path.join(BASE_DIR, "embedings", "embeddings_all-mpnet-base-v2.npy")
EMBEDEDINGS_FILEPATH_open_ai_api = os.path.join(BASE_DIR, "embedings", "embeddings_open_ai_api.npy")
EMBEDEDINGS_FILEPATH_tf_idf_bigrams = os.path.join(BASE_DIR, "embedings", "embeddings_tf_idf_bigrams.npy")
EMBEDEDINGS_FILEPATH_tf_idf_monograms = os.path.join(BASE_DIR, "embedings", "embeddings_tf_idf_monograms.npy")

CSV_FILEPATH_UNCHANGED_DATA = os.path.join(BASE_DIR, "winemag-data-130k-v2.csv")
# Output after data_exploration preprocessing (drops + vintage extraction)
CSV_FILEPATH_EXPLORATION_PREPROCESSED_DATA = os.path.join(
    BASE_DIR, "winemag-data-130k-v2-exploration-preprocessed.csv"
)
# Final, model-ready datasets after encoding in data_engineering
CSV_FILEPATH_MODEL_READY_DATA_POINTS = os.path.join(
    BASE_DIR, "winemag-data-130k-v2-model-ready.csv"
)
CSV_FILEPATH_MODEL_READY_DATA_PRICE = os.path.join(
    BASE_DIR, "winemag-data-130k-v2-model-ready-price.csv"
)
# Backward-compatible alias (points)
CSV_FILEPATH_MODEL_READY_DATA = CSV_FILEPATH_MODEL_READY_DATA_POINTS

FAIS_INDEX_FILEPATH = os.path.join(BASE_DIR, "Similarity_Wine_Search", "wine_search_index.index")

# Parameters for tfidf vectorizer
DIMENSIONS = 128
MIN_WORD_OCCURENCE = 10


def load_embeddings_and_data(embeddings_filepath, csv_filepath, isFirstColumnIndex = True, check_length=True):
    import numpy as np
    import pandas as pd

    embeddings = np.load(embeddings_filepath)
    embeddings_df = pd.DataFrame(embeddings)
    if isFirstColumnIndex:
        data = pd.read_csv(csv_filepath, index_col=0)
    else:
        data = pd.read_csv(csv_filepath)
    
    if check_length:
        if len(embeddings) != len(data):
            raise ValueError("The number of embeddings does not match the number of data entries.")

    combined_data = pd.concat([data, embeddings_df], axis=1)
    return combined_data
