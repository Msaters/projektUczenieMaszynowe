import faiss
import numpy as np
from config import *
import pandas as pd
import os

# to do function for normal preprocess
def fast_preprocess_df(df):
    df_nums = df.select_dtypes(include=['number'])
    df_nums = df_nums.dropna(subset=['price']).reset_index(drop=True)
    df_nums = df_nums.astype('float32')
    return df_nums

def find_similar_items(query_vector: np.ndarray, k: int):
    D, I = index.search(query_vector.reshape(1, -1), k) # type: ignore
    return D, I

def get_wines_by_indices(indices: np.ndarray, df: pd.DataFrame):
    wines_reviews_df = df.iloc[indices.flatten()]
    descriptions = wines_reviews_df['description'].tolist()
    print(descriptions)
    return wines_reviews_df


def build_faiss_index_from_df_nums(df_nums: pd.DataFrame): 
    df_nums_array = df_nums.to_numpy()
    normy = np.linalg.norm(df_nums_array, axis=1)
    indeksy_zerowe = np.where(normy == 0)[0]

    if len(indeksy_zerowe) > 0:
        print(f"Uwaga: Znaleziono {len(indeksy_zerowe)} win z pustymi wektorami.")
        # Rozwiązanie: Dodajemy minimalną wartość (epsilon), żeby uniknąć dzielenia przez 0
        # Dzięki temu wektor będzie "prawie zerowy", ale normalizacja zadziała.
        df_nums_array[indeksy_zerowe] += 1e-10

    # Upewniamy się, że tablica jest C-contiguous dla FAISS
    df_nums_array = np.ascontiguousarray(df_nums_array)
    faiss.normalize_L2(df_nums_array)
    dimensions = df_nums_array.shape[1]
    
    index = faiss.IndexFlatL2(dimensions)  
    index.add(df_nums_array)   # type: ignore
    return index, df_nums_array

def append_embedding_to_file(filepath, new_embedding):
    new_vector = np.array(new_embedding).reshape(1, -1)

    if os.path.exists(filepath):
        try:
            existing_data = np.load(filepath)
            updated_data = np.vstack((existing_data, new_vector))
            np.save(filepath, updated_data)
            
        except ValueError:
            print("Błąd: Wymiary nowego embeddingu nie pasują do istniejących danych.")
    else:
        # Jeśli plik nie istnieje zapisz wektor
        np.save(filepath, new_vector)