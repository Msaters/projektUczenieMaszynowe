import faiss
import numpy as np
from config import *
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# to do function for normal preprocess
def fast_preprocess_df(df):
    df_nums = df.select_dtypes(include=['number'])
    df_nums = df_nums.dropna(subset=['price']).reset_index(drop=True)
    df_nums = df_nums.astype('float32')
    return df_nums

def find_similar_items(index, query_vector: np.ndarray, k: int):
    query_vector = query_vector.astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector.reshape(1, -1), k) # type: ignore
    return D, I

def find_most_distant_items(index, query_vector: np.ndarray, k: int):
    query_vector = query_vector.astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    
    opposite_vector = -query_vector
    D, I = index.search(opposite_vector, k)
    return D, I


def get_wines_by_indices(indices: np.ndarray, df: pd.DataFrame):
    wines_reviews_df = df.iloc[indices.flatten()]
    descriptions_list = wines_reviews_df['description'].tolist()
    return wines_reviews_df, descriptions_list

def prepare_weighted_features(df, weight_dict=None, default_weight=4.0):
    df.columns = df.columns.astype(str)
    all_cols = df.columns.tolist()
    embedding_cols = [c for c in all_cols if str(c).isdigit() or str(c).startswith('svd_')]
    feature_cols = [c for c in all_cols if c not in embedding_cols]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=all_cols, index=df.index)

    if weight_dict:
        for col, w in weight_dict.items():
            if col in df_scaled.columns:
                df_scaled[col] *= w
    else:
        for col in feature_cols:
            df_scaled[col] *= default_weight
            if col == 'price':
                df_scaled[col] *= 2.5
        

    return df_scaled.astype('float32')

def build_faiss_index_from_df_nums(df_nums: pd.DataFrame): 
    df_nums_array = df_nums.to_numpy()
    normy = np.linalg.norm(df_nums_array, axis=1)
    indeksy_zerowe = np.where(normy == 0)[0]

    if len(indeksy_zerowe) > 0:
        print(f"Uwaga: Znaleziono {len(indeksy_zerowe)} win z pustymi wektorami.")
        # Dodajemy minimalną wartość (epsilon), żeby uniknąć dzielenia przez 0
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
    
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Utworzono brakujący folder: {dir_name}")


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

def save_index(index, fileName):
    faiss.write_index(index, fileName)

def faiss_read(index, fileName):
    index = faiss.read_index("wine_index.faiss")
    return index