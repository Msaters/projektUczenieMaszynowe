import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from config import *

df = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)
wine_descriptions = df['description'].str.replace('-', '_', regex=False)

my_additional_stop_words = {'wine', 'drink', 'bottle', 'flavor', 'taste', 'like', 'nose', 'palate', 'finish', 'aroma', 'notes', 'note', 'vineyard', 'shows', 'alongside', 'offers', 'feels'}
all_stop_words = list(ENGLISH_STOP_WORDS.union(my_additional_stop_words))

pipeline = Pipeline([
  ('tfidf', TfidfVectorizer(max_features=50000, min_df=MIN_WORD_OCCURENCE, stop_words=all_stop_words, max_df=0.3, token_pattern=r'(?u)\b[\w-]{2,}\b')),
  ('pca', TruncatedSVD(n_components=DIMENSIONS, random_state=RANDOM_STATE))
])

embeddings = pipeline.fit_transform(wine_descriptions)
embeddings = embeddings.astype(np.float32, copy=False)
np.save("embeddings.npy", embeddings)
print(embeddings.shape)
print(MIN_WORD_OCCURENCE)