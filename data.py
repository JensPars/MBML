import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.distributions import constraints
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import functools

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoGuideList, AutoDelta
from pyro.optim import ClippedAdam

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')



# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
punctuation = set(string.punctuation)

# Define the preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [stemmer.stem(word) for word in tokens]  # Stem the words
    return tokens

def add_mask_column(df, column_name, length, new_column_name='masked_list'):
    def pad_and_create_mask(lst):
        # Truncate if list is longer than the specified length
        truncated = lst[:length]

        # Create mask: 1s for actual values, 0s for padding
        mask = [1] * len(truncated)
        padding_length = max(0, length - len(truncated))
        mask += [0] * padding_length

        # Pad list with 0s if shorter than desired length
        padded_lst = truncated + [0] * padding_length

        return padded_lst, mask

    # Apply padding and mask creation
    result = df[column_name].apply(pad_and_create_mask)
    df[column_name] = result.apply(lambda x: x[0])  # overwrite with padded/truncated list
    df[new_column_name] = result.apply(lambda x: x[1])  # add mask column
    return df


def load_data():
    df = pd.read_csv("bbc-news-data.csv", sep='\t', encoding='utf-8')
    df['content'] = df['content'].apply(preprocess_text)
    dictionary = Dictionary(df['content'])
    df['content_indexed'] = df['content'].apply(lambda tokens: dictionary.doc2idx(tokens))
    df['content_length'] = df['content_indexed'].apply(len)
    df = df[df['content_length'] >= 20]

    # Create a mask column for the dataframe
    df = add_mask_column(df, 'content_indexed', length=60, new_column_name="content_masked")


    W = df["content_indexed"]
    Mask = df["content_masked"]

    # Convert to a LongTensor (2D tensor)
    tensor_W = torch.tensor(W.tolist(), dtype=torch.long).T
    tensor_Mask = torch.tensor(Mask.tolist(), dtype=torch.bool).T



# New function using CountVectorizer
def load_data_with_sklearn(sequence_length=60):
    df = pd.read_csv("tmdb_5000_movies.csv")
    df.dropna(subset=['overview'], inplace=True)
    # 3. Initialize and fit CountVectorizer
    vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, max_features=1000, stop_words="english"
    )
    vectorizer.fit(list(df['overview']))

    # 4. Get vocabulary (0-indexed)
    # sklearn_feature_names is an array of words, where index corresponds to word_id
    sklearn_feature_names = vectorizer.get_feature_names_out()
    pyro_idx_to_word_map = {i: word for i, word in enumerate(sklearn_feature_names)}
    pyro_word_to_idx_map = {word: i for i, word in enumerate(sklearn_feature_names)}
    
    num_vocab_words = len(sklearn_feature_names) # This is N for indices 0 to N-1

    # 5. Convert processed tokens to 0-based integer sequences
    def tokens_to_0based_indices(tokens):
        return [pyro_word_to_idx_map[token] for token in tokens if token in pyro_word_to_idx_map]

    df['content_indexed_0based'] = df['overview'].apply(tokens_to_0based_indices)
    
    # 6. Filter documents based on length after mapping to new vocabulary
    df['content_length_0based'] = df['content_indexed_0based'].apply(len)

    # 7. Pad sequences. Use a dedicated PAD_INDEX that is outside the vocabulary range.
    # Words are indexed 0 to num_vocab_words-1. Let PAD_INDEX = num_vocab_words.
    PAD_INDEX = num_vocab_words 
    df = add_mask_column(df, 'content_indexed_0based', 
                         length=sequence_length, 
                         new_column_name="content_masked",)
                         #pad_value=PAD_INDEX)

    # 8. Create tensors
    # tensor_W will contain word indices (0 to num_vocab_words-1) and PAD_INDEX for padding.
    # tensor_Mask will be 1 (True) for actual words, and 0 (False) for PAD_INDEX.
    W_padded_indexed = df["content_indexed_0based"]
    Mask = df["content_masked"]

    tensor_W = torch.tensor(W_padded_indexed.tolist(), dtype=torch.long).T
    tensor_Mask = torch.tensor(Mask.tolist(), dtype=torch.bool).T # Mask is 0 for pad, 1 for token

    # For the Pyro model:
    # - The `num_words` parameter should be `num_vocab_words`.
    # - The `topic_words` prior will be over `num_vocab_words` dimensions.
    # - The `pyro_idx_to_word_map` is used to interpret the topics.
    
    return tensor_W, tensor_Mask, pyro_idx_to_word_map, num_vocab_words, df, pyro_word_to_idx_map