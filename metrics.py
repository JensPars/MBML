import numpy as np
import torch
from collections import Counter
from scipy.spatial.distance import cosine
from torch import nn
import torch.nn.functional as F

def calculate_coherence_umass(topics, corpus, vocab, top_n=10):
    """
    Calculate UMass coherence for topics
    """
    coherence_scores = []
    
    for topic_words in topics:
        # Get top N words for this topic
        top_words = topic_words[:top_n]
        
        coherence = 0
        for i in range(1, len(top_words)):
            for j in range(i):
                word_i, word_j = top_words[i], top_words[j]
                
                # Count co-occurrences in corpus
                co_occur = sum(1 for doc in corpus if word_i in doc and word_j in doc)
                occur_j = sum(1 for doc in corpus if word_j in doc)
                
                if occur_j > 0:
                    coherence += np.log((co_occur + 1) / occur_j)
        
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)

# Extract topics from your model
def get_topics_from_model(model, vocab, top_n=10):
    """Extract top words for each topic"""
    beta = model.beta()  # Shape: (vocab_size, num_topics)
    topics = []
    
    for topic_idx in range(beta.shape[1]):
        # Get top word indices for this topic
        top_word_indices = beta[:, topic_idx].argsort(descending=True)[:top_n]
        top_words = [vocab[int(idx)] for idx in top_word_indices]
        topics.append(top_words)
    
    return topics



def calculate_perplexity(model, test_docs):
    """Calculate perplexity on test documents"""
    model.eval()
    total_log_likelihood = 0
    total_words = 0
    
    with torch.no_grad():
        # Get topic distributions for test docs
        logtheta_loc, logtheta_scale = model.encoder(test_docs)
        theta = F.softmax(logtheta_loc, dim=-1)
        
        # Get word probabilities
        word_probs = model.decoder(theta)
        
        # Calculate log likelihood
        for i, doc in enumerate(test_docs):
            doc_words = doc.sum().item()
            if doc_words > 0:
                # Normalized document
                doc_normalized = doc / doc_words
                # Calculate log likelihood for this document
                log_prob = (doc_normalized * torch.log(word_probs[i] + 1e-10)).sum()
                total_log_likelihood += log_prob.item() * doc_words
                total_words += doc_words
    
    # Calculate perplexity
    perplexity = np.exp(-total_log_likelihood / total_words)
    return perplexity


def calculate_topic_diversity(topics, top_n=10):
    """
    Calculate topic diversity as the percentage of unique words 
    in the top N words across all topics
    """
    all_top_words = []
    for topic in topics:
        all_top_words.extend(topic[:top_n])
    
    unique_words = len(set(all_top_words))
    total_words = len(all_top_words)
    
    return unique_words / total_words


#!pip install gensim nltk
import gensim
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import nltk
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# Ensure you have the 'punkt' tokenizer models
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def get_topics_from_model(model, vocab_words, top_n=10):
    """
    Extracts top N words for each topic from the model.
    Args:
        model: Trained ProdLDA model.
        vocab_words: A list or pd.Series of vocabulary words, where index corresponds to word ID.
        top_n: Number of top words to retrieve for each topic.
    Returns:
        topics: A list of lists, where each inner list contains the top N words for a topic.
    """
    topics = []
    beta = model.beta()  # Get the topic-word distributions
    for k in range(model.num_topics):
        top_words_indices = beta[k].topk(top_n).indices
        topic_words = [vocab_words[i.item()] for i in top_words_indices]
        topics.append(topic_words)
    return topics

def calculate_topic_coherence(topics, texts, dictionary, coherence_measure='c_v'):
    """
    Calculates topic coherence using Gensim's CoherenceModel.
    Args:
        topics: A list of lists, where each inner list contains the top N words for a topic.
        texts: A list of lists, where each inner list contains the tokenized words of a document.
        dictionary: Gensim Dictionary object created from the texts.
        coherence_measure: The coherence measure to use (e.g., 'c_v', 'u_mass').
    Returns:
        coherence_score: The calculated coherence score.
    """
    coherence_model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_measure
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score

# Prepare texts for coherence calculation (tokenize)
# We'll use the original reviews from the dataframe for this.
stop_words_nltk = stopwords.words('english')

def tokenize_texts(texts_series):
    tokenized_texts = []
    for text in texts_series:
        tokens = nltk.word_tokenize(text.lower()) # Convert to lowercase and tokenize
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words_nltk] # Remove punctuation, numbers, and stopwords
        tokenized_texts.append(tokens)
    return tokenized_texts

# Assuming your original text data is in df['review']
#original_texts_for_coherence = tokenize_texts(df['review'])

# Create Gensim Dictionary
#gensim_dictionary = Dictionary(original_texts_for_coherence)

# Example of how to use it (will be used properly in the training loop later)
# topics_from_trained_model = get_topics_from_model(prodLDA, vocab['word'], top_n=10)
# coherence_score_example = calculate_topic_coherence(topics_from_trained_model, original_texts_for_coherence, gensim_dictionary)
# print(f"Example Coherence Score (C_v): {coherence_score_example}")