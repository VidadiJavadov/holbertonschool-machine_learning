#!/usr/bin/env python3
"Bag of Words"
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: A list of sentences to analyze.
        vocab: A list of vocabulary words to use. If None, all words are used.

    Returns:
        embeddings: A numpy.ndarray of shape (s, f).
        features: A list of the features (words) used.
    """
    
    # 1. Preprocess sentences: Lowercase and extract words using regex
    # \b\w+\b matches word characters between word boundaries (removes punctuation)
    processed_sentences = [re.findall(r"\b\w+\b", s.lower()) for s in sentences]

    # 2. Determine the Vocabulary (Features)
    if vocab is None:
        unique_words = set()
        for sent in processed_sentences:
            unique_words.update(sent)
        # Sort the vocabulary for consistent column ordering
        features = sorted(list(unique_words))
    else:
        features = vocab

    # 3. Create a mapping from word to index for fast lookup
    feature_index = {word: i for i, word in enumerate(features)}

    # 4. Initialize the embedding matrix with zeros
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # 5. Fill the matrix
    for i, sent in enumerate(processed_sentences):
        for word in sent:
            if word in feature_index:
                j = feature_index[word]
                embeddings[i, j] += 1

    return embeddings, features
