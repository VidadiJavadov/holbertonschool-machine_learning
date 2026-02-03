#!/usr/bin/env python3
"""bagofwords"""
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    """
    # Preprocess sentences: 
    # 1. Convert to lowercase
    # 2. Use regex \b\w{2,}\b to find words with 2 or more characters.
    #    This removes punctuation and single letters (like 's' from "children's").
    processed_sentences = [re.findall(r"\b\w{2,}\b", s.lower()) for s in sentences]

    if vocab is None:
        # If vocab is None, collect all unique words from the sentences
        unique_words = set()
        for sent in processed_sentences:
            unique_words.update(sent)
        features = sorted(list(unique_words))
    else:
        # Use the provided vocabulary
        features = vocab

    # Create a dictionary for O(1) index lookup
    feature_to_index = {word: i for i, word in enumerate(features)}

    # Initialize the embeddings matrix with zeros
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Fill the matrix
    for i, sent in enumerate(processed_sentences):
        for word in sent:
            if word in feature_to_index:
                j = feature_to_index[word]
                embeddings[i, j] += 1

    return embeddings, features