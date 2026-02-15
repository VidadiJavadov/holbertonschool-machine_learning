#!/usr/bin/env python3
"""
Module to load and prep a dataset for machine translation.
"""
import tensorflow_datasets as tfds
import tensorflow as tf
from transformers import AutoTokenizer

class Dataset:
    """
    Handles loading and tokenizing the Portuguese-English TED dataset.
    """

    def __init__(self):
        """
        Initializes dataset splits and generates sub-word tokenizers.
        """
        # Load train/validation splits as supervised (pt, en) tuples
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # Build tokenizers based on training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Trains sub-word tokenizers from pre-trained BERT models.
        
        Args:
            data: tf.data.Dataset (pt, en)
            
        Returns:
            (tokenizer_pt, tokenizer_en): The trained tokenizers
        """
        # Base BERT models for Pt and En
        pt_path = "neuralmind/bert-base-portuguese-cased"
        en_path = "bert-base-uncased"
        
        # Load configurations
        tk_pt = AutoTokenizer.from_pretrained(pt_path)
        tk_en = AutoTokenizer.from_pretrained(en_path)

        def get_corpus(index):
            """Generator to yield decoded strings from the dataset."""
            for pt, en in data:
                yield pt.numpy().decode('utf-8') if index == 0 else en.numpy().decode('utf-8')

        # Train with vocab limit of 2**13 (8192)
        self.tokenizer_pt = tk_pt.train_new_from_iterator(get_corpus(0), vocab_size=2**13)
        self.tokenizer_en = tk_en.train_new_from_iterator(get_corpus(1), vocab_size=2**13)

        return self.tokenizer_pt, self.tokenizer_en
