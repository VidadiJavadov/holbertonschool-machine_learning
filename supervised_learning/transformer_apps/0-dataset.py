#!/usr/bin/env python3
"""
Module to load and prep a dataset for machine translation.
"""
import tensorflow_datasets as tfds
import transformers

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

        # Build tokenizers from the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Trains sub-word tokenizers using BERT base models.
        
        Args:
            data: tf.data.Dataset containing (pt, en) tuples
            
        Returns:
            tokenizer_pt, tokenizer_en: The trained tokenizers
        """
        # Base BERT models for Pt and En
        pt_path = "neuralmind/bert-base-portuguese-cased"
        en_path = "bert-base-uncased"
        
        # Load pre-trained configurations
        tk_pt = transformers.AutoTokenizer.from_pretrained(pt_path)
        tk_en = transformers.AutoTokenizer.from_pretrained(en_path)

        def get_corpus(index):
            """Generator to yield strings from the dataset."""
            for example in data:
                # index 0 is Portuguese, index 1 is English
                yield example[index].numpy().decode('utf-8')

        # Train new tokenizers with a vocab limit of 2**13 (8192)
        tokenizer_pt = tk_pt.train_new_from_iterator(
            get_corpus(0), 
            vocab_size=2**13
        )
        tokenizer_en = tk_en.train_new_from_iterator(
            get_corpus(1), 
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
