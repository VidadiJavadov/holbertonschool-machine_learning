#!/usr/bin/env python3
"""
Module to load and prep a dataset for machine translation.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and preps a dataset for machine translation:
    Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the dataset and creates sub-word tokenizers.
        """
        # Load train and validation splits as (pt, en) tuples
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # Generate tokenizers from training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset containing (pt, en) tuples.

        Returns:
            tokenizer_pt, tokenizer_en: The trained tokenizers.
        """
        # Base BERT models for configuration
        pt_path = "neuralmind/bert-base-portuguese-cased"
        en_path = "bert-base-uncased"

        # Load base tokenizers
        tk_pt = transformers.AutoTokenizer.from_pretrained(pt_path)
        tk_en = transformers.AutoTokenizer.from_pretrained(en_path)

        def get_corpus(index):
            """Yields decoded strings for the specific language index."""
            for example in data:
                # index 0: Portuguese, index 1: English
                yield example[index].numpy().decode('utf-8')

        # Train new tokenizers with vocab size 2**13 (8192)
        # Using WordPiece algorithm inherited from BERT
        tokenizer_pt = tk_pt.train_new_from_iterator(
            get_corpus(0),
            vocab_size=2**13
        )
        tokenizer_en = tk_en.train_new_from_iterator(
            get_corpus(1),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
