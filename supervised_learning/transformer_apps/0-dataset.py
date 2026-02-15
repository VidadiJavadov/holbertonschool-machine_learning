#!/usr/bin/env python3
"""
Module to load and prep a dataset for machine translation.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and preps a dataset for machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset instance with train and validation splits,
        and creates the sub-word tokenizers.
        """
        # Load the dataset splits as supervised (pt, en) tuples
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Initialize and train tokenizers on the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained BERT models.

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Load base tokenizers to inherit the WordPiece logic
        tk_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tk_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def get_corpus(index):
            """Generator to yield strings from the dataset for training."""
            # Each example is a tuple: (portuguese_tensor, english_tensor)
            for example in data:
                yield example[index].numpy().decode('utf-8')

        # Train new versions of the tokenizers with vocab size 2**13 (8192)
        tokenizer_pt = tk_pt.train_new_from_iterator(
            get_corpus(0),
            vocab_size=2**13
        )
        tokenizer_en = tk_en.train_new_from_iterator(
            get_corpus(1),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
