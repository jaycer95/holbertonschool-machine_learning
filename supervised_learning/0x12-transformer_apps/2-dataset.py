#!/usr/bin/env python3
""" Transformer Applications """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prep a dataset for machine translation"""

    def __init__(self):
        """initialization"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'], as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode).cache()
        self.data_valid = self.data_valid.map(self.tf_encode).cache()

    def tokenize_dataset(self, data):
        """ Create sub-word tokenizers for our dataset"""
        subtok = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        en = subtok(
            [en.numpy() for _, en in data], target_vocab_size=2**15)
        pt = subtok(
            [pt.numpy() for pt, _ in data], target_vocab_size=2**15)
        return pt, en

    def encode(self, pt, en):
        """ Encode a translation into tokens """

        pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')) + [self.tokenizer_pt.vocab_size + 1]
        en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')) + [self.tokenizer_en.vocab_size + 1]
        return pt, en

    def tf_encode(self, pt, en):
        """ Tensorflow wrapper for the encode instance method """
        tf_pt, tf_en = tf.py_function(
            self.encode, inp=[
                pt, en], Tout=[
                tf.int64, tf.int64])
        tf_pt.set_shape([None])
        tf_en.set_shape([None])
        return tf_pt, tf_en
