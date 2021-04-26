import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)


def build_vocabulary(texts, num_words=None):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words, oov_token='<UNK>', filters=''
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def tokenize(text):
    return t.tokenize(text)


def preprocess_dataset(texts):
    return ['<start> {} <end>'.format(text) for text in texts]


def preprocess_ja(texts):
    return [' '.join(tokenize(text)) for text in texts]


def create_dataset(en_texts, ja_texts, en_vocab, ja_vocab):
    en_seqs = en_vocab.texts_to_sequences(en_texts)
    ja_seqs = ja_vocab.texts_to_sequences(ja_texts)
    en_seqs = pad_sequences(en_seqs, padding='post')
    ja_seqs = pad_sequences(ja_seqs, padding='post')
    return [en_seqs, ja_seqs[:, :-1]], ja_seqs[:, 1:]
