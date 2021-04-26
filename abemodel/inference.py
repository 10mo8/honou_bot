"""
Inference API.
"""
import numpy as np


class InferenceAPI:
    """A model API that generates output sequence.

    Attributes:
        encoder_model: Model.
        decoder_model: Model.
        en_vocab: source language's vocabulary.
        ja_vocab: target language's vocabulary.
    """

    def __init__(self, encoder_model, decoder_model, en_vocab, ja_vocab):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.en_vocab = en_vocab
        self.ja_vocab = ja_vocab

    def predict(self, text):
        output, state = self._compute_encoder_output(text)
        sequence = self._generate_sequence(output, state)
        decoded = self._decode(sequence)
        return decoded

    def _compute_encoder_output(self, text):
        """Compute encoder output.

        Args:
            text : string, the input text.

        Returns:
            output: encoder's output.
            state : encoder's final state.
        """
        assert isinstance(text, str)
        x = self.en_vocab.texts_to_sequences([text])
        #print(x)
        output, state = self.encoder_model.predict(x)
        return output, state

    def _compute_decoder_output(self, target_seq, state, enc_output=None):
        """Compute decoder output.

        Args:
            target_seq: target sequence.
            state: hidden state.
            output: encoder's output.

        Returns:
            output: decoder's output.
            state: decoder's state.
        """
        output, state = self.decoder_model.predict([target_seq, state])
        return output, state

    def _generate_sequence(self, enc_output, state, max_seq_len=30):
        """Generate a sequence.

        Args:
            states: initial states of the decoder.

        Returns:
            sampled: a generated sequence.
        """
        target_seq = np.array([self.ja_vocab.word_index['<start>']])
        sequence = []
        indexlst = []
        for i in range(max_seq_len):
            output, state = self._compute_decoder_output(target_seq, state, enc_output)
            #print("output{}".format(output))
            
            if i == 0:
                output = output[0][0]
                new_output = np.power(output, 0.75)
                #変換後の確率分布の総和が1になるように変換
                new_output /= np.sum(new_output)
                out = np.random.choice(new_output, p=new_output)
                new_output = new_output.tolist()
                #先頭でサンプリングする文字の品詞を考える
                sampled_token_index = new_output.index(out)
            else:
                sampled_token_index = np.argmax(output[0][0])
            
            #sampled_token_index = np.argmax(output[0][0])

            if sampled_token_index == self.ja_vocab.word_index['<end>']:
                break
            sequence.append(sampled_token_index)
            target_seq = np.array([sampled_token_index])
        return sequence

    def _decode(self, sequence):
        """Decode a sequence.

        Args:
            sequence: a generated sequence.

        Returns:
            decoded: a decoded sequence.
        """
        decoded = self.ja_vocab.sequences_to_texts([sequence])
        decoded = decoded[0].split(' ')
        return decoded

class InferenceAPIforAttention(InferenceAPI):

    def _compute_decoder_output(self, target_seq, state, enc_output=None):
        output, state = self.decoder_model.predict([target_seq, enc_output, state])
        return output, state
