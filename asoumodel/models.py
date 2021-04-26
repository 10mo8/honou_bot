from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense, Input, Embedding, GRU, Dot, Activation, Concatenate


class BaseModel:

    def build(self):
        raise NotImplementedError()

    def save_as_json(self, filepath):
        model = self.build()
        with open(filepath, 'w') as f:
            f.write(model.to_json())

    @classmethod
    def load(cls, architecture_file, weight_file, by_name=True):
        with open(architecture_file) as f:
            model = model_from_json(f.read())
            model.load_weights(weight_file, by_name=by_name)
            return model


class Encoder(BaseModel):

    def __init__(self, input_dim, emb_dim=300, hid_dim=256, return_sequences=False):
        self.input = Input(shape=(None,), name='encoder_input')
        self.embedding = Embedding(input_dim=input_dim,
                                   output_dim=emb_dim,
                                   mask_zero=True,
                                   name='encoder_embedding')
        self.gru = GRU(hid_dim,
                       return_sequences=return_sequences,
                       return_state=True,
                       name='encoder_gru')

    def __call__(self):
        x = self.input
        embedding = self.embedding(x)
        output, state = self.gru(embedding)
        return output, state

    def build(self):
        output, state = self()
        return Model(inputs=self.input, outputs=[output, state])


class Decoder(BaseModel):

    def __init__(self, output_dim, emb_dim=300, hid_dim=256):
        self.input = Input(shape=(None,), name='decoder_input')
        self.embedding = Embedding(input_dim=output_dim,
                                   output_dim=emb_dim,
                                   mask_zero=True,
                                   name='decoder_embedding')
        self.gru = GRU(hid_dim,
                       return_sequences=True,
                       return_state=True,
                       name='decoder_gru')
        self.dense = Dense(output_dim, activation='softmax', name='decoder_output')

        # for inference.
        self.state_input = Input(shape=(hid_dim,), name='decoder_state_in')

    def __call__(self, states, enc_output=None):
        x = self.input
        embedding = self.embedding(x)
        outputs, state = self.gru(embedding, initial_state=states)
        outputs = self.dense(outputs)
        return outputs, state

    def build(self):
        decoder_output, decoder_state = self(states=self.state_input)
        return Model(
            inputs=[self.input, self.state_input],
            outputs=[decoder_output, decoder_state])


class LuongAttention:

    def __init__(self, units=300):
        self.dot = Dot(axes=[2, 2], name='dot')
        self.attention = Activation(activation='softmax', name='attention')
        self.context = Dot(axes=[2, 1], name='context')
        self.concat = Concatenate(name='concat')
        self.fc = Dense(units, activation='tanh', name='attn_out')

    def __call__(self, enc_output, dec_output):
        attention = self.dot([dec_output, enc_output])
        attention_weight = self.attention(attention)
        context_vector = self.context([attention_weight, enc_output])
        concat_vector = self.concat([context_vector, dec_output])
        output = self.fc(concat_vector)
        return output


class AttentionDecoder(Decoder):

    def __init__(self, output_dim, emb_dim=300, hid_dim=256):
        super().__init__(output_dim, emb_dim, hid_dim)
        self.attention = LuongAttention()
        self.enc_output = Input(shape=(None, hid_dim), name='encoder_output')

    def __call__(self, states, enc_output=None):
        x = self.input
        embedding = self.embedding(x)
        outputs, state = self.gru(embedding, initial_state=states)
        outputs = self.attention(enc_output, outputs)
        outputs = self.dense(outputs)
        return outputs, state

    def build(self):
        decoder_output, decoder_state = self(states=self.state_input,
                                             enc_output=self.enc_output)
        return Model(
            inputs=[self.input, self.enc_output, self.state_input],
            outputs=[decoder_output, decoder_state])


class Seq2seq(BaseModel):

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def build(self):
        encoder_output, state = self.encoder()
        decoder_output, _ = self.decoder(states=state, enc_output=encoder_output)
        return Model([self.encoder.input, self.decoder.input], decoder_output)
