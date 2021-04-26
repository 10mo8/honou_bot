from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from inference import InferenceAPI, InferenceAPIforAttention
from models import Seq2seq, Encoder, Decoder, AttentionDecoder
from preprocessing import build_vocabulary, preprocess_ja, preprocess_dataset, create_dataset
from utils import load_dataset, evaluate_bleu


def main():
    # Set hyper-parameters.
    batch_size = 32
    epochs = 100
    model_path = 'atmodel.h5'
    enc_arch = 'encoder.json'
    dec_arch = 'decoder.json'
    data_path = '../data/w16to19rennhouconv.txt'
    num_words = 5200
    num_data = 1771

    # Data loading.
    en_texts, ja_texts = load_dataset(data_path)
    en_texts, ja_texts = en_texts[:num_data], ja_texts[:num_data]

    # Preprocessings.
    #ja_texts = preprocess_ja(ja_texts)
    ja_texts = preprocess_dataset(ja_texts)
    en_texts = preprocess_dataset(en_texts)
    x_train, x_test, y_train, y_test = train_test_split(en_texts,
                                                        ja_texts,
                                                        test_size=0.2,
                                                        random_state=42)
    
    en_vocab = build_vocabulary(x_train, num_words)
    ja_vocab = build_vocabulary(y_train, num_words)
    print(x_train[:3])
    print(y_train[:3])
    x_train, y_train = create_dataset(x_train, y_train, en_vocab, ja_vocab)

    print(en_vocab.word_index)
    print(ja_vocab.word_index)

    # Build a simple model.
    encoder = Encoder(num_words)
    decoder = Decoder(num_words)
    # Build an attention model.
    #encoder = Encoder(num_words, return_sequences=True)
    #decoder = AttentionDecoder(num_words)
    seq2seq = Seq2seq(encoder, decoder)
    model = seq2seq.build()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Train the model.
    callbacks = [
        EarlyStopping(patience=10),
        ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
    ]
    """
    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_split=0.1)"""
    encoder.save_as_json(enc_arch)
    decoder.save_as_json(dec_arch)

    # Inference.
    encoder = Encoder.load(enc_arch, model_path)
    decoder = Decoder.load(dec_arch, model_path)
    api = InferenceAPI(encoder, decoder, en_vocab, ja_vocab)
    #api = InferenceAPIforAttention(encoder, decoder, en_vocab, ja_vocab)
    texts = sorted(set(en_texts[:50]), key=len)
    texts = ["お聞きしたいと思います", "さっき の 答弁 全く 納得 できません", "全く 納得 い き ません", "ありがとうございました", "おはようございます",\
            "よろしいでしょうか", "是非 よろしくお願いいたします", "もう少し 具体的に 教えて いただける と 助 か る んですけれども", "ちょっと 待 って", "質問 主 意 書 では 当然 混 同 は しておりません",\
            "正 式 な 要求 でいい んですか", "時間ですので まとめて ください", "ちょっと 静粛に お願いします", "よろしいですか", "静粛に お願いします",\
            "答弁 を まとめて ください", "時間 ですから", "驚 き の答弁 ですね", "それは いつ ごろ でしょうか", "そのとおり です"
    ]
    for text in texts:
        decoded = api.predict(text=text)
        print('入力: {}'.format(text))
        print('応答: {}'.format(decoded))

    y_test = [y.split(' ')[1:-1] for y in y_test]
    bleu_score = evaluate_bleu(x_test, y_test, api)
    print('BLEU: {}'.format(bleu_score))


if __name__ == '__main__':
    main()
