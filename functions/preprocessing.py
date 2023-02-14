# Standard
#import os
import re
import numpy as np
from collections import Counter
import logging
import time
import pickle
import itertools

# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#from sklearn.manifold import TSNE
#from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
# , Activation, Flatten, Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Embedding, LSTM
#from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
#import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2vec
import gensim


class Preprocessing:
    """
    create docstring
    """

    def __init__(self):
        """

        """
        # Set log
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # DATASET
        dataset_columns = ["target", "ids", "date", "flag", "user", "text"]
        dataset_encoding = "ISO-8859-1"
        train_size = 0.8

        # TEXT CLENAING
        text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

        # WORD2VEC
        w2v_size = 300
        w2v_window = 7
        w2v_epoch = 32
        w2v_min_count = 10

        # KERAS
        sequence_length = 300
        epochs = 8
        batch_size = 1024

        # SENTIMENT
        positive = "POSITIVE"
        negative = "NEGATIVE"
        neutral = "NEUTRAL"
        # thresholds for what is determined as negative,neutral,positive
        sentiment_thresholds = (0.4, 0.7)

        # EXPORT
        keras_model = "model.h5"
        word2vec_model = "model.w2v"
        tokenizer_model = "tokenizer.pkl"
        encoder_model = "encoder.pkl"

        # Load Dataset
        dataset_path = "pre-processing/data/tweets.csv"
        print("Open file:", dataset_path)

        try:
            df = pd.read_csv(
                dataset_path, encoding=dataset_encoding, names=dataset_columns)

        except FileNotFoundError:
            logging.error("CSV load failed. Please check csv path")

        print("Dataset size:", len(df))

        decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

        def encode_sentiment(label):
            return decode_map[int(label)]

        df.target = df.target.apply(lambda x: encode_sentiment(x))

        target_cnt = Counter(df.target)

        plt.figure(figsize=(16, 8))
        plt.bar(target_cnt.keys(), target_cnt.values())
        plt.title("Dataset labels distribuition")
        plt.savefig('dataset_distribuition.png')

        # Pre-Processing dataset
        stop_words = stopwords.words("english")
        stemmer = SnowballStemmer("english")

        def preprocess(text, stem=False):

            # Remove link,user and special characters
            text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
            tokens = []
            for token in text.split():
                if token not in stop_words:
                    if stem:
                        tokens.append(stemmer.stem(token))
                    else:
                        tokens.append(token)
            return " ".join(tokens)

        df.text = df.text.apply(lambda x: preprocess(x))

        # split train and test
        df_train, df_test = train_test_split(
            df, test_size=1-train_size, random_state=42)
        print("TRAIN size:", len(df_train))
        print("TEST size:", len(df_test))

        # Word2vec
        documents = [_text.split() for _text in df_train.text]

        w2v_model = gensim.models.word2vec.Word2Vec(size=w2v_size,
                                                    window=w2v_window,
                                                    min_count=w2v_min_count,
                                                    workers=8)

        w2v_model.build_vocab(documents)

        words = w2v_model.wv.vocab.keys()
        vocab_size = len(words)
        print("Vocab size", vocab_size)

        w2v_model.train(documents, total_examples=len(
            documents), epochs=w2v_epoch)

        w2v_model.most_similar("love")

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df_train.text)

        vocab_size = len(tokenizer.word_index) + 1
        print("Total words", vocab_size)

        x_train = pad_sequences(tokenizer.texts_to_sequences(
            df_train.text), maxlen=sequence_length)
        x_test = pad_sequences(tokenizer.texts_to_sequences(
            df_test.text), maxlen=sequence_length)

        # Encoder
        labels = df_train.target.unique().tolist()
        labels.append(neutral)

        encoder = LabelEncoder()
        encoder.fit(df_train.target.tolist())

        y_train = encoder.transform(df_train.target.tolist())
        y_test = encoder.transform(df_test.target.tolist())

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        print("y_train", y_train.shape)
        print("y_test", y_test.shape)
        print("x_train", x_train.shape)
        print("x_test", x_test.shape)

        embedding_matrix = np.zeros((vocab_size, w2v_size))
        for word, i in tokenizer.word_index.items():
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
            print(embedding_matrix.shape)

        # embedding layer
        embedding_layer = Embedding(vocab_size, w2v_size, weights=[
                                    embedding_matrix], input_length=sequence_length, trainable=False)

        # build model
        model = Sequential()
        model.add(embedding_layer)
        model.add(Dropout(0.5))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])

        # Define callbacks
        callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                     EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

        # train
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=callbacks)

        # evaluate
        score = model.evaluate(x_test, y_test, batch_size=batch_size)

        print("ACCURACY:", score[1])
        print("LOSS:", score[0])

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig('Training_and_validation_loss.png')

        # predict
        def decode_sentiment(score, include_neutral=True):
            if include_neutral:
                label = neutral
                if score <= sentiment_thresholds[0]:
                    label = negative
                elif score >= sentiment_thresholds[1]:
                    label = positive
                return label
            else:
                return negative if score < 0.5 else positive

        def predict(text, include_neutral=True):
            start_at = time.time()
            # Tokenize text
            x_test = pad_sequences(tokenizer.texts_to_sequences(
                [text]), maxlen=sequence_length)
            # Predict
            score = model.predict([x_test])[0]
            # Decode sentiment
            label = decode_sentiment(score, include_neutral=include_neutral)

            return {"label": label, "score": float(score),
                    "elapsed_time": time.time()-start_at}

        predict("I love the music")
        predict("I hate the rain")
        predict("i don't know what i'm doing")

        # confusion matrix
        y_pred_1d = []
        y_test_1d = list(df_test.target)
        scores = model.predict(x_test, verbose=1, batch_size=8000)
        y_pred_1d = [decode_sentiment(
            score, include_neutral=False) for score in scores]

        def plot_confusion_matrix(cm, classes,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title, fontsize=30)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
            plt.yticks(tick_marks, classes, fontsize=22)

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label', fontsize=25)
            plt.xlabel('Predicted label', fontsize=25)

            cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
            plt.figure(figsize=(12, 12))
            plot_confusion_matrix(
                cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")
            plt.savefig('confusion_matrix.png')

            # classification report
            print(classification_report(y_test_1d, y_pred_1d))

            # accuracy score
            accuracy_score(y_test_1d, y_pred_1d)

            # save model
            model.save(keras_model)
            w2v_model.save(word2vec_model)
            pickle.dump(tokenizer, open(tokenizer_model, "wb"), protocol=0)
            pickle.dump(encoder, open(encoder_model, "wb"), protocol=0)
