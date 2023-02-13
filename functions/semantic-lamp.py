import speech_recognition as sr
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import time
import pickle
import pygame


class SemanticLamp:
    def __init__(self):

        # check if files exist, if not, run preprocessing.py

        return

    def capture():
        """
        Get audio recording
        """
        # Capture audio

        rec = sr.Recognizer()

        with sr.Microphone() as source:
            print('LISTENING...')
            audio = rec.listen(source, phrase_time_limit=5)

        try:
            text = rec.recognize_google(audio, language='en-US')
            return text

        except sr.UnknownValueError:
            print('Sorry, I could not understand what you said.')
            return 0

    def process_text(input):
        # Process what is said

        print("You said: " + input + ".")

        predict(input)
        return

    def decode_sentiment(score, include_neutral=True):
        POSITIVE = "POSITIVE"
        NEGATIVE = "NEGATIVE"
        NEUTRAL = "NEUTRAL"
        SENTIMENT_THRESHOLDS = (0.4, 0.7)

        if include_neutral:
            label = NEUTRAL
            if score <= SENTIMENT_THRESHOLDS[0]:
                label = NEGATIVE
            elif score >= SENTIMENT_THRESHOLDS[1]:
                label = POSITIVE

            return label
        else:
            return NEGATIVE if score < 0.5 else POSITIVE

    def predict(text, include_neutral=True):
        start_at = time.time()
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences(
            [text]), maxlen=SEQUENCE_LENGTH)
        # Predict
        score = model.predict([x_test])[0]
        # Decode sentiment
        label = decode_sentiment(score, include_neutral=include_neutral)

        print("Predicted sentiment: {}, Score: {} , elapsed_time: {}".format(
            label, float(score), time.time()-start_at))

        update_colour(label, score)
        return

    def update_colour(label, score):
        global green
        global red
        norm_score = (score - 0.5) * 100

        if label == "POSITIVE":
            green = green + norm_score
            red = red - norm_score
            print("positive")

        elif label == "NEGATIVE":
            green = green + norm_score
            red = red - norm_score
            print("negative")
        else:
            green = green
            red = red

        if green > 255:
            green = 255
        if green < 0:
            green = 0

        if red > 255:
            red = 255
        if red < 0:
            red = 0

        color = (red, green, 0)
        print(color)
        # Changing surface color
        surface.fill(color)
        pygame.display.update()

        return

    def load_model(self):
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)

        SEQUENCE_LENGTH = 300

        # Load a word2vec model stored in the C *text* format.
        wv_from_text = KeyedVectors.load_word2vec_format(
            datapath('word2vec_pre_kv_c'), binary=False)

        # Load keras model
        model = keras.models.load_model('model.h5')

    def load_pygame_environment(self):
        pygame.init()

        # Initializing surface
        surface = pygame.display.set_mode((400, 300))

        # Initialing RGB Color
        green = 123
        red = 123
        color = (red, green, 0)
        surface.fill(color)
        pygame.display.flip()
