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

        # initialize tokenizer
        try:
            with open('tokenizer.pkl', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except:
            raise Exception("Path to tokenizer missing")

        # initalizer keras model
        # Load keras model
        self.model = keras.models.load_model('model.h5')

        # Initializing surface
        pygame.display.init()
        self.lamp_sim = pygame.display.set_mode((400, 300))

        self.red = 123
        self.green = 123
        # Initialing RGB Color
        self.lamp_sim.fill((self.red, self.green, 0))
        pygame.display.flip()

        return

    def capture(self):
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
            return None

    def predict(self, text, include_neutral=True):

        # Tokenize text
        x_test = pad_sequences(self.tokenizer.texts_to_sequences(
            [text]), maxlen=300)  # match with tokenizer created in pre-processing
        # Predict
        score = self.model.predict([x_test])[0]
        # Decode sentiment
        sentiment_thresholds = (0.4, 0.7)

        if include_neutral:
            label = "NEUTRAL"
            if score <= sentiment_thresholds[0]:
                label = "NEGATIVE"
            elif score >= sentiment_thresholds[1]:
                label = "POSITIVE"
        else:
            if score < 0.5:
                label = "NEGATIVE"
            else:
                label = "POSTIVE"

        return label, score

    def update_colour(self, label, score):

        self.norm_score = (score - 0.5) * 100

        if label == "POSITIVE":
            self.green = self.green + self.norm_score
            self.red = self.red - self.norm_score
            print("positive")

        elif label == "NEGATIVE":
            self.green = self.green + self.norm_score
            self.red = (self.red - self.norm_score)
            print("negative")
        else:
            pass

        self.green = 0 if self.green < 0 else 255 if self.green > 255 else self.green
        self.red = 0 if self.red < 0 else 255 if self.red > 255 else self.red

        color = (self.red, self.green, 0)
        print(color)
        # Changing surface color
        self.lamp_sim.fill(color)
        pygame.display.update()

        return
