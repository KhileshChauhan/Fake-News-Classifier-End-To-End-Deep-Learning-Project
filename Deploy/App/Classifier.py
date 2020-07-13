from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'LSTM_Bi_dropout.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'tokenizer.pkl')
VOCAB_SIZE = 5000
FIXED_LENGTH = 20


class LSTMPipeline(object):

    def __init__(self, txt):
        self.model = load_model(MODEL_PATH)
        self.txt = txt
        self.ps = PorterStemmer()
        with open(TOKENIZER_PATH, 'rb') as self.pi:
            self.tok = pickle.load(self.pi)
        self.onehot_repr = None
        
    def process(self):
        self.txt = re.sub('[^A-Za-z]', ' ', self.txt).lower().split()
        self.txt = [self.ps.stem(word) for word in self.txt if word not in stopwords.words('english')] 
        self.txt = ' '.join(self.txt)
        return self
    
    def transform(self):
        # self.txt = np.array(self.txt)
        self.txt = self.tok.texts_to_sequences([self.txt])
        self.txt = pad_sequences(self.txt, padding='pre', maxlen=FIXED_LENGTH)
        return self
    
    def predict(self):
        self.process()
        self.transform()
        self.pred = np.squeeze(self.model.predict_classes(self.txt))
        if self.pred == 1:
            return 'Not Fake'
        else:
            return 'Fake'

model = LSTMPipeline('A was Plane crashed near india-china border')
print(model.predict())