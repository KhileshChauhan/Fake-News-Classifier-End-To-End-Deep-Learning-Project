from nltk.stem.porter import PorterStemmer
from nltk.util import pad_sequence
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),'LSTM_Bi_dropout.h5')
VOCAB_SIZE = 5000
FIXED_LENGTH = 20


class LSTMPipeline(object):

    def __init__(self, txt):
        self.model = load_model(MODEL_PATH)
        self.txt = txt
        self.ps = PorterStemmer()
        self.onehot_repr = None

    def process(self):
        self.txt = re.sub('[^A-Za-z]', ' ', self.txt).lower().split()
        self.txt = [self.ps.stem(word) for word in self.txt if word not in stopwords.words('english')] 
        self.txt = ' '.join(self.txt)
        return self
    
    def transform(self):
        self.onehot_repr = one_hot(self.txt, VOCAB_SIZE)
        self.txt = pad_sequences([self.onehot_repr], padding='pre', maxlen=FIXED_LENGTH)
        self.txt = np.array(self.txt)
        return self
    
    def predict(self):
        self.process()
        self.transform()
        return self.model.predict_classes(self.txt)

# model = LSTMPipeline('A was Plane crashed near india-china border')
# print(model.predict())