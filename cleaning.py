import nltk
import string
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import re 

nltk.download('stopwords')
nltk.download('punkt')

stemming = PorterStemmer()
stops = set(stopwords.words("english"))

# cleaning text
def text_cleaning(raw):
  
  # lowering word
  lower_case = raw.lower()

  # hapus punctuatioation & lower
  hasil_punctuation = lower_case.translate(str.maketrans("","",string.punctuation))
  
  # hapus whitespace
  hasil_whitespace = hasil_punctuation.strip()

  # hapus angka
  hasil_hapusangka = re.sub(r"\d+", "", hasil_whitespace)
  
  # tokenisasi 
  tokens = nltk.tokenize.word_tokenize(hasil_hapusangka)
  
  # Stemming
  stemmed_words = [stemming.stem(w) for w in tokens]
  
  # Remove stop words
  meaningful_words = [w for w in stemmed_words if not w in stops]
  
  # Rejoin meaningful stemmed words
  joined_words = ( " ".join(meaningful_words))
  
  # Return cleaned data
  return joined_words  

# applying
def apply_cleaning(hasil):
    cleaned_hasil = []
    for element in hasil:
        cleaned_hasil.append(text_cleaning(element))
    return cleaned_hasil
