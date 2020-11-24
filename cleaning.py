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

# fungsi ini digunakan untuk mengecek data secara keseluruhan dataset tertentu
def fulldataset(index0, index1):
  x1 = pd.ExcelFile(index0)
  dfs = {sh:x1.parse(sh) for sh in x1.sheet_names}[index1]
  return dfs

# normalizer
def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

# doc_term_matrix_l2 = []
# for vec in doc_array:
#     doc_term_matrix_l2.append(l2_normalizer(vec))

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

def freq(term, document):
  return document.split().count(term)

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount 

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)

def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

