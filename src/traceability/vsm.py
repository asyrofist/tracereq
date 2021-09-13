import numpy as np
import pandas as pd
import math
from tabulate import tabulate
from scipy.sparse import data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from traceability.preprocessing_evaluation import pengukuranEvaluasi

class measurement:
  def __init__(self, cleaned_data):
      self.data = cleaned_data
  
  def bagOfWords(self, data_raw, req):
      b = CountVectorizer(data_raw) # dilakukan vektorisasi
      c = b.fit(data_raw) # dilakukan fiting 
      d = b.get_feature_names() # diambil namanya, sebagai kolom
      e = b.transform(data_raw).toarray() #data 
      bow_df= pd.DataFrame(e, req, d) #data, indeks, kolom
      return bow_df

  def l2_normalizer(self, vec):
      denom = np.sum([el**2 for el in vec])
      return [(el / math.sqrt(denom)) for el in vec]

  def build_lexicon(self, corpus):
      lexicon = set()
      for doc in corpus:
          lexicon.update([word for word in doc.split()])
      return lexicon

  def freq(self, term, document):
    return document.split().count(term)

  def numDocsContaining(self, word, doclist):
      doccount = 0
      for doc in doclist:
          if measurement.freq(self, term= word, document= doc) > 0:
              doccount +=1
      return doccount 

  def idf(self, word, doclist):
      n_samples = len(doclist)
      df = measurement.numDocsContaining(self, word, doclist)
      return np.log(n_samples / 1+df)

  def build_idf_matrix(self, idf_vector):
      idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
      np.fill_diagonal(idf_mat, idf_vector)
      return idf_mat

  def cosine_measurement(self, data, req):
      X = np.array(data[0:])
      Y = np.array(data)
      cosine_similaritas = pairwise_kernels(X, Y, metric='linear')
      frequency_cosine = pd.DataFrame(cosine_similaritas, index=req, columns=req)
      return frequency_cosine    

  def threshold_value(self, threshold, data):
      dt = data.values >= threshold
      dt1 = pd.DataFrame(dt, index= data.index, columns= data.columns)
      mask = dt1.isin([True])
      dt3 = dt1.where(mask, other= 0)
      mask2 = dt3.isin([False])
      th_cosine1 = dt3.where(mask2, other= 1)
      return th_cosine1

if __name__ == "__main__":

      myVSMMeasurement = measurement(data)
      mydoclist = data
      bow = myVSMMeasurement.bagOfWords(mydoclist)
      vocabulary = myVSMMeasurement.build_lexicon(mydoclist)

      # tfidf normal
      my_idf_vector = [myVSMMeasurement.idf(word, mydoclist) for word in vocabulary] # vektor idf
      my_idf_matrix = myVSMMeasurement.build_idf_matrix(my_idf_vector) # membuat matriks idf
      doc_term_matrix_tfidf = [np.dot(tf_vector, my_idf_matrix) for tf_vector in bow.values] 
      dt_cosine = myVSMMeasurement.cosine_measurement(doc_term_matrix_tfidf.values)
      th_cosine = myVSMMeasurement.threshold_value(0.2, dt_cosine)
      print(tabulate(dt_cosine, headers = 'keys', tablefmt = 'psql'))
      myUkur = pengukuranEvaluasi(dt_cosine, th_cosine)
      hasil_ukur1 = myUkur.ukur_evaluasi()


      # tfidf dengan l2 normalizer
      doc_term_matrix_l2 = [myVSMMeasurement.l2_normalizer(vec) for vec in bow.values]
      doc_term_matrix_tfidf_l2 = [myVSMMeasurement.l2_normalizer(tf_vector) for tf_vector in doc_term_matrix_tfidf]
      dt_cosine_l2 = myVSMMeasurement.cosine_measurement(doc_term_matrix_tfidf_l2.values)
      th_cosine_l2 = myVSMMeasurement.threshold_value(0.2, dt_cosine_l2)
      print(tabulate(dt_cosine, headers = 'keys', tablefmt = 'psql'))
      myUkur = pengukuranEvaluasi(dt_cosine_l2, th_cosine_l2)
      hasil_ukur2 = myUkur.ukur_evaluasi()