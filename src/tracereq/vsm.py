import math, numpy as np, pandas as pd
from scipy.sparse import data
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tracereq.preprocessing_evaluation import pengukuranEvaluasi


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

  def main(self, threshold, data, output= ['normal', 'l2_normalizer']):
    bow = measurement.bagOfWords(self, data)
    vocabulary = measurement.build_lexicon(self, data)
    my_idf_vector = [measurement.idf(self, word, data) for word in vocabulary] # vektor idf
    my_idf_matrix = measurement.build_idf_matrix(self, my_idf_vector) # membuat matriks idf
    doc_term_matrix_tfidf = [np.dot(tf_vector, my_idf_matrix) for tf_vector in bow.values] 
    dt_cosine = measurement.cosine_measurement(self, doc_term_matrix_tfidf.values)
    th_cosine = measurement.threshold_value(threshold, dt_cosine)# tfidf normal
    myUkur = pengukuranEvaluasi(dt_cosine, th_cosine).ukur_evaluasi()
    doc_term_matrix_l2 = [measurement.l2_normalizer(self, vec) for vec in bow.values]
    doc_term_matrix_tfidf_l2 = [measurement.l2_normalizer(self, tf_vector) for tf_vector in doc_term_matrix_tfidf]
    dt_cosine_l2 = measurement.cosine_measurement(self, doc_term_matrix_tfidf_l2.values)
    th_cosine_l2 = measurement.threshold_value(self, threshold, dt_cosine_l2)# tfidf dengan l2 normalizer
    myUkur = pengukuranEvaluasi(dt_cosine_l2, th_cosine_l2).ukur_evaluasi()
    if 'normal' in output:
        return my_idf_vector, my_idf_matrix, doc_term_matrix_tfidf, dt_cosine, th_cosine, myUkur
    if 'l2_normalizer' in output:
        return doc_term_matrix_l2, doc_term_matrix_tfidf_l2, dt_cosine_l2, th_cosine_l2, myUkur

if __name__ == "__main__":
  try:
    a = measurement(data).main(0.2, data, 'l2_normalizer')
    print(tabulate(a[0], headers = 'keys', tablefmt = 'psql'))
    print(tabulate(a[1], headers = 'keys', tablefmt = 'psql'))
    print(tabulate(a[2], headers = 'keys', tablefmt = 'psql'))
    print(tabulate(a[3], headers = 'keys', tablefmt = 'psql'))
    print(tabulate(a[4], headers = 'keys', tablefmt = 'psql'))


  except OSError as err:
    print("OS error: {0}".format(err))
