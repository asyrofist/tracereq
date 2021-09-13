import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from traceability.preprocessing_evaluation import pengukuranEvaluasi
from tabulate import tabulate


class latentSemantic:
  def __init__(self, data_raw):
      self.data = data_raw

  def ukurLSA(self, req):
      vectorizer = TfidfVectorizer(stop_words='english', 
                                    max_features= 1000, # keep top 1000 terms 
                                    max_df = 0.5, 
                                    smooth_idf=True)
      X = vectorizer.fit_transform(self.data)
      svd_model = TruncatedSVD(n_components=len(self.data), algorithm='randomized', n_iter=100, random_state=122)
      svd_model.fit(X)
      terms = vectorizer.get_feature_names()
      return pd.DataFrame(svd_model.components_, index= req, columns= terms)

  def threshold_value(self, threshold, data):
      dt = data.values >= threshold
      dt1 = pd.DataFrame(dt, index= data.index, columns= data.columns)
      mask = dt1.isin([True])
      dt3 = dt1.where(mask, other= 0)
      mask2 = dt3.isin([False])
      th_cosine1 = dt3.where(mask2, other= 1)
      return th_cosine1

if __name__ == "__main__":
    # lsa measurement 
    myLSA = latentSemantic()
    dt_lsa = myLSA.ukurLSA()
    print(tabulate(dt_lsa, headers = 'keys', tablefmt = 'psql'))
    th_lsa = myLSA.threshold_value(0.2, dt_lsa)
    print(tabulate(th_lsa, headers = 'keys', tablefmt = 'psql'))
    myUkur = pengukuranEvaluasi(dt_lsa, th_lsa)
    hasil_ukur3 = myUkur.ukur_evaluasi()