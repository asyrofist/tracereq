import pandas as pd
from scipy.sparse import data
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate

from traceability.preprocessing_evaluation import pengukuranEvaluasi


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

  def main(self, threshold, data):
      dt_lsa = latentSemantic.ukurLSA(self, data)
      th_lsa = latentSemantic.threshold_value(self, threshold, dt_lsa)
      myUkur = pengukuranEvaluasi(dt_lsa, th_lsa).ukur_evaluasi()
      return dt_lsa, th_lsa, myUkur

if __name__ == "__main__":
  try:
    a = latentSemantic(data).main(0.2, data)
    print(tabulate(a[0], headers = 'keys', tablefmt = 'psql'))
    print(tabulate(a[1], headers = 'keys', tablefmt = 'psql'))
    print(a[2])

  except OSError as err:
    print("OS error: {0}".format(err))
