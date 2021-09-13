import pandas as pd
from scipy.sparse import data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from traceability.preprocessing_evaluation import pengukuranEvaluasi
from tabulate import tabulate


class latentDirichlet:
  def __init__(self, data_raw):
      self.data = data_raw
      self.n_features = len(self.data)

  def ukur_tfidf_vectorizer(self):
      # Use tf-idf features for NMF.
      tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_features,
                                        stop_words='english')
      tfidf = tfidf_vectorizer.fit_transform(self.data)
      return tfidf_vectorizer, tfidf

  def ukur_tf(self):
      tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                      max_features=self.n_features,
                                      stop_words='english')
      tf = tf_vectorizer.fit_transform(self.data)
      return tf_vectorizer, tf

  def Frobenius_norm_feature(self, req):
      nmf = NMF(n_components=len(self.data), random_state=1, alpha=.1, l1_ratio=.5).fit(latentDirichlet.ukur_tfidf_vectorizer(self)[1])
      nmf_tfidf = latentDirichlet.ukur_tfidf_vectorizer(self)[0].get_feature_names()
      fitur_frb_tfidf = (nmf_tfidf)
      data_frb_tfidf = (nmf.components_)
      dt_df =  pd.DataFrame(data_frb_tfidf, index= req, columns= fitur_frb_tfidf)
      return dt_df

  def Kullback_feature(self, req):
      nmf = NMF(n_components=len(self.data), random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1, l1_ratio=.5).fit(latentDirichlet.ukur_tfidf_vectorizer(self)[1])
      tfidf_feature_names = latentDirichlet.ukur_tfidf_vectorizer(self)[0].get_feature_names()
      fitur_kll_tfidfi = (tfidf_feature_names)
      data_kll_tfidf = (nmf.components_)
      dt_df =  pd.DataFrame(data_kll_tfidf, index= req, columns= fitur_kll_tfidfi)
      return dt_df

  def lda_feature(self, req):
      lda = LatentDirichletAllocation(n_components=len(self.data), max_iter=5, learning_method='online', learning_offset=50., random_state=0)
      lda.fit(latentDirichlet.ukur_tf(self)[1])
      tf_feature_names = latentDirichlet.ukur_tf(self)[0].get_feature_names()
      fitur_lda = (tf_feature_names)
      nmf = NMF(n_components=len(self.data), random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1, l1_ratio=.5).fit(latentDirichlet.ukur_tfidf_vectorizer(self)[1])
      data_lda = (nmf.components_)
      dt_df =  pd.DataFrame(data_lda, index= req, columns= fitur_lda)
      return dt_df

  def threshold_value(self, threshold, data):
      dt = data.values >= threshold
      dt1 = pd.DataFrame(dt, index= data.index, columns= data.columns)
      mask = dt1.isin([True])
      dt3 = dt1.where(mask, other= 0)
      mask2 = dt3.isin([False])
      th_cosine1 = dt3.where(mask2, other= 1)
      return th_cosine1

if __name__ == "__main__":
      myLDA = latentDirichlet(data)
      dt_fr = myLDA.Frobenius_norm_feature(data)
      print(tabulate(dt_fr, headers = 'keys', tablefmt = 'psql'))
      th_fr = myLDA.threshold_value(0.2, dt_fr)
      print(tabulate(th_fr, headers = 'keys', tablefmt = 'psql'))
      myUkur = pengukuranEvaluasi(dt_fr, th_fr)
      hasil_ukur1 = myUkur.ukur_evaluasi()

      dt_kl = myLDA.Kullback_feature(data)
      print(tabulate(dt_kl, headers = 'keys', tablefmt = 'psql'))
      th_kl = myLDA.threshold_value(0.2, dt_kl)
      print(tabulate(th_kl, headers = 'keys', tablefmt = 'psql'))
      myUkur = pengukuranEvaluasi(dt_kl, th_kl)
      hasil_ukur2 = myUkur.ukur_evaluasi()

      dt_lda = myLDA.lda_feature(data)
      print(tabulate(dt_lda, headers = 'keys', tablefmt = 'psql'))
      th_lda = myLDA.threshold_value(0.2, dt_lda)
      print(tabulate(th_lda, headers = 'keys', tablefmt = 'psql'))
      myUkur = pengukuranEvaluasi(dt_lda, th_lda)
      hasil_ukur3 = myUkur.ukur_evaluasi()