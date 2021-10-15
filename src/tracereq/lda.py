"""Modul ini mengkalkulasi setiap bobot dengan penggunaa lda, dari setiap dokumen yang 
dibuat fungsi ini digunakan untuk memproses lda yang baik dan benar.
"""
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tracereq.preprocessing_evaluation import pengukuranEvaluasi, cleaned_text, id_req, tabulate


class latentDirichlet:
  def __init__(self, cleaned_text):
      self.data = cleaned_text
      self.n_features = len(self.data)

  def ukur_tfidf_vectorizer(self):
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

  def tabulasi_fr(self, th, id_req, param= ['fr', 'th', 'ukur']):
      dt_fr = latentDirichlet.Frobenius_norm_feature(self, id_req)
      th_fr = pengukuranEvaluasi.threshold_value(self, th, dt_fr)
      myUkur = pengukuranEvaluasi.ukur_evaluasi(self, dt_fr, th_fr)
      if 'fr' in param:
          return dt_fr
      elif 'th' in param:
          return th_fr
      elif 'ukur' in param:
          return myUkur
      
  def tabulasi_kl(self, th, id_req, param= ['kl', 'th', 'ukur']):
      dt_kl = latentDirichlet.Kullback_feature(self, id_req)
      th_kl = pengukuranEvaluasi.threshold_value(self, th, dt_kl)
      myUkur = pengukuranEvaluasi.ukur_evaluasi(self, dt_kl, th_kl)
      if 'kl' in param:
          return dt_kl
      elif 'th' in param:
          return th_kl
      elif 'ukur' in param:
          return myUkur

  def tabulasi_lda(self, th, id_req, param= ['lda', 'th', 'ukur']):
      dt_lda = latentDirichlet.lda_feature(id_req)
      th_lda = pengukuranEvaluasi.threshold_value(th, dt_lda)
      myUkur = pengukuranEvaluasi.ukur_evaluasi(self, dt_lda, th_lda)
      if 'lda' in param:
          return dt_lda
      elif 'th' in param:
          return th_lda
      elif 'ukur' in param:
          return myUkur

  def main(self, th, id_req, output= ['fr', 'kl', 'lda']):
      if 'fr'in output:
        return latentDirichlet.tabulasi_fr(self, th, id_req,'fr')
      elif 'kl'in output:
        return latentDirichlet.tabulasi_kl(self, th, id_req, 'kl')
      elif 'lda'in output:
        return latentDirichlet.tabulasi_lda(self, th, id_req, 'lda')


if __name__ == "__main__":
  try:
    a = latentDirichlet(cleaned_text).main(0.2, id_req, 'lda')
    print(tabulate(a, headers = 'keys', tablefmt = 'psql'))

  except OSError as err:
    print("OS error: {0}".format(err))
