"""Modul ini mengkalkulasi setiap bobot dengan penggunaa lda, dari setiap dokumen yang 
dibuat fungsi ini digunakan untuk memproses lda yang baik dan benar.
"""
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tracereq.preprocessing_evaluation import prosesData, tabulate


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

  def main(self, id_req, output= ['fr', 'kl', 'lda']):
      dt_fr = latentDirichlet.Frobenius_norm_feature(self, id_req)
      dt_kl = latentDirichlet.Kullback_feature(self, id_req)
      dt_lda = latentDirichlet.lda_feature(id_req)
      if 'fr'in output:
        return dt_fr
      elif 'kl'in output:
        return dt_kl
      elif 'lda'in output:
        return dt_lda

if __name__ == "__main__":
  try:
    myData = prosesData() # myData.preprocessing()
    req = myData.fulldataset() # myData.fulldataset(inputSRS)
    id_req  = req['ID']
    text_to_clean = list(req['Requirement Statement'])
    cleaned_text = myData.apply_cleaning_function_to_list(text_to_clean)
    print(tabulate(cleaned_text, headers = 'keys', tablefmt = 'psql'))
    a = latentDirichlet(cleaned_text).main(id_req, 'lda')
    print(tabulate(a, headers = 'keys', tablefmt = 'psql'))

  except OSError as err:
    print("OS error: {0}".format(err))
