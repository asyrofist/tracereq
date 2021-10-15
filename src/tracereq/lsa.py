"""Modul ini mengkalkulasi setiap bobot dengan penggunaa lsa, dari setiap dokumen yang 
dibuat fungsi ini digunakan untuk memproses lsa yang baik dan benar.
"""
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tracereq.preprocessing_evaluation import prosesData, pengukuranEvaluasi, tabulate


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

  def main(self, threshold, data, output= ['lsa', 'th', 'ukur']):
      '''fungsi ini merupakan fungsi utama dari pengukuran evaluasi.
      cukup dengan menggunakan inisialisasi dari dengan syntax ini.
      pengukuranEvaluasi(cleaned_text).main(threshold, data, output= 'lsa')
      '''
      dt_lsa = latentSemantic.ukurLSA(self, data)
      th_lsa = pengukuranEvaluasi.threshold_value(self, threshold, dt_lsa)
      myUkur = pengukuranEvaluasi.ukur_evaluasi(self, dt_lsa, th_lsa)
      if 'lsa' in output:
        return dt_lsa
      elif 'th' in output:
        return th_lsa
      elif 'ukur' in output:
        return myUkur

if __name__ == "__main__":
  try:
    myData = prosesData() # myData.preprocessing()
    req = myData.fulldataset() # myData.fulldataset(inputSRS)
    id_req  = req['ID']
    text_to_clean = list(req['Requirement Statement'])
    cleaned_text = myData.apply_cleaning_function_to_list(text_to_clean)
    print(tabulate(cleaned_text, headers = 'keys', tablefmt = 'psql'))
    a = latentSemantic(cleaned_text).main(0.2, id_req, 'lsa')
    print(tabulate(a, headers = 'keys', tablefmt = 'psql'))

  except OSError as err:
    print("OS error: {0}".format(err))
