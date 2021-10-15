"""Modul ini digunakan untuk melakukan proses preprocessing dan evaluasi.
   Sehingga dengan jelas, jika ada penggunaan preprocessing seperti cleaning text.
   Maka modul ini digunakan, agar, digunakan lebih baik.
   dari modul ini pula dapat digunakan untuk mengeveluasi setiap model yang dikerjakan.
"""
import numpy as np, pandas as pd
from pyAutoML.ml import ML, EncodeCategorical
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from tabulate import tabulate


class prosesData:  
  def __init__(self, namaFile):
      '''fungi ini digunakan untuk menginisialisasi data
      sehingga perlu dilakukan proses construct data.
      '''
      self.dataFile = namaFile

  def fulldataset(self, inputSRS= 'SRS1'):
      '''fungi ini digunakan untuk menampilkan data secara spesifik,
      sehingga menunjukkan data yang diinginkan.
      syntax yang digunakan adalah sebagai berikut:
      prosesData(data).fulldataset(inputSRS)
      '''
      xl = pd.ExcelFile(self.dataFile)
      dfs = {sh:xl.parse(sh) for sh in xl.sheet_names}[inputSRS]
      return dfs

  def preprocessing(self):
      '''fungsi ini digunakan untuk mengecek data yang digunakan 
      untuk melihat data yang digunakan. sehingga syntax yang digunakan
      yaitu adalah sebagai berikut:
      prosesData(data).preprocessing()
      '''
      xl = pd.ExcelFile(self.dataFile)
      for sh in xl.sheet_names:
        df = xl.parse(sh)
        print('Processing: [{}] ...'.format(sh))
        print(df.head())

  def apply_cleaning_function_to_list(self, X):
      '''fungsi ini digunakan untuk mengaplikasikan fungsi pembersihan data, 
      sesuai dengan list yang dipilih. Sehingga maasuk ke tahapan selanjutnya.
      prosesData(data).apply_cleaning_function_to_list(X)
      '''
      cleaned_X = [prosesData.clean_text(self, raw_text= element)for element in X]
      return cleaned_X

  def clean_text(self, raw_text):
      '''fungsi ini digunakan untuk mengaplikasikan fungsi pembersihan data, 
      sesuai dengan list yang dipilih. Sehingga maasuk ke tahapan selanjutnya.
      prosesData(data).clean_text(raw_text)
      '''
      nlp = English()
      tokenizer = Tokenizer(nlp.vocab)
      tokens = tokenizer(raw_text)
      lemma_list = [token.lemma_.lower() for token in tokens if token.is_stop is False 
      and token.is_punct is False and token.is_alpha is True]
      joined_words = ( " ".join(lemma_list))
      return joined_words  

class pengukuranEvaluasi:
  def __init__(self):
      '''fungsi ini digunakan untuk menginisialisasi data
      sehingga perlu dilakukan proses construct data.
      '''
      pass

  def threshold_value(self, threshold, data):
      '''fungsi ini digunakan untuk mendapatkan nilai threshold yang diingikan.
      Fungsi yang digunakan cukup dengan 
      pengukuranEvaluasi().threshold_value(threshold, data)
      '''
      dt = data.values >= threshold
      dt1 = pd.DataFrame(dt, index= data.index, columns= data.columns)
      mask = dt1.isin([True])
      dt3 = dt1.where(mask, other= 0)
      mask2 = dt3.isin([False])
      th_cosine1 = dt3.where(mask2, other= 1)
      return th_cosine1

  def kmeans_cluster(self, data1, data2, nilai_cluster= 3):
      '''fungsi ini digunakan untuk mengklaster data menggunakan metode kmeans.
      Dari fungsi ini dapat menciptakan data korelasi yang dimuat hasil klaster.
      pengukuranEvaluasi().kmeans_cluster(data1, data2)
      '''
      XVSM = np.array(data1)
      yVSM = np.array(data2)
      kmeans = KMeans(n_clusters=nilai_cluster) # You want cluster the passenger records into 2: Survived or Not survived
      kmeans.fit(XVSM)
      correct = 0
      for i in range(len(XVSM)):
          predict_me = np.array(XVSM[i].astype(float))
          predict_me = predict_me.reshape(-1, len(predict_me))
          prediction = kmeans.predict(predict_me)
          if prediction[0] == yVSM.all():
              correct += 1
      scaler = MinMaxScaler()
      XVSM_scaled = scaler.fit_transform(yVSM)
      print("data_correction {}".format(correct/len(XVSM)))
      return (XVSM_scaled)


  def ukur_evaluasi(self, data1, data2, nilai_cluster=3, test_size= 0.3, random_state= 100, size= 0.4, max_iter= 7000):
      '''fungsi ini digunakan untuk mengukur evaluasi dari data 1, dan data2.
      pengukuranEvaluasi().ukur_evaluasi(data1, data2)
      '''
      X_train, X_test, y_train, y_test = \
      train_test_split(pengukuranEvaluasi.kmeans_cluster(self, data1, data2, nilai_cluster), 
      data2, test_size) # 70% training and 30% test
      y_train = y_train.argmax(axis= 1)
      y_train = EncodeCategorical(y_train)
      return ML(X_train, y_train, size, SVC(), RandomForestClassifier(), 
      DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(max_iter))      

if __name__ == "__main__":
    myData = prosesData() # myData.preprocessing()
    req = myData.fulldataset() # myData.fulldataset(inputSRS)
    id_req  = req['ID']
    text_to_clean = list(req['Requirement Statement'])
    cleaned_text = myData.apply_cleaning_function_to_list(text_to_clean)
    print(tabulate(cleaned_text, headers = 'keys', tablefmt = 'psql'))
