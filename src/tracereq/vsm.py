"""Modul ini mengkalkulasi setiap bobot dengan penggunaa vsm, dari setiap dokumen yang 
dibuat fungsi ini digunakan untuk memproses vsm yang baik dan benar.
"""
import  numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_kernels
from tracereq.preprocessing_evaluation import prosesData, pengukuranEvaluasi, tabulate

class measurement:
    def __init__(self):
        pass
  
    def bow(self, data_raw, id_req, stopwords_list):
        vectorizer = CountVectorizer(stop_words= stopwords_list)
        X = vectorizer.fit_transform(data_raw)
        name_fitur = vectorizer.get_feature_names()
        df = pd.DataFrame(X.toarray(), index= id_req, columns = name_fitur)
        return df

    def tfidf(self, data_raw, id_req, norm_list, stopwords_list):
        vectorizer = TfidfVectorizer(norm= norm_list, stop_words= stopwords_list)
        X = vectorizer.fit_transform(data_raw)
        name = vectorizer.get_feature_names()
        df = pd.DataFrame(X.toarray(), index= id_req, columns= name)
        return df

    def cosine_measurement(self, data, req):
        X = np.array(data[0:])
        Y = np.array(req)
        cosine_similaritas = pairwise_kernels(X, Y, metric='linear')
        frequency_cosine = pd.DataFrame(cosine_similaritas, index=X, columns=Y)
        return frequency_cosine    

    def main(self, cleaned_text, id_req, stopwords_list = 'english' , 
            norm_list= ['l1', 'l2', 'max'], output= ['bow', 'tfidf']):
        data_bow = measurement.bow(self, cleaned_text, id_req, stopwords_list)
        data_tfidf = measurement.tfidf(self, cleaned_text, id_req, norm_list, stopwords_list)
        if 'bow' in output:
            return data_bow
        elif 'tfidf' in output:
            return data_tfidf

if __name__ == "__main__":
  try:
    myData = prosesData() # myData.preprocessing()
    req = myData.fulldataset() # myData.fulldataset(inputSRS)
    id_req  = req['ID']
    text_to_clean = list(req['Requirement Statement'])
    cleaned_text = myData.apply_cleaning_function_to_list(text_to_clean)
    print(tabulate(cleaned_text, headers = 'keys', tablefmt = 'psql'))
    a = measurement().main(cleaned_text, id_req, norm_list= 'l2', output= 'tfidf')
    print(tabulate(a, headers = 'keys', tablefmt = 'psql'))


  except OSError as err:
    print("OS error: {0}".format(err))
