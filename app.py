import pandas as pd
import streamlit as st
import numpy as np
import string #allows for format()
from sklearn.feature_extraction.text import CountVectorizer
from cleaning import apply_cleaning, fulldataset
from cleaning import l2_normalizer, build_lexicon, freq, numDocsContaining, idf, build_idf_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


#file upload
index0 = st.file_uploader("Choose a file") 
if index0 is not None:
     st.sidebar.header('Dataset Parameter')
     x1 = pd.ExcelFile(index0)
     index1 = st.sidebar.selectbox( 'What Dataset you choose?', x1.sheet_names)

     # Load data example (dari functional maupun nonfunctional)
     st.header('Dataset parameters')
     statement = fulldataset(index0, index1)

     # Get text to clean (dari row yang diinginkan)
     text_to_clean = list(statement['Requirement Statement'])

     # Clean text
     print("Loading Original & Cleaned Text...")
     cleaned_text = apply_cleaning(text_to_clean)

     # Show first example
     text_df = pd.DataFrame([text_to_clean, cleaned_text],index=['ORIGINAL','CLEANED'], columns= statement['ID']).T
     st.write(text_df)
     
     st.header('Traceability parameters')
     id_requirement = fulldataset(index0, index1)['ID']
     
     genre = st.sidebar.radio("What do you choose?",('Information_Retrieval', 'Ontology', 'IR+LSA', 'IR+LDA'))
     if genre == 'Information_Retrieval':
          st.subheader("bag of words")
          count_vector = CountVectorizer(cleaned_text)
          count_vector.fit(cleaned_text)
          kolom_df = count_vector.get_feature_names()
          doc_array = count_vector.transform(cleaned_text).toarray()
          frequency_matrix = pd.DataFrame(doc_array, index= id_requirement, columns= kolom_df)
          st.write(frequency_matrix)

          # l2 normalizer
          vocabulary = build_lexicon(cleaned_text)
          mydoclist = cleaned_text

          my_idf_vector = [idf(word, mydoclist) for word in vocabulary]
          my_idf_matrix = build_idf_matrix(my_idf_vector)
          
          doc_term_matrix_tfidf = []
          #performing tf-idf matrix multiplication
          for tf_vector in doc_array:
              doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))

          #normalizing
          doc_term_matrix_tfidf_l2 = []
          for tf_vector in doc_term_matrix_tfidf:
              doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
          hasil_tfidf = np.matrix(doc_term_matrix_tfidf_l2)
          
          st.subheader("l2 tfidf normalizer")
          frequency_TFIDF = pd.DataFrame(hasil_tfidf,index= id_requirement, columns= kolom_df)
          st.write(frequency_TFIDF)
          
          # klaster
          klaster_value = st.sidebar.slider("Berapa Cluster?", 0, 20, len(id_requirement))
          kmeans = KMeans(n_clusters= klaster_value) # You want cluster the passenger records into 2: Survived or Not survived
          kmeans_df = kmeans.fit(hasil_tfidf)
          st.subheader("K-Means Cluster")
          
          correct = 0
          for i in range(len(hasil_tfidf)):
              predict_me = np.array(cosine_similaritas[i].astype(float))
              predict_me = predict_me.reshape(-1, len(predict_me))
              prediction = kmeans.predict(predict_me)
              if prediction[0] == hasil_tfidf[i].all():
                  correct += 1
          st.sidebar.write(correct/len(hasil_tfidf))
          
          klasterkm = kmeans.cluster_centers_
          klaster_df = pd.DataFrame(klasterkm, columns= id_requirement)
          st.write(klaster_df)

     elif genre == 'Ontology':
          st.write("ontology.")
          
     elif genre == 'IR+LSA':
          st.write("LSA.")
          feature_value = st.sidebar.slider("Berapa Feature?", 10, 100, 1000)
          df_value = st.sidebar.slider("Berapa df?", 0.0, 0.9, 0.5)
          feature_value = st.sidebar.slider('Berapa Max Feature Model?', 0, 10, 1000)
          iterasi_value = st.sidebar.slider('Berapa Dimension Model?', 0, 200, 100)
          random_value = st.sidebar.slider('Berapa Random Model?', 0, 300, 122)
          
          vectorizer = TfidfVectorizer(stop_words='english', 
          max_features= feature_value, # keep top 1000 terms 
          max_df = df_value, 
          smooth_idf=True)
          X = vectorizer.fit_transform(cleaned_text)
          fitur_id = vectorizer.get_feature_names()
          svd_model = TruncatedSVD(n_components= (X.shape[0]), algorithm='randomized', n_iter= iterasi_value, random_state= random_value)
          svd_model.fit(X)
          jumlah_kata = svd_model.components_
          tabel_lsa = pd.DataFrame(jumlah_kata, index= id_requirement, columns= fitur_id)
          st.dataframe(tabel_lsa)
          
          st.subheader("VSM using cosine")
          X = np.array(jumlah_kata[0:])
          Y = np.array(jumlah_kata)
          cosine_similaritas = pairwise_kernels(X, Y, metric='linear')
          cosine_df = pd.DataFrame(cosine_similaritas,index= id_requirement, columns= id_requirement)
          st.write(cosine_df)
          
          # klaster
          klaster_value = st.sidebar.slider("Berapa Cluster?", 0, 5, len(id_requirement))
          kmeans = KMeans(n_clusters= klaster_value) # You want cluster the passenger records into 2: Survived or Not survived
          kmeans_df = kmeans.fit(cosine_similaritas)
          st.subheader("K-Means Cluster")
          
          correct = 0
          for i in range(len(cosine_similaritas)):
              predict_me = np.array(cosine_similaritas[i].astype(float))
              predict_me = predict_me.reshape(-1, len(predict_me))
              prediction = kmeans.predict(predict_me)
              if prediction[0] == cosine_similaritas[i].all():
                  correct += 1
          st.sidebar.write(correct/len(cosine_similaritas))
          
          klasterkm = kmeans.cluster_centers_
          klaster_df = pd.DataFrame(klasterkm, columns= id_requirement)
          st.write(klaster_df)
          
     elif genre == 'IR+LDA':
          st.write("LDA.")
