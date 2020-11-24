import pandas as pd
import streamlit as st
import numpy as np
import math
import string #allows for format()
from sklearn.feature_extraction.text import CountVectorizer
from cleaning import apply_cleaning, fulldataset

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
     genre = st.sidebar.radio("What do you choose?",('Information_Retrieval', 'Ontology', 'IR+LSA', 'IR+LDA'))
     if genre == 'Information_Retrieval':
          st.subheader('Information Retrieval Parameter')
          count_vector = CountVectorizer(cleaned_text)
          count_vector.fit(cleaned_text)
          kolom_df = count_vector.get_feature_names()
          doc_array = count_vector.transform(cleaned_text).toarray()
          id_requirement = fulldataset(index0, index1)['ID']
          frequency_matrix = pd.DataFrame(doc_array, index= id_requirement, columns= kolom_df)
          st.write(frequency_matrix)
          
          # normalizer
          def l2_normalizer(vec):
              denom = np.sum([el**2 for el in vec])
              return [(el / math.sqrt(denom)) for el in vec]

          doc_term_matrix_l2 = []
          for vec in doc_array:
              doc_term_matrix_l2.append(l2_normalizer(vec))
          
          st.write(doc_term_matrix_l2)
          
          def build_lexicon(corpus):
              lexicon = set()
              for doc in corpus:
                  lexicon.update([word for word in doc.split()])
              return lexicon

          def freq(term, document):
            return document.split().count(term)

          def numDocsContaining(word, doclist):
              doccount = 0
              for doc in doclist:
                  if freq(word, doc) > 0:
                      doccount +=1
              return doccount 

          def idf(word, doclist):
              n_samples = len(doclist)
              df = numDocsContaining(word, doclist)
              return np.log(n_samples / 1+df)

          vocabulary = build_lexicon(cleaned_text)
          mydoclist = cleaned_text

          my_idf_vector = [idf(word, mydoclist) for word in vocabulary]
          
          def build_idf_matrix(idf_vector):
              idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
              np.fill_diagonal(idf_mat, idf_vector)
              return idf_mat

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
          st.write (hasil_tfidf)
          

          # np.matrix() just to make it easier to look at
          
     elif genre == 'Ontology':
          st.write("ontology.")
     elif genre == 'IR+LSA':
          st.write("LSA.")
     elif genre == 'IR+LDA':
          st.write("LDA.")
