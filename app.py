import pandas as pd
import streamlit as st
from cleaning import apply_cleaning, fulldataset

#file upload
index0 = st.file_uploader("Choose a file") 
if index0 is not None:
     st.sidebar.header('Dataset Parameter')
     x1 = pd.ExcelFile(index0)
     index1 = st.sidebar.selectbox( 'What Dataset you choose?', x1.sheet_names)

     # Load data example (dari functional maupun nonfunctional)
     st.subheader('Dataset parameters')
     statement = fulldataset(index0, index1)

     # Get text to clean (dari row yang diinginkan)
     text_to_clean = list(statement['Requirement Statement'])

     # Clean text
     print("Loading Original & Cleaned Text...")
     cleaned_text = apply_cleaning(text_to_clean)

     # Show first example
     text_df = pd.DataFrame([text_to_clean, cleaned_text],index=['ORIGINAL','CLEANED'], columns= statement['ID'])
     st.write(text_df)
