import streamlit as st
from cleaning import apply_cleaning

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.read()
     st.write(bytes_data)

    # Load data example (dari functional maupun nonfunctional)
    statement = bytes_data

    # Get text to clean (dari row yang diinginkan)
    text_to_clean = list(statement['Requirement Statement'])

    # Clean text
    print("Loading Original & Cleaned Text...")
    cleaned_text = apply_cleaning(text_to_clean)

    # Show first example
    text_df = pd.DataFrame([text_to_clean, cleaned_text],index=['ORIGINAL','CLEANED'], columns= statement['ID'])
    st.write(text_df)
