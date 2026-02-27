# filepath: /home/prichai/AI_ML/RAG_Project/RAG_Streamlit_App/pages/1_MainPage.py
import os
import tempfile
import streamlit as st

# import the processing function
from embedandchunk import process_pdf_for_rag

st.set_page_config(page_title="RAG Streamlit App - Main", layout="centered")

# Require login
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

st.title("Upload a PDF File")
st.write(f"Logged in as: {st.session_state.user_email}")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    data_folder = "/home/prichai/AI_ML/RAG_Project/RAG_Streamlit_App/data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded successfully and saved as: {file_path}")

    # Run RAG preprocessing on the uploaded PDF
    with st.spinner("Processing PDF for RAG (this may take a while)..."):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        pages_csv = os.path.join(data_folder, f"{base_name}_pages.csv")
        chunks_csv = os.path.join(data_folder, f"{base_name}_chunks.csv")
        embeddings_pt = os.path.join(data_folder, f"{base_name}_embeddings.pt")

        process_pdf_for_rag(
            pdf_path=file_path,
            pages_csv_path=pages_csv,
            chunks_csv_path=chunks_csv,
            embeddings_tensor_path=embeddings_pt,
        )

    st.success("PDF processed successfully for RAG!")
    st.info(f"Pages CSV: {pages_csv}")
    st.info(f"Chunks CSV: {chunks_csv}")
    st.info(f"Embeddings tensor: {embeddings_pt}")

if st.button("Logout"):
    st.session_state.user_logged_in = False
    st.session_state.user_email = None
    st.switch_page("app.py")