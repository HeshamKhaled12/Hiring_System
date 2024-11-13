import os
import streamlit as st
from hiring_system import CVHiringSystem

CV_TEMP_DIR = "temp_cvs"

HUGGING_FACE_TOKEN = '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'
QDRANT_API_KEY = 'v3nBB4OdDHbEvmVdqwkD0wlPaqF_zb8S9j2z4HWfUMqvz3tOnuogtA'
QDRANT_URL = 'https://3627359d-1a29-4061-9502-3bea64dca6c3.us-east4-0.gcp.cloud.qdrant.io:6333'
OUTPUT_JSON = 'cv_data.json'

if not os.path.exists(CV_TEMP_DIR):
    os.makedirs(CV_TEMP_DIR)

st.title("CV Hiring System")
st.write("Upload a folder containing multiple CV PDFs, and the system will extract relevant data.")

collection_name = st.text_input("Enter the Qdrant collection name:", "cv_collection")

uploaded_files = st.file_uploader("Choose CV PDFs", type="pdf", accept_multiple_files=True)

cv_hiring_system = CVHiringSystem(
    hug_token=HUGGING_FACE_TOKEN,
    cv_dir=CV_TEMP_DIR,
    output_json=OUTPUT_JSON,
    qdrant_url=QDRANT_URL,
    qdrant_api=QDRANT_API_KEY,
    collection_name=collection_name
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(CV_TEMP_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success("Files uploaded successfully!")

    if st.button("Process CVs"):
        st.text("Extracting text and processing CVs...")
        extracted_data = cv_hiring_system.pdf_txt_extract()
        st.text("Extracting structured information from CVs...")
        structured_data = cv_hiring_system.getting_info_cvs(extracted_data)
        st.text("Saving extracted data...")
        cv_hiring_system.creating_json(structured_data)

        st.success("CV processing complete!")

        st.subheader("Extracted Data")
        st.json(structured_data)

        job_description = st.text_area("Enter Job Description", "")
        if job_description and st.button("Match Candidates"):
            st.text("Matching candidates with job description...")
            matches = cv_hiring_system.match_candidates(job_description)
            st.subheader("Matched Candidates")
            st.write(matches)

        for uploaded_file in uploaded_files:
            os.remove(os.path.join(CV_TEMP_DIR, uploaded_file.name))
