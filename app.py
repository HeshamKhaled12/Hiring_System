import streamlit as st
from pathlib import Path
from hiring_system import CVHiringSystem


huggin_face = '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'
qdrant_url = 'https://3627359d-1a29-4061-9502-3bea64dca6c3.us-east4-0.gcp.cloud.qdrant.io:6333'
qdrant_api = 'nBU4G05b0JnnLWiOOPVf_BNRAbOtR_6wZeTRsWHt3TDxwJSEWU578g'

def main():
    st.title('CV Hiring System')
    st.write("Welcome to the CV Hiring System!")
    
    
    st.write(f"Using predefined Hugging Face API Token: {huggin_face[:4]}... (hidden)")
    st.write("The system will use these API credentials to process the CVs.")

    exist_collection=st.checkbox('Use an existing collection')
    if exist_collection:
        collection_name= st.text_input('Enter the name of the existing Qdrant collection:')
    else:
        collection_name = st.text_input("Enter Collection Name for Qdrant:")

    if st.button("Confirm Collection Name"):
        st.success(f"Collection name '{collection_name}' confirmed.")

    
    uploaded_files = st.file_uploader('Upload CV PDFs', type='pdf', accept_multiple_files=True)
    if uploaded_files:
        cv_dir = Path('uploaded_cv_folder')
        cv_dir.mkdir(exist_ok=True)
        
        
        for uploaded_file in uploaded_files:
            file_path = cv_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
        st.success("PDFs uploaded successfully.")
    else:
        cv_dir = None

    if st.button("Initialize System") and collection_name and cv_dir:
        cv_system = CVHiringSystem(
            hug_token=huggin_face,
            cv_dir=str(cv_dir),
            output_json='output.json',
            qdrant_url=qdrant_url,
            qdrant_api=qdrant_api,
            collection_name=collection_name
        )
        st.success("System initialized successfully.")

        if exist_collection:
            st.write('Loading existing CV data from Qdrant ...')
            loaded_data=cv_system.qdrant_client.scroll(collection_name=collection_name,limit=100)
            for result in loaded_data:
                st.json(result.payload)
            st.write('Existing data loaded successfully.')
        else:
        
            st.write("Extracting text from PDFs...")
            pdf_data = cv_system.pdf_txt_extract()
            for file in pdf_data:
                st.write(f"Filename: {file['filename']}")
                st.text_area('Extracted Text', file['text'], height=300)

            
            st.write("Processing CV information...")
            results = cv_system.getting_info_cvs(pdf_data)
            for result in results:
                st.json(result)

            
            cv_system.creating_json(results)
            st.success("CV data extracted and saved successfully.")
            
            
            st.write("Embedding and storing CV data in Qdrant...")
            cv_system.embedd_and_storing_cv(results)
            st.success("CV data embedded and stored successfully in Qdrant.")


    job_desc = st.text_area("Enter the Job Description for Candidate Matching")
    num_matches = st.number_input("Number of matches to display", min_value=1, max_value=15, value=5)

    if job_desc and st.button("Match Candidates"):
        st.write(f"Matching candidates to job description: {job_desc}")
        matches = cv_system.match_candidates(job_desc)

        
        st.write(f"Top {num_matches} matches:")
        for match in matches[:num_matches]:
            st.write(f"Candidate ID: {match['candidate_id']} - Similarity: {match['similarity_score']}")
            st.text_area('Candidate Info', match['text'], height=150)

if __name__ == '__main__':
    main()
