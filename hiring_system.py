import os
import json
import pdfplumber 
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from huggingface_hub import login



class CVHiringSystem:
  def __init__(self, hug_token: str, cv_dir: str, output_json: str, qdrant_url: str, qdrant_api: str, collection_name: str):
    """
        Initializes the CVHiringSystem with necessary API keys, models, and paths.

        Args:
            hf_token (str): Hugging Face token for authentication.
            qdrant_url (str): URL for the Qdrant instance.
            qdrant_api_key (str): API key for the Qdrant instance.
            cv_dir (str): Directory path where CV PDF files are stored.
            output_json (str): Path to the output JSON file where extracted CV data will be saved.
        """
    self.cv_dir=cv_dir
    self.output_json= output_json
    self.qdrant_client=QdrantClient(url=qdrant_url,api_key=qdrant_api)
    self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
    self.collection_name=collection_name
    self.hugging_login(hug_token)
    self.pipeline=pipeline('text-generation',model='meta-llama/Llama-3.2-3B-Instruct',max_new_tokens=2000,device=self.device)
    self.embedding_model= SentenceTransformer('paraphrase-mpnet-base-v2',device=self.device)


  def hugging_login(self,hug_token: str) -> None:
    """
        Logs in to Hugging Face using the provided token.

        Args:
            hf_token (str): Hugging Face API token.
        """
    login(token=hug_token)

  def extract_txt (self, pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
      return ''.join([page.extract_text() or '' for page in pdf.pages])

  def pdf_txt_extract(self) -> list :
    """
        Extracts text from all PDF files in the specified directory and returns their content.

        Returns:
            list: A list of dictionaries where each dictionary contains the filename and extracted text content.
        """
    data=[]
    cv_files=[f for f in os.listdir(self.cv_dir) if f.endswith('.pdf')]
    for filename in cv_files:
      file_path= os.path.join(self.cv_dir, filename)
      extracted_text= self.extract_txt(file_path)
      data.append({'filename':filename,'text':extracted_text})
    return data

  def getting_info_cvs(self,data: list) -> list:
     
     """
        Extracts structured information from CV text using a language model.

        Args:
            data (list): List of dictionaries containing filenames and their corresponding text data.

        Returns:
            list: A list of dictionaries where each dictionary contains the filename and extracted structured data.
        """
     
     results=[]

     for file in data:
       filename=file["filename"]
       text=file['text']
       prompt = {
            "Name": f"""<s>[INST] <<SYS>>
        Extract only the full name from the following CV text. Provide the result in JSON format as shown:
        {{
            "Name": "<Full Name>"
        }}

        CV text:
        {text}
        [/INST]</s>""",

            "Education": f"""<s>[INST] <<SYS>>
        Extract only the educational background from the following CV text. Provide the result in JSON format as shown:
        {{
            "Education": "<Degrees and Institutions>"
        }}

        CV text:
        {text}
        [/INST]</s>""",

            "Skills": f"""<s>[INST] <<SYS>>
        Extract only the key skills from the following CV text. Provide the result in JSON format as shown:
        {{
            "Skills": "<Key Skills List>"
        }}

        CV text:
        {text}
        [/INST]</s>""",

            "WorkExperience": f"""<s>[INST] <<SYS>>
        Extract only the work experience from the following CV text, including job titles, companies, dates, and job details (descriptions of the responsibilities and achievements). Provide the result in JSON format as shown:
        {{
            "WorkExperience": "<Job Titles, Companies, Dates, and Job Details>"
        }}

        CV text:
        {text}
        [/INST]</s>""",

            "Certifications": f"""<s>[INST] <<SYS>>
        Extract only the certifications from the following CV text. Provide the result in JSON format as shown:
        {{
            "Certifications": "<Certifications and Issuing Organizations>"
        }}

        CV text:
        {text}
        [/INST]</s>"""
        }
       input_data=[{'role':'user','content':prompt}] 
       result=self.pipeline(input_data,temperature=0.000000001)[0]['generated_text']
        
       results.append({'filename':file['filename'],'results':result})
     return results

  def creating_json(self, results):
    """
        Saves the structured CV data to a JSON file.

        Args:
            results (list): List of structured data extracted from CVs.
        """
    with open (self.output_json,'w',encoding='utf-8') as f:
      json.dump(results,f,indent=4, ensure_ascii=False)

  def loading_json(self):
    """
        Loads data from the output JSON file.

        Returns:
            list: Data loaded from the JSON file.
        """
    with open(self.output_json,'r') as file:
      return json.load(file)

  def _extract_cleaned_fields(self, cv_info):
        """
        Extracts and cleans specific fields from the CV information.

        Args:
            cv_info (dict): Dictionary containing raw extracted information.

        Returns:
            dict: Dictionary with cleaned field data.
        """
        def clean_field(data, field):
            if field in data and data[field].get('generated_text'):
                for entry in data[field][0]['generated_text']:
                    if entry['role'] == 'assistant':
                        try:
                            content= entry['content']
                            parsed_content = content.split(f'"{field}":')[1].split('\n')[0].strip(' "{}')
                            return parsed_content
                        except IndexError:
                            return ""
            return ""

        name = clean_field(cv_info, "Name")
        education = clean_field(cv_info, "Education")
        skills = clean_field(cv_info, "Skills")
        experience = clean_field(cv_info, "WorkExperience")
        certifications = clean_field(cv_info, "Certifications")

        return {
            'Name': name,
            'Education': education,
            'Skills': skills,
            'WorkExperience': experience,
            'Certifications': certifications
        }

  def embedd_and_storing_cv(self,data):
    """
        Embeds and stores CV data into Qdrant.

        Args:
            data (list): List of structured data extracted from CVs.
        """
    text_data=[]
    for cv in data:
        
        cv_info= cv['results']
        cleaned_data=self._extract_cleaned_fields(cv_info)
        combined_text = (f"Name: {cleaned_data.get('Name')}\nEducation: {cleaned_data.get('Education')}\n"
                         f"Skills: {cleaned_data.get('Skills')}\nExperience: {cleaned_data.get('WorkExperience')}\n"
                         f"Certifications: {cleaned_data.get('Certifications')}")
        text_data.append(combined_text)
    embeddings= np.array(self.embedding_model.encode(text_data,show_progress_bar=True))
    vector_config=VectorParams(size=embeddings.shape[1],distance='Cosine')
    self.qdrant_client.recreate_collection(collection_name=self.collection_name,vectors_config=vector_config)

    for idx,embedding in enumerate(embeddings):
         
        point=PointStruct(id=idx, vector=embedding.tolist(), payload={'text':text_data[idx]})
        self.qdrant_client.upsert(collection_name=self.collection_name,points=[point])

  def match_candidates(self,job_description):
    """
        Matches job description with stored CV embeddings.

        Args:
            job_description (str): Job description to match with candidate CVs.

        Returns:
            list: List of matched candidates with similarity scores.
        """
    job_embedds= self.embedding_model.encode(job_description).tolist()
    search_result= self.qdrant_client.search(collection_name=self.collection_name,query_vector=job_embedds,limit=10)

    matches = [{'candidate_id': result.id, 'text': result.payload['text'], 'similarity_score': result.score} for result in search_result]
    return matches
