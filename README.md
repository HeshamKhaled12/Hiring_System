# CV Hiring System

## Description
The CV Hiring System is an AI-powered solution designed to automate the hiring process by analyzing and matching CVs to job descriptions. This system extracts key information from CVs, embeds them into a vector space, and stores them in a Qdrant database. The system then compares candidate CVs to job descriptions to find the best matches. Built with a focus on efficiency and scalability, it leverages natural language processing (NLP) and machine learning techniques to provide a seamless experience for recruiters.

## Model Used
The project uses the **Sentence-Transformers** model to embed text from CVs. The embeddings are then stored in a Qdrant database for efficient retrieval and comparison. The matching process is based on cosine similarity to match candidates with job descriptions.

## Requirements

### Hardware/Software Requirements:
- **Python 3.7+**
- **Streamlit** (for the web interface)
- **Sentence-Transformers** (for embedding generation)
- **Qdrant** (for storing and querying embeddings)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **pdfplumber** (for extracting text from PDFs)

### Computational Power:
- **GPU** recommended for faster embedding generation, especially if dealing with large datasets.
- The model can run on a regular CPU, but performance will be slower for large CV collections.

### How to Run the Project:

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/cv-hiring-system.git
   ```

2. **Install Dependencies**:
   Install the required libraries by running the following command in your terminal:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   Navigate to the project directory and run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. **Upload CVs**:
   Once the app is running, you can upload CVs in PDF format. The system will extract the relevant information and store it in the Qdrant database.

5. **Match with Job Description**:
   Enter a job description in the provided input box, and the system will find the top candidates that best match the job requirements.