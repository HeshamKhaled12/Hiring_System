# CV Hiring System with Embeddings and Qdrant Integration

## Description

This project provides an automated CV hiring system that processes and analyzes candidate CVs by extracting key information such as name, education, skills, work experience, and certifications. The system embeds the extracted data into vector representations and stores them in a Qdrant database for efficient similarity search and matching with job descriptions. 

The project uses a pre-trained language model to convert CVs into meaningful vector embeddings and leverages the power of Qdrant for fast and scalable retrieval of candidates based on job requirements.

## Model Used

- **Embedding Model**: The project uses a pre-trained transformer-based model (e.g., sentence-transformers) to generate embeddings for the text data extracted from CVs. The embeddings capture the semantic meaning of the text, allowing the system to compare candidates' qualifications to job descriptions.
  
- **Similarity Search**: The embeddings are stored in Qdrant, an open-source vector database, to perform fast nearest neighbor searches based on cosine similarity, helping recruiters find the best-matched candidates efficiently.

## Requirements

### Computation Power
- **GPU (Optional but recommended)**: A GPU is highly recommended for fast embedding generation using models like sentence-transformers. The project can run on a CPU but will be slower for large datasets.
- **RAM**: At least 8GB of RAM to handle multiple CVs and perform embedding generation effectively.
- **Disk Space**: Depending on the number of CVs and the embeddings, sufficient disk space is required to store data and model files.
  
### Software
- **Python 3.8+**
- **Required Libraries**: 
    - `sentence-transformers`
    - `qdrant-client`
    - `numpy`
    - `pdfplumber`
    - `requests` (for API integration if applicable)
    - `json` (for data processing)
    - `torch` (if using models that require PyTorch)

Install dependencies using the following:
```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   cd your-repository-name
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python main.py
   ```

4. The system will process the CVs, generate embeddings, and store them in Qdrant. Afterward, you can match candidates against job descriptions using the API or command line interface.

## Usage

- **Extract and Embed CVs**: Upload CVs in PDF format, and the system will automatically extract key information and generate embeddings.
  
- **Job Description Matching**: Input a job description, and the system will return the top candidates whose CVs are the best match based on the embedded text.
