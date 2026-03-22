# Smart Resume Analyzer

Smart Resume Analyzer is an AI-powered system for analyzing, comparing, and ranking multiple resumes using modern NLP techniques. It uses a Retrieval-Augmented Generation (RAG) pipeline to extract relevant information from resumes and generate structured insights based on user queries or job descriptions.

## Overview

The application processes resumes by converting them into semantic embeddings using Sentence Transformers. These embeddings are stored in a FAISS vector index, enabling fast and accurate similarity search. When a query is provided, the system retrieves the most relevant sections from each resume and uses a local Large Language Model (via Ollama) to generate responses, including candidate ranking and reasoning.

## Features

- Upload multiple resumes (PDF, DOCX, TXT)
- Semantic search using embeddings
- Fast retrieval with FAISS
- Cross-resume question answering
- Candidate ranking based on job description
- Interview question generation
- Interactive UI built with Streamlit
- Runs locally using Ollama (no external API required)

## Tech Stack

- Python  
- Streamlit  
- FAISS (Vector Database)  
- Sentence Transformers (`all-MiniLM-L6-v2`)  
- Ollama (LLMs)  
- PyPDF / python-docx  

## How It Works

1. Upload resumes and extract text  
2. Split text into smaller chunks  
3. Convert chunks into embeddings  
4. Store embeddings in FAISS index  
5. Convert user query into embedding  
6. Retrieve most relevant chunks  
7. Generate answer using LLM  

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd smart-resume-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama:
   ```bash
   ollama serve
   ```

4. Pull a model:
   ```bash
   ollama pull qwen2.5:0.8b
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Upload multiple resumes  
- Ask questions across all resumes  
- Provide a job description to rank candidates  
- Generate interview questions for selected candidates  

## Future Improvements

- Candidate scoring system  
- Skill gap analysis  
- Resume summarization  
- Export reports  

## License

This project is for educational and demonstration purposes.
