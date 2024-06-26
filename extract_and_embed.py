import fitz  # PyMuPDF
import os
import pandas as pd
from openai import OpenAI
import numpy as np

# Initialize OpenAI client
client = OpenAI(api_key="your_api_key")  # Ensure you set your OpenAI API key

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Directory containing PDF files
pdf_directory = 'path_to_your_pdf_directory'  # Replace with your actual directory path

# Read and process PDF files
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
data = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    text = extract_text_from_pdf(pdf_path)
    embedding = get_embedding(text)
    data.append({'file': pdf_file, 'text': text, 'embedding': embedding})

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Save the dataframe with embeddings
df.to_csv('./website-with-embeddings.csv', index=False)
df.to_pickle('./website-with-embeddings.pkl')
