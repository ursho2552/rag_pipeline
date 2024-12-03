import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')

# Function to check if the uploaded file is allowed (only PDF files)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'csv'}

# Function to save the uploaded file to the temporary folder
def save_file(file):
    # Save the uploaded file with a secure filename and return the file path
    ct = datetime.now()
    ts = ct.timestamp()
    filename = str(ts) + "_" + secure_filename(file.filename)
    file_path = os.path.join(TEMP_FOLDER, filename)
    file.save(file_path)

    return file_path

# Function to load and split the data from the PDF file
def load_and_split_data(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()

    # New: Improved splitter for better semantic coherence
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced chunk size for better embedding granularity
        chunk_overlap=50  # Increased overlap for continuity
    )
    chunks = text_splitter.split_documents(data)
    return chunks

import pandas as pd

# Function to load CSV data and prepare embeddings
def embed_csv(file):
    try:
        if file.filename != '' and file and allowed_file(file.filename):
            # Save the file temporarily
            file_path = save_file(file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Convert each row to a text chunk with metadata
            rows = []
            for index, row in df.iterrows():
                row_text = ', '.join([f"{col}: {value}" for col, value in row.items()])
                rows.append({"content": row_text, "metadata": {"index": index, "source": file.filename}})

            # Add rows to the database
            db = get_vector_db()
            db.add_documents(rows)

            # Remove the temporary file
            os.remove(file_path)
            return True

    except Exception as e:
        return {'success': False, 'error': str(e)}



def embed(file):
    try:
        if file.filename != '' and file and allowed_file(file.filename):
            file_path = save_file(file)
            chunks = load_and_split_data(file_path)
            db = get_vector_db()

            # Add metadata to each chunk
            metadata = {"filename": file.filename, "timestamp": str(datetime.now())}
            for chunk in chunks:
                chunk.metadata.update(metadata)  # Add metadata to each document chunk

            db.add_documents(chunks)  # Add documents with metadata
            os.remove(file_path)
            return {'success': True, 'num_chunks': len(chunks)}  # Return more meaningful feedback

    except Exception as e:
        return {'success': False, 'error': str(e)}
