# RAG Pipeline with Missing Value Inference

This project is a retrieval-augmented generation (RAG) pipeline that combines document embeddings, retrieval, and a language model to enable advanced querying and data completion. It supports PDFs and CSV files, allowing you to upload datasets or documents, query them, and infer missing values in CSV datasets.

This project is based on the work of Nasser Maronie's post "Build Your Own RAG App: A Step-by-Step Guide to Setup LLM locally using Ollama, Python, and ChromaDB" (https://github.com/firstpersoncode/local-rag)

## Features
- Embedding PDFs and CSVs:
    - Process and embed documents into a vector database (Chroma) for later retrieval.
    - Includes metadata for filtering and contextual relevance.
- Query System:
    - Ask questions about uploaded documents or datasets using a natural language query interface.
    - Retrieves relevant chunks and answers based on the embedded context.
- Fill Missing Values in CSVs:
    - Upload a CSV with missing values (e.g., marked as "missing") in certain columns.
    - The system uses context from an uploaded "golden dataset" to infer and fill the missing values.
- Session-Based Chat History:
    - Retains query and response history in the session for continuity during a browsing session.
- Support for Multiple File Types:
    - Handles both PDFs and CSVs dynamically.

## Setup and Installation
1. Prerequisites
    - Python 3.8 or later
    - A modern browser (for frontend interaction)
    - Flask (backend framework)
    - Chroma (vector database)
    - Ollama or a compatible LLM backend
2. Clone the Project

3. Install Dependencies

Install the required Python packages using pip:

```
pip install -r requirements.txt
```

4. Install Ollama (https://ollama.com/download)

Pull your model of choice
```
ollama pull mistral
```

Pull the text embedding model
```
ollama pull nomic-embed-text
```

Then start the ollama server
```
ollama serve
```
Or on mac you can just start ollama on the desktop.


4. Set Up Environment Variables

Create a .env file in the root directory with the following values:

```
TEMP_FOLDER = './_temp'
CHROMA_PATH = 'chroma'
COLLECTION_NAME = 'local-rag'
LLM_MODEL = 'mistral'
TEXT_EMBEDDING_MODEL = 'nomic-embed-text'

```

5. Run the Application

Start the Flask server:

```
python app.py
```
The application will be accessible at http://127.0.0.1:8080.

## Usage
1. Embedding Documents

Navigate to the home page (http://127.0.0.1:8080) and upload a PDF or CSV to embed its content.


2. Query the Embedded Data

Ask questions about the uploaded documents using natural language. The system retrieves relevant chunks and generates responses.


3. Filling Missing Values

Upload a CSV with some columns marked as "missing".

The system fills missing values based on context from an uploaded golden dataset. Note that it may take information from the uploaded pdf if one was embedded before. To avoid this, remove the contents of _temp, chorma and flask_session, then you can only upload the reference dataset and the missing values one to infer the missing values.

## Project Structure
**app.py:**
Main Flask application handling routes and session management. Note that the query for the missing values in the csv file is very specific. In case you want to adapt this, you can change the query on line 159

**embed.py:**
Functions to embed PDFs and CSVs into the Chroma database.

**query.py:**
Handles queries and retrieval from the Chroma database.

**get_vector_db.py:**
Initializes the Chroma database and embedding functions.



