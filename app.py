import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify, render_template, session, Response
from flask_session import Session

from embed import embed, embed_csv
from query import query
from get_vector_db import get_vector_db
from multiprocessing import active_children

from embed import save_file
from query import LLM_MODEL
from langchain_ollama import ChatOllama

import pandas as pd

import re
import time  # Simulating progress for demonstration
import traceback


TEMP_FOLDER = os.getenv("TEMP_FOLDER", "./_temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)
first_request = True

app = Flask(__name__)

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data in files
app.config['SESSION_FILE_DIR'] = './flask_session'  # Path for session files
app.config['SESSION_PERMANENT'] = False  # Sessions expire when the browser closes
app.config['SECRET_KEY'] = 'supersecretkey'  # Required for signing cookies

# Initialize server-side sessions
Session(app)


@app.route('/embed', methods=['POST'])
def route_embed():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    file_extension = file.filename.rsplit('.', 1)[1].lower()

    # Handle PDF files
    if file_extension == 'pdf':
        embedded = embed(file)
        if embedded:
            return jsonify({'message': 'PDF embedded successfully'}), 200

    # Handle CSV files
    elif file_extension == 'csv':
        embedded = embed_csv(file)
        if embedded:
            return jsonify({'message': 'CSV embedded successfully'}), 200

    return jsonify({'error': 'Unsupported file type'}), 400


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        result = embed(file)
        if result.get('success'):
            return jsonify({"message": f"Embedded successfully! {result.get('num_chunks')} chunks added."}), 200
        else:
            return jsonify({"error": result.get('error', 'Unknown error')}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    session.clear()  # Clear session data
    return jsonify({"message": "Session cleared"}), 200

@app.route('/list_pdfs', methods=['GET'])
def list_pdfs():
    db = get_vector_db()
    # Query all unique filenames in the metadata
    # Retrieve all documents in the collection
    collection_data = db._collection.get()  # Fetch all data from the collection
    metadata_list = collection_data.get('metadatas', [])

    # Extract filenames from metadata
    filenames = {meta.get('filename', 'Unknown') for meta in metadata_list if 'filename' in meta}
    return jsonify(list(filenames))


@app.before_request
def before_first_request():
    global first_request
    if first_request:
        # do your thing like create_db or whataver
        first_request = False


@app.teardown_appcontext
def cleanup_semaphores(exception=None):
    """Clean up any lingering semaphore objects."""
    for child in active_children():
        child.terminate()  # Terminate any active child processes
        child.join()


@app.route('/ask', methods=['POST'])
def ask():
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Retrieve chat history
    if 'chat_history' not in session:
        session['chat_history'] = []

    response = query(query_text)
    if response:
        # Append query and response to chat history
        session['chat_history'].append({"query": query_text, "response": response})
        return jsonify(session['chat_history']), 200
    return jsonify({"error": "Query processing failed"}), 500

@app.route('/chat_history', methods=['GET'])
def chat_history():
    return jsonify(session.get('chat_history', []))


@app.route('/fill_missing', methods=['POST'])
def fill_missing():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the file temporarily
    file_path = save_file(file)
    df = pd.read_csv(file_path)

    # Process rows with missing values
    llm = ChatOllama(model=LLM_MODEL)
    db = get_vector_db()
    retriever = db.as_retriever()

    # Fill missing values
    for index, row in df.iterrows():
        for col in df.columns:
            if str(row[col]).lower() == "missing":  # Check for missing values
                # Query the retriever for context
                query_text = f"Fill the missing value for '{col}'. Put your one word response between two _. Use the context of this row: {row.to_dict()}"
                context_docs = retriever.invoke(query_text)  # Pass query as a string

                # Combine content of retrieved documents
                context = " ".join([doc.page_content for doc in context_docs])

                # Use the LLM to predict the missing value
                response = llm.invoke(f"Context: {context}\n\n{query_text}")  # Input as a string

                # Replace missing value with the response
                inferred_value = response.content.strip()

                # Extract the word between underscores using a regular expression
                match = re.search(r"_(.*?)_", inferred_value)
                if match:
                    inferred_value = match.group(1)  # Extract the word inside underscores
                else:
                    inferred_value = 'Unknown'
                # Clean up the inferred value (optional: strip excess text if needed)
                inferred_value = inferred_value.split('\n')[0].strip()
                df.at[index, col] = inferred_value

    # Save the completed file
    completed_file_path = file_path.replace(".csv", "_completed.csv")
    df.to_csv(completed_file_path, index=False)

    # Return the completed file
    return jsonify({"message": "Missing values filled successfully.", "completed_file": completed_file_path}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

# curl --request POST \
# --url http://localhost:8080/embed \
# --header 'Content-Type: multipart/form-data' \
# --form file=@/Users/urshofmannelizondo/Documents/PhD/ocean4_model_report.pdf

# curl --request POST \
#   --url http://localhost:8080/query \
#   --header 'Content-Type: application/json' \
#   --data '{ "query": "The study performed a sensitivity analysis. Each model parameter was changed by a relative amount. I want to know the amount by which each parameter was changed in relative terms. Be as short in your answer as possible." }'