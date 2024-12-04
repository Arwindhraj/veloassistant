from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import json
import faiss
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
import numpy as np
import PyPDF2
from google.auth import compute_engine

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure Flask App
app = Flask(__name__)
CORS(app)

# Secret key for session management (must be set to a secure value in production)
app.secret_key = os.urandom(24)

# Configure Generative AI Model
genai.configure(api_key="AIzaSyDaqVHWmp2hw0bFT3syrf5oJb3v3VVdd88")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

# Embedding Model Configuration
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Global Constants
MAX_INTERACTIONS = 5  # Maximum number of back-and-forth interactions

def imgcap(filename):
    filename = str(filename)
    myfile = genai.upload_file(f'app/images/{filename}')
    input_text = "Give a detailed description of the image, should be a summary"
    imgcap = model.generate_content([myfile, "\n\n", input_text])
    return imgcap.text.strip()


def json_details():
    file_path = 'app/issue/selections.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Check if data is a list
    if isinstance(data, list):
        data = data[-1] if data else {}

    selected_data = {
        'vehicleType': data.get('vehicleType'),
        'fuelType': data.get('fuelType'),
        'brand': data.get('brand'),
        'model': data.get('model'),
        'year': data.get('year')
    }

    return selected_data


def generate_vehicle_description(details, image_caption):
    vehicle_type = details['vehicleType']
    vehicle_fuel_type = details['fuelType']
    vehicle_comp = details['brand']
    vehicle_name = details['model']
    vehicle_year = details['year']
    comp_name = f"{vehicle_comp}_{vehicle_name}_{vehicle_year}_{vehicle_fuel_type}"

    instruction = f"""You have been provided with specific details about a vehicle:
                        - Vehicle Type: {vehicle_type}
                        - Company: {vehicle_comp}
                        - Model: {vehicle_name}
                        - Fuel Type: {vehicle_fuel_type}
                        - Year: {vehicle_year}, with the following image description provided:
                        {image_caption}.  

                        IMPORTANT RULES:
                        For the FIRST message:
                        - Start with something related to "hello, what seems to be the issue with your {vehicle_name}?"
                        - DO NOT ask questions about details already known from the above information
                        - Ask only ONE specific question at a time
                        
                        For ALL SUBSEQUENT messages:
                        You MUST follow this EXACT format (including the empty lines and numbers):
                        [Your diagnostic question]

                        1. [First answer option]
                        2. [Second answer option]
                        3. [Third answer option]
                        4. Others: [Describe your specific situation]

                        EXAMPLE - You must follow this exact formatting:
                        What type of sound do you hear when starting the engine?

                        1. Grinding metal sound
                        2. Clicking sound
                        3. Whining sound
                        4. Others: Please describe the specific sound

                        General Rules:
                        - ALWAYS include the numbers 1-4 with the exact format shown above
                        - Options must be answers/statements, not questions
                        - Each option should be short and clear
                        - Always use "4. Others: " as the last option
                        - Progress from basic to more complex diagnostics
                        - Use simple, clear English

                        Begin by asking a precise, targeted question about the vehicle's current problem or condition."""

    return instruction, comp_name


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image.'}), 400

    try:
        image_path = os.path.join('app', 'images', file.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        file.save(image_path)
        filename = file.filename
        image_caption = imgcap(filename)  # Generate image caption

        # Store the image caption in the session to isolate it per user
        session['image_caption'] = image_caption

        return jsonify({'message': 'Image uploaded and processed successfully.', 'image_caption': image_caption}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to process the image.'}), 500


@app.route('/chat', methods=['POST'])
def process_request():
    user_id = request.remote_addr  # You can use session ID or remote IP to track users
    if 'chat_history' not in session:
        session['chat_history'] = []
        session['interaction_count'] = 0

    chat_history = session['chat_history']
    interaction_count = session['interaction_count']
    
    details = json_details()
    image_caption = session.get('image_caption', '')

    instruction, comp_name = generate_vehicle_description(details, image_caption)
    data = request.get_json()
    user_answer = data.get('user_answer', '').strip()

    # If it's the first request (no user answer), generate initial bot question
    if not user_answer:
        response = chat.send_message(instruction)
        bot_question = response.text.strip()

        # Initialize chat history and counter for the session
        chat_history.clear()
        session['interaction_count'] = 0
        chat_history.append({'role': 'bot', 'message': bot_question})
        session['chat_history'] = chat_history

        return jsonify({
            'bot_question': bot_question, 
            'is_first_message': True,
            'interactions_remaining': MAX_INTERACTIONS
        }), 200

    # Add user's answer to chat history
    chat_history.append({'role': 'user', 'message': user_answer})
    session['chat_history'] = chat_history

    # Increment interaction counter
    interaction_count += 1
    session['interaction_count'] = interaction_count

    # Check if we've reached the maximum number of interactions
    if interaction_count >= MAX_INTERACTIONS:
        chat_summary = generate_chat_summary(chat_history)

        folder_path = "python_Script/pdfs"
        pdf_file_path = find_pdf_by_comp_name(comp_name, folder_path)

        if pdf_file_path is None:
            return jsonify({'response': 'PDF not found for the given vehicle.'}), 404

        text = extract_text_from_pdf(pdf_file_path)
        chunks = split_text(text)   
        embeddings = embed_text_chunks(chunks)
        index = create_faiss_index(embeddings)

        final_response = generate_response(chat_summary, index, chunks)
        return jsonify({
            'response': final_response.text.strip(),
            'is_final': True,
            'interactions_remaining': 0
        }), 200

    instruction = f"{instruction} Previous user response: {user_answer}"
    response = chat.send_message(instruction)
    bot_question = response.text.strip()

    chat_history.append({'role': 'bot', 'message': bot_question})
    session['chat_history'] = chat_history

    return jsonify({
        'bot_question': bot_question,
        'is_first_message': False,
        'interactions_remaining': MAX_INTERACTIONS - interaction_count
    }), 200


# Utility functions remain the same
def extract_text_from_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

def generate_chat_summary(chat_history):
    history_text = "\n".join([f"{entry['role']}: {entry['message']}" for entry in chat_history])
    
    summary_prompt = f"""Summarize the following conversation, focusing on the key diagnostic information about the vehicle:

    {history_text}

    Provide a concise summary that highlights:
    - The main vehicle issue or problem discussed
    - Key symptoms or observations
    - Any potential diagnostic insights
    - Recommended next steps"""
    
    summary_response = model.generate_content(summary_prompt)
    return summary_response.text.strip()

def find_pdf_by_comp_name(comp_name, folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().startswith(comp_name.lower()) and filename.endswith(".pdf"):
            return os.path.join(folder_path, filename)
    return None

def split_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = embed_text_chunks([query])[0].reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def generate_response(query, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = "\n".join(relevant_chunks)
    input_text = f"Context:\n{context}\n\nQuery: {query}"
    response = model.generate_content(input_text)
    return response


if __name__ == '__main__':
    app.run(debug=True, port=4000)