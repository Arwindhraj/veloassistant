from flask import Flask, request, jsonify
import json
import faiss
import torch
import PyPDF2
import numpy as np
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel

genai.configure(api_key="AIzaSyDaqVHWmp2hw0bFT3syrf5oJb3v3VVdd88")
model = genai.GenerativeModel("gemini-1.5-flash")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

chat = model.start_chat(history=[])


app = Flask(__name__)

def json_details():
    file_path = 'app/issue/selections.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    if isinstance(data, list):
        return data[-1]


@app.route('/questions', methods=['POST'])
def questions():
    details = json_details()

    vehicle_type = details['vehicleType']
    vehicle_fuel_type = details['fuelType']
    vehicle_comp = details['brand']
    vehicle_name = details['model']
    vehicle_year = details['year']

    comp_name = f"{vehicle_comp}{vehicle_name}{vehicle_year}_{vehicle_fuel_type}"

    instruction = f"""The following are questions related to vehicle repair for the vehicle type {vehicle_type}, 
    vehicle company {vehicle_comp}, vehicle name {vehicle_name}, and fuel type {vehicle_fuel_type}.
    Please ask one question at a time, progressively linking them to previous answers."""
    chat_history = []
    summary = f"{details} = "

    while True:
        response = chat.send_message(instruction)
        bot_question = response.text.strip()

        user_answer = request.json.get('user_answer', '').strip()
        if user_answer.lower() == 'stop':
            break

        chat_history.append(f"Bot: {bot_question}")
        chat_history.append(f"You: {user_answer}")

        instruction += f" Previous user response: {user_answer}"
        summary += f", bot: {bot_question}, user: {user_answer} "

    summary += " ."
    ai_summary_instruction = f"Generate a single-line summary based on the following conversation: {summary}"
    ai_response = chat.send_message(ai_summary_instruction)
    query = ai_response.text.strip()

    return jsonify({"query": query, "comp_name": comp_name})


def extract_text_from_pdf(comp_name):
    reader = PyPDF2.PdfReader(comp_name)
    # reader = PyPDF2.PdfReader(f"pds/{comp_name}.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

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

@app.route('/generate_response', methods=['POST'])
def generate_response():
    query = request.json.get('query')
    comp_name = 'D:/Project_Files/VehicleLLM/Vehicle_Fault_DataCollection.pdf'

    text = extract_text_from_pdf(comp_name)
    chunks = split_text(text)
    embeddings = embed_text_chunks(chunks)
    index = create_faiss_index(embeddings)

    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = "\n".join(relevant_chunks)
    input_text = f"Context:\n{context}\n\nQuery: {query}"
    response = model.generate_content(input_text)

    return jsonify({"response": response.text.strip()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)