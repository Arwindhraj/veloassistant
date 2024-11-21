import io
import os
import fitz
import torch
import faiss
import PyPDF2
import numpy as np
from PIL import Image
import google.generativeai as genai
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

yt_api_key = "AIzaSyAPxSWlIH4LHjPqDZswoBAXEWLyRVAXyuU"
genai.configure(api_key="AIzaSyDaqVHWmp2hw0bFT3syrf5oJb3v3VVdd88")
model = genai.GenerativeModel("gemini-1.5-flash")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def search_youtube(api_key, query, max_results):
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.search().list(
        part='snippet',
        q=query,
        maxResults=max_results
    )
    response = request.execute()

    video_urls = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_urls.append(video_url)

    return video_urls

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
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

def generate_response(query, index, chunks, image):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = "\n".join(relevant_chunks)
    input_text = f"Context:\n{context}\n\nQuery: {query}"
    if image != 0:
        response = model.generate_content([image,"\n\n",input_text])
        return response
    else:
        response = model.generate_content(input_text)
    return response


pdf_path = "D:/Project_Files/VehicleLLM/[TATA] NEXON Owners Manual.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = split_text(text)
embeddings = embed_text_chunks(chunks)
index = create_faiss_index(embeddings)

query = "what are the DASHBOARD AND FEATURES in tata nexon"
image = 0
video = 0
maxResults = 5
response = generate_response(query, index, chunks, image)
print(response.text.strip())

if video !=0:
    video_urls = search_youtube(yt_api_key, query, maxResults)
    for i in range(maxResults):
        print(video_urls[i])