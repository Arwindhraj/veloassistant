import io
import os
import fitz
import torch
import faiss
import PyPDF2
import numpy as np
from PIL import Image
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

genai.configure(api_key="AIzaSyDaqVHWmp2hw0bFT3syrf5oJb3v3VVdd88")
model = genai.GenerativeModel("gemini-1.5-flash")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

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

def generate_response(query, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = "\n".join(relevant_chunks)
    input_text = f"Context:\n{context}\n\nQuery: {query}"
    response = model.generate_content(input_text)
    return response

def extract_images_from_pdf(pdf_path):
    images = []
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    document.close()
    return images

def display_response_with_image(response, images, query):
    print(f"Response for '{query}':\n{response.text.strip()}")
    
    # Display relevant images if available
    if images:
        print("\nRelevant Image(s):")
        for i, image in enumerate(images):
            image.show()  # For Jupyter, or use image.show() for standalone scripts
            if i >= 2:  # Limit display to 3 images
                break
    else:
        print("No relevant images found in the PDF.")

# pdf_path = "D:/Project_Files/VehicleLLM/[TATA] NEXON Owners Manual.pdf"
# images = extract_images_from_pdf(pdf_path)
comp_name = 'D:/Project_Files/VehicleLLM/Vehicle_Fault_DataCollection.pdf'
text = extract_text_from_pdf(comp_name)
chunks = split_text(text)
embeddings = embed_text_chunks(chunks)
index = create_faiss_index(embeddings)
query = 'Overheating Engine what to do?'
# response = generate_response(query1, index, chunks)
# response = model.generate_content(query)
# response = response.text.strip()
# print(response)
# query = "what are the DASHBOARD AND FEATURES in tata nexon"
response = generate_response(query, index, chunks)
response = response.text.strip()
print(type(response))
# display_response_with_image(response, images, query)