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

class PDFChunk:
    def __init__(self, text, page_num, images=None):
        self.text = text
        self.page_num = page_num
        self.images = images if images else []

def extract_text_and_images_from_pdf(pdf_path):
    chunks = []
    current_chunk = ""
    current_page = 0
    chunk_images = []
    
    # Open PDF with PyMuPDF (fitz)
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text()
        
        # Extract images
        image_list = page.get_images()
        page_images = []
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                page_images.append(image)
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {e}")
                
        # Create chunk with both text and images
        chunks.append(PDFChunk(text, page_num, page_images))
    
    doc.close()
    return chunks

def split_text(pdf_chunks, chunk_size=512):
    processed_chunks = []
    
    for pdf_chunk in pdf_chunks:
        words = pdf_chunk.text.split()
        text_chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        # Create new PDFChunk objects for each text chunk, maintaining page number and images
        for text in text_chunks:
            processed_chunks.append(PDFChunk(text, pdf_chunk.page_num, pdf_chunk.images))
    
    return processed_chunks

def embed_text_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk.text, return_tensors="pt", padding=True, truncation=True)
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
    query_embedding = embed_text_chunks([PDFChunk(query, 0)])[0].reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def generate_response(query, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    
    # Create context from text of relevant chunks
    context = "\n".join([chunk.text for chunk in relevant_chunks])
    input_text = f"Context:\n{context}\n\nQuery: {query}"
    response = model.generate_content(input_text)
    
    return response, relevant_chunks

def display_chunk_with_images(chunk):
    print(f"\nText from page {chunk.page_num + 1}:")
    print(chunk.text)
    
    if chunk.images:
        print(f"\nFound {len(chunk.images)} images on this page:")
        for i, image in enumerate(chunk.images):
            print(f"Image {i + 1} dimensions: {image.size}")
            # Display the image (if in a notebook environment)
            image.show()

# Main execution
pdf_path = "D:/Project_Files/VehicleLLM/SAFARI_STORME_OSB_1.20-FINAL_FEB_2017.pdf"
pdf_chunks = extract_text_and_images_from_pdf(pdf_path)
processed_chunks = split_text(pdf_chunks)
embeddings = embed_text_chunks(processed_chunks)
index = create_faiss_index(embeddings)

query = "DOORS LOCKING AND UNLOCKING in car"
response, relevant_chunks = generate_response(query, index, processed_chunks)

# Print response and display relevant chunks with images
print("Generated Response:")
print(response.text.strip())
print("\nRelevant sections from the PDF with images:")
for chunk in relevant_chunks:
    display_chunk_with_images(chunk)