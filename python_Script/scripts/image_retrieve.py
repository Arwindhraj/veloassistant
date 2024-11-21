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

genai.configure(api_key="AIzaSyCdcHN1Trcc_r7SGe5ZMI8rLvtuiMqH8VE")
model = genai.GenerativeModel("gemini-pro")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

class PDFChunk:
    def __init__(self, text, page_num, images=None):
        self.text = text
        self.page_num = page_num
        self.images = images if images else []

def extract_text_and_images_from_pdf(pdf_path):
    chunks = []

    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]

        text = page.get_text()

        image_list = page.get_images()
        page_images = []
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                image = Image.open(io.BytesIO(image_bytes))
                page_images.append(image)
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {e}")

        chunks.append(PDFChunk(text, page_num, page_images))
    
    doc.close()
    return chunks

def split_text_chunks(pdf_chunks, chunk_size=512):
    processed_chunks = []
    
    for pdf_chunk in pdf_chunks:
        words = pdf_chunk.text.split()
        text_chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

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

def retrieve_relevant_image(query, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    for chunk in relevant_chunks:
        if chunk.images:
            return chunk.images[1], chunk
    return None, None  

def display_chunk_with_images(chunk):
    print(f"\nText from page {chunk.page_num + 1}:")
    print(chunk.text)
    
    if chunk.images:
        print(f"\nFound {len(chunk.images)} images on this page:")
        for i, image in enumerate(chunk.images):
            print(f"Image {i + 1} dimensions: {image.size}")
            image.show()

if __name__ == "__main__":
    pdf_path = r"C:\Users\Kishore\Desktop\Crawfords-Auto-Repair-Guide.pdf"
    pdf_chunks = extract_text_and_images_from_pdf(pdf_path)
    processed_chunks = split_text_chunks(pdf_chunks)
    embeddings = embed_text_chunks(processed_chunks)
    index = create_faiss_index(embeddings)

    query = "brake"
    relevant_image, chunk_with_image = retrieve_relevant_image(query, index, processed_chunks)

    if chunk_with_image:
        print("\nRelevant text section:")
        print(chunk_with_image.text)

    if relevant_image:
        print("\nDisplaying the most relevant image:")
        relevant_image.show()
    else:
        print("\nNo relevant image found.")
