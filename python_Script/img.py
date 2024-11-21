import google.generativeai as genai
genai.configure(api_key="AIzaSyDaqVHWmp2hw0bFT3syrf5oJb3v3VVdd88")
model = genai.GenerativeModel("gemini-1.5-flash")
def imgcap(filename):

    filename = str(filename)
    myfile = genai.upload_file(f'app/images/{filename}')
    input_text = "Give a detailed description of the image"
    imgcap = model.generate_content([myfile,"\n\n",input_text])
    print(imgcap.text.strip())
    imgDes = imgcap
    return imgcap