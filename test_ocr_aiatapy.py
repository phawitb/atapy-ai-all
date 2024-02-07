
'''
POST : http://127.0.0.1:8013/aiatapy
Body : {
  "imgsrc": "https://eng.m.fontke.com/d/file/image/91/dc/91bdc776ab2efbb5f94acb5169d2c8ea.png",
  "aimethod" : 0
}

aimethod = 0 >> use pyinterect&easyocr
aimethod = 1 >> use pyinterect&easyocr&wordcorrection

uvicorn main_aiatapy:app --reload --port 8001

conda env : yolov5
'''

# from fastapi import FastAPI
from pydantic import BaseModel
import base64
from PIL import Image
import requests
from io import BytesIO
from urllib.parse import urlparse
import easyocr
from pythainlp.spell import spell
from pythainlp.corpus import thai_words
from pythainlp.tokenize import word_tokenize
import pytesseract
import time
import streamlit as st


def reduce_image_size(byte_data, target_size_kb):
    # Open the image from byte data
    image = Image.open(BytesIO(byte_data))

    # Calculate the compression ratio needed to achieve the target size
    current_size_kb = len(byte_data) / 1024.0
    compression_ratio = target_size_kb / current_size_kb

    # Reduce the image size
    new_width = int(image.width * compression_ratio)
    new_height = int(image.height * compression_ratio)
    
    # Use Image.ANTIALIAS as a constant value for antialiasing
    resized_image = image.resize((new_width, new_height))

    # Save the resized image to a byte buffer
    output_buffer = BytesIO()
    resized_image.save(output_buffer, format='JPEG')  # You can change the format as needed (JPEG, PNG, etc.)
    output_byte_data = output_buffer.getvalue()

    return output_byte_data

# def aiocr(reader,img_path,iscorrection):
#     def is_base64_image(encoded_text):
#         try:
#             decoded_data = base64.b64decode(encoded_text)
#             if isinstance(decoded_data, bytes):
#                 return True
#             else:
#                 return False
#         except Exception as e:
#             return False
    
#     def save_base64_to_jpg(base64_data):
#         try:
#             decoded_data = base64.b64decode(base64_data)
#             image_stream = BytesIO(decoded_data)
#             image = Image.open(image_stream)
#             image.save('current_img.jpg', format='JPEG')
#         except Exception as e:
#             print(f"Error: {e}")
        
#     def loadimgfromurl(url):
#         # url = 'https://images.squarespace-cdn.com/content/v1/54981918e4b0bff4592d6daa/1489005122989-X3H9A1RGI08CPC1QFCST/fee.jpg?format=2500w'
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content))
#         img.save("current_img.jpg")
    
#     def is_valid_url(url):
#         try:
#             result = urlparse(url)
#             return all([result.scheme, result.netloc])
#         except ValueError:
#             return False
        
#     def ocr_tesseract(img_path):
#         # img_path = 'imgs/191984.jpg'
#         im = Image.open(img_path)
        
#         im = im.convert("L") #แปลงให้เป็นภาพขาวดำ
#         # pytesseract.image_to_string(im, lang='tha+eng',preserve_interword_spaces=1)
#         custom_config = r'-l tha+eng --dpi 300 --oem 3 --psm 4'  # OCR Engine Mode 3, Page Segmentation Mode 6
#         text = pytesseract.image_to_string(im, config=custom_config)
#         return text
        
#     def ocr_easyocr(reader,path):
#         result = reader.readtext(path,detail = 0)
#         return '\n'.join(result)
        
#     def correction(result):
#         result = result.split('\n')
#         R = ''
#         for r in result:
#             w = word_tokenize(r, engine="newmm")
#             # print(w)
#             S = []
#             for ww in w:
#                 # print(ww)
#                 if len(ww) > 2:
#                     s = spell(ww,  engine="pn")
#                     S.append(s[0])
#                 else:
#                     S.append(ww)
#             R += ''.join(S) + '\n'
#             # print(''.join(S))
#         return R
    
#     start = time.time()

#     #check is url
#     if is_valid_url(img_path):
#         print('is url')
#         loadimgfromurl(img_path)
#         img_path = 'current_img.jpg'
#     #check is base64
#     elif is_base64_image(img_path):
#         print('is base64')
#         save_base64_to_jpg(img_path)
#         img_path = 'current_img.jpg'
        
#     #check text in image by tesseract
#     text_tesseract = ocr_tesseract(img_path)

#     tesseract_time = time.time()
    
#     #easyocr
#     text_easyocr = None
#     if text_tesseract:
#         text_easyocr = ocr_easyocr(reader,img_path)
#         # print(text_easyocr)

#     easyocr_time = time.time()

#     #correction
#     if iscorrection and text_easyocr:
#         text_easyocr = correction(text_easyocr)

#     correction_time = time.time()

#     print('tesseract_time',tesseract_time - start)
#     print('easyocr_time',easyocr_time - tesseract_time)
#     print('correction_time',correction_time - easyocr_time)
#     print('Total time',time.time()-start)
        
#     return text_easyocr

with st.spinner('Load model...'):
    if 'reader' not in st.session_state:
        with st.spinner('load model...'):
            st.session_state.reader = easyocr.Reader(['th','en'])

reader = st.session_state.reader

# reader = easyocr.Reader(['th','en'])
iscorrection = False

img_input = st.text_input('Image URL')

if img_input:

    strat_time = time.time()
    with st.spinner('Wait for it...'):
        # text_ocr = aiocr(reader,img_input,iscorrection)
        result = reader.readtext(img_input,detail = 0)
        text_ocr = '\n'.join(result)

    text_ocr = f'[{round(time.time()-strat_time,2)}sec]{text_ocr}'

    cols = st.columns(2)
    with cols[0]:
        st.image(img_input)
    with cols[1]:
        st.write(text_ocr)

else:
    uploaded_files = st.file_uploader("Choose a Images file", accept_multiple_files=True,type=['png','jpeg','jpg'])
    for uploaded_file in uploaded_files:
        # if uploaded_file:
        bytes_data = uploaded_file.read()

        bytes_data = reduce_image_size(bytes_data, 100)


        strat_time = time.time()
        with st.spinner('OCR..Wait for it...'):
            # text_ocr = aiocr(reader,bytes_data,iscorrection)
            result = reader.readtext(bytes_data,detail = 0)
            text_ocr = '\n'.join(result)

        text_ocr = f'[{round(time.time()-strat_time,2)}sec]{text_ocr}'
        
        cols = st.columns(2)
        with cols[0]:
            st.image(bytes_data)
        with cols[1]:
            st.write(text_ocr)






# app = FastAPI()

# class Ai(BaseModel):
#     imgsrc: str
#     aimethod: int = None

# @app.post("/aiatapy/")
# async def ai_img(ai: Ai):
#     iscorrection = False
#     if ai.aimethod==1:
#         iscorrection = True

#     text_ocr = aiocr(reader,ai.imgsrc,iscorrection)

#     return {
#         # "imgsrc": ai.imgsrc,
#         "aimethod": ai.aimethod,
#         "text_ocr":text_ocr
#     }

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)



