
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

from fastapi import FastAPI
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

def aiocr(reader,img_path,iscorrection):
    def is_base64_image(encoded_text):
        try:
            decoded_data = base64.b64decode(encoded_text)
            if isinstance(decoded_data, bytes):
                return True
            else:
                return False
        except Exception as e:
            return False
    
    def save_base64_to_jpg(base64_data):
        try:
            decoded_data = base64.b64decode(base64_data)
            image_stream = BytesIO(decoded_data)
            image = Image.open(image_stream)
            image.save('current_img.jpg', format='JPEG')
        except Exception as e:
            print(f"Error: {e}")
        
    def loadimgfromurl(url):
        # url = 'https://images.squarespace-cdn.com/content/v1/54981918e4b0bff4592d6daa/1489005122989-X3H9A1RGI08CPC1QFCST/fee.jpg?format=2500w'
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save("current_img.jpg")
    
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
        
    def ocr_tesseract(img_path):
        # img_path = 'imgs/191984.jpg'
        im = Image.open(img_path)
        
        im = im.convert("L") #แปลงให้เป็นภาพขาวดำ
        # pytesseract.image_to_string(im, lang='tha+eng',preserve_interword_spaces=1)
        custom_config = r'-l tha+eng --dpi 300 --oem 3 --psm 4'  # OCR Engine Mode 3, Page Segmentation Mode 6
        text = pytesseract.image_to_string(im, config=custom_config)
        return text
        
    def ocr_easyocr(reader,path):
        result = reader.readtext(path,detail = 0)
        return '\n'.join(result)
        
    def correction(result):
        result = result.split('\n')
        R = ''
        for r in result:
            w = word_tokenize(r, engine="newmm")
            # print(w)
            S = []
            for ww in w:
                # print(ww)
                if len(ww) > 2:
                    s = spell(ww,  engine="pn")
                    S.append(s[0])
                else:
                    S.append(ww)
            R += ''.join(S) + '\n'
            # print(''.join(S))
        return R
    
    start = time.time()

    #check is url
    if is_valid_url(img_path):
        print('is url')
        loadimgfromurl(img_path)
        img_path = 'current_img.jpg'
    #check is base64
    elif is_base64_image(img_path):
        print('is base64')
        save_base64_to_jpg(img_path)
        img_path = 'current_img.jpg'
        
    #check text in image by tesseract
    text_tesseract = ocr_tesseract(img_path)

    tesseract_time = time.time()
    
    #easyocr
    text_easyocr = None
    if text_tesseract:
        text_easyocr = ocr_easyocr(reader,img_path)
        # print(text_easyocr)

    easyocr_time = time.time()

    #correction
    if iscorrection and text_easyocr:
        text_easyocr = correction(text_easyocr)

    correction_time = time.time()

    print('tesseract_time',tesseract_time - start)
    print('easyocr_time',easyocr_time - tesseract_time)
    print('correction_time',correction_time - easyocr_time)
    print('Total time',time.time()-start)
        
    return text_easyocr


reader = easyocr.Reader(['th','en'])

iscorrection = False
imgsrc = 'https://eng.m.fontke.com/d/file/image/91/dc/91bdc776ab2efbb5f94acb5169d2c8ea.png'
text_ocr = aiocr(reader,imgsrc,iscorrection)

print('text_ocr:',text_ocr)


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




