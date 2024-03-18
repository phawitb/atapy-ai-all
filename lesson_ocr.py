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

from face_recognition import FaceRecognition
import cv2
import json

def loadimgfromurl(url):
    # url = 'https://images.squarespace-cdn.com/content/v1/54981918e4b0bff4592d6daa/1489005122989-X3H9A1RGI08CPC1QFCST/fee.jpg?format=2500w'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save("current_img.jpg")

def ocr_tesseract(img_path):
    # img_path = 'imgs/191984.jpg'
    im = Image.open(img_path)
    
    im = im.convert("L") #แปลงให้เป็นภาพขาวดำ
    # pytesseract.image_to_string(im, lang='tha+eng',preserve_interword_spaces=1)
    custom_config = r'-l tha+eng --dpi 300 --oem 3 --psm 4'  # OCR Engine Mode 3, Page Segmentation Mode 6
    text = pytesseract.image_to_string(im, config=custom_config)
    return text
        
def ocr_easyocr(reader,path):
    def calculate_area(box):
        return (box[2][0] - box[0][0]) * (box[2][1] - box[0][1])
    
    result = reader.readtext(path,detail = 0)
    text_all = '\n'.join(result)

    result = reader.readtext(path)
    result_text = [x[-2] for x in result]
    result_box_area = [calculate_area(x[0]) for x in result]
    result_box_sort_index = [i for i, _ in sorted(enumerate(result_box_area), key=lambda x: x[1], reverse=True)]

    text_list = [x.strip() for x in result_text]
    text_sort = [result_text[x].strip() for x in result_box_sort_index]

    return text_all,text_list,text_sort


reader = easyocr.Reader(['th','en'])

img_url = "https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg"
loadimgfromurl(img_url)

img_path = "current_img.jpg"

text_tesseract = ocr_tesseract(img_path)
print('text_tesseract',text_tesseract)
print("--------------")

text_all,text_list,text_sort = ocr_easyocr(reader,img_path)
print('text_all',text_all)
print('text_list',text_list)
print('text_sort',text_sort)
