# atapy-ocr

## API
POST : http://127.0.0.1:8013/aiatapy  
Body : {  
  "imgsrc": "https://eng.m.fontke.com/d/file/image/91/dc/91bdc776ab2efbb5f94acb5169d2c8ea.png",  
  "aimethod" : 0  
}  

aimethod = 0 >> use pyinterect&easyocr  
aimethod = 1 >> use pyinterect&easyocr&wordcorrection  

uvicorn main_aiatapy:app --reload --port 8001  


## Streamlit
test_ocr_aiatapy.py  
https://atapy-ocr.streamlit.app/  
