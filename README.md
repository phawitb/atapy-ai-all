# atapy-ocr

## API
```
uvicorn main_aiatapy_all:app --reload --port 8001    
uvicorn main_aiatapy_ocr:app --reload --port 8001
```
```
POST : http://127.0.0.1:8013/aiatapy  
Body : {
  "imgsrc": "https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg",
  "isocr": 1,
  "isface_recognition": 1,
  "isword_correction": 0,
  "resize_img": 150,
  "isOnlyBigText": 1
}
```


## Streamlit  
```
test_ocr_aiatapy.py    
https://atapy-ocr.streamlit.app/  
```

## Install
#### pip
```
git clone https://github.com/phawitb/atapy-ai-all.git
pip install -r requirements.txt
conda install conda-forge::tesseract
```
#### conda
```
conda env create -f atapy-ai-all.yml
conda activate atapy-ai-all
```
#### note
```
conda env export -n atapy-ai-all -f atapy-ai-all.yml
```

