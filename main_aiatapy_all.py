
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

from face_recognition import FaceRecognition
import cv2
import json

# def aiocr(reader,fr,img_path,aimethod):
def aiocr(reader,fr,img_path,isocr,isword_correction,isface_recognition,resize_img,isageface):

    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes

    def ageFace(frame):
        resultImg,faceBoxes=highlightFace(faceNet,frame)

        age_face = {}

        if not faceBoxes:
            print("No face detected")
        else:
            print("detected faces :",faceBoxes)

        for i,faceBox in enumerate(faceBoxes):
            # print(faceBox)
            face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            # print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            # print(f'Age: {age[1:-1]} years')

            age_face[i] = {
                'gender' : gender,
                'age' : age[1:-1],
                'faceBox' : faceBox

            }

        return age_face

    def resize_image_with_target_size(input_path, output_path, target_size_kb):
        image = cv2.imread(input_path)
        quality = 95
        _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        current_size_kb = len(encoded_image.tobytes()) / 1024
        while current_size_kb > target_size_kb and quality > 0:
            quality -= 5
            _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            current_size_kb = len(encoded_image.tobytes()) / 1024

        resized_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        cv2.imwrite(output_path, resized_image)

    def aiface(fr,img):
        img =  cv2.imread(img)
        result = fr.predict(img, threshold=0.3)
        R = []
        for r in result['predictions']:
            if r['person'] in class_dict.keys():
                r['person_name'] = class_dict[r['person']]
            else:
                r['person_name'] = 'unknown'
            R.append(r)

        return R

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
            S = []
            for ww in w:
                if len(ww) > 2:
                    s = spell(ww,  engine="pn")
                    S.append(s[0])
                else:
                    S.append(ww)
            R += ''.join(S) + '\n'
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

    if resize_img != -1:
        resize_image_with_target_size(img_path, img_path, resize_img)

    text_easyocr = None
    if isocr:
        
        #check text in image by tesseract
        text_tesseract = ocr_tesseract(img_path)
        print('text_tesseract',text_tesseract)
        tesseract_time = time.time()-start
        start = time.time()

        #easyocr
        if text_tesseract:
            text_easyocr = ocr_easyocr(reader,img_path)
            print('text_easyocr',text_easyocr)

        easyocr_time = time.time() - start
        start = time.time()

        #word correction
        if isword_correction and text_easyocr:
            text_easyocr = correction(text_easyocr)

        correction_time = time.time() - start

        print('tesseract_time',tesseract_time)
        print('easyocr_time',easyocr_time)
        print('correction_time',correction_time)
        print('Total time',time.time()-start)
    
    #face recognition
    face_recognition = None
    if isface_recognition:
        start_time = time.time()
        face_recognition = aiface(fr,img_path)
        face_recognition_time = time.time() - start_time

        print('face_recognition_time',face_recognition_time)
        print('face_recognition_time',face_recognition_time)
        print('face_recognition',face_recognition)

    #age face
    if isageface:
        frame = cv2.imread(img_path)
        result_age_face = ageFace(frame)


    return text_easyocr,face_recognition,result_age_face


CLASS_DICT_PATH = 'class_dict.json' 
# TEST_IMAGE_FILE = "https://s.isanook.com/fi/0/fp/397/1988417/tagline-template-update-april.jpg"  #"data/train/class2/img1.jpg" #"./Datasets/Test2/"
MODEL_PATH = "model_v1.pkl"
CONFIDENCE = 0.7
with open(CLASS_DICT_PATH, 'r') as file:
    class_dict = json.load(file)


#load model face
fr = FaceRecognition()
fr.load(MODEL_PATH)
#load model easyocr
reader = easyocr.Reader(['th','en'])

#prepare agedace model
faceProto="model/opencv_face_detector.pbtxt"
faceModel="model/opencv_face_detector_uint8.pb"
ageProto="model/age_deploy.prototxt"
ageModel="model/age_net.caffemodel"
genderProto="model/gender_deploy.prototxt"
genderModel="model/gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
padding=20
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


app = FastAPI()

class Ai(BaseModel):
    imgsrc: str
    # aimethod: int = None
    isocr: bool = True
    isword_correction: bool = False
    isface_recognition: bool = True
    resize_img: int = 100
    isageface: bool = True


@app.post("/aiatapy/")
async def ai_img(ai: Ai):

    text_ocr,face_recognition,result_age_face = aiocr(reader,fr,ai.imgsrc,ai.isocr,ai.isword_correction,ai.isface_recognition,ai.resize_img,ai.isageface)

    return {
        "text_ocr":text_ocr,
        'face_recognition':face_recognition,
        'age_face' : result_age_face
    }

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)




