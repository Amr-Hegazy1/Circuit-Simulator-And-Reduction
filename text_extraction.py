import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

import copy

import pytesseract

# import easyocr
# reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\emerg\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def extract_values(image_name):



    model = YOLO('text.pt')  # pretrained YOLOv8n model



    results = model(image_name) 
    results_data = results[0].boxes.data

    classes = results[0].names

    boxes = []

    for result in results_data:
        
        class_index = int(result[5])
        
        
        class_name = classes[class_index]
        
        
        if class_name == 'text':
        
            box = {'class':class_name,'cords':[int(result[0]),int(result[1]),int(result[2]),int(result[3])]}
        
            boxes.append(box)
            
            



    # display the image with bounding boxes

    img = cv2.imread(image_name)

    texts = []

    for box in boxes:
        
        if box['class'] == 'text':
            
            x1,y1,x2,y2 = box['cords']
            
            ohm_space = int((x2-x1)/2.25)
            
        
            
            
            
            crop_img = copy.deepcopy(img[y1:y2, x1:(x2 - ohm_space)])
            
            # img = cv2.rectangle(img,(x1,y1),(x2-ohm_space,y2),(0,255,0),2)
            
            
            
            
            
            
            
            # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # crop_img = cv2.resize(crop_img,(0,0),fx=2,fy=2)
            
            # crop_img = cv2.GaussianBlur(crop_img,(5,5),0)
            
            # crop_img = cv2.threshold(crop_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
            # cv2.imshow(f'crop_img{x2}',crop_img)
            
            
            #digit=.,1234567890
            
            
            custom_config = r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789knmpuKM'
            
            text = pytesseract.image_to_string(crop_img,config=custom_config)
            
            
            
            # img = cv2.putText(img,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            text_data = {'text':text,'cords':[x1,y1,x2,y2]}

            texts.append(text_data)
    return texts
        
# cv2.imshow('img',img)

# cv2.waitKey(0)
    



# easyocr_results = reader.readtext('circuit5.png')


# img = cv2.imread('circuit5.png')

# for result in easyocr_results:
        
#     cords = result[0]
    
#     text = result[1]
    
#     x1 = int(cords[0][0])
#     y1 = int(cords[0][1])
#     x2 = int(cords[2][0])
#     y2 = int(cords[2][1])
    
#     img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    
#     img = cv2.putText(img,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
# cv2.imshow('img',img)

# cv2.waitKey(0)

# print(easyocr_results)